using Random
using Distributions
using Statistics
using StatsBase
using Base.Threads
using LoopVectorization
using Distributed
using Einsum

using LaTeXStrings
using PyPlot
using QuadGK
using SpecialFunctions
using GLM
using DataFrames
using CSV
using JLD2
using Distributed

pygui(true)
latex(s) = latexstring(replace(s," " => "\\,\\,"))

function h5(c, r; λ= 2., ν = 2)
    ## c in increasing order
    gap_min = minimum( (c[2:end] .- r[2:end]) .- (c[1:(end-1)] .+ r[1:(end-1)]) )

    if gap_min <= 0
        return 0
    else
        return exp(-λ/(gap_min^ν))   
    end

end

function truncatedMH(xt, tgt_f, lower, upper; rng=nothing, tgt_it=2000, max_it=20000)
    ## xt: initial value
    ## tgt_f: unformalized targeted function (can be untruncated)
    ## lower, upper: lower and upper bound of the truncated density
    ## tgt_it: minimum number of acceptances
    ## max_it: maximum number of MCMC iterations
    if rng === nothing
        rng = MersenneTwister()
    end

    sigma = (upper - lower) / 4
    x_old = xt
    accept = 0
    t = 0

    #samples = Vector{Float64}()

    while (accept < tgt_it) & (t < max_it)
        t += 1
        x_new = rand(rng, truncated(Normal(x_old, sigma), lower=lower, upper=upper), 1)[1]
        q_old_new = pdf(truncated(Normal(x_old, sigma), lower=lower, upper=upper), x_new)
        q_new_old = pdf(truncated(Normal(x_new, sigma), lower=lower, upper=upper), x_old)
        R = (tgt_f(x_new) * q_new_old) / (tgt_f(x_old) * q_old_new)
        threshold = minimum([1, R])
        indicator = rand(rng, Uniform(0, 1), 1)[1]
        if indicator < threshold
            x_old = x_new
            accept += 1
        end
        #append!(samples, x_old)
    end

    #acc_rate = accept / t

    #print("acceptance rate: $(acc_rate) \n")

    return x_old #, samples
end

function map_gz(i, xi, gi, zi; rng=nothing, K, uk_star, ik_star, w, betas, phis, sigmas)
    ## function for updating ui, gi, zi for each sample tuple (i, xi, gi, zi)

    if rng === nothing
        rng = MersenneTwister()
    end

    ustar_x = uk_star[gi][zi]
    istar_x = ik_star[gi][zi]
    beta_x = betas[gi][zi]

    if i == istar_x
        ui = ustar_x
    else
        ui = rand(rng, Uniform(ustar_x, beta_x), 1)[1]
    end

    betas_indx = Dict{Int64,Vector{Int64}}()
    pdf_x = Dict{Int64,Vector{Float64}}()
    for k in 1:K
        indk = findall(p -> p > ui, betas[k])
        betas_indx[k] = indk
        if isempty(indk)
            pdf_x[k] = [0.0]
        else
            pdf_x[k] = pdf.(Normal.(phis[k][indk], sigmas[k][indk]), xi)
        end
    end

    pk = w .* [sum(pdf_x[k]) for k in 1:K]

    if !all(pk .== 0.0)
        gi_new = sample(rng, 1:K, Weights(pk ./ sum(pk)))
    else
        gi_new = gi
    end

    if sum(pdf_x[gi_new]) == 0.0
        zi_new = sample(rng, betas_indx[gi_new])
    else
        zi_new = sample(rng, betas_indx[gi_new], Weights(pdf_x[gi_new] ./ sum(pdf_x[gi_new])))
    end

    return [gi_new, zi_new]

end

function reduce_ϕσ(pair, dict; rng=nothing, sigmas, centers, σ0, int_lb, int_ub, αs, θs)
    ## pair (comp_k, clst_l)
    ## center_k is the center for the k th component
    ## intk_lb, intk_ub are the lower bound and upper bound for the interval of the k th component

    if rng === nothing
        rng = MersenneTwister()
    end

    comp_k, clst_l = pair
    sigma_kl = sigmas[comp_k][clst_l]
    center_k = centers[comp_k]
    intk_lb = int_lb[comp_k]
    intk_ub = int_ub[comp_k]

    x_kl = dict[pair] # all samples in this cluster
    n_kl = length(x_kl)

    ## posterior parameters for ϕₖₗ
    mu_pst = (σ0^2 * sum(x_kl) + center_k * sigma_kl^2) / (σ0^2 * n_kl + sigma_kl^2) # parameter μₖₗ in posterior for ϕₖₗ
    sigma_pst = 1 / sqrt(n_kl / (sigma_kl^2) + 1 / (σ0^2)) # parameter σₖ in posterior for ϕₖₗ

    ## generate new ϕₖₗ
    phi_kl_new = rand(rng, truncated(Normal(mu_pst, sigma_pst), lower=intk_lb, upper=intk_ub), 1)[1]

    ## posterior parameters for σₖₗ
    αs_pst = αs + n_kl / 2
    θs_pst = θs + sum((x_kl .- phi_kl_new) .^ 2) / 2

    ## generate new σₖₗ
    sigma_kl_new = sqrt(rand(rng, InverseGamma(αs_pst, θs_pst), 1)[1])

    return pair, n_kl, phi_kl_new, sigma_kl_new

end

function MDP7(x; inits, h=h5, map_gz=map_gz, reduce_ϕσ=reduce_ϕσ, truncatedMH=truncatedMH,
                 alpha=nothing, γ=1.0, τ=2.0, κ=3.0, θ=1.0, λ=2.0, ν=2, σ0=1., 
                 vs=6., σs=1., burnin = 0, iteration = 500, seed=nothing)
    ## Mxiture of DP with K components ∑ₖwₖDP(γfₖ)       
    ## Using slice sampling 
    ## x: samples 
    ## K: number of clusters
    ## inits: initial value tuple (w, c, r), w,c,r ∈ R^{K×1}, cₖ in increasing order  
    ## h: function for calculating repulsive prior based on current components
    ## map_gz: map function to update component indicator g and cluster indicator z
    ## reduce_ϕσ: reduce function to update cluster parameters ϕ and σ
    ## alpha: vector [α₁,⋯, αₖ] of length K, α₁,⋯,αₖ >0, prior parameters for w~Dirichlet(alpha)
    ## γ: γ ∈ R₊, concentration parameter γ of DPs
    ## τ: τ ∈ R₊, for prior N(0, τ²) of each cₖ
    ## κ: κ ∈ R₊, shape parameter for gamma(κ,θ) of each rₖ
    ## θ: θ ∈ R₊, scale parameter for gamma(κ,θ) of each rₖ
    ## λ: λ ∈ R₊, parameter in h(c,r) = minᵢⱼexp{-λ/max((|cᵢ-cⱼ|-(rᵢ+rⱼ)),0)^ν}
    ## ν: ν ∈ N₊, parameter in h(c,r) = minᵢⱼexp{-λ/max((|cᵢ-cⱼ|-(rᵢ+rⱼ)),0)^ν}
    ## σ0: parameter in truncated normal prior fₖ for ϕₖᵢ, fₖ:ϕₖᵢ~ N(cₖ, σ0²)1{cₖ-rₖ ≤ x ≤ cₖ+rₖ}
    ## vs: parameter in InverseGamma prior gₖ for σₖᵢ², σₖᵢ² ~ InverseGamma(vs/2, σs²vs/2)
    ## σs: parameter in InverseGamma prior gₖ for σₖᵢ², σₖᵢ² ~ InverseGamma(vs/2, σs²vs/2)
    ## seed: random seed to ensure reproducibility
  
  ### global settings
    if seed === nothing
        seed = rand(RandomDevice(), UInt64) # generate random seed
    end       
    rng_glb = MersenneTwister(seed)
  ###
  
  ### check initial values
    w, c, r = inits

    if length(w) != length(c) || length(c) != length(r) 
        error("Initial values are not of the same length.")
    end

    if !all( (c[1:(end-1)] .- c[2:end]) .< 0)
        error("c is not in increasing orde.")
    end

    if !all((c[1:(end-1)] .+ r[1:(end-1)]) .< (c[2:end] .- r[2:end]))
        error("Iₖ's are not disjoint.")
    end

    w = Float64.(w)
    c = Float64.(c)
    r = Float64.(r)
    K = length(w)
  ###

  ### set hyperparameters
    n = length(x) 
    cc = 0. # prior N(cc, τ²) for cₖ
    αs = vs/2 # parameter for InverseGamma(αs, θs)
    θs = σs^2 * vs/2 # parameter for InverseGamma(αs, θs)
    if alpha === nothing
        alpha = 2.5 .* ones(K)
    end
  ###
  
  ### initialization

   ## initializing each variable in the first MCMC iteration
    w_new = w
    c_new = c
    r_new = r
    uh_new = rand(rng_glb, Uniform(0, h(c_new, r_new; λ=λ, ν=ν)), 1)[1]
    int_lb = c_new .- r_new
    int_ub = c_new .+ r_new
    fk_new = map((x, y, z) -> truncated(x, lower=y, upper=z), Normal.(c_new, σ0), int_lb, int_ub) # vector of base measure fₖ's
    g_new = sample(rng_glb, 1:K, Weights(w_new), n) #indicator of DP component gᵢ=k for DP(γfₖ)
    z_new = ones(Int64, n)  #indicator of cluster within the DP_gᵢ for each sample
    nk = countmap(g_new) # number of samples in each component
    betas_new = Dict{Int64,Vector{Float64}}() # the kth item for betas in the kth component
    phis_new = Dict{Int64,Vector{Float64}}() # the kth item for phis in the kth component
    sigmas_new = Dict{Int64,Vector{Float64}}() # the kth item for sigmas in the kth component
    uk_star = Dict{Int64,Vector{Float64}}() # the minimum of u of each beta in each component k
    ik_star = Dict{Int64,Vector{Int64}}() # the indicator of uk_star of each beta in each component k (global index)
    for k in 1:K
        if haskey(nk, k)
            betas_new[k] = rand(rng_glb, Dirichlet([nk[k], γ]), 1)[:, 1]
            uk_star[k] = rand(rng_glb, Beta(1, nk[k]), 1) .* betas_new[k][1]
            indk = findall(x -> x == k, g_new)
            ik_star[k] = [indk[sample(rng_glb, 1:nk[k])]]
        else
            betas_new[k] = rand(rng_glb, Dirichlet([1, γ]), 1)[:, 1]
        end
        phis_new[k] = rand(rng_glb, fk_new[k], 1)
        sigmas_new[k] = sqrt.(rand(rng_glb, InverseGamma(αs, θs), 1))
    end
   ##

   ## initializing matrices to store MCMC results 
    ncol = iteration - burnin
    c_all = Matrix{Float64}(undef, K, ncol)
    r_all = Matrix{Float64}(undef, K, ncol)
    w_all = Matrix{Float64}(undef, K, ncol) # vector to store w's in each MCMC iteration 
    g_all = Matrix{Int64}(undef, n, ncol) # matrix to store gs in each column
    z_all = Matrix{Int64}(undef, n, ncol) # matrix to store zs in each column
    ind_all = Matrix{Int64}(undef, K, ncol) # starting locations for betas, phis, sigmas in each MCMC iteration for component k
    β_all = Dict{Int64,Vector{Float64}}() # βs for the K components 
    σ_all = Dict{Int64,Vector{Float64}}() # σs for the K components
    ϕ_all = Dict{Int64,Vector{Float64}}() # ϕs for the K components
    for k in 1:K
        β_all[k] = []
        σ_all[k] = []
        ϕ_all[k] = []
    end
   ##

  ###

  ### Gibbs sampling (slice sampler)
    for t in 1:iteration 

      ### update gᵢ and zᵢ
        u_star = minimum(minimum.(values(uk_star))) # find u_star for all components

       ## instantiate beta, phis, sigmas based on u_star
       ## multi-threading: not recommended when n is small
        # rngs = [MersenneTwister(seed + t) for t in 1:Threads.nthreads()]
        # @threads for k in 1:K 
        #     local_rng = rngs[Threads.threadid()] 
        #     while betas_new[k][end] >= u_star
        #         nu = rand(local_rng, Beta(1, γ), 1)[1]
        #         append!(betas_new[k], betas_new[k][end] * (1-nu))
        #         betas_new[k][end-1] *= nu
        #         append!(phis_new[k], rand(local_rng, fk_new[k], 1))
        #         append!(sigmas_new[k], sqrt.(rand(local_rng, InverseGamma(αs, θs), 1)))
        #     end
        # end
       ##       
        for k in 1:K
            while betas_new[k][end] >= u_star
                nu = rand(rng_glb, Beta(1, γ), 1)[1]
                append!(betas_new[k], betas_new[k][end] * (1 - nu))
                betas_new[k][end-1] *= nu
                append!(phis_new[k], rand(rng_glb, fk_new[k], 1))
                append!(sigmas_new[k], sqrt.(rand(rng_glb, InverseGamma(αs, θs), 1)))
            end
        end
       ##
        
       ## update gi, zi for each sample xi
        if n < 10^6
            # when sample size n < O(10⁶), use normal for-loops
            results = Matrix{Int64}(undef, n, 2)
            for i in 1:n
                #print("g_new[i]: $(g_new[i])\n")
                #print("z_new[i]: $(z_new[i])\n")
                #print("ik_star[g_new[i]]: $(ik_star[g_new[i]])\n")
                #print("uk_star[g_new[i]]: $(uk_star[g_new[i]])\n")
                result_gz = map_gz(i, x[i], g_new[i], z_new[i]; rng=rng_glb, K=K, uk_star=uk_star, ik_star=ik_star,
                                    w=w_new, betas=betas_new, phis=phis_new, sigmas=sigmas_new)
                results[i, :] = result_gz
            end
            g_new = results[:, 1]
            z_new = results[:, 2]
        else
            # when sample size n > O(10⁶), use multi-threading to optimize performance
            results = Matrix{Int64}(undef, n, 2)
            rngs = [MersenneTwister(seed + t) for t in 1:Threads.nthreads()]
            @threads for i in 1:n
                local_rng = rngs[Threads.threadid()]
                result_gz = map_gz(i, x[i], g_new[i], z_new[i]; rng=local_rng, K=K, uk_star=uk_star, ik_star=ik_star, 
                                    w=w_new, betas=betas_new, phis=phis_new, sigmas=sigmas_new)
                results[i, :] = result_gz
            end
            g_new = results[:,1]
            z_new = results[:,2]
        end
       ##
      ###
                   
      ### update weights w
        nk = countmap(g_new)
        alpha_pst = Vector{Float64}(undef, K)
        for k in 1:K
            if haskey(nk, k)
                alpha_pst[k] = alpha[k] + nk[k]
            else
                alpha_pst[k] = alpha[k]
            end
        end
        w_new = rand(rng_glb, Dirichlet(alpha_pst), 1)[:,1]
      ###
    
      ### middle steps
       ### delete idle clusters, update z_new accordingly, update ϕs, σs, and βs for empty components 
        for k in 1:K
            if haskey(nk, k)
                indk = findall(x -> x == k, g_new)
                zks = copy(z_new[indk])
                zks_seq = sort(unique(zks))
                delete_i = filter!(x -> x ∉ zks_seq, collect(eachindex(phis_new[k])))
                if !isempty(delete_i)
                    betas_new[k][end] += sum(betas_new[k][delete_i])
                    deleteat!(betas_new[k], delete_i)
                    deleteat!(phis_new[k], delete_i)
                    deleteat!(sigmas_new[k], delete_i)
                end

                for (i, val) in enumerate(zks_seq)
                    if i != val
                        ind_val = findall(x -> x == val, zks)
                        zks[ind_val] .= i
                    end
                end

                z_new[indk] = zks

            else
                ## update β, ϕ and σ for empty components
                phis_new[k] = rand(rng_glb, fk_new[k], 1)
                sigmas_new[k] = sqrt.(rand(rng_glb, InverseGamma(αs, θs), 1))
                betas_new[k] = rand(rng_glb, Dirichlet([1, γ]), 1)[:, 1]
            end

        end

       ### create a dictionary with key (gi,zi) and value xi's ∈ component gi, cluster zi
        clst_dict = Dict{Tuple{Int64,Int64},Vector{Float64}}()
        for i in 1:n
            key = (g_new[i], z_new[i])
            if !haskey(clst_dict, key)
                clst_dict[key] = [x[i]]
            else
                append!(clst_dict[key], x[i])
            end
        end
      ###
    
      ### update ϕₖs,σₖs for non-empty clusters
        nkl_dict = Dict{Tuple{Int64,Int64}, Int64}()
        for key in keys(clst_dict)

            #### get new parameters for each cluster
            pair, n_kl, phi_kl_new, sigma_kl_new = reduce_ϕσ(key, clst_dict; rng=rng_glb, sigmas=sigmas_new, centers=c_new,
                σ0=σ0, int_lb=int_lb, int_ub=int_ub, αs=αs, θs=θs)
            comp_k, clst_l = pair
            nkl_dict[pair] = n_kl

            ## update phis and sigmas
            phis_new[comp_k][clst_l] = phi_kl_new
            sigmas_new[comp_k][clst_l] = sigma_kl_new

        end
      ###
    
      ### update βs for all clusters, update uk_star and ik_star for non-empty clusters
      ### update ϕₖs,σₖs for empty clusters

        uk_star = Dict{Int64,Vector{Float64}}()
        ik_star = Dict{Int64,Vector{Int64}}()

        for k in 1:K
            if haskey(nk, k)
                alphak_pst = [nkl_dict[(k, l)] for l in eachindex(phis_new[k])]
                append!(alphak_pst, γ)
                betas_new[k] = rand(rng_glb, Dirichlet(alphak_pst), 1)[:, 1]
                uk_star[k] = [rand(rng_glb, f, 1)[1] for f in Beta.(1, alphak_pst[1:(end-1)])] .* betas_new[k][1:(end-1)]

                ik_star[k] = []
                for l in eachindex(phis_new[k])
                    ind_kl = findall(x -> x == 1, (g_new .== k) .&& (z_new .== l))
                    append!(ik_star[k], ind_kl[sample(rng_glb, 1:alphak_pst[l])])
                end
            
            end
        end
      ###

      ### update c
        for k in 1:K
            if haskey(nk, k)
                phiks = phis_new[k]
                n_phik = length(phiks)
                phik_min = minimum(phiks)
                phik_max = maximum(phiks)

                ### posterior parameters
                if k == K
                    ub = phik_min + r_new[k]
                else
                    ub = min((phik_min + r_new[k]), (c_new[k+1] - r_new[k+1] - r_new[k] - (λ / (-log(uh_new)))^(1 / ν)))
                end

                if k == 1
                    lb = phik_max - r_new[k]
                else
                    lb = max((c_new[k-1] + r_new[k-1] + r_new[k] + (λ / (-log(uh_new)))^(1 / ν)), phik_max - r_new[k])
                end

                mu_c = (cc * σ0^2 + τ^2 * sum(phiks)) / (σ0^2 + n_phik * τ^2)
                sigma_c = 1 / sqrt((1 / (τ^2)) + (n_phik / (σ0^2)))

                c_new[k] = rand(rng_glb, truncated(Normal(mu_c, sigma_c), lower=lb, upper=ub), 1)[1]

            else
                if k == K
                    lb = c_new[k-1] + r_new[k-1] + r_new[k] + (λ / (-log(uh_new)))^(1 / ν)
                    c_new[k] = rand(truncated(Normal(cc, τ), lower=lb), 1)[1]
                elseif k == 1
                    ub = c_new[k+1] - r_new[k+1] - r_new[k] - (λ / (-log(uh_new)))^(1 / ν)
                    c_new[k] = rand(truncated(Normal(cc, τ), upper=ub), 1)[1]
                else
                    lb = c_new[k-1] + r_new[k-1] + r_new[k] + (λ / (-log(uh_new)))^(1 / ν)
                    ub = c_new[k+1] - r_new[k+1] - r_new[k] - (λ / (-log(uh_new)))^(1 / ν)
                    c_new[k] = rand(truncated(Normal(cc, τ), lower=lb, upper=ub), 1)[1]
                end
            end
        end
      ###

      ### update r
        for k in 1:K
            if haskey(nk, k)
                phiks = phis_new[k]
                n_phik = length(phiks)
                phik_min = minimum(phiks)
                phik_max = maximum(phiks)

                ## posterior parameters for r
                tgt_f = x -> pdf.(Gamma(κ, θ), x) .* (1 ./ ((1 .- 2 .* cdf.(Normal(0, σ0), -x)) .^ n_phik))
                if k == K
                    lb = max(abs(phik_min - c_new[k]), abs(phik_max - c_new[k]))
                    ub = c_new[k] - c_new[k-1] - r_new[k-1] - (λ / (-log(uh_new)))^(1 / ν)
                elseif k == 1
                    lb = max(abs(phik_min - c_new[k]), abs(phik_max - c_new[k]))
                    ub = c_new[k+1] - c_new[k] - r_new[k+1] - (λ / (-log(uh_new)))^(1 / ν)
                else
                    lb = max(abs(phik_min - c_new[k]), abs(phik_max - c_new[k]))
                    ub = min(c_new[k] - c_new[k-1] - r_new[k-1] - (λ / (-log(uh_new)))^(1 / ν),
                        c_new[k+1] - c_new[k] - r_new[k+1] - (λ / (-log(uh_new)))^(1 / ν))
                end
                r_new[k] = truncatedMH((ub + lb) / 2, tgt_f, lb, ub; rng=rng_glb)
            else
                if k == K
                    lb = 0
                    ub = c_new[k] - c_new[k-1] - r_new[k-1] - (λ / (-log(uh_new)))^(1 / ν)
                elseif k == 1
                    lb = 0
                    ub = c_new[k+1] - c_new[k] - r_new[k+1] - (λ / (-log(uh_new)))^(1 / ν)
                else
                    lb = 0
                    ub = min(c_new[k] - c_new[k-1] - r_new[k-1] - (λ / (-log(uh_new)))^(1 / ν),
                        c_new[k+1] - c_new[k] - r_new[k+1] - (λ / (-log(uh_new)))^(1 / ν))
                end
                r_new[k] = rand(truncated(Gamma(κ, θ), lower=lb, upper=ub), 1)[1]
            end
        end

      ###

      ### update uh
        uh_new = rand(rng_glb, Uniform(0, h(c_new, r_new; λ=λ, ν=ν)), 1)[1]
      ###
    
      ### update int_lb, int_ub and fk_new
        int_lb = c_new .- r_new
        int_ub = c_new .+ r_new
        fk_new = map((x, y, z) -> truncated(x, lower=y, upper=z), Normal.(c_new, σ0), int_lb, int_ub) # vector of base measure fₖ's
      ### 
        
      ### store values
        if t > burnin
            icol = t - burnin
            w_all[:, icol] = w_new
            c_all[:, icol] = c_new
            r_all[:, icol] = r_new
            g_all[:,icol] = g_new
            z_all[:,icol] = z_new
            for k in 1:K
                ind_all[k, icol] = length(β_all[k]) + 1
                append!(β_all[k], betas_new[k][1:(end-1)])
                append!(ϕ_all[k], phis_new[k])
                append!(σ_all[k], sigmas_new[k])
            end
 
        end
      ###

    end
  ###

    return w_all, c_all, r_all, g_all, z_all, β_all, ϕ_all, σ_all, ind_all

end

