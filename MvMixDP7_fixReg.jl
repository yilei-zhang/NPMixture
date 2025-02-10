using Random
using Distributions
using Statistics
using StatsBase
using Base.Threads
using LoopVectorization
using Distributed
using RCall
using LinearAlgebra
using PDMats

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

R"library(TruncatedNormal)"

function h5_mv(c, r; λ=2.0, ν=2, ρ=1.0)
    ## repulsive prior for c and r
    ## c: K×d matrix, r: K×1 vector
    ## λ, ν, ρ: parameters in h(c,r) = minᵢⱼexp{-λ/max((||cᵢ-cⱼ||-ρ(rᵢ+rⱼ)),0)^ν}   

    K = length(r)
    gap_min = +Inf

    for i in 1:(K-1)
        for j in (i+1):K
            gap = norm(c[i,:] .- c[j,:]) - ρ*(r[i] + r[j])
            if gap <= 0
                return 0.0
            elseif gap < gap_min
                gap_min = gap
            end
        end
    end

    return exp(-λ / (gap_min^ν))

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
    f_old = tgt_f(x_old)
    accept = 0
    t = 0

    #samples = []

    while (accept < tgt_it) & (t < max_it)
        t += 1
        x_new = rand(rng, truncated(Normal(x_old, sigma), lower=lower, upper=upper), 1)[1]
        f_new = tgt_f(x_new)
        q_old_new = pdf(truncated(Normal(x_old, sigma), lower=lower, upper=upper), x_new)
        q_new_old = pdf(truncated(Normal(x_new, sigma), lower=lower, upper=upper), x_old)
        R = (f_new * q_new_old) / (f_old * q_old_new)
        threshold = minimum([1, R])
        indicator = rand(rng, Uniform(0, 1), 1)[1]
        if indicator < threshold
            x_old = x_new
            f_old = f_new
            accept += 1
        end
        #append!(samples, x_old)
    end

    #acc_rate = accept / t

    #println("acceptance rate: $(acc_rate)")
    #println("iterations: $(t)")

    return x_old #, samples
end

function mv_map_gz(i, xi, gi, zi; rng=nothing, K, uk_star, ik_star, w, betas, phis, Sigmas)
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

    betas_indx = Dict{Int64, Vector{Int64}}()
    pdf_x = Dict{Int64, Vector{Float64}}()
    for k in 1:K
        indk = findall(p -> p > ui, betas[k])
        betas_indx[k] = indk
        if isempty(indk)
            pdf_x[k] = [0.0]
        else
            pdf_x[k] = [pdf(MvNormal(phis[k][:, id], Sigmas[k][:, :, id]), xi) for id in indk]
        end
    end

    pk = w .* [sum(pdf_x[k]) for k in 1:K]

    if !all(pk .≈ 0.0)
        gi_new = sample(rng, 1:K, Weights(pk ./ sum(pk)))
    else
        gi_new = gi
    end

    if sum(pdf_x[gi_new]) ≈ 0.0
        zi_new = sample(rng, betas_indx[gi_new])
    else
        zi_new = sample(rng, betas_indx[gi_new], Weights(pdf_x[gi_new] ./ sum(pdf_x[gi_new])))
    end

    return [gi_new, zi_new]

end

function mv_reduce_ϕΣ(pair, dict; rng=nothing, Sigmas, centers, inv_Σ0, radius, v0, P0)
    ## pair: (comp_k, clst_l)
    ## center_k is the center for the comp_k th component
    ## rect_lb, rect_ub are the lower bound and upper bound for the interval of the comp_k th component

    if rng === nothing
        rng = MersenneTwister()
    end

    comp_k, clst_l = pair
    Sigma_kl = Sigmas[comp_k][:, :, clst_l]
    center_k = centers[comp_k, :]
    if radius[comp_k, :] != [0.0, 0.0]
        rectk_lb = centers[comp_k, :] - radius[comp_k, :]
        rectk_ub = centers[comp_k, :] + radius[comp_k, :]
    end

    X_kl = dict[pair] # all samples in this cluster
    n_kl = size(X_kl, 1)
    inv_Sigma_kl = inv(Sigma_kl)

    if radius[comp_k, :] != [0.0, 0.0]
       ## posterior parameters for ϕₖₗ
        M_kl = n_kl * inv_Sigma_kl + inv_Σ0
        b_kl = n_kl * inv_Sigma_kl * mean(X_kl, dims=1)' + inv_Σ0 * center_k
        Sigma_pst = inv(M_kl)
        mu_pst = Sigma_pst * b_kl

       ## generate new ϕₖₗ
        phi_kl_new = rcopy(R"rtmvnorm(n=1, mu=$(mu_pst), sigma=$(Sigma_pst), lb=$(rectk_lb), ub=$(rectk_ub))")
    else    
        phi_kl_new = center_k
    end

    ## posterior parameters for Σₖₗ
    X_kl_c = X_kl .- phi_kl_new'
    nS_kl = X_kl_c' * X_kl_c

    ## generate new Σₖₗ
    Sigma_kl_new = rand(rng, InverseWishart(n_kl + v0, nS_kl + P0))

    return pair, n_kl, phi_kl_new, Sigma_kl_new

end

function MvMDP7_fixReg(X; w, r, c, map_gz=mv_map_gz, reduce_ϕΣ=mv_reduce_ϕΣ,
                 truncatedMH=truncatedMH, alpha=nothing, γ=1.0,  
                 σ0=1.0, v0=nothing, P0=nothing, burnin = 0, 
                 iteration = 500, seed=nothing)

    ## Mxiture of DP with K components ∑ₖwₖDP(γfₖ) with fixed region support for fₖ      
    ## Using slice sampling 
    ## X: n×d matrix, n samples, each of dimension d 
    ## w: w ∈ R^{K}, initial value of w, K is the number of clusters 
    ## r,c: r ∈ R^{K×d}, c=[c₁,⋯,cₖ] ∈ R^{K×d}, r and c will be fixed through the sampling
    ## map_gz: map function to update component indicator g and cluster indicator z
    ## reduce_ϕΣ: reduce function to update cluster parameters ϕ and Σ
    ## alpha: vector [α₁,⋯,αₖ] of length K, α₁,⋯,αₖ >0, prior parameters for w~Dirichlet(alpha)
    ## γ: γ ∈ R₊, concentration parameter γ of DPs 
    ## σ0: Σ0 = σ0*I ∈ R^{d×d} is the parameter in truncated normal prior fₖ for ϕₖᵢ, fₖ:ϕₖᵢ~ N(cₖ, Σ0)1{cₖ-rₖ ≤ x ≤ cₖ+rₖ}
    ## v0: v0 ∈ R, v0 > (d-1), hyperparameter in prior for Σk: InverseWishart(v0,P0)
    ## P0: P0 ∈ R^{d×d} hyperparameter in prior for Σk: InverseWishart(v0,P0)
    ## seed: random seed to ensure reproducibility
  
  ### global settings
    if seed === nothing
        seed = rand(RandomDevice(), UInt64) # generate random seed
    end       
    rng_glb = MersenneTwister(seed)
    seed_r = rand(rng_glb, 1:1e6, 1)[1]
    R"set.seed($(seed_r))"
  ###
  
  ### check initial values
    if length(w) != size(r, 1) || size(c, 1) != size(r, 1) 
        error("Initial values do not have the same number of components.")
    end

    if size(c, 2) != size(X, 2) || size(r, 2) != size(X, 2)
        error("Dimension of c and X are not the same.")
    end

    w = Float64.(w)

    if !(sum(w) ≈ 1.0)
        error("Initial weights do not sum to 1.")
    end

    c = Float64.(c)
    r = Float64.(r)

    K = length(w)
    d = size(X, 2)

    if P0 !== nothing
        if size(P0) != (d, d)
            error("Dimension of P0 is not the same as the dimension of X.")
        elseif !isposdef(P0)
            error("P0 is not positive definite.")
        end
    end
  ###

  ### set hyperparameters
    n = size(X, 1) 
    
    if alpha === nothing
        alpha = 2.5 .* ones(K)
    end

    if v0 === nothing
        v0 = d + 2.5
    end

    if P0 === nothing
        P0 = PDMat(Matrix{Float64}(d * v0 * I, d, d))
    end

    Σ0 = PDMat(Matrix{Float64}(σ0^2 * I, d, d))
    inv_Σ0 = inv(Σ0)

  ###
  
  ### initialization

   ## initializing each variable in the first MCMC iteration
    w_new = w
    c_new = c
    r_new = r
    rect_lb = c_new - r_new
    rect_ub = c_new + r_new

    g_new = sample(rng_glb, 1:K, Weights(w_new), n) #indicator of DP component gᵢ=k for DP(γfₖ)
    z_new = ones(Int64, n)  #indicator of cluster within the DP_gᵢ for each sample
    nk = countmap(g_new) # number of samples in each component
    betas_new = Dict{Int64, Vector{Float64}}() # the kth item is for betas in the kth component
    phis_new = Dict{Int64, Matrix{Float64}}() # the kth item is for phis in the kth component
    Sigmas_new = Dict{Int64, Array{Float64}}() # the kth item is for Sigmas in the kth component
    uk_star = Dict{Int64, Vector{Float64}}() # the minimum of u of each beta in each component k
    ik_star = Dict{Int64, Vector{Int64}}() # the indicator of uk_star of each beta in each component k (global index)
    for k in 1:K
        if haskey(nk, k)
            betas_new[k] = rand(rng_glb, Dirichlet([nk[k], γ]), 1)[:, 1]
            uk_star[k] = rand(rng_glb, Beta(1, nk[k]), 1) .* betas_new[k][1]
            indk = findall(x -> x == k, g_new)
            ik_star[k] = [indk[sample(rng_glb, 1:nk[k])]]
        else
            betas_new[k] = rand(rng_glb, Dirichlet([1, γ]), 1)[:, 1]
        end

        if r_new[k, :] != [0., 0.]
            phis_new[k] = reshape(rcopy(R"rtmvnorm(n=1, mu=$(c_new[k,:]), sigma=$(Σ0), lb=$(rect_lb[k,:]), ub=$(rect_ub[k,:]))"), d, 1)
        else
            phis_new[k] = reshape(c_new[k, :], d, 1)
        end

        Sigmas_new[k] = rand(rng_glb, InverseWishart(v0, P0))

    end
   ##

   ## initializations to store MCMC results 
    nit = iteration - burnin
    w_all = Matrix{Float64}(undef, K, nit) # vector to store w's in each MCMC iteration 
    g_all = Matrix{Int64}(undef, n, nit) # matrix to store gs in each column
    z_all = Matrix{Int64}(undef, n, nit) # matrix to store zs in each column
    ind_all = Matrix{Int64}(undef, K, nit) # starting locations for betas, phis, Sigmas in each MCMC iteration for component k
    β_all = Dict{Int64, Vector{Float64}}() # βs for the K components
    ϕ_all = Dict{Int64, Matrix{Float64}}() # ϕs for the K components
    Σ_all = Dict{Int64, Array{Float64}}() # Σs for the K components
    for k in 1:K
        β_all[k] = []
    end
   ##

  ###

  ### Gibbs sampling (slice sampler)
    for t in 1:iteration 

      ### update gᵢ and zᵢ
        u_star = minimum(minimum.(values(uk_star))) # find u_star for all components

       ## instantiate beta, phis, Sigmas based on u_star
       ## multi-threading: not recommended when n is small
        # rngs = [MersenneTwister(seed + t) for t in 1:Threads.nthreads()]
        # @threads for k in 1:K 
        #     local_rng = rngs[Threads.threadid()] 
        #     while betas_new[k][end] >= u_star
        #         nu = rand(local_rng, Beta(1, γ), 1)[1]
        #         append!(betas_new[k], betas_new[k][end] * (1-nu))
        #         betas_new[k][end-1] *= nu
        #         append!(phis_new[k], rand(local_rng, fk_new[k], 1))
        #         append!(Sigmas_new[k], sqrt.(rand(local_rng, InverseGamma(αs, θs), 1)))
        #     end
        # end
       ##       
        for k in 1:K
            while betas_new[k][end] >= u_star
                nu = rand(rng_glb, Beta(1, γ), 1)[1]
                append!(betas_new[k], betas_new[k][end] * (1 - nu))
                betas_new[k][end-1] *= nu
                if r_new[k,:] != [0., 0.]
                    phis_new[k] = hcat(phis_new[k],
                        rcopy(R"rtmvnorm(n=1, mu=$(c_new[k,:]), sigma=$(Σ0), lb=$(rect_lb[k,:]), ub=$(rect_ub[k,:]))"))
                else
                    phis_new[k] = hcat(phis_new[k], c_new[k, :])
                end
                Sigmas_new[k] = [Sigmas_new[k];;; rand(rng_glb, InverseWishart(v0, P0))]
            end
        end
       ##
        
       ## update gi, zi for each sample xi
        if n < 10^6
            # when sample size n < O(10⁶), use normal for-loops
            results = Matrix{Int64}(undef, n, 2)
            for i in 1:n
                result_gz = map_gz(i, X[i, :], g_new[i], z_new[i]; rng=rng_glb, K=K, uk_star=uk_star, ik_star=ik_star,
                                    w=w_new, betas=betas_new, phis=phis_new, Sigmas=Sigmas_new)
                results[i, :] = result_gz
            end
            g_new = results[:, 1]
            z_new = results[:, 2]
        else
            # when sample size n > O(10⁶), use multi-threading to optimize the performance
            results = Matrix{Int64}(undef, n, 2)
            rngs = [MersenneTwister(seed + t) for t in 1:Threads.nthreads()]
            @threads for i in 1:n
                local_rng = rngs[Threads.threadid()]
                result_gz = map_gz(i, X[i, :], g_new[i], z_new[i]; rng=local_rng, K=K, uk_star=uk_star, ik_star=ik_star, 
                                    w=w_new, betas=betas_new, phis=phis_new, Sigmas=Sigmas_new)
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
        w_new = rand(rng_glb, Dirichlet(alpha_pst), 1)[:, 1]
      ###
    
      ### middle steps
       ### delete idle clusters, update z_new accordingly, update ϕs, Σs, and βs for empty components 
        for k in 1:K
            if haskey(nk, k)
                indk = findall(x -> x == k, g_new)
                zks = copy(z_new[indk])
                zks_seq = sort(unique(zks))
                delete_i = filter(x -> x ∉ zks_seq, 1:size(phis_new[k], 2))
                keep_i = setdiff(1:size(phis_new[k], 2), delete_i)
                if !isempty(delete_i)
                    betas_new[k][end] += sum(betas_new[k][delete_i])
                    deleteat!(betas_new[k], delete_i)
                    phis_new[k] = phis_new[k][:, keep_i]
                    Sigmas_new[k] = Sigmas_new[k][:, :, keep_i]
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
                if r_new[k, :] != [0.0, 0.0]
                    phis_new[k] = reshape(rcopy(R"rtmvnorm(n=1, mu=$(c_new[k,:]), sigma=$(Σ0), lb=$(rect_lb[k,:]), ub=$(rect_ub[k,:]))"), d, 1)
                else
                    phis_new[k] = reshape(c_new[k, :], d, 1)
                end
                Sigmas_new[k] = rand(rng_glb, InverseWishart(v0, P0))
                betas_new[k] = rand(rng_glb, Dirichlet([1, γ]), 1)[:, 1]
            end

        end

       ### create a dictionary with key (gi,zi) and value xi's ∈ component gi, cluster zi
        clst_dict = Dict{Tuple{Int64,Int64},Matrix{Float64}}()
        for i in 1:n
            key = (g_new[i], z_new[i])
            if !haskey(clst_dict, key)
                clst_dict[key] = reshape(X[i, :], 1, d)
            else
                clst_dict[key] = vcat(clst_dict[key], X[i, :]')
            end
        end
      ###
    
      ### update ϕₖs,Σₖs for non-empty clusters
        nkl_dict = Dict{Tuple{Int64,Int64}, Int64}()
        for key in keys(clst_dict)

            #### get new parameters for each cluster
            pair, n_kl, phi_kl_new, Sigma_kl_new = reduce_ϕΣ(key, clst_dict; rng=rng_glb, Sigmas=Sigmas_new, centers=c_new,
                                                 inv_Σ0=inv_Σ0, radius=r_new, v0=v0, P0=P0)
            comp_k, clst_l = pair
            nkl_dict[pair] = n_kl

            ## update phis and Sigmas
            phis_new[comp_k][:, clst_l] = phi_kl_new
            Sigmas_new[comp_k][:, :, clst_l] = Sigma_kl_new

        end
      ###
    
      ### update βs for all clusters, update uk_star and ik_star for non-empty clusters
      ### update ϕₖs,Σₖs for empty clusters
        uk_star = Dict{Int64,Vector{Float64}}()
        ik_star = Dict{Int64,Vector{Int64}}()

        for k in 1:K
            if haskey(nk, k)
                alphak_pst = [nkl_dict[(k, l)] for l in 1:size(phis_new[k],2)]
                append!(alphak_pst, γ)
                betas_new[k] = rand(rng_glb, Dirichlet(alphak_pst), 1)[:, 1]
                uk_star[k] = [rand(rng_glb, f, 1)[1] for f in Beta.(1, alphak_pst[1:(end-1)])] .* betas_new[k][1:(end-1)]

                ik_star[k] = []
                for l in 1:size(phis_new[k], 2)
                    ind_kl = findall(x -> x == 1, (g_new .== k) .&& (z_new .== l))
                    append!(ik_star[k], ind_kl[sample(rng_glb, 1:alphak_pst[l])])
                end       
            end
        end
      ###

        
      ### store values
        if t > burnin
            it = t - burnin
            w_all[:, it] = w_new
            g_all[:, it] = g_new
            z_all[:, it] = z_new
            if it == 1
                for k in 1:K
                    ind_all[k, 1] = 1
                    append!(β_all[k], betas_new[k][1:(end-1)])
                    ϕ_all[k] =  phis_new[k]
                    Σ_all[k] = Sigmas_new[k]
                end
            else
                for k in 1:K
                    ind_all[k, it] = length(β_all[k]) + 1
                    append!(β_all[k], betas_new[k][1:(end-1)])
                    ϕ_all[k] = hcat(ϕ_all[k], phis_new[k])
                    Σ_all[k] = [Σ_all[k];;; Sigmas_new[k]]
                end 
            end
        
        end
      ###

    end
  ###
    return w_all, g_all, z_all, β_all, ϕ_all, Σ_all, ind_all
end



