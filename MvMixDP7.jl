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

    betas_indx = Dict{Int64,Vector{Int64}}()
    pdf_x = Dict{Int64,Vector{Float64}}()
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

function mv_reduce_ϕΣ(pair, dict; rng=nothing, Sigmas, centers, inv_Σ0, rect_lb, rect_ub, v0, P0)
    ## pair: (comp_k, clst_l)
    ## center_k is the center for the comp_k th component
    ## rect_lb, rect_ub are the lower bound and upper bound for the interval of the comp_k th component

    if rng === nothing
        rng = MersenneTwister()
    end

    comp_k, clst_l = pair
    Sigma_kl = Sigmas[comp_k][:, :, clst_l]
    center_k = centers[comp_k, :]
    rectk_lb = rect_lb[comp_k, :]
    rectk_ub = rect_ub[comp_k, :]

    X_kl = dict[pair] # all samples in this cluster
    n_kl = size(X_kl, 1)
    inv_Sigma_kl = inv(Sigma_kl)

    ## posterior parameters for ϕₖₗ
    M_kl = n_kl * inv_Sigma_kl + inv_Σ0
    b_kl = n_kl * inv_Sigma_kl * mean(X_kl, dims=1)' + inv_Σ0 * center_k
    Sigma_pst = inv(M_kl)
    mu_pst = Sigma_pst * b_kl
    
    ## generate new ϕₖₗ
    phi_kl_new = rcopy(R"rtmvnorm(n=1, mu=$(mu_pst), sigma=$(Sigma_pst), lb=$(rectk_lb), ub=$(rectk_ub))")

    ## posterior parameters for Σₖₗ
    X_kl_c = X_kl .- phi_kl_new'
    nS_kl = X_kl_c' * X_kl_c

    ## generate new Σₖₗ
    Sigma_kl_new = rand(rng, InverseWishart(n_kl + v0, nS_kl + P0))

    return pair, n_kl, phi_kl_new, Sigma_kl_new

end

function intersection_point(ck_old, mu_c, cks_concern, min_dists)
    ## ck_old: the old center of the component k
    ## mu_c: the posterior center of ck
    ## cks_concern: the centers of other components that are close to mu_c
    ## min_dists: the minimum distances required between ck and cks_concern

    P = ck_old # starting point of the line
    D = mu_c - ck_old # direction of the line
    t = []
    for i in eachindex(min_dists)
        C = cks_concern[i, :] # center of the sphere
        r = min_dists[i]

        a = dot(D, D)
        b = 2 * dot(D, P - C)
        c = dot(P - C, P - C) - r^2

        if b^2 - 4 * a * c < 0
            continue
        end
        
        t1 = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a)
        t2 = (-b - sqrt(b^2 - 4 * a * c)) / (2 * a)

        if t1 >= 0.0 && t1 <= 1.0
            push!(t, t1)
        end

        if t2 >= 0.0 && t2 <= 1.0
            push!(t, t2)
        end
    end

    if isempty(t)
        return mu_c
    else
        t_max = maximum(t)
        return P + t_max * D
    end

end

function MvMDP7(X; inits, h=h5_mv, intersection_point=intersection_point, map_gz=mv_map_gz, reduce_ϕΣ=mv_reduce_ϕΣ, truncatedMH=truncatedMH,
                 alpha=nothing, γ=1.0, cc=nothing, Γ=nothing, κ=3.0, θ=1.0, λ=2.0, ν=2, ρ=nothing, σ0=1.0, 
                 v0=nothing, P0=nothing, burnin = 0, iteration = 500, seed=nothing)
    ## Mxiture of DP with K components ∑ₖwₖDP(γfₖ)       
    ## Using slice sampling 
    ## X: n×d matrix, n samples, each of dimension d
    ## inits: initial value tuple (w, r, c), w,r ∈ R^{K}, c=[c₁,⋯,cₖ] ∈ R^{K×d}, K is the number of clusters
    ## h: function for calculating repulsive prior based on current components
    ## map_gz: map function to update component indicator g and cluster indicator z
    ## reduce_ϕΣ: reduce function to update cluster parameters ϕ and Σ
    ## alpha: vector [α₁,⋯, αₖ] of length K, α₁,⋯,αₖ >0, prior parameters for w~Dirichlet(alpha)
    ## γ: γ ∈ R₊, concentration parameter γ of DPs
    ## cc: cc ∈ R^d, prior N(cc, Γ) for cₖ
    ## Γ: Γ ∈ R^{d×d} covariance matrix for prior N(cc, Γ) of each cₖ
    ## κ: κ ∈ R₊, shape parameter for gamma(κ,θ) of each rₖ
    ## θ: θ ∈ R₊, scale parameter for gamma(κ,θ) of each rₖ
    ## λ: λ ∈ R₊, parameter in h(c,r) = minᵢⱼexp{-λ/max((||cᵢ-cⱼ||-ρ(rᵢ+rⱼ)),0)^ν}
    ## ν: ν ∈ N₊, parameter in h(c,r) = minᵢⱼexp{-λ/max((||cᵢ-cⱼ||-ρ(rᵢ+rⱼ)),0)^ν}
    ## ρ: ρ ∈ R₊, parameter in h(c,r) = minᵢⱼexp{-λ/max((||cᵢ-cⱼ||-ρ(rᵢ+rⱼ)),0)^ν}
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
    w, r, c = inits

    if length(w) != length(r) || size(c, 1) != length(r) 
        error("Initial values are not of the same length.")
    end

    if size(c, 2) != size(X, 2)
        error("Dimension of c and X are not the same.")
    end

    w = Float64.(w)

    if sum(w) != 1.0
        error("Initial weights do not sum to 1.")
    end

    c = Float64.(c)
    r = Float64.(r)

    K = length(w)
    d = size(X, 2)

    if ρ === nothing
        ρ = sqrt(d)
    end

    for i in 1:(K-1)
        for j in (i+1):K
            if norm(c[i, :] - c[j, :]) <= ρ * (r[i] + r[j])
                error("Initial intervals are not well separated.")
            end
        end
    end

    if Γ !== nothing
        if size(Γ) != (d, d)
            error("Dimension of Γ is not the same as the dimension of X.")
        elseif !isposdef(Γ)
            error("Γ is not positive definite.") 
        end
    end

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

    if cc === nothing
        cc = zeros(Float64, d) # prior N(cc, Γ) for cₖ
    end
    
    if alpha === nothing
        alpha = 2.5 .* ones(K)
    end

    if v0 === nothing
        v0 = d + 2.5
    end

    if Γ === nothing
        Γ = PDMat(Matrix{Float64}(d * I, d, d))
    end

    if P0 === nothing
        P0 = PDMat(Matrix{Float64}(d * I, d, d))
    end

    Σ0 = PDMat(Matrix{Float64}(σ0^2 * I, d, d))
    inv_Σ0 = inv(Σ0)
    inv_Γ = inv(Γ)
  ###
  
  ### initialization

   ## initializing each variable in the first MCMC iteration
    w_new = w
    c_new = c
    r_new = r
    uh_new = rand(rng_glb, Uniform(0, h(c_new, r_new; λ=λ, ν=ν, ρ=ρ)), 1)[1]
    rect_lb = c_new .- r_new
    rect_ub = c_new .+ r_new

    g_new = sample(rng_glb, 1:K, Weights(w_new), n) #indicator of DP component gᵢ=k for DP(γfₖ)
    z_new = ones(Int64, n)  #indicator of cluster within the DP_gᵢ for each sample
    nk = countmap(g_new) # number of samples in each component
    betas_new = Dict{Int64,Vector{Float64}}() # the kth item is for betas in the kth component
    phis_new = Dict{Int64,Matrix{Float64}}() # the kth item is for phis in the kth component
    Sigmas_new = Dict{Int64,Array{Float64}}() # the kth item is for Sigmas in the kth component
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
        phis_new[k] = reshape(rcopy(R"rtmvnorm(n=1, mu=$(c_new[k,:]), sigma=$(Σ0), lb=$(rect_lb[k,:]), ub=$(rect_ub[k,:]))"), d, 1)
        Sigmas_new[k] = rand(rng_glb, InverseWishart(v0, P0))
    end
   ##

   ## initializations to store MCMC results 
    nit = iteration - burnin
    c_all = Array{Float64}(undef, K, d, nit)
    r_all = Matrix{Float64}(undef, K, nit)
    w_all = Matrix{Float64}(undef, K, nit) # vector to store w's in each MCMC iteration 
    g_all = Matrix{Int64}(undef, n, nit) # matrix to store gs in each column
    z_all = Matrix{Int64}(undef, n, nit) # matrix to store zs in each column
    ind_all = Matrix{Int64}(undef, K, nit) # starting locations for betas, phis, Sigmas in each MCMC iteration for component k
    β_all = Dict{Int64,Vector{Float64}}() # βs for the K components
    ϕ_all = Dict{Int64,Matrix{Float64}}() # ϕs for the K components
    Σ_all = Dict{Int64,Array{Float64}}() # σs for the K components
    for k in 1:K
        β_all[k] = []
    end
   ##

  ###

  ### Gibbs sampling (slice sampler)
    ck_not_update = 0
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
                phis_new[k] = hcat(phis_new[k], 
                    rcopy(R"rtmvnorm(n=1, mu=$(c_new[k,:]), sigma=$(Σ0), lb=$(rect_lb[k,:]), ub=$(rect_ub[k,:]))"))
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
        w_new = rand(rng_glb, Dirichlet(alpha_pst), 1)[:,1]
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
                phis_new[k] = reshape(rcopy(R"rtmvnorm(n=1, mu=$(c_new[k,:]), sigma=$(Σ0), lb=$(rect_lb[k,:]), ub=$(rect_ub[k,:]))"), d, 1)
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
                                                 inv_Σ0=inv_Σ0, rect_lb=rect_lb, rect_ub=rect_ub, v0=v0, P0=P0)
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

      ### update c and r
        for k in 1:K
            
            ind_notk = setdiff(1:K, k)
            r_notk = r_new[ind_notk]
            c_notk = c_new[ind_notk, :]

            if haskey(nk, k)
                phiks = phis_new[k]
                rk = r_new[k]
                ck_old = c_new[k, :]
                n_phik = size(phiks, 2)
                phiks_min = minimum(phiks, dims=2)[:,1]
                phiks_max = maximum(phiks, dims=2)[:,1]

              ### update cₖ
               ### posterior parameters
                Mk = n_phik * inv_Σ0 + inv_Γ
                bk = n_phik * inv_Σ0 * mean(phiks, dims=2)

                Sigma_c = inv(Mk)
                mu_c = (Sigma_c * bk)[:, 1]

               ### get lower bound and upper bound
                bd_close = 0 #indicator of whether the bounds are close to ck_old
                lb_ck = phiks_max .- rk
                ub_ck = phiks_min .+ rk
                vertex_to_mu = zeros(d)
                ### adjust bounds towards the direction of mu_c
                for j in 1:d
                    if mu_c[j] < ck_old[j]
                        ub_ck[j] = ck_old[j]
                        vertex_to_mu[j] = lb_ck[j]
                    elseif mu_c[j] > ck_old[j]
                        lb_ck[j] = ck_old[j]
                        vertex_to_mu[j] = ub_ck[j]
                    else
                        lb_ck[j] = (ck_old[j] + lb_ck[j]) / 2
                        ub_ck[j] = (ck_old[j] + ub_ck[j]) / 2
                        vertex_to_mu[j] = ck_old[j]
                    end
                end

                if norm(lb_ck - ub_ck) < 0.05 * sqrt(d)
                    bd_close = 1
                    ck_not_update += 1
                else

                    min_dist_to_ck = ρ .* (r_notk .+ rk) .+ (-λ / log(uh_new))^(1 / ν)
                    ## adjust lb and ub if mu_c is not far enough from other cₖ's
                    if any([norm(c_notk[i, :] - vertex_to_mu) <= min_dist_to_ck[i] for i in eachindex(min_dist_to_ck)])

                        ind = findall(i -> norm(c_notk[i, :] - vertex_to_mu) <= min_dist_to_ck[i], eachindex(min_dist_to_ck))
                        cks_concern = c_notk[ind, :]
                        min_dists = min_dist_to_ck[ind]
                        pt = intersection_point(ck_old, mu_c, cks_concern, min_dists)
                        #println("within bounds, pt to ck_old norm:$(norm(pt - ck_old))")

                        if norm(pt - ck_old) < 0.05 * sqrt(d)
                            bd_close = 1
                            ck_not_update += 1
                            #println("close bounds: ck not updated")
                        else
                            lb_ck_ajst = minimum([ck_old pt], dims=2)[:, 1]
                            ub_ck_ajst = maximum([ck_old pt], dims=2)[:, 1]
                            for j in 1:d
                                if lb_ck_ajst[j] ≈ ub_ck_ajst[j]
                                    lb_ck_ajst[j] = lb_ck[j]
                                    ub_ck_ajst[j] = ub_ck[j]
                                end

                                if lb_ck_ajst[j] < lb_ck[j]
                                    lb_ck_ajst[j] = lb_ck[j]
                                end

                                if ub_ck_ajst[j] > ub_ck[j]
                                    ub_ck_ajst[j] = ub_ck[j]
                                end

                            end
                            lb_ck = lb_ck_ajst
                            ub_ck = ub_ck_ajst

                            if norm(lb_ck - ub_ck) < 0.05 * sqrt(d)
                                bd_close = 1
                                ck_not_update += 1
                            end
                        end
                    end

                end               

               ### generate new cₖ
                if bd_close != 1
                    n_sim = 0
                    achieve = 0
                    while n_sim < 5 ## rejection sampling to update cₖ
                        n_sim += 1
                        c_sim_batch = rcopy(R"rtmvnorm(n=1000, mu=$(mu_c), sigma=$(Sigma_c), lb=$(lb_ck), ub=$(ub_ck))")
                        for c_sim in eachrow(c_sim_batch)
                            if all([norm(c_notk[i, :] - c_sim) > min_dist_to_ck[i] for i in eachindex(min_dist_to_ck)])
                                c_new[k, :] = c_sim
                                achieve = 1
                                break
                            end
                        end

                        if achieve == 1
                            break
                        end            
                    end

                    #println("n_sim: $n_sim")

                    if n_sim == 5
                        ck_not_update += 1
                        #println("nonempty cluster: cₖ not updated")
                        #println("ck_not_update: $ck_not_update")
                        #println("c_notk = $c_notk\nmin_dist_to_ck = $min_dist_to_ck\nmu_c = $mu_c\nck_old = $ck_old\nSigma_c = $Sigma_c\nlb_ck = $lb_ck\nub_ck = $ub_ck\nbd_close = $bd_close")
                    end
                end
              ###

              ### update rₖ
                ck = c_new[k, :]
                lb_rk = max(maximum(ck - phiks_min), maximum(phiks_max - ck))
                ub_rk = minimum([(norm(c_notk[i, :] - ck) - (-λ / log(uh_new))^(1 / ν))/ρ - r_notk[i] for i in eachindex(r_notk)])
                tgt_f = x -> pdf.(Gamma(κ, θ), x) .* (1 ./ ((2 .* cdf.(Normal(0, σ0), x) - 1) .^ (2 * n_phik)))
                r_new[k] = truncatedMH((ub_rk + lb_rk) / 2, tgt_f, lb_rk, ub_rk; rng=rng_glb)

              ###

            else
               ### generate new cₖ
                min_dist_to_ck = ρ .* (r_notk .+ r_new[k]) .+ (-λ / log(uh_new))^(1 / ν)
                n_sim = 0
                while n_sim < 10000 ## rejection sampling to update cₖ
                    n_sim += 1
                    c_sim = rand(MvNormal(cc, Γ), 1)[:,1]
                    if all([norm(c_notk[i, :] - c_sim) > min_dist_to_ck[i] for i in eachindex(min_dist_to_ck)])
                        c_new[k, :] = c_sim
                        break
                    end                   
                end
                
                #println("n_sim: $n_sim")

                if n_sim == 10000
                    println("empty cluster: cₖ not updated")
                end

               ### generate new rₖ
                lb_rk = 0
                ub_rk = minimum([(norm(c_notk[i, :] - c_new[k, :]) - (-λ / log(uh_new))^(1 / ν))/ρ - r_notk[i] for i in eachindex(r_notk)])
                r_new[k] = rand(rng_glb, truncated(Gamma(κ, θ), lower=lb_rk, upper=ub_rk), 1)[1]
                
            end
        end
      ###

      ### update uh
        uh_new = rand(rng_glb, Uniform(0, h(c_new, r_new; λ=λ, ν=ν, ρ=ρ)), 1)[1]
      ###
    
      ### update int_lb, int_ub and fk_new
        rect_lb = c_new .- r_new
        rect_ub = c_new .+ r_new
      ### 
        
      ### store values
        if t > burnin
            it = t - burnin
            w_all[:, it] = w_new
            r_all[:, it] = r_new
            c_all[:, :, it] = c_new
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
    println("ck_not_update: $ck_not_update")
    return w_all, c_all, r_all, g_all, z_all, β_all, ϕ_all, Σ_all, ind_all
end




