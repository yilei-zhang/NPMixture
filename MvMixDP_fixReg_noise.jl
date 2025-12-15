using Random
using Distributions
using Statistics
using StatsBase
using Base.Threads
using RCall
using LinearAlgebra
using PDMats
using PyPlot
using SpecialFunctions
using DataFrames
using CSV
using QuadGK


function mv_map_gz_mt_noise(X, K, R, uk_star, ik_star, w, betas, phis, Sigmas,
    g_cur::Vector{Int64}, z_cur::Vector{Int64};
    rngs)
    """
    Threaded mapping pass for multivariate case with a noise component (index K+1).

    Inputs:
    - X :: m×n matrix, column i is x_i
    - K :: number of "signal" components (noise is K+1)
    - R :: scalar, noise density is 1/R per eligible stick
    - uk_star, ik_star :: Vector{Vector} slice thresholds / special indices
    - w :: length K+1 weights
    - betas :: Vector{Vector{Float64}} (length K+1)
    - phis  :: Vector{Matrix{Float64}}  (for k≤K, size m×Lk)
    - Ls    :: Vector{Vector{LowerTriangular}}  (Cholesky factors for k≤K)
    - logc  :: Vector{Vector{Float64}}  (log constants for k≤K)
    - g_cur, z_cur :: current assignments (Vector{Int})
    - g_out, z_out :: outputs (Vector{Int})

    Reproducible given `seed`.  Uses per-thread scratch buffers.
    """

    n, d = size(X)
    T = Threads.nthreads()

    g_out = Vector{Int64}(undef, n)
    z_out = Vector{Int64}(undef, n)

    # Precompute max number of clusters to size scratch buffers
    Lmax = maximum(length, betas)  # handles K+1 as well

    # Static chunking for determinism
    @threads for tid in 1:T
        # tid = Threads.threadid()
        rng = rngs[tid]

        w_k = Vector{Float64}(undef, K + 1)
        betas_indx = [Vector{Int}(undef, Lmax) for _ in 1:K]
        pdf_x = [Vector{Float64}(undef, Lmax) for _ in 1:K]
        nidxs = Vector{Int}(undef, K)
       
        # contiguous slice for this thread
        start = div((tid - 1) * n, T) + 1
        stop = div(tid * n, T)
        for i in start:stop
            x_i = view(X, i, :)
            gi = g_cur[i]
            zi = z_cur[i]

            # slice u
            ustar_x = uk_star[gi][zi]
            istar_x = ik_star[gi][zi]
            beta_x = betas[gi][zi]
            ui = (i == istar_x) ? ustar_x : (rand(rng) * (beta_x - ustar_x) + ustar_x)

            # ── component weights w_k[k] = w[k] * sum_l p(x|k,l) over β_{k,l}>u
            total = 0.0
            # signal components 1...K
            for k in 1:K
                βk = betas[k]
                nidx = 0
                s = 0.0
                for l in eachindex(βk)
                    if βk[l] > ui
                        nidx += 1
                        betas_indx[k][nidx] = l
                        pdf_x[k][nidx] = pdf(MvNormal(phis[k][:, l], Sigmas[k][:, :, l]), x_i)
                        s += pdf_x[k][nidx]
                    end
                end
                nidxs[k] = nidx
                if nidx == 0
                    w_k[k] = 0.0
                else
                    w_k[k] = w[k] * s
                end
                total += w_k[k]
            end

            # noise component K+1: each eligible stick contributes 1/R
            knoise = K + 1
            w_k[knoise] = w[knoise] * (1 / R)
            total += w_k[knoise]

            # sample component g_new
            g_new = gi
            if total > 0.0
                r = rand(rng) * total
                acc = 0.0
                for k in 1:(K+1)
                    acc += w_k[k]
                    if r <= acc
                        g_new = k
                        break
                    end
                end
            end

            # sample cluster z_new within g_new
            if g_new == knoise
                z_new = 1
            else
                nidx = nidxs[g_new]
                s = sum(pdf_x[g_new][1:nidx])
                if s == 0.0
                    # fallback: choose uniformly among candidates
                    z_new = sample(rng, betas_indx[g_new][1:nidx])
                else
                    r = rand(rng) * s
                    acc = 0.0
                    for t2 in 1:nidx
                        acc += pdf_x[g_new][t2]
                        if r <= acc
                            z_new = betas_indx[g_new][t2]
                            break
                        end
                    end
                end

            end

            g_out[i] = g_new
            z_out[i] = z_new

        end

    end

    return g_out, z_out
end

function mv_map_gz_noise!(X, K, R, uk_star, ik_star, w, betas, phis, Sigmas,
    g_cur::Vector{Int64}, z_cur::Vector{Int64},
    g_out::Vector{Int64}, z_out::Vector{Int64};
    rng)
    """
    Threaded mapping pass for multivariate case with a noise component (index K+1).

    Inputs:
    - X :: m×n matrix, column i is x_i
    - K :: number of "signal" components (noise is K+1)
    - R :: scalar, noise density is 1/R per eligible stick
    - uk_star, ik_star :: Vector{Vector} slice thresholds / special indices
    - w :: length K+1 weights
    - betas :: Vector{Vector{Float64}} (length K+1)
    - phis  :: Vector{Matrix{Float64}}  (for k≤K, size m×Lk)
    - Ls    :: Vector{Vector{LowerTriangular}}  (Cholesky factors for k≤K)
    - logc  :: Vector{Vector{Float64}}  (log constants for k≤K)
    - g_cur, z_cur :: current assignments (Vector{Int})
    - g_out, z_out :: outputs (Vector{Int})

    Reproducible given `seed`.  Uses per-thread scratch buffers.
    """

    n, d = size(X)

    # Precompute max number of clusters to size scratch buffers
    Lmax = maximum(length, betas)  # handles K+1 as well
    w_k = Vector{Float64}(undef, K + 1)
    betas_indx = [Vector{Int}(undef, Lmax) for _ in 1:K]
    pdf_x = [Vector{Float64}(undef, Lmax) for _ in 1:K]
    nidxs = Vector{Int}(undef, K)

    @inbounds for i in 1:n
        x_i = view(X, i, :)
        gi = g_cur[i]
        zi = z_cur[i]

        # slice u
        ustar_x = uk_star[gi][zi]
        istar_x = ik_star[gi][zi]
        beta_x = betas[gi][zi]
        ui = (i == istar_x) ? ustar_x : (rand(rng) * (beta_x - ustar_x) + ustar_x)

        # ── component weights w_k[k] = w[k] * sum_l p(x|k,l) over β_{k,l}>u
        total = 0.0
        # signal components 1...K
        for k in 1:K
            βk = betas[k]
            # collect eligible l's
            nidx = 0
            s = 0.0
            for l in eachindex(βk)
                if βk[l] > ui
                    nidx += 1
                    betas_indx[k][nidx] = l
                    pdf_x[k][nidx] = pdf(MvNormal(phis[k][:, l], Sigmas[k][:, :, l]), x_i)
                    s += pdf_x[k][nidx]
                end
            end
            nidxs[k] = nidx
            if nidx == 0
                w_k[k] = 0.0
            else
                w_k[k] = w[k] * s
            end
            total += w_k[k]
        end

        # noise component K+1: each eligible stick contributes 1/R
        knoise = K + 1
        w_k[knoise] = w[knoise] * (1 / R)
        total += w_k[knoise]

        # sample component g_new
        g_new = gi
        if total > 0.0
            r = rand(rng) * total
            acc = 0.0
            for k in 1:(K+1)
                acc += w_k[k]
                if r <= acc
                    g_new = k
                    break
                end
            end
        end

        # sample cluster z_new within g_new

        if g_new == knoise
            z_new = 1
        else
            nidx = nidxs[g_new]
            s = sum(pdf_x[g_new][1:nidx])
            if s == 0.0
                # fallback: choose uniformly among candidates
                z_new = rand(rng, betas_indx[g_new][1:nidx])
            else
                r = rand(rng) * s
                acc = 0.0
                for t2 in 1:nidx
                    acc += pdf_x[g_new][t2]
                    if r <= acc
                        z_new = betas_indx[g_new][t2]
                        break
                    end
                end
            end

        end

        g_out[i] = g_new
        z_out[i] = z_new

    end

    return g_out, z_out
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

function remap_stats(old2new, d_in::Dict{Tuple{Int64,Int64},T}) where {T}
    d_out = Dict{Tuple{Int64,Int64},T}()
    for ((k, l), v) in d_in
        m = old2new[k][l]
        m == 0 && continue   # drop empty clusters
        d_out[(k, m)] = v
    end
    return d_out
end

function MvMDP_fixReg_noise(X; w, r, c, R, map_gz_mt=mv_map_gz_mt_noise, map_gz=mv_map_gz_noise!,
    truncatedMH=truncatedMH, remap_stats=remap_stats, alpha=nothing, wlb=0.005, wub=0.995, γ=1.0, σ0=1.0,
    v0=nothing, P0=nothing, burnin=0, iteration=500, thin=1, multithreads=false, seed=nothing)

    ## Mixture of DP with K components ∑ₖwₖDP(γfₖ) with fixed region support for fₖ      
    ## Using slice sampling 
    ## X: n×d matrix, n samples, each of dimension d 
    ## w: w ∈ R^{K+1}, initial value of w, K is the number of clusters 
    ## r,c: r ∈ R^{K×d}, c=[c₁,⋯,cₖ] ∈ R^{K×d}, r and c will be fixed through the sampling
    ## bg_lb, bg_ub: ∈ Rᵈ lower and upper bounds for the background noise
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
    rngs = [MersenneTwister(seed + t) for t in 1:Threads.nthreads()]

    seed_r = rand(rng_glb, 1:1e6, 1)[1]
    R"set.seed($(seed_r))"
    R"library(TruncatedNormal)"
    ###

    ### check initial values
    if length(w) != size(r, 1) + 1 || size(c, 1) != size(r, 1)
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

    K = length(w) - 1
    d = size(X, 2)
    wlb = wlb .* ones(K + 1)
    wub = wub .* ones(K + 1)

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
        alpha = 2.5 .* ones(K + 1)
    end

    if v0 === nothing
        v0 = d + 4.0
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
    rect_lb = c - r
    rect_ub = c + r
    #R = prod(bg_ub - bg_lb)

    g_new = sample(rng_glb, 1:(K+1), Weights(w_new), n) #indicator of DP component gᵢ=k for DP(γfₖ)
    z_new = ones(Int64, n)  #indicator of cluster within the DP_gᵢ for each sample
    nk = countmap(g_new) # number of samples in each component
    betas_new = Vector{Vector{Float64}}(undef, K+1) # the kth item is for betas in the kth component
    phis_new = Vector{Matrix{Float64}}(undef, K) # the kth item is for phis in the kth component
    Sigmas_new = Vector{Array{Float64}}(undef, K) # the kth item is for Sigmas in the kth component
    uk_star = Dict{Int64,Vector{Float64}}() # the minimum of u of each beta in each component k
    ik_star = Dict{Int64,Vector{Int64}}() # the indicator of uk_star of each beta in each component k (global index)

    for k in 1:K
        if haskey(nk, k)
            betas_new[k] = rand(rng_glb, Dirichlet([nk[k], γ]))
            uk_star[k] = rand(rng_glb, Beta(1, nk[k]), 1) .* betas_new[k][1]
            indk = findall(x -> x == k, g_new)
            ik_star[k] = sample(rng_glb, indk, 1)
        else
            betas_new[k] = rand(rng_glb, Dirichlet([1, γ]))
        end

        if r[k, :] != [0.0, 0.0]
            phis_new[k] = reshape(rcopy(R"rtmvnorm(n=1, mu=$(c[k,:]), sigma=$(Σ0), lb=$(rect_lb[k,:]), ub=$(rect_ub[k,:]))"), d, 1)
        else
            phis_new[k] = reshape(c[k, :], d, 1)
        end

        Sigmas_new[k] = rand(rng_glb, InverseWishart(v0, P0))

    end

    k = K + 1
    if haskey(nk, k)
        betas_new[k] = [1.0]
        uk_star[k] = rand(rng_glb, Beta(1, nk[k]), 1) .* betas_new[k][1]
        indk = findall(x -> x == k, g_new)
        ik_star[k] = sample(rng_glb, indk, 1)
    else
        betas_new[k] = [1.0]
    end

    ##

    ## initializations to store MCMC results 
    nit = (iteration - burnin - 1) ÷ thin + 1
    w_all = Matrix{Float64}(undef, K + 1, nit) # vector to store w's in each MCMC iteration 
    g_count = zeros(Int64, n, K + 1) # matrix to store gs in each column
    #z_all = Matrix{Int64}(undef, n, nit) # matrix to store zs in each column
    ind_all = Matrix{Int64}(undef, K, nit) # starting locations for betas, phis, Sigmas in each MCMC iteration for component k
    β_all = Vector{Vector{Float64}}(undef, K) # the kth item is for betas in the kth component
    ϕ_all = Vector{Matrix{Float64}}(undef, K) # the kth item is for phis in the kth component
    Σ_all = Vector{Array{Float64}}(undef, K) # the kth item is for Sigmas in the kth component

    for k in 1:K
        β_all[k] = []
    end
    ##


    ### Gibbs sampling (slice sampler)
    for t in 1:((nit-1)*thin+1+burnin)

        ### update gᵢ and zᵢ
        u_star = minimum(minimum.(values(uk_star))) # find u_star for all components
    
        for k in 1:K
            m = 0
            tail = betas_new[k][end]
            nus = Float64[]
            while tail >= u_star
                nu = rand(rng_glb, Beta(1, γ))
                push!(nus, nu)
                tail *= (1 - nu)
                m += 1
            end

            m == 0 && continue

            L0 = length(betas_new[k])
            resize!(betas_new[k], L0 + m)
            tail = betas_new[k][L0]
            for j in 1:m
                nu = nus[j]
                betas_new[k][L0+(j-1)] = tail * nu
                tail = tail * (1 - nu)
                betas_new[k][L0+j] = tail
            end

            if r[k, :] != [0.0, 0.0]
                Z = rcopy(R"rtmvnorm(n=$m, mu=$(c[k,:]), sigma=$(Σ0),
                                      lb=$(rect_lb[k,:]), ub=$(rect_ub[k,:]))")
                if m == 1
                    phis_new[k] = hcat(phis_new[k], Z)
                else
                    phis_new[k] = hcat(phis_new[k], Z')
                end
            else
                phis_new[k] = hcat(phis_new[k], repeat(c[k, :], 1, m))
            end

            sigs = rand(rng_glb, InverseWishart(v0, P0), m)
            Sigmas_new[k] = cat(Sigmas_new[k], cat(sigs...; dims=3); dims=3)  # one 3D cat
        end
        ##


        ## update gi, zi for each sample xi
        if !multithreads
            g_old = copy(g_new)
            z_old = copy(z_new)
            map_gz(X, K, R, uk_star, ik_star, w_new, betas_new, phis_new, Sigmas_new,
                g_old, z_old, g_new, z_new; rng=rng_glb)
        else
            g_old = copy(g_new)
            z_old = copy(z_new)
            g_new, z_new = map_gz_mt(X, K, R, uk_star, ik_star, w_new, betas_new, phis_new, Sigmas_new,
                g_old, z_old; rngs=rngs)
        end
        ##
        ###

        ### update weights w
        nk = countmap(g_new)
        alpha_pst = Vector{Float64}(undef, K + 1)
        for k in 1:(K + 1)
            if haskey(nk, k)
                alpha_pst[k] = alpha[k] + nk[k]
            else
                alpha_pst[k] = alpha[k]
            end
        end
        w_new = tdirichlet(1; rng=rng_glb, lb=wlb, ub=wub, γ=alpha_pst)
        if any(isnan.(w_new))
            w_new = tdirichlet(1; rng=rng_glb, lb=wlb, ub=wub, γ=alpha)
        end
        ###

      ### middle steps
        if multithreads

            T = Threads.nthreads()
            L = [size(phis_new[k], 2) for k in 1:K] # number of clusters in each component
            used_local = [[falses(L[k]) for k in 1:K] for _ in 1:T]
            local_maps = [Dict{Tuple{Int64,Int64},Vector{Int64}}() for _ in 1:T] # local maps for each thread
            nkl_local = [Dict{Tuple{Int64,Int64},Int64}() for _ in 1:T]
            sx_local = [Dict{Tuple{Int64,Int64}, Vector{Float64}}() for _ in 1:T]
            sxx_local = [Dict{Tuple{Int64,Int64}, Matrix{Float64}}() for _ in 1:T]
   
            @threads for tid in 1:T
                #tid = Threads.threadid()
                start = div((tid - 1) * n, T) + 1
                stop = div(tid * n, T)

                for i in start:stop                
                    gi = g_new[i]
                    gi === K + 1 && continue # skip noise component

                    xi = view(X, i, :)

                    zi = z_new[i]
                    used_local[tid][gi][zi] = true

                    key = (gi, zi)
                    v = get!(local_maps[tid], key, Int64[])
                    push!(v, i)
                    nkl = get!(nkl_local[tid], key, 0)
                    nkl_local[tid][key] = nkl + 1

                    sx = get!(sx_local[tid], key, zeros(Float64, d))
                    sx_local[tid][key] = sx .+ xi
                    sxx = get!(sxx_local[tid], key, zeros(Float64, d, d))
                    sxx_local[tid][key] = sxx .+ xi * xi'
                end


            end

            # merge serially
            clst_idx = Dict{Tuple{Int64,Int64},Vector{Int64}}()
            used_glb = [falses(L[k]) for k in 1:K]
            nkl_glb = Dict{Tuple{Int64,Int64}, Int64}()
            sx_glb = Dict{Tuple{Int64,Int64}, Vector{Float64}}()
            sxx_glb = Dict{Tuple{Int64,Int64}, Matrix{Float64}}()  # Updated to Matrix{Float64}

            for tid in 1:T
                for k in 1:K
                    used_glb[k] .|= used_local[tid][k]
                end

                m = nkl_local[tid]
                for (k, nkl) in m
                    nkl_glb[k] = get!(nkl_glb, k, 0) + nkl
                    sx_glb[k] = get!(sx_glb, k, zeros(Float64, d)) .+ sx_local[tid][k]
                    sxx_glb[k] = get!(sxx_glb, k, zeros(Float64, d, d)) .+ sxx_local[tid][k]
                    v = get!(clst_idx, k, Int64[])
                    append!(v, local_maps[tid][k])
                end
            end

        else
            L = [size(phis_new[k], 2) for k in 1:K] # number of clusters in each component
            used_glb = [falses(L[k]) for k in 1:K]
            clst_idx = Dict{Tuple{Int64,Int64},Vector{Int64}}()
            nkl_glb = Dict{Tuple{Int64,Int64},Int64}()
            sx_glb = Dict{Tuple{Int64,Int64},Vector{Float64}}()  # Updated to Vector{Float64}
            sxx_glb = Dict{Tuple{Int64,Int64},Matrix{Float64}}()  # Updated to Matrix{Float64}
            @inbounds for i in 1:n
                gi = g_new[i]
                gi === K + 1 && continue # skip noise component

                xi = view(X, i, :)
                zi = z_new[i]
                used_glb[gi][zi] = true

                key = (gi, zi)
                v = get!(clst_idx, key, Int64[])
                push!(v, i)
                nkl = get!(nkl_glb, key, 0)
                nkl_glb[key] = nkl + 1

                sx = get!(sx_glb, key, zeros(Float64, d))
                sx_glb[key] = sx .+ xi
                sxx = get!(sxx_glb, key, zeros(Float64, d, d))
                sxx_glb[key] = sxx .+ xi * xi'
            end

        end

       ### delete idle clusters, update z_new accordingly, update ϕs, σs, and βs for empty components
        old2new = Vector{Vector{Int64}}(undef, K)
        for k in 1:K
            usedk = used_glb[k]
            Lk = length(usedk)
            old2new[k] = zeros(Int64, Lk)
            if all(usedk)
                # identity mapping
                for l in 1:Lk
                    old2new[k][l] = l
                end
                continue
            end

            # absorb deleted β mass into the last kept label
            # (choose lastkeep first, before we mutate the arrays)
            lastkeep = findlast(usedk)
            if isnothing(lastkeep) ## update β, ϕ and σ for empty components
                if r[k, :] != [0.0, 0.0]
                    phis_new[k] = reshape(rcopy(R"rtmvnorm(n=1, mu=$(c[k,:]), sigma=$(Σ0), lb=$(rect_lb[k,:]), ub=$(rect_ub[k,:]))"), d, 1)
                else
                    phis_new[k] = reshape(c[k, :], d, 1)
                end
                Sigmas_new[k] = rand(rng_glb, InverseWishart(v0, P0))
                betas_new[k] = rand(rng_glb, Dirichlet([1, γ]), 1)[:, 1]              
                continue
            end

            delmass = 0.0
            dels = Int64[]
            keeps = Int64[]
            newl = 0
            for l in 1:Lk
                if usedk[l]
                    newl += 1
                    old2new[k][l] = newl
                    push!(keeps, l)
                else
                    delmass += betas_new[k][l]
                    push!(dels, l)
                end
            end
            # Add mass to last kept stick
            betas_new[k][end] += delmass

            # Drop deleted entries from parameter arrays
            # (Use keeps to preserve order)
            deleteat!(betas_new[k], dels)
            phis_new[k] = phis_new[k][:, keeps]
            Sigmas_new[k] = Sigmas_new[k][:, :, keeps]
        end

       ### Remap keys in the global stats dictionaries to the new labels
        nkl_glb = remap_stats(old2new, nkl_glb)
        clst_idx = remap_stats(old2new, clst_idx)
        sx_glb = remap_stats(old2new, sx_glb)
        sxx_glb = remap_stats(old2new, sxx_glb)

       ### Update z_new according to the new labels
        for i in eachindex(z_new)
            g_new[i] == K + 1 && continue
            z_new[i] = old2new[g_new[i]][z_new[i]]
        end

      ### Reduce Functionn: update ϕₖs,σₖs for non-empty clusters
        for key in sort!(collect(keys(nkl_glb)))

            k, l = key

            # read-only lookups
            nkl = nkl_glb[(k, l)]
            skl = sx_glb[(k, l)]
            sxx = sxx_glb[(k, l)]

            center_k = c[k, :]
            Sigma_kl = Sigmas_new[k][:, :, l]

            inv_Sigma_kl = inv(Sigma_kl)

            if r[k, :] != [0.0, 0.0]
                ## posterior parameters for ϕₖₗ
                M_kl = nkl * inv_Sigma_kl + inv_Σ0
                b_kl = inv_Sigma_kl * skl + inv_Σ0 * center_k
                Sigma_pst = inv(M_kl)
                mu_pst = Sigma_pst * b_kl

                Ψ = Symmetric((Sigma_pst + transpose(Sigma_pst)) / 2)
                if !isposdef(Ψ)
                    jitter = 1e-10 * tr(Ψ) / d
                    Ψ = Symmetric(Matrix(Ψ) + jitter * I)
                end

                ## generate new ϕₖₗ
                phi_kl_new = rcopy(R"rtmvnorm(n=1, mu=$(mu_pst), sigma=$(Ψ), lb=$(rect_lb[k, :]), ub=$(rect_ub[k, :]))")
            else
                phi_kl_new = center_k
            end

            ## posterior parameters for Σₖₗ
            nS_kl = sxx + phi_kl_new * transpose(phi_kl_new) * nkl - phi_kl_new * transpose(skl) - skl * transpose(phi_kl_new)

            Ψ = Symmetric((nS_kl + P0 + transpose(nS_kl + P0)) / 2)
            if !isposdef(Ψ)
                jitter = 1e-10 * tr(Ψ) / d
                Ψ = Symmetric(Matrix(Ψ) + jitter * I)
            end

            Sigma_kl_new = rand(rng_glb, InverseWishart(nkl + v0, PDMat(Ψ)))

            # write to disjoint cells (thread-safe)
            Sigmas_new[k][:, :, l] = Sigma_kl_new
            phis_new[k][:, l] = phi_kl_new


        end

      ###


        ### update βs for all clusters, update uk_star and ik_star for non-empty clusters
        ### update ϕₖs,Σₖs for empty clusters
        uk_star = Dict{Int64, Vector{Float64}}()
        ik_star = Dict{Int64, Vector{Int64}}()

        for k in 1:K
            if haskey(nk, k)
                alphak_pst = [Float64(nkl_glb[(k, l)]) for l in 1:size(phis_new[k], 2)]
                append!(alphak_pst, γ)
                betas_new[k] = rand(rng_glb, Dirichlet(alphak_pst))
                uk_star[k] = [rand(rng_glb, f) for f in Beta.(1, alphak_pst[1:(end-1)])] .* betas_new[k][1:(end-1)]

                ik_star[k] = []
                for l in 1:size(phis_new[k], 2)
                    ind_kl = clst_idx[(k, l)]
                    append!(ik_star[k], sample(rng_glb, ind_kl))
                end
            end
        end

        k = K + 1
        if haskey(nk, k)
            uk_star[k] = rand(rng_glb, Beta(1, nk[k])) .* betas_new[k]
            ind_kl = findall(x -> x == 1, g_new .== k)
            ik_star[k] = sample(rng_glb, ind_kl, 1)
        end

        ###


        ### store values
        if t > burnin && ((t - burnin - 1) % thin == 0)
            it = (t - burnin - 1) ÷ thin + 1
            w_all[:, it] = w_new
            for (i, gi) in enumerate(g_new)
                g_count[i, gi] += 1
            end
            #g_all[:, it] = g_new
            #z_all[:, it] = z_new
            if it == 1
                for k in 1:K
                    ind_all[k, 1] = 1
                    append!(β_all[k], betas_new[k][1:(end-1)])
                    ϕ_all[k] = phis_new[k]
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

        if t % 3000 == 0
            println("Iteration $(t) completed.")
            flush(stdout)
        end

    end
    ###

    g_result = [argmax(row) for row in eachrow(g_count)]

    return w_all, g_result, β_all, ϕ_all, Σ_all, ind_all
end

function tdirichlet(n; rng=nothing, lb, ub, γ)
    if rng === nothing
        rng = MersenneTwister()
    end

    if sum(lb) > 1
        error("sum of lower bounds must be less than or equal to 1.")
    end

    if sum(ub) < 1
        error("sum of upper bounds must be greater than or equal to 1.")
    end

    K = length(γ)

    if n == 1
        sample = Vector{Float64}(undef, K)
        for k in (K-1):-1:1
            if k == K - 1
                lbk = max(lb[k], 1 - sum(ub) + ub[k])
                ubk = min(ub[k], 1 - sum(lb) + lb[k])
                α1 = γ[k]
                α2 = sum(γ) - γ[k]
                sample[k] = rand(rng, truncated(Beta(α1, α2); lower=lbk, upper=ubk), 1)[1]
            else
                denom = 1 - sum(sample[(k+1):(K-1)])
                lbk = max(lb[k] / denom, 1 - (sum(ub) - sum(ub[k:(K-1)])) / denom)
                ubk = min(ub[k] / denom, 1 - (sum(lb) - sum(lb[k:(K-1)])) / denom)
                α1 = γ[k]
                α2 = sum(γ) - sum(γ[k:(K-1)])
                sample[k] = rand(rng, truncated(Beta(α1, α2); lower=lbk, upper=ubk), 1)[1] * denom
            end
        end

        sample[K] = 1 - sum(sample[1:(K-1)])
    else
        sample = Matrix{Float64}(undef, K, n)
        for i in 1:n
            sample[:, i] = tdirichlet(1; lb=lb, ub=ub, γ=γ)
        end
    end
    return sample
end

function mv_comp_density_f(t, bp_ind, beta, phi, Sigma, burnin=0)
    ## recover the component density in the t-th MCMC iteration

    if t < burnin
        error("t must be greater than $(burnin)")
    end

    start = bp_ind[t-burnin]
    if (t - burnin) == length(bp_ind)
        last = length(beta)
    else
        last = bp_ind[t-burnin+1] - 1
    end
    beta_t = beta[start:last]
    phi_t = phi[:, start:last]
    Sigma_t = Sigma[:, :, start:last]
    f = MixtureModel(MvNormal, [(phi_t[:, i], Sigma_t[:, :, i]) for i in eachindex(beta_t)], (beta_t ./ sum(beta_t)))
    return f
end