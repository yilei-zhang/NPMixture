using Random
using Distributions
using Statistics
using StatsBase
using Base.Threads
using LinearAlgebra
using PDMats
using PyPlot
using SpecialFunctions
using DataFrames
using CSV
using JLD2
using QuadGK

const INV_SQRT2PI = inv(sqrt(2π))
@inline normalpdf(x::AbstractFloat, μ::AbstractFloat, σ::AbstractFloat) = (INV_SQRT2PI / σ) * exp(-0.5 * ((x - μ) / σ)^2)


function map_gz_mt(x, K, uk_star, ik_star, w, betas, phis, sigmas,
    g_cur::Vector{Int64}, z_cur::Vector{Int64}; rngs)

    ## function for updating ui, gi, zi for each sample tuple (i, xi, gi, zi)

    n = length(x)
    T = Threads.nthreads()

    g_out = Vector{Int64}(undef, n)
    z_out = Vector{Int64}(undef, n)

    Lmax = maximum(length, betas)

    @threads for tid in 1:T
        rng = rngs[tid]

        w_k = Vector{Float64}(undef, K + 1)
        betas_indx = [Vector{Int}(undef, Lmax) for _ in 1:K]
        pdf_x = [Vector{Float64}(undef, Lmax) for _ in 1:K]
        nidxs = Vector{Int}(undef, K)

        start = div((tid - 1) * n, T) + 1
        stop = div(tid * n, T)

        for i in start:stop
            xi = x[i]
            gi = g_cur[i]
            zi = z_cur[i]

            ustar_x = uk_star[gi][zi]
            istar_x = ik_star[gi][zi]
            beta_x = betas[gi][zi]
            ui = (i == istar_x) ? ustar_x : rand(rng)[1] * (beta_x - ustar_x) + ustar_x

            tot = 0.0
            for k in 1:K
                βk = betas[k]
                nidx = 0
                s = 0.0
                ϕk = phis[k]
                σk = sigmas[k]
                for l in eachindex(βk)
                    if βk[l] > ui
                        nidx += 1
                        betas_indx[k][nidx] = l
                        pdf_x[k][nidx] = normalpdf(xi, ϕk[l], σk[l])
                        s += pdf_x[k][nidx]
                    end
                end
                nidxs[k] = nidx
                if nidx == 0
                    w_k[k] = 0.0
                else
                    w_k[k] = w[k] * s
                end
                tot += w_k[k]
            end

            g_new = gi
            if tot > 0.0
                r = rand(rng) * tot
                acc = 0.0
                for k in 1:K
                    acc += w_k[k]
                    if r <= acc
                        g_new = k
                        break
                    end
                end
            end

            # choose z within component gnew
            nidx = nidxs[g_new]
            s = sum(pdf_x[g_new][1:nidx])
            if s == 0.0
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

            g_out[i] = g_new
            z_out[i] = z_new

        end

    end

    return g_out, z_out

end

function map_gz!(x, K, uk_star, ik_star, w, betas, phis, sigmas,
    g_cur::Vector{Int64}, z_cur::Vector{Int64},
    g_out::Vector{Int64}, z_out::Vector{Int64}; rng)

    n = length(x)
    Lmax = maximum(length, betas)
    w_k = Vector{Float64}(undef, K)
    betas_indx = [Vector{Int}(undef, Lmax) for _ in 1:K]
    pdf_x = [Vector{Float64}(undef, Lmax) for _ in 1:K]
    nidxs = Vector{Int}(undef, K)

    for i in 1:n
        xi = x[i]
        gi = g_cur[i]
        zi = z_cur[i]

        ustar_x = uk_star[gi][zi]
        istar_x = ik_star[gi][zi]
        beta_x = betas[gi][zi]
        ui = (i == istar_x) ? ustar_x : rand(rng) * (beta_x - ustar_x) + ustar_x

        tot = 0.0
        for k in 1:K
            βk = betas[k]
            nidx = 0
            s = 0.0
            ϕk = phis[k]
            σk = sigmas[k]
            for l in eachindex(βk)
                if βk[l] > ui
                    nidx += 1
                    betas_indx[k][nidx] = l
                    pdf_x[k][nidx] = normalpdf(xi, ϕk[l], σk[l])
                    s += pdf_x[k][nidx]
                end
            end
            nidxs[k] = nidx
            if nidx == 0
                w_k[k] = 0.0
            else
                w_k[k] = w[k] * s
            end
            tot += w_k[k]
        end

        g_new = gi
        if tot > 0.0
            r = rand(rng) * tot
            acc = 0.0
            for k in 1:K
                acc += w_k[k]
                if r <= acc
                    g_new = k
                    break
                end
            end
        end

        # choose z within component gnew
        nidx = nidxs[g_new]
        s = sum(pdf_x[g_new][1:nidx])
        if s == 0.0
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

        g_out[i] = g_new
        z_out[i] = z_new

    end

    return g_out, z_out

end

function remap_stats(old2new, d_in::Dict{Tuple{Int64,Int64},T}) where {T}
    d_out = Dict{Tuple{Int64,Int64},T}()
    @inbounds for ((k, l), v) in d_in
        m = old2new[k][l]
        m == 0 && continue   # drop empty clusters
        d_out[(k, m)] = v
    end
    return d_out
end

function MDP_sigma(x; inits, map_gz_mt=map_gz_mt, map_gz=map_gz!, alpha=nothing, σl, σu, wlb=0.05, wub=0.95, 
            γ=1.0, σ0=1., vs=6., σs=1., burnin=0, iteration=500, thin=1, 
            multithreads=false, seed=nothing)
    ## Mxiture of DP with K components ∑ₖwₖDP(γfₖ)       
    ## Using slice sampling 
    ## x: samples 
    ## K: number of clusters
    ## inits: initial value tuple (w, c, r), w ∈ R^{2×1}, c ∈ R, r ∈ R+ 
    ## σl: component 1 has σ<σl 
    ## σu: component 2 has σ>σu
    ## map_gz: map function to update component indicator g and cluster indicator z
    ## reduce_ϕσ: reduce function to update cluster parameters ϕ and σ
    ## alpha: vector [α₁,α₂], α₁,α₂ >0, prior parameters for w~Beta(alpha)
    ## γ: γ ∈ R₊, concentration parameter γ of DPs
    ## σ0: parameter in truncated normal prior fₖ for ϕₖᵢ, fₖ:ϕₖᵢ~ N(cₖ, σ0²)1{cₖ-rₖ ≤ x ≤ cₖ+rₖ}
    ## vs: parameter in InverseGamma prior gₖ for σₖᵢ², σₖᵢ² ~ InverseGamma(vs/2, σs²vs/2)
    ## σs: parameter in InverseGamma prior gₖ for σₖᵢ², σₖᵢ² ~ InverseGamma(vs/2, σs²vs/2)
    ## seed: random seed to ensure reproducibility
  
  ### global settings
    if seed === nothing
        seed = rand(RandomDevice(), UInt64) # generate random seed
    end
    rng_glb = MersenneTwister(seed)
    rngs = [MersenneTwister(seed + t) for t in 1:Threads.nthreads()]
  ###
  
  ### check initial values
    w, c, r = inits

    w = Float64.(w)
    c = Float64.(c)
    r = Float64.(r)
    K = length(w)

    wlb = wlb .* ones(K)
    wub = wub .* ones(K)
  ###

  ### set hyperparameters
    n = length(x) 
    αs = vs/2 # parameter for InverseGamma(αs, θs)
    θs = σs^2 * vs/2 # parameter for InverseGamma(αs, θs)
    if alpha === nothing
        alpha = 1/K .* ones(K)
    end
  ###
  
  ### initialization

   ## initializing each variable in the first MCMC iteration
    w_new = w
    
    int_lb = c - r
    int_ub = c + r
    fk_new = truncated(Normal(c, σ0),lower=int_lb, upper=int_ub) # base measure fₖ's for location parameter
    hk_new = [truncated(InverseGamma(αs, θs), upper=σl), truncated(InverseGamma(αs, θs), lower=σu)] # base measure h₁'s for scale parameter
    
    g_new = sample(rng_glb, 1:K, Weights(w_new), n) #indicator of DP component gᵢ=k for DP(γfₖ)
    z_new = ones(Int64, n)  #indicator of cluster within the DP_gᵢ for each sample
    nk = countmap(g_new) # number of samples in each component
    betas_new = Vector{Vector{Float64}}(undef, K)
    phis_new = Vector{Vector{Float64}}(undef, K)
    sigmas_new = Vector{Vector{Float64}}(undef, K)
    uk_star = Dict{Int64,Vector{Float64}}() # the minimum of u of each beta in each component k
    ik_star = Dict{Int64,Vector{Int64}}() # the indicator of uk_star of each beta in each component k (global index)

    for k in 1:K
        if haskey(nk, k)
            betas_new[k] = rand(rng_glb, Dirichlet([nk[k], γ]))
            uk_star[k] = rand(rng_glb, Beta(1, nk[k]), 1) .* betas_new[k][1]
            indk = findall(x -> x == k, g_new)
            ik_star[k] = [sample(rng_glb, indk)]
        else
            betas_new[k] = rand(rng_glb, Dirichlet([1, γ]))
        end
        phis_new[k] = rand(rng_glb, fk_new, 1)
        sigmas_new[k] = sqrt.(rand(rng_glb, hk_new[k], 1))
    end
   ##

   ## initializing matrices to store MCMC results 
    ncol = (iteration - burnin - 1) ÷ thin + 1
    w_all = Matrix{Float64}(undef, K, ncol) # vector to store w's in each MCMC iteration 
    #g_all = Matrix{Int64}(undef, n, ncol) # matrix to store gs in each column
    #z_all = Matrix{Int64}(undef, n, ncol) # matrix to store zs in each column
    g_count = zeros(Int64, n, K)
    ind_all = Matrix{Int64}(undef, K, ncol) # starting locations for betas, phis, sigmas in each MCMC iteration for component k
    β_all = Vector{Vector{Float64}}(undef, K) # βs for the K components 
    σ_all = Vector{Vector{Float64}}(undef, K) # σs for the K components
    ϕ_all = Vector{Vector{Float64}}(undef, K) # ϕs for the K components
    for k in 1:K
        β_all[k] = []
        σ_all[k] = []
        ϕ_all[k] = []
    end
   ##

  ###

  ### Gibbs sampling (slice sampler)
    for t in 1:((ncol-1)*thin+1+burnin)

      ### update gᵢ and zᵢ
        u_star = minimum(minimum.(values(uk_star))) # find u_star for all components

       ## instantiate beta, phis, sigmas based on u_star  
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

            append!(phis_new[k], rand(rng_glb, fk_new, m))
            append!(sigmas_new[k], sqrt.(rand(rng_glb, hk_new[k], m)))

        end
       ##
        
       ## update gi, zi for each sample xi
        if !multithreads
            g_old = copy(g_new)
            z_old = copy(z_new)
            map_gz(x, K, uk_star, ik_star, w_new, betas_new, phis_new, sigmas_new,
                g_old, z_old, g_new, z_new; rng=rng_glb)
        else
            g_old = copy(g_new)
            z_old = copy(z_new)
            g_new, z_new = map_gz_mt(x, K, uk_star, ik_star, w_new, betas_new, phis_new, sigmas_new,
                g_old, z_old; rngs=rngs)
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
        #w_new = rand(rng_glb, Dirichlet(alpha_pst), 1)[:,1]
        w_new = tdirichlet(1; rng=rng_glb, lb=wlb, ub=wub, γ=alpha_pst)
        if any(isnan.(w_new))
            w_new = tdirichlet(1; rng=rng_glb, lb=wlb, ub=wub, γ=alpha)
        end
      ###
        
      ### middle steps
        ### building clusters indices and their sufficient statistics
        if multithreads
            T = Threads.nthreads()
            L = [length(phis_new[k]) for k in 1:K] # number of clusters in each component
            used_local = [[falses(L[k]) for k in 1:K] for _ in 1:T]
            local_maps = [Dict{Tuple{Int64,Int64},Vector{Int64}}() for _ in 1:T] # local maps for each thread
            nkl_local = [Dict{Tuple{Int64,Int64},Int64}() for _ in 1:T]
            sx_local = [Dict{Tuple{Int64,Int64},Float64}() for _ in 1:T]
            sxx_local = [Dict{Tuple{Int64,Int64},Float64}() for _ in 1:T]

            @threads for tid in 1:T
                #tid = Threads.threadid()
                start = div((tid - 1) * n, T) + 1
                stop = div(tid * n, T)

                for i in start:stop
                    gi = g_new[i]
                    zi = z_new[i]
                    used_local[tid][gi][zi] = true

                    key = (gi, zi)
                    v = get!(local_maps[tid], key, Int64[])
                    push!(v, i)
                    nkl = get!(nkl_local[tid], key, 0)
                    nkl_local[tid][key] = nkl + 1
                    sx = get!(sx_local[tid], key, 0.0)
                    sx_local[tid][key] = sx + x[i]
                    sxx = get!(sxx_local[tid], key, 0.0)
                    sxx_local[tid][key] = sxx + x[i]^2

                end

            end

            # merge serially
            clst_idx = Dict{Tuple{Int64,Int64},Vector{Int64}}()
            used_glb = [falses(L[k]) for k in 1:K]
            nkl_glb = Dict{Tuple{Int64,Int64},Int64}()
            sx_glb = Dict{Tuple{Int64,Int64},Float64}()
            sxx_glb = Dict{Tuple{Int64,Int64},Float64}()

            for tid in 1:T
                for k in 1:K
                    used_glb[k] .|= used_local[tid][k]
                end

                m = nkl_local[tid]
                for (k, nkl) in m
                    nkl_glb[k] = get!(nkl_glb, k, 0) + nkl
                    sx_glb[k] = get!(sx_glb, k, 0.0) + sx_local[tid][k]
                    sxx_glb[k] = get!(sxx_glb, k, 0.0) + sxx_local[tid][k]
                    v = get!(clst_idx, k, Int64[])
                    append!(v, local_maps[tid][k])
                end
            end

        else
            L = [length(phis_new[k]) for k in 1:K] # number of clusters in each component
            used_glb = [falses(L[k]) for k in 1:K]
            clst_idx = Dict{Tuple{Int64,Int64},Vector{Int64}}()
            nkl_glb = Dict{Tuple{Int64,Int64},Int64}()
            sx_glb = Dict{Tuple{Int64,Int64},Float64}()
            sxx_glb = Dict{Tuple{Int64,Int64},Float64}()
            for i in 1:n
                gi = g_new[i]
                zi = z_new[i]
                used_glb[gi][zi] = true

                key = (gi, zi)
                v = get!(clst_idx, key, Int64[])
                push!(v, i)
                nkl = get!(nkl_glb, key, 0)
                nkl_glb[key] = nkl + 1
                sx = get!(sx_glb, key, 0.0)
                sx_glb[key] = sx + x[i]
                sxx = get!(sxx_glb, key, 0.0)
                sxx_glb[key] = sxx + x[i]^2
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
                phis_new[k] = rand(rng_glb, fk_new[k], 1)
                sigmas_new[k] = sqrt.(rand(rng_glb, InverseGamma(αs, θs), 1))
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
            phis_new[k] = phis_new[k][keeps]
            sigmas_new[k] = sigmas_new[k][keeps]
        end

        # Remap keys in the global stats dictionaries to the new labels
        nkl_glb = remap_stats(old2new, nkl_glb)
        sx_glb = remap_stats(old2new, sx_glb)
        sxx_glb = remap_stats(old2new, sxx_glb)
        clst_idx = remap_stats(old2new, clst_idx)

        # Update z_new according to the new labels
        for i in eachindex(z_new)
            z_new[i] = old2new[g_new[i]][z_new[i]]
        end
      ###


      ### Reduce Function 
        if multithreads
            # materialize & size-balance work (bigger nkl first)
            pairs = collect(keys(nkl_glb))                       # Vector{Tuple{Int,Int}}
            ord = sortperm(pairs; by=k -> nkl_glb[k], rev=true)

            @threads :static for idx in ord
                local_rng = rngs[Threads.threadid()]
                k, l = pairs[idx]

                # read-only lookups
                nkl = nkl_glb[(k, l)]
                skl = sx_glb[(k, l)]
                sxx = sxx_glb[(k, l)]

                σkl = sigmas_new[k][l]
                c_k = c
                lb, ub = c-r, c+r

                # ϕ posterior
                μ_post = (σ0^2 * skl + c_k * σkl^2) / (σ0^2 * nkl + σkl^2)
                σ_post = 1 / sqrt(nkl / (σkl^2) + 1 / (σ0^2))
                ϕ_new = rand(local_rng, truncated(Normal(μ_post, σ_post), lower=lb, upper=ub))

                # σ posterior
                ssq = sxx + ϕ_new^2 * nkl - 2 * ϕ_new * skl
                αs_post = αs + nkl / 2
                θs_post = θs + ssq / 2
                hkl = InverseGamma(αs, θs)
                if k == 1
                    try
                        # code that might throw an error
                        hkl = truncated(InverseGamma(αs_post, θs_post), upper=σl)
                    catch e
                        # code to run if an error occurs
                        hkl = truncated(InverseGamma(αs, θs), upper=σl)
                    end
                else
                    try
                        # code that might throw an error
                        hkl = truncated(InverseGamma(αs_post, θs_post), lower=σu)
                    catch e
                        # code to run if an error occurs
                        hkl = truncated(InverseGamma(αs, θs), lower=σu)
                    end
                end
                σ_new = sqrt(rand(local_rng, hkl, 1)[1])  # Updated to use local_rng

                # write to disjoint cells (thread-safe)
                sigmas_new[k][l] = σ_new
                phis_new[k][l] = ϕ_new
            end

        else

            for key in sort!(collect(keys(nkl_glb)))

                k, l = key

                nkl = nkl_glb[(k, l)]
                skl = sx_glb[(k, l)]
                sxx = sxx_glb[(k, l)]

                σkl = sigmas_new[k][l]
                c_k = c
                lb, ub = c - r, c + r
                
                # ϕ posterior
                μ_post = (σ0^2 * skl + c_k * σkl^2) / (σ0^2 * nkl + σkl^2)
                σ_post = 1 / sqrt(nkl / (σkl^2) + 1 / (σ0^2))
                ϕ_new = rand(rng_glb, truncated(Normal(μ_post, σ_post), lower=lb, upper=ub))

                # σ posterior
                ssq = sxx + ϕ_new^2 * nkl - 2 * ϕ_new * skl
                αs_post = αs + nkl / 2
                θs_post = θs + ssq / 2

                hkl = InverseGamma(αs, θs)
                if k == 1
                    try
                        # code that might throw an error
                        hkl = truncated(InverseGamma(αs_post, θs_post), upper=σl)
                    catch e
                        # code to run if an error occurs
                        hkl = truncated(InverseGamma(αs, θs), upper=σl)
                    end
                else
                    try
                        # code that might throw an error
                        hkl = truncated(InverseGamma(αs_post, θs_post), lower=σu)
                    catch e
                        # code to run if an error occurs
                        hkl = truncated(InverseGamma(αs, θs), lower=σu)
                    end
                end
                σ_new = sqrt(rand(rng_glb, hkl, 1)[1])  # Updated to use hkl

                # write to disjoint cells (thread-safe)
                sigmas_new[k][l] = σ_new
                phis_new[k][l] = ϕ_new

            end
        end
      ###
    
      ### update βs for all clusters, update uk_star and ik_star for non-empty clusters
      ### update ϕₖs,σₖs for empty clusters

        uk_star = Dict{Int64,Vector{Float64}}()
        ik_star = Dict{Int64,Vector{Int64}}()

        for k in 1:K
            if haskey(nk, k)
                alphak_pst = Float64.([nkl_glb[(k, l)] for l in eachindex(phis_new[k])])
                append!(alphak_pst, γ)
                betas_new[k] = rand(rng_glb, Dirichlet(alphak_pst))
                uk_star[k] = [rand(rng_glb, f) for f in Beta.(1, alphak_pst[1:(end-1)])] .* betas_new[k][1:(end-1)]

                ik_star[k] = []
                for l in eachindex(phis_new[k])
                    ind_kl = clst_idx[(k, l)]
                    append!(ik_star[k], sample(rng_glb, ind_kl))
                end
            
            end
        end
      ###
        
      ### store values
        if t > burnin && ((t - burnin - 1) % thin == 0)
            icol = (t - burnin - 1) ÷ thin + 1
            w_all[:, icol] = w_new
            #g_all[:,icol] = g_new
            #z_all[:,icol] = z_new
            for (i, gi) in enumerate(g_new)
                g_count[i, gi] += 1
            end

            for k in 1:K
                ind_all[k, icol] = length(β_all[k]) + 1
                append!(β_all[k], betas_new[k][1:(end-1)])
                append!(ϕ_all[k], phis_new[k])
                append!(σ_all[k], sigmas_new[k])
            end
 
        end
      ###
        if t % 5000 == 0
            println("Iteration $(t) completed.")
            flush(stdout)
        end

    end
  ###
    g_result = [argmax(row) for row in eachrow(g_count)]

    return w_all, g_result, β_all, ϕ_all, σ_all, ind_all

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

function comp_density_f(t, bp_ind, beta, phi, sigma, burnin=0)
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
    phi_t = phi[start:last]
    sigma_t = sigma[start:last]
    f = MixtureModel(Normal, [(phi_t[i], sigma_t[i]) for i in eachindex(phi_t)], (beta_t ./ sum(beta_t)))

    return f
end

