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
@inline normalpdf(x::AbstractFloat, μ::AbstractFloat, σ::AbstractFloat) =
    (INV_SQRT2PI / σ) * exp(-0.5 * ((x - μ) / σ)^2)


function h(c, r; λ= 2., ν = 2.)
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

    @inbounds for i in 1:n
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

# Remap keys in the global stats dictionaries to the new labels
function remap_stats(old2new, d_in::Dict{Tuple{Int64,Int64},T}) where {T}
    d_out = Dict{Tuple{Int64,Int64},T}()
    @inbounds for ((k, l), v) in d_in
        m = old2new[k][l]
        m == 0 && continue   # drop empty clusters
        d_out[(k, m)] = v
    end
    return d_out
end


function MDP(x; inits, h=h, map_gz_mt=map_gz_mt, map_gz=map_gz!, truncatedMH=truncatedMH, remap_stats=remap_stats,
                 alpha=nothing, wlb = 0.05, wub = 0.95, γ=1.0, τ=2.0, κ=3.0, θ=1.0, λ=2.0, ν=2, σ0=1., 
                 vs=6., σs=1., burnin = 0, iteration = 500, thin=1, multithreads=false, seed=nothing)
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
    rngs = [MersenneTwister(seed + t) for t in 1:Threads.nthreads()]

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

    wlb = wlb .* ones(K)
    wub = wub .* ones(K)

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
    # betas_new = Dict{Int64,Vector{Float64}}() # the kth item is for betas in the kth component
    # phis_new = Dict{Int64,Vector{Float64}}() # the kth item is for phis in the kth component
    # sigmas_new = Dict{Int64,Vector{Float64}}() # the kth item is for sigmas in the kth component
    # uk_star = Dict{Int64,Vector{Float64}}() # the minimum of u of each beta in each component k
    # ik_star = Dict{Int64,Vector{Int64}}() # the indicator of uk_star of each beta in each component k (global index)

    betas_new = Vector{Vector{Float64}}(undef, K)
    phis_new = Vector{Vector{Float64}}(undef, K)
    sigmas_new = Vector{Vector{Float64}}(undef, K)
    uk_star = Dict{Int64,Vector{Float64}}() # the minimum of u of each beta in each component k
    ik_star = Dict{Int64,Vector{Int64}}() # the indicator of uk_star of each beta in each component k (global index)


    @inbounds for k in 1:K
        if haskey(nk, k)
            betas_new[k] = rand(rng_glb, Dirichlet([nk[k], γ]))
            uk_star[k] = rand(rng_glb, Beta(1, nk[k]), 1) .* betas_new[k][1]
            indk = findall(x -> x == k, g_new)
            ik_star[k] = [indk[sample(rng_glb, 1:nk[k])]]
        else
            betas_new[k] = rand(rng_glb, Dirichlet([1, γ]))
        end
        phis_new[k] = rand(rng_glb, fk_new[k], 1)
        sigmas_new[k] = sqrt.(rand(rng_glb, InverseGamma(αs, θs), 1))
    end
   ##

   ## initializing matrices to store MCMC results 
    ncol = (iteration - burnin - 1) ÷ thin + 1
    c_all = Matrix{Float64}(undef, K, ncol)
    r_all = Matrix{Float64}(undef, K, ncol)
    w_all = Matrix{Float64}(undef, K, ncol) # vector to store w's in each MCMC iteration
    g_count = zeros(Int64, n, K)  
    #g_all = Matrix{Int64}(undef, n, ncol) # matrix to store gs in each column
    #z_all = Matrix{Int64}(undef, n, ncol) # matrix to store zs in each column
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
    
        for k in 1:K
            # while betas_new[k][end] >= u_star
            #     nu = rand(rng_glb, Beta(1, γ), 1)[1]
            #     append!(betas_new[k], betas_new[k][end] * (1 - nu))
            #     betas_new[k][end-1] *= nu
            #     append!(phis_new[k], rand(rng_glb, fk_new[k], 1))
            #     append!(sigmas_new[k], sqrt.(rand(rng_glb, InverseGamma(αs, θs), 1)))
            # end
            
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

            append!(phis_new[k], rand(rng_glb, fk_new[k], m))
            append!(sigmas_new[k], sqrt.(rand(rng_glb, InverseGamma(αs, θs), m)))

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
        w_new = tdirichlet(1;rng=rng_glb, lb=wlb, ub=wub, γ=alpha_pst)
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
            @inbounds for i in 1:n
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
    
      ### Reduce Functionn: update ϕₖs,σₖs for non-empty clusters
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
                c_k = c_new[k]
                lb, ub = int_lb[k], int_ub[k]

                # ϕ posterior
                μ_post = (σ0^2 * skl + c_k * σkl^2) / (σ0^2 * nkl + σkl^2)
                σ_post = 1 / sqrt(nkl / (σkl^2) + 1 / (σ0^2))
                ϕ_new = rand(local_rng, truncated(Normal(μ_post, σ_post), lower=lb, upper=ub))

                # σ posterior
                ssq = sxx + ϕ_new^2 * nkl - 2 * ϕ_new * skl
                αs_post = αs + nkl / 2
                θs_post = θs + ssq / 2
                σ_new = sqrt(rand(local_rng, InverseGamma(αs_post, θs_post)))

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
                c_k = c_new[k]
                lb, ub = int_lb[k], int_ub[k]

                # ϕ posterior
                μ_post = (σ0^2 * skl + c_k * σkl^2) / (σ0^2 * nkl + σkl^2)
                σ_post = 1 / sqrt(nkl / (σkl^2) + 1 / (σ0^2))
                ϕ_new = rand(rng_glb, truncated(Normal(μ_post, σ_post), lower=lb, upper=ub))

                # σ posterior
                ssq = sxx + ϕ_new^2 * nkl - 2 * ϕ_new * skl
                αs_post = αs + nkl / 2
                θs_post = θs + ssq / 2
                σ_new = sqrt(rand(rng_glb, InverseGamma(αs_post, θs_post)))

                # write to disjoint cells (thread-safe)
                sigmas_new[k][l] = σ_new
                phis_new[k][l] = ϕ_new


            end
        end
        
      ###
    
      ### update βs for all clusters, update uk_star and ik_star for non-empty clusters

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

                c_new[k] = rand(rng_glb, truncated(Normal(mu_c, sigma_c), lower=lb, upper=ub))
                continue
            end

            if k == K
                lb = c_new[k-1] + r_new[k-1] + r_new[k] + (λ / (-log(uh_new)))^(1 / ν)
                c_new[k] = rand(rng_glb, truncated(Normal(cc, τ), lower=lb))
            elseif k == 1
                ub = c_new[k+1] - r_new[k+1] - r_new[k] - (λ / (-log(uh_new)))^(1 / ν)
                c_new[k] = rand(rng_glb, truncated(Normal(cc, τ), upper=ub))
            else
                lb = c_new[k-1] + r_new[k-1] + r_new[k] + (λ / (-log(uh_new)))^(1 / ν)
                ub = c_new[k+1] - r_new[k+1] - r_new[k] - (λ / (-log(uh_new)))^(1 / ν)
                c_new[k] = rand(rng_glb, truncated(Normal(cc, τ), lower=lb, upper=ub))
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
                continue
            end
            
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
            r_new[k] = rand(rng_glb, truncated(Gamma(κ, θ), lower=lb, upper=ub))
            
        end

      ###

      ### update uh
        uh_new = rand(rng_glb, Uniform(0, h(c_new, r_new; λ=λ, ν=ν)))
      ###
    
      ### update int_lb, int_ub and fk_new
        int_lb = c_new .- r_new
        int_ub = c_new .+ r_new
        fk_new = map((x, y, z) -> truncated(x, lower=y, upper=z), Normal.(c_new, σ0), int_lb, int_ub) # vector of base measure fₖ's
      ### 
        
      ### store values
        if t > burnin && ((t - burnin - 1) % thin == 0)
            icol = (t - burnin - 1) ÷ thin + 1
            w_all[:, icol] = w_new
            c_all[:, icol] = c_new
            r_all[:, icol] = r_new
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

    return w_all, c_all, r_all, g_result, β_all, ϕ_all, σ_all, ind_all

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
            if k == K-1
                lbk = max(lb[k], 1-sum(ub)+ub[k])
                ubk = min(ub[k], 1-sum(lb)+lb[k])
                α1 = γ[k]
                α2 = sum(γ) - γ[k]
                sample[k] = rand(rng, truncated(Beta(α1, α2);lower=lbk, upper=ubk), 1)[1]
            else
                denom = 1 - sum(sample[(k+1):(K-1)])
                lbk = max(lb[k]/denom, 1- (sum(ub)-sum(ub[k:(K-1)]))/denom)
                ubk = min(ub[k]/denom, 1 - (sum(lb)-sum(lb[k:(K-1)]))/denom)
                α1 = γ[k]
                α2 = sum(γ) - sum( γ[k:(K-1)] )
                sample[k] = rand(rng, truncated(Beta(α1, α2); lower=lbk, upper=ubk), 1)[1] * denom
            end
        end

        sample[K] = 1 - sum(sample[1:(K-1)])
    else
        sample = Matrix{Float64}(undef, K, n)
        for i in 1:n
            sample[:,i] = tdirichlet(1; lb=lb, ub=ub, γ=γ)
        end
    end
    return sample
end
