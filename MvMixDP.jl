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


struct Rect
    x1::Float64
    y1::Float64
    x2::Float64
    y2::Float64
end

function h_mv(c, r; λ=2.0, ν=2, ρ=1.0)
    ## repulsive prior for c and r
    ## c: K×d matrix, r: K×1 vector
    ## λ, ν, ρ: parameters in h(c,r) = minᵢⱼexp{-λ/max((||cᵢ-cⱼ||-ρ(rᵢ+rⱼ)),0)^ν}   

    K = length(r)
    gap_min = +Inf

    for i in 1:(K-1)
        for j in (i+1):K
            gap = norm(c[i,:] .- c[j,:], Inf) - ρ*(r[i] + r[j])
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

function mv_map_gz_mt(X, K, uk_star, ik_star, w, betas, phis, Sigmas,
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
        #tid = Threads.threadid()
        rng = rngs[tid]

        w_k = Vector{Float64}(undef, K)
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


            # sample component g_new
            g_new = gi
            if total > 0.0
                r = rand(rng) * total
                acc = 0.0
                for k in 1:K
                    acc += w_k[k]
                    if r <= acc
                        g_new = k
                        break
                    end
                end
            end

            # sample cluster z_new within g_new
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

            g_out[i] = g_new
            z_out[i] = z_new

        end

    end
    return g_out, z_out
end

function mv_map_gz!(X, K, uk_star, ik_star, w, betas, phis, Sigmas,
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
    w_k = Vector{Float64}(undef, K)
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

        # sample component g_new
        g_new = gi
        if total > 0.0
            r = rand(rng) * total
            acc = 0.0
            for k in 1:K
                acc += w_k[k]
                if r <= acc
                    g_new = k
                    break
                end
            end
        end

        # sample cluster z_new within g_new
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

        g_out[i] = g_new
        z_out[i] = z_new

    end

    return g_out, z_out
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

function remap_stats(old2new, d_in::Dict{Tuple{Int64,Int64},T}) where {T}
    d_out = Dict{Tuple{Int64,Int64},T}()
    for ((k, l), v) in d_in
        m = old2new[k][l]
        m == 0 && continue   # drop empty clusters
        d_out[(k, m)] = v
    end
    return d_out
end

function intersect_rect(a::Rect, b::Rect)
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    (x1 < x2 && y1 < y2) ? Rect(x1, y1, x2, y2) : nothing
end

# Build sorted unique breakpoints from s1 and clipped obstacles
function build_breaks(s1::Rect, obs::Vector{Rect})
    xs = Set([s1.x1, s1.x2])
    ys = Set([s1.y1, s1.y2])
    for r in obs
        push!(xs, r.x1)
        push!(xs, r.x2)
        push!(ys, r.y1)
        push!(ys, r.y2)
    end
    return sort!(collect(xs)), sort!(collect(ys))
end

# Check if cell (i,j) intersects rectangle r
function cell_intersects(r::Rect, xs, ys, i, j)
    cx1, cx2 = xs[j], xs[j+1]
    cy1, cy2 = ys[i], ys[i+1]
    # open-interval style overlap (interior-disjoint)
    return !(cx2 <= r.x1 || r.x2 <= cx1 || cy2 <= r.y1 || r.y2 <= cy1)
end

function build_mask(clipped::Vector{Rect}, xs, ys)
    nrows = length(ys) - 1
    ncols = length(xs) - 1
    blocked = falses(nrows, ncols)
    for i in 1:nrows, j in 1:ncols
        # skip cells fully outside s1 (shouldn't happen if xs,ys from s1 bounds)
        isblocked = false
        for r in clipped
            if cell_intersects(r, xs, ys, i, j)
                isblocked = true
                break
            end
        end
        blocked[i, j] = isblocked
    end
    return blocked
end

function largest_rect_in_hist(H::Vector{Float64}, dx::Vector{Float64})
    stack = Int[]            # indices with strictly increasing H
    best_area = 0.0
    best_l = 1
    best_r = 0
    best_h = 0.0
    # prefix sums of dx for O(1) span-width queries
    ps = cumsum([0.0; dx])
    # helper to width over [l..r]
    width(l, r) = ps[r+1] - ps[l]

    N = length(H)
    for j in 1:(N+1)
        curH = (j <= N) ? H[j] : -Inf
        while !isempty(stack) && H[stack[end]] > curH
            top = pop!(stack)
            h = H[top]
            l = isempty(stack) ? 1 : (stack[end] + 1)
            r = j - 1
            w = width(l, r)
            area = h * w
            if area > best_area
                best_area = area
                best_l = l
                best_r = r
                best_h = h
            end
        end
        push!(stack, j)
    end
    return best_area, best_l, best_r, best_h
end

function largest_empty_axis_rect(lb_ck, ub_ck, c_notk, r_notk, rk, slice_dist; intersect_rect=intersect_rect,
    build_breaks=build_breaks, build_mask=build_mask,
    largest_rect_in_hist=largest_rect_in_hist)

    s1 = Rect(lb_ck[1], lb_ck[2], ub_ck[1], ub_ck[2])
    obstacles = Rect[]
    for i in eachindex(r_notk)
        c = c_notk[i, :]
        r = r_notk[i] + rk + slice_dist
        x1, y1 = c .- r
        x2, y2 = c .+ r
        push!(obstacles, Rect(x1, y1, x2, y2))
    end

    # 1) clip to s1 and keep nonempty
    clipped = Rect[]
    for r in obstacles
        ir = intersect_rect(s1, r)
        if ir !== nothing
            push!(clipped, ir)
        end
    end

    # 2) discretize on all obstacle edges + s1 edges
    xs, ys = build_breaks(s1, clipped)
    dx = diff(xs)
    dy = diff(ys)
    nrows = length(dy)
    ncols = length(dx)

    # 3) blocked mask
    blocked = build_mask(clipped, xs, ys)

    # 4) sweep rows, accumulate physical heights, run weighted histogram
    H = zeros(Float64, ncols)
    best_area = 0.0
    best_rect = Rect(s1.x1, s1.y1, s1.x1, s1.y1)

    for i in 1:nrows
        # update histogram heights
        for j in 1:ncols
            H[j] = blocked[i, j] ? 0.0 : (H[j] + dy[i])
        end
        area, l, r, h = largest_rect_in_hist(H, dx)
        if area > best_area && l <= r
            # height h spans upward from current strip top
            y2 = ys[i+1]
            y1 = y2 - h
            x1 = xs[l]
            x2 = xs[r+1]
            best_area = area
            best_rect = Rect(x1, y1, x2, y2)
        end
    end
    return best_rect, best_area
end

function MvMDP(X; inits, h=h_mv, map_gz_mt=mv_map_gz_mt, map_gz=mv_map_gz!,
    truncatedMH=truncatedMH, largest_empty_axis_rect=largest_empty_axis_rect, 
    remap_stats=remap_stats, alpha=nothing, wlb=0.05, wub=0.95, γ=1.0,
    cc=nothing, Γ=nothing, κ=3.0, θ=1.0, λ=2.0, ν=2, σ0=1.0,
    v0=nothing, P0=nothing, multithreads=false, burnin=0, iteration=500, thin=1, seed=nothing)
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
    rngs = [MersenneTwister(seed + t) for t in 1:Threads.nthreads()]

    seed_r = rand(rng_glb, 1:1e6, 1)[1]
    R"set.seed($(seed_r))"
    R"library(TruncatedNormal)"
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


    for i in 1:(K-1)
        for j in (i+1):K
            if norm(c[i, :] - c[j, :], Inf) <= r[i] + r[j]
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

    wlb = wlb .* ones(K)
    wub = wub .* ones(K)
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
        v0 = d + 4.0
    end

    if Γ === nothing
        Γ = PDMat(Matrix{Float64}(d * I, d, d))
    end

    if P0 === nothing
        P0 = PDMat(Matrix{Float64}(d * v0 * I, d, d))
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
    uh_new = rand(rng_glb, Uniform(0, h(c_new, r_new; λ=λ, ν=ν)), 1)[1]
    rect_lb = c_new .- r_new
    rect_ub = c_new .+ r_new

    g_new = sample(rng_glb, 1:K, Weights(w_new), n) #indicator of DP component gᵢ=k for DP(γfₖ)
    z_new = ones(Int64, n)  #indicator of cluster within the DP_gᵢ for each sample
    nk = countmap(g_new) # number of samples in each component
    betas_new = Vector{Vector{Float64}}(undef, K) # the kth item is for betas in the kth component
    phis_new = Vector{Matrix{Float64}}(undef, K) # the kth item is for phis in the kth component
    Sigmas_new = Vector{Array{Float64}}(undef, K) # the kth item is for Sigmas in the kth component
    uk_star = Dict{Int64,Vector{Float64}}() # the minimum of u of each beta in each component k
    ik_star = Dict{Int64,Vector{Int64}}() # the indicator of uk_star of each beta in each component k (global index)
    for k in 1:K
        if haskey(nk, k)
            betas_new[k] = rand(rng_glb, Dirichlet([nk[k], γ]))
            uk_star[k] = rand(rng_glb, Beta(1, nk[k]), 1) .* betas_new[k][1]
            indk = findall(x -> x == k, g_new)
            ik_star[k] = [indk[sample(rng_glb, 1:nk[k])]]
        else
            betas_new[k] = rand(rng_glb, Dirichlet([1, γ]))
        end

        if r_new[k] != 0.0
            phis_new[k] = reshape(rcopy(R"rtmvnorm(n=1, mu=$(c_new[k,:]), sigma=$(Σ0), lb=$(rect_lb[k,:]), ub=$(rect_ub[k,:]))"), d, 1)
        else
            phis_new[k] = reshape(c_new[k, :], d, 1)
        end

        Sigmas_new[k] = rand(rng_glb, InverseWishart(v0, P0))
    end
    ##

    ## initializations to store MCMC results 
    nit = (iteration - burnin - 1) ÷ thin + 1
    c_all = Array{Float64}(undef, K, d, nit)
    r_all = Matrix{Float64}(undef, K, nit)
    w_all = Matrix{Float64}(undef, K, nit) # vector to store w's in each MCMC iteration 
    g_all = Matrix{Int64}(undef, n, nit) # matrix to store gs in each column
    #z_all = Matrix{Int64}(undef, n, nit) # matrix to store zs in each column
    ind_all = Matrix{Int64}(undef, K, nit) # starting locations for betas, phis, Sigmas in each MCMC iteration for component k
    β_all = Vector{Vector{Float64}}(undef, K) # the kth item is for betas in the kth component
    ϕ_all = Vector{Matrix{Float64}}(undef, K) # the kth item is for phis in the kth component
    Σ_all = Vector{Array{Float64}}(undef, K) # the kth item is for Sigmas in the kth component

    for k in 1:K
        β_all[k] = []
    end
    ##

    ###

    ### Gibbs sampling (slice sampler)
    ck_not_update = 0
    for t in 1:((nit-1)*thin+1+burnin)

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

            if r_new[k] != 0.0
                Z = rcopy(R"rtmvnorm(n=$m, mu=$(c_new[k,:]), sigma=$(Σ0),
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

        ### update gi, zi for each sample xi
        if !multithreads #!multithreads
            g_old = copy(g_new)
            z_old = copy(z_new)
            map_gz(X, K, uk_star, ik_star, w_new, betas_new, phis_new, Sigmas_new,
                g_old, z_old, g_new, z_new; rng=rng_glb)
        else
            g_old = copy(g_new)
            z_old = copy(z_new)
            g_new, z_new = map_gz_mt(X, K, uk_star, ik_star, w_new, betas_new, phis_new, Sigmas_new,
                g_old, z_old; rngs=rngs)
        end
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
        w_new = tdirichlet(1; rng=rng_glb, lb=wlb, ub=wub, γ=alpha_pst)
        ###

        ### middle steps
        if multithreads

            T = Threads.nthreads()
            L = [size(phis_new[k], 2) for k in 1:K] # number of clusters in each component
            used_local = [[falses(L[k]) for k in 1:K] for _ in 1:T]
            local_maps = [Dict{Tuple{Int64,Int64},Vector{Int64}}() for _ in 1:T] # local maps for each thread
            nkl_local = [Dict{Tuple{Int64,Int64},Int64}() for _ in 1:T]
            sx_local = [Dict{Tuple{Int64,Int64},Vector{Float64}}() for _ in 1:T]
            sxx_local = [Dict{Tuple{Int64,Int64},Matrix{Float64}}() for _ in 1:T]

            @threads for tid in 1:T
                #tid = Threads.threadid()
                start = div((tid - 1) * n, T) + 1
                stop = div(tid * n, T)

                for i in start:stop
                    gi = g_new[i]

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
            nkl_glb = Dict{Tuple{Int64,Int64},Int64}()
            sx_glb = Dict{Tuple{Int64,Int64},Vector{Float64}}()
            sxx_glb = Dict{Tuple{Int64,Int64},Matrix{Float64}}()  # Updated to Matrix{Float64}

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
                if r_new[k] != 0.0
                    phis_new[k] = reshape(rcopy(R"rtmvnorm(n=1, mu=$(c_new[k,:]), sigma=$(Σ0), lb=$(rect_lb[k,:]), ub=$(rect_ub[k,:]))"), d, 1)
                else
                    phis_new[k] = reshape(c_new[k, :], d, 1)
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
            z_new[i] = old2new[g_new[i]][z_new[i]]
        end
        ###

        ### Reduce Functionn: update ϕₖs,σₖs for non-empty clusters
        for key in sort!(collect(keys(nkl_glb)))
            k, l = key

            # read-only lookups
            nkl = nkl_glb[(k, l)]
            skl = sx_glb[(k, l)]
            sxx = sxx_glb[(k, l)]

            center_k = c_new[k, :]
            Sigma_kl = Sigmas_new[k][:, :, l]

            inv_Sigma_kl = inv(Sigma_kl)

            if r_new[k] != 0.0
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
                phi_kl_new = rcopy(R"rtmvnorm(n=1, mu=$(mu_pst), sigma=$(Ψ), lb=$(rect_lb[k,:]), ub=$(rect_ub[k,:]))")

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

            ## generate new Σₖₗ
            Sigma_kl_new = rand(rng_glb, InverseWishart(nkl + v0, PDMat(Ψ)))


            # write to disjoint cells (thread-safe)
            Sigmas_new[k][:, :, l] = Sigma_kl_new
            phis_new[k][:, l] = phi_kl_new


        end
        ###

        ### update βs for all clusters, update uk_star and ik_star for non-empty clusters
        ### update ϕₖs,Σₖs for empty clusters
        uk_star = Dict{Int64,Vector{Float64}}()
        ik_star = Dict{Int64,Vector{Int64}}()

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
        ###

        ### update c and r
        for k in 1:K

            ind_notk = setdiff(1:K, k)
            r_notk = r_new[ind_notk]
            c_notk = c_new[ind_notk, :]

            if haskey(nk, k)
                phiks = phis_new[k]
                rk = r_new[k]
                n_phik = size(phiks, 2)
                phiks_min = minimum(phiks, dims=2)[:, 1]
                phiks_max = maximum(phiks, dims=2)[:, 1]

                ### update cₖ
                ### posterior parameters
                Mk = n_phik * inv_Σ0 + inv_Γ
                bk = n_phik * inv_Σ0 * mean(phiks, dims=2)

                Sigma_c = inv(Mk)
                mu_c = (Sigma_c*bk)[:, 1]

                ### get lower bound and upper bound
                lb_ck = phiks_max .- rk
                ub_ck = phiks_min .+ rk

                best_rect, best_area = largest_empty_axis_rect(lb_ck, ub_ck, c_notk, r_notk, rk, (-λ / log(uh_new))^(1 / ν))

                if best_area <= 0.01^d
                    ck_not_update += 1
                else
                    lb_ck = [best_rect.x1, best_rect.y1]
                    ub_ck = [best_rect.x2, best_rect.y2]
                    c_new[k, :] = rcopy(R"rtmvnorm(n=1, mu=$(mu_c), sigma=$(Sigma_c), lb=$(lb_ck), ub=$(ub_ck))")
                end

                ### update rₖ
                ck = c_new[k, :]
                lb_rk = max(maximum(ck - phiks_min), maximum(phiks_max - ck))
                ub_rk = minimum([(norm(c_notk[i, :] - ck, Inf) - (-λ / log(uh_new))^(1 / ν)) - r_notk[i] for i in eachindex(r_notk)])
                tgt_f = x -> pdf.(Gamma(κ, θ), x) .* (1 ./ ((2 .* cdf.(Normal(0, σ0), x) - 1) .^ (2 * n_phik)))
                r_new[k] = truncatedMH((ub_rk + lb_rk) / 2, tgt_f, lb_rk, ub_rk; rng=rng_glb)
                ###

            else
                ### generate new cₖ
                min_dist_to_ck = (r_notk .+ r_new[k]) .+ (-λ / log(uh_new))^(1 / ν)
                n_sim = 0
                while n_sim < 10000 ## rejection sampling to update cₖ
                    n_sim += 1
                    c_sim = rand(rng_glb, MvNormal(cc, Γ), 1)[:, 1]
                    if all([norm(c_notk[i, :] - c_sim, Inf) > min_dist_to_ck[i] for i in eachindex(min_dist_to_ck)])
                        c_new[k, :] = c_sim
                        break
                    end
                end

                if n_sim == 10000
                    println("empty cluster: cₖ not updated")
                    ck_not_update += 1
                end

                ### generate new rₖ
                lb_rk = 0
                ub_rk = minimum([(norm(c_notk[i, :] - c_new[k, :], Inf) - (-λ / log(uh_new))^(1 / ν)) - r_notk[i] for i in eachindex(r_notk)])
                r_new[k] = rand(rng_glb, truncated(Gamma(κ, θ), lower=lb_rk, upper=ub_rk), 1)[1]

            end
        end
        ###

        ### update uh
        uh_new = rand(rng_glb, Uniform(0, h(c_new, r_new; λ=λ, ν=ν)), 1)[1]
        ###

        ### update int_lb, int_ub and fk_new
        rect_lb = c_new .- r_new
        rect_ub = c_new .+ r_new
        ### 

        ### store values
        if t > burnin && ((t - burnin - 1) % thin == 0)
            it = (t - burnin - 1) ÷ thin + 1
            w_all[:, it] = w_new
            r_all[:, it] = r_new
            c_all[:, :, it] = c_new
            g_all[:, it] = g_new
            # z_all[:, it] = z_new 
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
    g_result = [mode(row) for row in eachrow(g_all)]
    return ck_not_update, w_all, c_all, r_all, g_result, β_all, ϕ_all, Σ_all, ind_all
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
