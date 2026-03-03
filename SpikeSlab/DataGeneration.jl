function trapz(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y) ≥ 2
    s = zero(eltype(y))
    @inbounds for i in 1:length(x)-1
        s += (x[i+1]-x[i]) * (y[i+1]+y[i]) / 2
    end
    return s
end

function cumtrapz(x::AbstractVector, y::AbstractVector)
    n = length(x)
    out = similar(y, n)
    out[1] = zero(eltype(y))
    s = zero(eltype(y))
    @inbounds for i in 1:n-1
        s += (x[i+1]-x[i]) * (y[i+1]+y[i]) / 2
        out[i+1] = s
    end
    return out
end

# ---------- Hermite functions (orthonormal in L2) ----------
const πm14 = π^(-1/4)


function hermite_vec(x::Real, L::Integer)
    ## hermite_vec(x, L)
    ## Return ψ_0(x),…,ψ_{L-1}(x) using physicists’ Hermite functions.
    ## Recurrence: ψ_{n+1} = √(2/(n+1)) x ψ_n - √(n/(n+1)) ψ_{n-1}

    L ≤ 0 && return Float64[]
    ψ = Vector{Float64}(undef, L)
    ψ[1] = πm14 * exp(-x^2/2)
    if L ≥ 2
        ψ[2] = √2 * x * ψ[1]
        @inbounds for n in 2:L-1
            ψ[n+1] = sqrt(2/n) * x * ψ[n] - sqrt((n-1)/n) * ψ[n-1]
        end
    end
    return ψ
end

function hermite_density_on_grid(x::AbstractVector{<:Real};
    μ::Real, s::Real, a::AbstractVector{<:Real}, eps::Real=1e-8)
    invsqrt_s = 1/sqrt(s)

    ## hermite_density_on_grid(x; μ, s, a, eps=1e-8)
    ## p(x) ∝ (∑ a_n * (1/√s) ψ_n((x-μ)/s))^2 + eps  (normalized over x)

    y = Vector{Float64}(undef, length(x))
    @inbounds for i in eachindex(x)
        z = (x[i]-μ)/s
        ψ = hermite_vec(z, length(a))
        v = invsqrt_s * dot(a, ψ)
        y[i] = v*v + eps
    end
    y ./= trapz(x, y)
    return y
end

function mixture_pdf_on_grid(x::AbstractVector, Ps::Vector{<:AbstractVector}, w::AbstractVector)
    @assert length(Ps) == length(w)
    f = zeros(Float64, length(x))
    @inbounds for k in eachindex(Ps)
        @. f += w[k] * Ps[k]
    end
    f ./= trapz(x, f)
    return f
end

function sample_two_component_mixture(
    N::Integer;
    xgrid::AbstractVector{<:Real},
    P_spike::AbstractVector{<:Real},
    P_slab::AbstractVector{<:Real},
    w::NTuple{2,Float64}=(0.4, 0.6),
    seed::Integer=0)
    ## sample_two_component_mixture(N; xgrid, P_spike, P_slab, w=[0.4,0.6], seed=0)
    ## Sample N points by first drawing component label ~ Categorical(w), then inverse-CDF
    ## on that component’s grid.

    rng = MersenneTwister(seed)
    F1 = cumtrapz(xgrid, P_spike); T1 = F1[end]
    F2 = cumtrapz(xgrid, P_slab);  T2 = F2[end]

    labels = rand(rng, Categorical(collect(w)), N)
    X = Vector{Float64}(undef, N)

    @inbounds for i in 1:N
        if labels[i] == 1
            u = rand(rng) * T1
            j = searchsortedfirst(F1, u)
            j ≤ 1 ? X[i] = xgrid[1] :
            j > length(xgrid) ? X[i] = xgrid[end] :
            begin
                dj = F1[j]-F1[j-1]; t = dj ≤ eps() ? 0.0 : (u - F1[j-1])/dj
                X[i] = (1-t)*xgrid[j-1] + t*xgrid[j]
            end
        else
            u = rand(rng) * T2
            j = searchsortedfirst(F2, u)
            j ≤ 1 ? X[i] = xgrid[1] :
            j > length(xgrid) ? X[i] = xgrid[end] :
            begin
                dj = F2[j]-F2[j-1]; t = dj ≤ eps() ? 0.0 : (u - F2[j-1])/dj
                X[i] = (1-t)*xgrid[j-1] + t*xgrid[j]
            end
        end
    end
    return X, labels
end

function rand_spike_coeffs_sparse_asym(rng::AbstractRNG, L::Int;
    M_active::Int=3,                  # number of active Hermite modes
    pool=nothing,  # restrict to mid-range orders
    tau::Real=1.0, alpha::Real=1.6,      # magnitude and smoothness
    nonneg::Bool=true)                    # use positive weights

    if pool === nothing
        # choose some asymmetric orders on purpose
        pool = [2, 3, 5, 6, 7]
        pool = filter(n -> n < L, pool)
    end
    a = zeros(Float64, L)
    # choose a few distinct orders in [nmin, nmax]
    #pool = collect(nmin:nmax)
    
    M = min(M_active, length(pool))
    act = rand(rng, sample(rng, pool, M; replace=false))
    for n in act
        σ = tau / (n+1)^(alpha/2)
        c = randn(rng) * σ
        a[n+1] = nonneg ? abs(c) : c
    end
    return a
end

function rand_slab_coeffs(rng::AbstractRNG, L::Int;
    tau0::Real=0.7, alpha::Real=3.2, L_low::Int=min(6, L))
    a = zeros(Float64, L)
    for n in 0:L-1
        base = (n < L_low) ? tau0 : 0.35*tau0
        σ = base / (n+1)^(alpha/2)
        a[n+1] = randn(rng) * σ
    end
    return a
end
