include("../MixDP_sigma.jl")
pygui(true)

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

function draw_spike_slab_components(
    x::AbstractVector{<:Real};
    center::Real=0.0,
    s_spike::Real=0.9, s_slab::Real=2.0,
    L_spike::Int=24, L_slab::Int=14,
    seed::Integer=0)
    
    rng = MersenneTwister(seed)

    a_spike = rand_spike_coeffs(rng, L_spike; M_active=6, tau=0.95, alpha=1.3, bias_high=true)
    a_slab  = rand_slab_coeffs(rng,  L_slab;  tau0=0.7,  alpha=3.0, L_low=min(6,L_slab))

    P_spike = hermite_density_on_grid(x; μ=center, s=s_spike, a=a_spike)
    P_slab  = hermite_density_on_grid(x; μ=center, s=s_slab,  a=a_slab)

    specs = (center=center, s_spike=s_spike, s_slab=s_slab,
             L_spike=L_spike, L_slab=L_slab,
             a_spike=a_spike, a_slab=a_slab)
    return P_spike, P_slab, specs
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


burnin = 150000
iteration = 400000
thin = 15
N = 10000
multithreads = false


seed1 = 52595
seeds = 163

rng = MersenneTwister(seed1)
L_spike=7
L_slab=8
center = 0.0
x = collect(range(-20, 20; length=5001))
a_spike = rand_spike_coeffs_sparse_asym(rng, L_spike; M_active=2, tau=2.6, alpha=1.5, nonneg=false)
for n in 1:3   # low orders
    a_spike[n] += 0.1 * randn(rng)
end
a_slab  = rand_slab_coeffs(rng, L_slab; tau0=0.9, alpha=5.2, L_low=min(6, L_slab))

s_spike=1.8
s_slab=14.5


δ = 0.1
P_spike = hermite_density_on_grid(x; μ=center+δ, s=max(1.0, s_spike), a=a_spike)
γ = 5.8
P_spike .= P_spike .^ γ
P_spike ./= trapz(x, P_spike)
P_slab  = hermite_density_on_grid(x; μ=center-δ, s=s_slab, a=a_slab)

w = (0.68, 0.32)
X, z = sample_two_component_mixture(N;
        xgrid=x, P_spike=P_spike, P_slab=P_slab, w=w, seed=seeds)

blue = "#1f77b4"
orange = "#ff7f0e"
green = "#2ca02c"
red = "#d62728"

fig, ax = subplots(figsize=(8, 4))
title = "True densities"
ind = findall(x -> x>=-15 && x<=15, x)
Xind = findall(x -> x>=-15 && x<=15, X)
mix = mixture_pdf_on_grid(x[ind], [P_spike[ind], P_slab[ind]], collect(w))
ax.plot(x[ind], mix; color=blue, linewidth=2, alpha=0.9, label=L"$w_1f_1 + w_2f_2$")
ax.plot(x[ind], P_spike[ind]; color=orange, linestyle="--", alpha=0.95, label=L"$f_1$ ($w_1$" * "=$(collect(w)[1]))")
ax.plot(x[ind], P_slab[ind]; color=green, linestyle="--", alpha=0.95, label=L"$f_2$ ($w_2$" * "=$(collect(w)[2]))")
ax.set_title(title); ax.legend(frameon=false)
ax.set_xlabel("x"); ax.set_ylabel("density")
ax.hist(X[Xind]; bins=130, color=red, density=true, alpha=0.25, label="sample hist")
ax.legend(frameon=false)
savefig("truth_ss_8-4_$(N)_$(seeds)_it$(iteration).png", dpi=300, bbox_inches="tight", pad_inches=0.05)


σl = 0.5
σu = 3.0

w = [0.6, 0.4]
c = 0.
r = 20.
@time w_all, g_result, beta_all,
phi_all, sigma_all, ind_all = MDP_sigma(X; inits=(w, c, r), burnin=burnin, iteration=iteration, thin=thin,
                                wlb = 0.1, wub = 0.9, σl=σl, σu=σu, σ0=3.0, vs=6., σs=0.7, multithreads=multithreads,seed=seeds)
                           
w_hat = mean(w_all, dims=2)
T = size(w_all, 2)
K = size(w_all, 1)
fis = Array{Distribution{Univariate, Continuous}}(undef, K, T)
for t in 1:T
    for k in 1:K
        fis[k, t] = comp_density_f(t, ind_all[k, :], beta_all[k], phi_all[k], sigma_all[k])
    end
end

x = x[ind]
fis_pdf = Array{Float64}(undef, K, length(x), T)
fs_pdf = Matrix{Float64}(undef, length(x), T)
for t in 1:T
    for k in 1:K
        fis_pdf[k, :, t] = pdf.(fis[k, t], x)
    end
    fs_pdf[:, t] = w_all[:, t]' * fis_pdf[:, :, t]
end

fis_mean = mean(fis_pdf, dims=3)
fis_025perc = Array{Float64}(undef, K, length(x))
fis_975perc = Array{Float64}(undef, K, length(x))
for k in 1:K
    fis_025perc[k, :] = quantile.(eachrow(fis_pdf[k, :, :]), 0.025)
    fis_975perc[k, :] = quantile.(eachrow(fis_pdf[k, :, :]), 0.975)
end
f_mean = mean(fs_pdf, dims=2)
f_025perc = quantile.(eachrow(fs_pdf), 0.025)
f_975perc = quantile.(eachrow(fs_pdf), 0.975)


wl = quantile.(eachrow(w_all), 0.025)
wu = quantile.(eachrow(w_all), 0.975)

fig, ax = subplots(figsize=(8, 4))
ax.plot(x, P_spike[ind]; color=orange,linestyle="--", alpha=0.95, label=L"true $f_1$")
ax.fill_between(x, fis_025perc[1, :], fis_975perc[1, :], alpha=0.4, facecolor=orange, label=L"fitted $f_1$-95% CI")
ax.plot(x, P_slab[ind]; color=green, linestyle="--", alpha=0.95, label=L"true $f_2$")
ax.fill_between(x, fis_025perc[2, :], fis_975perc[2, :], alpha=0.4, facecolor=green, label=L"fitted $f_2$-95% CI")
ax.text(-15, 0.535, L"$\widehat{w}_1$" * " 95% CI: ($(round(wl[1]; digits=3)), $(round(wu[1]; digits=3)))", fontsize=10, color="black")
ax.text(-15, 0.495, L"$\widehat{w}_2$" * " 95% CI: ($(round(wl[2]; digits=3)), $(round(wu[2]; digits=3)))", fontsize=10, color="black")
ax.set_title("Fitted component densities")
ax.set_xlabel("x"); ax.set_ylabel("density")
ax.legend(frameon=false)
savefig("fitted_ss_8-4_$(N)_$(seeds)_it$(iteration)_w.png", dpi=300, bbox_inches="tight", pad_inches=0.05)


fig, ax = subplots(figsize=(8, 4))
ax.plot(x, P_spike[ind]; color=orange,linestyle="--", alpha=0.95, label=L"true $f_1$")
ax.fill_between(x, fis_025perc[1, :], fis_975perc[1, :], alpha=0.4, facecolor=orange, label=L"fitted $f_1$-95% CI")
ax.plot(x, P_slab[ind]; color=green, linestyle="--", alpha=0.95, label=L"true $f_2$")
ax.fill_between(x, fis_025perc[2, :], fis_975perc[2, :], alpha=0.4, facecolor=green, label=L"fitted $f_2$-95% CI")
ax.set_title("Fitted component densities")
ax.set_xlabel("x"); ax.set_ylabel("density")
ax.legend(frameon=false)
savefig("fitted_ss_8-4_$(N)_$(seeds)_it$(iteration).png", dpi=300, bbox_inches="tight", pad_inches=0.05)
