include("../MixDP.jl")
pygui(true)

function trapz(x::AbstractVector, y::AbstractVector)
 ### integration of `y` w.r.t. `x`.
 ### ∫ y(x) dx over a monotone grid `x`.
    @assert length(x) == length(y) ≥ 2
    acc = zero(eltype(y))
    @inbounds for i in 1:length(x)-1
        acc += (x[i+1] - x[i]) * (y[i+1] + y[i]) / 2
    end
    return acc
end

function cumtrapz(x::AbstractVector, y::AbstractVector)
  ## Cumulative trapezoid; same length as `x`, starting at 0.
    n = length(x)
    out = similar(y, n)
    out[1] = zero(eltype(y))
    s = zero(eltype(y))
    @inbounds for i in 1:n-1
        s += (x[i+1] - x[i]) * (y[i+1] + y[i]) / 2
        out[i+1] = s
    end
    return out
end

const πm14 = π^(-1/4)

function hermite_vec(x::Real, L::Integer)
  ## return a fresh vector with ψ_0(x),…,ψ_{L-1}(x).
  ## Orthonormal "physicists" Hermite functions ψ_n(x):
  ## ψ_0(x) = π^(-1/4) exp(-x^2/2)
  ## ψ_{n+1}(x) = sqrt(2/(n+1)) * x * ψ_n(x) - sqrt(n/(n+1)) * ψ_{n-1}(x)

    if L ≤ 0
        return Float64[]
    end
    ψ = Vector{Float64}(undef, L)
    ψ[1] = πm14 * exp(-x^2/2)
    if L == 1
        return ψ
    end
    ψ[2] = √2 * x * ψ[1]
    if L ≥ 3
        @inbounds for n in 2:L-1
            # build ψ_n given ψ_{n-1}, ψ_{n-2}
            # We want to compute ψ_{n} for n≥2 using:
            # ψ_{n} = sqrt(2/n) * x * ψ_{n-1} - sqrt((n-1)/n) * ψ_{n-2}
            ψ[n] = ψ[n] # no-op; placeholder to keep indices valid
        end
        # redo with proper forward fill
        # we already have ψ1 (index 1) and ψ2 (index 2)
        # now compute indices 3..L
        @inbounds for n in 2:L-1
            ψ[n+1] = sqrt(2/n) * x * ψ[n] - sqrt((n-1)/n) * ψ[n-1]
        end
    end
    return ψ
end

Base.@kwdef struct SobolevCompSpec
    μ::Float64
    s::Float64 = 1.5
    L::Int     = 12           # number of basis functions (0..L-1)
    alpha::Float64 = 2.5      # coefficient decay exponent (H1-friendly)
    tau::Float64   = 0.6      # overall coefficient scale
end

function random_coeffs(rng::AbstractRNG, L::Int; alpha::Real=2.5, tau::Real=0.6)
  ## Draw zero-mean Gaussian coefficients with variance ~ (tau^2) / (n+1)^(alpha).
    a = Vector{Float64}(undef, L)
    @inbounds for n in 0:L-1
        #alpha_n = alpha + 0.3 * randn(rng)
        σ = tau / (n+1)^(alpha/2)
        a[n+1] = randn(rng) * σ
    end
    return a
end

function sobolev_component_density(
    x::AbstractVector{<:Real};
    spec::SobolevCompSpec,
    rng::AbstractRNG = Random.default_rng(),
    coeffs::Union{Nothing,AbstractVector{<:Real}} = nothing,
    eps::Real = 1e-8)
  ## Build a nonnegative density on grid `x` via a truncated Hermite expansion
  ## around (μ, s). Uses a *squared* expansion to enforce nonnegativity, then normalizes.
  ## p_k(x) ∝ (∑_{n=0}^{L-1} a_n * (1/√s) ψ_n((x-μ)/s))^2 + eps

    L = spec.L
    a = coeffs === nothing ? random_coeffs(rng, L; alpha=spec.alpha, tau=spec.tau) : collect(coeffs)
    μ, s = spec.μ, spec.s
    invsqrt_s = 1 / sqrt(s)

    y = Vector{Float64}(undef, length(x))
    # Evaluate expansion on the grid:
    @inbounds for i in eachindex(x)
        z = (x[i] - μ) / s
        ψ = hermite_vec(z, L)
        v = dot(a, ψ) * invsqrt_s # scale ensures L2 normalization under change of variable
        #v += 0.05 * ((x[i] - μ) / s)
        y[i] = v*v + eps
        #y[i] = v + eps
    end
    Z = trapz(x, y)
    y ./= Z
    return y, a
end

function draw_sobolev_components(
    x::AbstractVector{<:Real};
    K::Int=3,
    centers::AbstractVector{<:Real} = [-4.0, 0.0, 4.0],
    scales::AbstractVector{<:Real}  = fill(1.5, 3),
    L::Int=12,
    alpha::Real=2.5,
    tau::Real=0.6,
    seed::Integer=0)
  ### Construct K component densities on grid `x`. Returns (Ps, specs, coeffs_list).

    rng = MersenneTwister(seed)
    @assert length(centers) ≥ K
    @assert length(scales) ≥ K
    specs = SobolevCompSpec[]
    Ps    = Vector{Vector{Float64}}(undef, K)
    coeffs_list = Vector{Vector{Float64}}(undef, K)
    for k in 1:K
        spec = SobolevCompSpec(μ=centers[k], s=scales[k], L=L, alpha=alpha, tau=tau)
        Pk, ak = sobolev_component_density(x; spec, rng)
        push!(specs, spec)
        Ps[k] = Pk
        coeffs_list[k] = ak
    end
    return Ps, specs, coeffs_list
end

function mixture_pdf_on_grid(x::AbstractVector{<:Real}, Ps::Vector{<:AbstractVector}, w::AbstractVector)
    ### Combine component PDFs `Ps` with weights `w` (length K)
    @assert length(Ps) == length(w)
    K = length(w)
    f = zeros(Float64, length(x))
    @inbounds for k in 1:K
        @. f += w[k] * Ps[k]
    end
    # numerical renormalization guard (should already be normalized)
    Z = trapz(x, f)
    f ./= Z
    return f
end

function _inverse_cdf_sample(x::AbstractVector{<:Real}, p::AbstractVector{<:Real}, U::Real)
  ## Given grid `x` and density values `p` (sum to 1 under trapz),
  ## sample one value using inverse-CDF with linear interpolation.
    F = cumtrapz(x, p)
    total = F[end]
    u = U * total
    j = searchsortedfirst(F, u)
    if j == 1
        return x[1]
    elseif j > length(x)
        return x[end]
    else
        # linear interpolation within [x[j-1], x[j]]
        # avoid divide-by-zero:
        dj = F[j] - F[j-1]
        if dj <= eps(eltype(F))
            return x[j]
        else
            t = (u - F[j-1]) / dj
            return (1 - t)*x[j-1] + t*x[j]
        end
    end
end

function sample_sobolev_mixture(
    N::Integer;
    K::Int=3,
    xgrid::AbstractVector{<:Real}=collect(range(-10, 10; length=4001)),
    Ps::Union{Nothing,Vector{<:AbstractVector}}=nothing,
    specs::Union{Nothing,Vector{SobolevCompSpec}}=nothing,
    coeffs_list::Union{Nothing,Vector{<:AbstractVector}}=nothing,
    centers::AbstractVector{<:Real} = [-4.0, 0.0, 4.0],
    scales::AbstractVector{<:Real}  = fill(1.6, 3),
    L::Int=12, alpha::Real=2.5, tau::Real=0.6,
    seed::Integer=0,
    w=nothing)
    ## Draw N samples from the mixture defined by component PDFs `Ps` over `xgrid` and weights `w`.
    ## If `Ps`/`specs` are not provided, they are drawn randomly with separated basis locations
    ## Returns samples
    rng = MersenneTwister(seed)

    if Ps === nothing
        Ps, specs, coeffs_list = draw_sobolev_components(
            xgrid; K, centers, scales, L, alpha, tau, seed = rand(rng, 1:10^9)
        )
    else
        @assert specs !== nothing && coeffs_list !== nothing
    end

    if w === nothing
        # random weights:
        dirichlet_α = fill(1.0, K)    
        w = rand(rng, Dirichlet(dirichlet_α))
    else
        @assert length(w) == K
    end

    # Draw latent component per sample for bookkeeping:
    z = rand(rng, Categorical(w), N)

    # Sample values using component-wise inverse CDFs (for clearer z==k mapping).
    # To allow overlapping components while retaining z, we sample from each component with prob w_k.
    samples = Vector{Float64}(undef, N)
    for i in 1:N
        k = z[i]
        samples[i] = _inverse_cdf_sample(xgrid, Ps[k], rand(rng))
    end
    return samples
end

function sample_mixture(N; w, lp_paras, sep_paras, xgrid, Ps, specs, coeffs_list, seed)
    rng = MersenneTwister(seed)
    cat = Categorical(w)
    sample_inds = rand(rng, cat, N)
    samples = similar(sample_inds, Float64)
    N1, N2, N3 = sum(sample_inds .== 1), sum(sample_inds .== 2), sum(sample_inds .== 3)
    samples1 = sample_sobolev_mixture(
        N1; K=1, xgrid=xgrid, Ps=Ps, specs=specs, coeffs_list=coeffs_list, w=[1.0], seed=seed)
    samples2 = rand(rng, Laplace(lp_paras[1], lp_paras[2]), N2)
    samples3 = rand(rng, SkewedExponentialPower(sep_paras[1], sep_paras[2], sep_paras[3], sep_paras[4]), N3)
    samples[sample_inds.==1] = samples1
    samples[sample_inds.==2] = samples2
    samples[sample_inds.==3] = samples3
    return samples
end


burnin = 100000
iteration = 350000
thin = 20
N = 20000
seeds = 16815


seed = 69833
x = collect(range(-15.0, 15.0; length=4001))
Ps, specs, coeffs = draw_sobolev_components(x; K=1,
    centers=[-7.0], scales=[0.8], L=14, alpha=1.6, tau=0.7, seed=seed)

w = [0.3, 0.45, 0.25]
samples = sample_mixture(N; w=w, lp_paras=[0.0, 1.1], sep_paras=[6.2, 1.1, 0.7, 0.7],
    xgrid=x, Ps=Ps, specs=specs, coeffs_list=coeffs, seed=seeds)

pdf1 = pdf.(Laplace(0.0, 1.1), x)
pdf2 = pdf.(SkewedExponentialPower(6.2, 1.1, 0.7, 0.7), x)
pdfs = [pdf1, pdf2]

blue = "#1f77b4"
orange = "#ff7f0e"
green = "#2ca02c"
red = "#d62728"
purple = "#9467bd"
brown = "#8c564b"

#### true densities plot
fig, ax = subplots(figsize=(9, 4.5))
mix = mixture_pdf_on_grid(x, Ps, [1.0])
ax.plot(x, w[1] .* mix .+ w[2] .* pdfs[1] .+ w[3] .* pdfs[2]; color=blue, linewidth=2, alpha=0.9, label=L"$w_1f_1+w_2f_2+w_3f_3$")
ax.plot(x, Ps[1]; color=orange, linestyle="--", alpha=0.9, label=L"$f_1$ ($w_1$=" * " $(round(w[1]; digits=2)))")
ax.plot(x, pdfs[1]; color=green, linestyle="--", alpha=0.9, label=L"$f_2$ ($w_2$=" * " $(round(w[2]; digits=2)))")
ax.plot(x, pdfs[2]; color=purple, linestyle="--", alpha=0.9, label=L"$f_3$ ($w_3$=" * " $(round(w[3]; digits=2)))")
ind = findall(x -> x >= -15 && x <= 15, samples)
ax.hist(samples[ind]; color=red, bins=130, density=true, alpha=0.25, label="sample hist")
ax.set_title("True densities")
ax.legend(frameon=false)
ax.set_xlabel("x")
ax.set_ylabel("density")
savefig("truth_loc_mix_9-4p5_$(N)_$(seeds)_it$(iteration).png", dpi=300, bbox_inches="tight", pad_inches=0.05)


w = [0.4, 0.4, 0.2]
c = [-3.0, 0., 3.0]
r = [1.0, 1.0, 1.0]

@time w_all, c_all, r_all, g_result, beta_all,
phi_all, sigma_all, ind_all = MDP(samples; inits=(w, c, r), λ=2.0, ν=2, γ=2.0, vs=5., σs=0.35, wlb=0.1, 
    wub=0.9, burnin=burnin, iteration=iteration, thin=thin, multithreads=false, seed=seeds)


T = size(w_all, 2)
K = size(w_all, 1)
fis = Array{Distribution{Univariate,Continuous}}(undef, K, T)
for t in 1:T
    for k in 1:K
        fis[k, t] = comp_density_f(t, ind_all[k, :], beta_all[k], phi_all[k], sigma_all[k])
    end
end

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
c_mean = mean(c_all, dims=2)
r_mean = mean(r_all, dims=2)
Ilb = c_mean - r_mean
Iub = c_mean + r_mean

fig, ax = subplots(figsize=(9, 4.5))
ax.plot(x, Ps[1]; color=orange, linestyle="--", alpha=0.9, label=L"true $f_1$")
ax.fill_between(x, fis_025perc[1, :], fis_975perc[1, :], alpha=0.4, facecolor=orange, label=L"fitted $f_1$-95% CI")
ax.plot(x, pdfs[1]; color=green, linestyle="--", alpha=0.9, label=L"true $f_2$")
ax.fill_between(x, fis_025perc[2, :], fis_975perc[2, :], alpha=0.4, facecolor=green, label=L"fitted $f_2$-95% CI")
ax.plot(x, pdfs[2]; color=purple, linestyle="--", alpha=0.9, label=L"true $f_3$")
ax.fill_between(x, fis_025perc[3, :], fis_975perc[3, :], alpha=0.4, facecolor=purple, label=L"fitted $f_3$-95% CI")
ax.text(0.04, 0.95, L"$\widehat{w}_1$" * " 95% CI: ($(round(wl[1]; digits=3)), $(round(wu[1]; digits=3)))", transform=ax.transAxes, ha="left", va="top", fontsize=10, color="black")
ax.text(0.04, 0.90, L"$\widehat{w}_2$" * " 95% CI: ($(round(wl[2]; digits=3)), $(round(wu[2]; digits=3)))", transform=ax.transAxes, ha="left", va="top", fontsize=10, color="black")
ax.text(0.04, 0.85, L"$\widehat{w}_3$" * " 95% CI: ($(round(wl[3]; digits=3)), $(round(wu[3]; digits=3)))", transform=ax.transAxes, ha="left", va="top", fontsize=10, color="black")
ax.set_title("Fitted component densities")
ax.legend(frameon=false)
ax.set_xlabel("x")
ax.set_ylabel("density")
colors = [orange, green, purple]
for i in 1:K
    a = Ilb[i]
    b = Iub[i]
    x_mid = (a + b) / 2
    offset = 0.08 * (b - a)
    ylow_end = -0.1 * maximum(vcat(Ps...))
    ylow = -0.07 * maximum(vcat(Ps...))
    ax.plot([a, a], [ylow_end, 0], color=colors[i], linewidth=2.0, clip_on=false)
    ax.plot([b, b], [ylow_end, 0], color=colors[i], linewidth=2.0, clip_on=false)
    ax.text(x_mid, ylow, L"\widehat{I}_{%$i}", ha="center", va="center", fontsize=12, color=colors[i], transform=ax.transData, clip_on=false)
    ax.annotate("", xy=(a, ylow), xytext=(x_mid - offset, ylow),
        arrowprops=Dict(
            "arrowstyle" => "->",
            "linewidth" => 1.5,
            "color" => colors[i]),
        annotation_clip=false)
    ax.annotate("", xy=(b, ylow), xytext=(x_mid + offset, ylow),
        arrowprops=Dict(
            "arrowstyle" => "->",
            "linewidth" => 1.5,
            "color" => colors[i]),
        annotation_clip=false)
end

savefig("fitted_loc_mix_9-4p5_$(seeds)_it$(iteration).png", dpi=300, bbox_inches="tight", pad_inches=0.05)


