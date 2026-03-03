
include("MixDP_sigma.jl")
include("DataGeneration.jl")

seed1 = 52595
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

blue = "#1f77b4"
orange = "#ff7f0e"
green = "#2ca02c"
red = "#d62728"

results = load_object("../results/spike_slab_mcmc.jld2")
w_all, g_result, beta_all, phi_all, sigma_all, ind_all = results

w_hat = mean(w_all, dims=2)
T = size(w_all, 2)
K = size(w_all, 1)
fis = Array{Distribution{Univariate, Continuous}}(undef, K, T)
for t in 1:T
    for k in 1:K
        fis[k, t] = comp_density_f(t, ind_all[k, :], beta_all[k], phi_all[k], sigma_all[k])
    end
end

ind = findall(x -> x>=-15 && x<=15, x)
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
savefig("../results/fig2d.pdf", bbox_inches="tight", pad_inches=0.05)

