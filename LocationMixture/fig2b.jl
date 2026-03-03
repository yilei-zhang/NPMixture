include("MixDP.jl")
include("DataGeneration.jl")

N = 10000
seed1 = 69833
seed2 = 78335

x = collect(range(-15.0, 15.0; length=4001))
Ps, specs, coeffs = draw_sobolev_components(x; K=1,
    centers=[-7.0], scales=[0.8], L=14, alpha=1.6, tau=0.7, seed=seed1)

w = [0.3, 0.35, 0.35]
samples = sample_mixture(N; w=w, lp_paras=[6.3, 1.1], sep_paras=[.9, 1.1, 0.7, 0.7],
    xgrid=x, Ps=Ps, specs=specs, coeffs_list=coeffs, seed=seed2)

pdf1 = pdf.(SkewedExponentialPower(.9, 1.1, 0.7, 0.7), x)
pdf2 = pdf.(Laplace(6.3, 1.1), x)
pdfs = [pdf1, pdf2]

blue = "#1f77b4"
orange = "#ff7f0e"
green = "#2ca02c"
red = "#d62728"
purple = "#9467bd"
brown = "#8c564b"

results = load_object("../results/loc_mix_mcmc.jld2")
w_all, c_all, r_all, g_result, beta_all,
phi_all, sigma_all, ind_all = results

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

savefig("../results/fig2b.pdf", bbox_inches="tight", pad_inches=0.05)
