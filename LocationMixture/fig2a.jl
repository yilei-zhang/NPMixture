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

#### true densities plot
fig, ax = subplots(figsize=(9, 4.5))
mix = mixture_pdf_on_grid(x, Ps, [1.0])
ax.plot(x, w[1] .* mix .+ w[2] .* pdfs[1] .+ w[3] .* pdfs[2]; color=blue, linewidth=2, alpha=0.9, label=L"$w_1f_1+w_2f_2+w_3f_3$")
ax.plot(x, Ps[1]; color=orange, linestyle="--", alpha=0.9, label=L"$f_1$ ($w_1$=" * " $(round(w[1]; digits=2)))")
ax.plot(x, pdfs[1]; color=green, linestyle="--", alpha=0.9, label=L"$f_2$ ($w_2$=" * " $(round(w[2]; digits=2)))")
ax.plot(x, pdfs[2]; color=purple, linestyle="--", alpha=0.9, label=L"$f_3$ ($w_3$=" * " $(round(w[3]; digits=2)))")
ind = findall(x -> x >= -15 && x <= 15, samples)
ax.hist(samples[ind]; color=red, bins=130, density=true, alpha=0.25, label="sample hist")
ax.set_title("True densities (n=10,000)")
ax.legend(frameon=false)
ax.set_xlabel("x")
ax.set_ylabel("density")
savefig("../results/fig2a.pdf", bbox_inches="tight", pad_inches=0.05)
