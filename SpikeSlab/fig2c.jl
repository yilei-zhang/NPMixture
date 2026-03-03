include("MixDP_sigma.jl")
include("DataGeneration.jl")

N = 10000

seed1 = 52595
seed2 = 163

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
        xgrid=x, P_spike=P_spike, P_slab=P_slab, w=w, seed=seed2)

blue = "#1f77b4"
orange = "#ff7f0e"
green = "#2ca02c"
red = "#d62728"

fig, ax = subplots(figsize=(8, 4))
title = "True densities (n=10,000)"
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
#savefig("../results/fig2c.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
savefig("../results/fig2c.pdf", bbox_inches="tight", pad_inches=0.05)

