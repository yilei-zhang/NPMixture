
include("MixDP_sigma.jl")
include("DataGeneration.jl")


println("Number of threads: ", Threads.nthreads())

multithreads = true
if length(ARGS) >= 1
    arg = lowercase(ARGS[1])
    if arg == "true"
        multithreads = true
    elseif arg == "false"
        multithreads = false
    else
        error("Argument must be 'true' or 'false'. Example: julia mcmc.jl false")
    end
end
println("Using threads: ", multithreads)

burnin = 150000
iteration = 400000
thin = 15
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

σl = 0.5
σu = 3.0

w0 = [0.6, 0.4]
c0 = 0.
r0 = 20.
@time w_all, g_result, beta_all,
phi_all, sigma_all, ind_all = MDP_sigma(X; inits=(w0, c0, r0), burnin=burnin, iteration=iteration, thin=thin,
                                wlb = 0.1, wub = 0.9, σl=σl, σu=σu, σ0=3.0, vs=6., σs=0.7, multithreads=multithreads,seed=seed2)

results = (w_all, g_result, beta_all, phi_all, sigma_all, ind_all)
save_object("../results/spike_slab_mcmc.jld2", results)
