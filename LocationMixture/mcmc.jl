include("MixDP.jl")
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
iteration = 450000
thin = 20
N = 10000
seed1 = 69833
seed2 = 78335

x = collect(range(-15.0, 15.0; length=4001))
Ps, specs, coeffs = draw_sobolev_components(x; K=1,
    centers=[-7.0], scales=[0.8], L=14, alpha=1.6, tau=0.7, seed=seed1)

w = [0.3, 0.35, 0.35]
samples = sample_mixture(N; w=w, lp_paras=[6.3, 1.1], sep_paras=[0.9, 1.1, 0.7, 0.7],
    xgrid=x, Ps=Ps, specs=specs, coeffs_list=coeffs, seed=seed2)

pdf1 = pdf.(SkewedExponentialPower(0.9, 1.1, 0.7, 0.7), x)
pdf2 = pdf.(Laplace(6.3, 1.1), x)
pdfs = [pdf1, pdf2]

blue = "#1f77b4"
orange = "#ff7f0e"
green = "#2ca02c"
red = "#d62728"
purple = "#9467bd"
brown = "#8c564b"

w = [0.4, 0.4, 0.2]
c = [-3.0, 0., 3.0]
r = [1.0, 1.0, 1.0]

@time w_all, c_all, r_all, g_result, beta_all,
phi_all, sigma_all, ind_all = MDP(samples; inits=(w, c, r), λ=2.0, ν=2, γ=2.0, vs=5., σs=0.35, wlb=0.1, 
    wub=0.9, burnin=burnin, iteration=iteration, thin=thin, multithreads=multithreads, seed=seed2)

#println(w_all[end])
results = (w_all, c_all, r_all, g_result, beta_all,
phi_all, sigma_all, ind_all)
save_object("../results/loc_mix_mcmc.jld2", results)
