include("MixDP.jl")

df = CSV.read("sharkdata.csv", DataFrame)

samples = df[:, 1]
n = length(samples)
burnin = 50000
iteration = 100000
thin = 20
w = [0.4, 0.4, 0.2]
c = [-4.0, -3.2, -2.5]
r = [0.4, 0.2, 0.4]

@time w_all, c_all, r_all, g_result, β_all, 
ϕ_all, σ_all, ind_all = MDP(samples; inits=(w, c, r), λ=0.8, ν=1, γ=2.0, vs=5., σs=0.072, wub=0.63,
    burnin=burnin, iteration=iteration, thin=thin, multithreads=false, seed=1465)

results = (w_all, c_all, r_all, g_result, β_all, ϕ_all, σ_all, ind_all)
save_object("../results/shark_mcmc.jld2", results)

