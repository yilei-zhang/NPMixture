include("MvMixDP.jl")

println("Number of threads: ", Threads.nthreads())
multithreads = false
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

burnin = 100000
iteration = 150000
thin = 10
seed = 69480

rng = MersenneTwister(73)
c1 = [-2, 2]
r1 = 2.
c2 = [2, -2]
r2 = 2.
N = 100
θ = range(0, 2π, length=N)[1:end-1]
x1 = c1[1] .+ r1 .* cos.(θ)
y1 = c1[2] .+ r1 .* sin.(θ)

x2 = c2[1] .+ r2 .* cos.(θ)
y2 = c2[2] .+ r2 .* sin.(θ)

Sigma_dist1 = InverseWishart(6, PDMat([0.8 0.0; 0.0 0.8]))
Sigma_dist2 = InverseWishart(6, PDMat([4.5 0.0; 0.0 4.5]))

mus1 = [x1 y1]
mus2 = [x2 y2]
mus = [mus1; mus2]
Sigmas = rand(rng, Sigma_dist1, 2 * (N - 1))
Sigmas_l = rand(rng, Sigma_dist2, 22)
Sigmas[82:92] = Sigmas_l[1:11]
Sigmas[(N-1+32):(N-1+42)] = Sigmas_l[12:22]

mix_p1 = rand(rng, Dirichlet(N - 1, 1), 1)
mix_p2 = rand(rng, Dirichlet(N - 1, 1), 1)
true_gmm = MixtureModel(MvNormal, [(mus[i, :], Sigmas[i]) for i in 1:2(N-1)], vec([1 / 2 .* mix_p1; 1 / 2 .* mix_p2]))
data = rand(rng, true_gmm, 20000)
samples = transpose(data)


### run MCMC

inits = ([1 / 3, 2 / 3], [1., 1.], [0.0 5.0; 0.0 -5.0])
sigma_P0 = 1.4

@time ck_not_update, w_all, c_all, r_all, g_result, 
β_all, ϕ_all, Σ_all, ind_all = MvMDP(samples; inits=inits, burnin=burnin, iteration=iteration, thin=thin,
                multithreads=multithreads, σ0=2.0, P0=PDMat(Matrix{Float64}(sigma_P0 * I, 2, 2)), seed=seed)

results = (inits, ck_not_update, w_all, c_all, r_all, g_result, β_all, ϕ_all, Σ_all, ind_all)
save_object("../results/mv_mcmc.jld2", results)
