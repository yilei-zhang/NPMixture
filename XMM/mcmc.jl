include("MvMixDP_fixReg_noise.jl")

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

burnin = 15000
iteration = 25000
thin = 1
sigma_P = 4.0

dat = CSV.read("XMM.csv", DataFrame)
n = 758869
xy_dat = Matrix{Float64}(undef, n, 2)
xy_dat[:, 1] = dat[:, 1]
xy_dat[:, 2] = dat[:, 2]

w0 = [3 / 10, 1 / 2, 1 / 5]
c1 = [121.0 124.2]
r1 = [3.0 2.2]
c2 = [121.0 128.2]
r2 = [3.0 1.72]
c = [c1; c2]
r = [r1; r2]

R = (14.68466 + 14.6579) / 2 * (13.1059 + 13.03028) / 2
σ0=1.2
P0=PDMat(Matrix{Float64}(sigma_P * I, 2, 2))

#### running MCMC 
@time w_all, g_result, β_all,
ϕ_all, Σ_all, ind_all = MvMDP_fixReg_noise(xy_dat; w=w0, c=c, r=r, R=R, σ0=σ0, P0=P0,
    burnin=burnin, iteration=iteration, thin=thin, multithreads=multithreads, seed=3793)

results = (c, r, σ0, P0, w_all, g_result, β_all, ϕ_all, Σ_all, ind_all)
save_object("../results/xmm_full_mcmc.jld2", results)