include("MixDP.jl")

df = CSV.read("sharkdata.csv", DataFrame)
adsp = CSV.read("adSP.csv", DataFrame)
fpsp = CSV.read("fpSP.csv", DataFrame)

samples = df[:, 1]

results = load_object("../results/shark_mcmc.jld2")
w_all, c_all, r_all, g_result, β_all, 
ϕ_all, σ_all, ind_all = results

T = size(w_all, 2)
f1s = [comp_density_f(t, ind_all[1, :], β_all[1], ϕ_all[1], σ_all[1]) for t in 1:T]
f2s = [comp_density_f(t, ind_all[2, :], β_all[2], ϕ_all[2], σ_all[2]) for t in 1:T]
f3s = [comp_density_f(t, ind_all[3, :], β_all[3], ϕ_all[3], σ_all[3]) for t in 1:T]

x = collect(range(-5.5, -1; length=5000))
f1_density = Matrix{Float64}(undef, length(f1s), length(x))
for i in eachindex(f1s)
    f1_density[i, :] = pdf.(f1s[i], x) .* w_all[1, i]
end

f2_density = Matrix{Float64}(undef, length(f2s), length(x))
for i in eachindex(f2s)
    f2_density[i, :] = pdf.(f2s[i], x) .* w_all[2, i]
end

f3_density = Matrix{Float64}(undef, length(f3s), length(x))
for i in eachindex(f3s)
    f3_density[i, :] = pdf.(f3s[i], x) .* w_all[3, i]
end

f1_mean = [mean(f1_density[:, i]) for i in eachindex(x)]
f1_025perc = [quantile!(f1_density[:, i], 0.025) for i in eachindex(x)]
f1_975perc = [quantile!(f1_density[:, i], 0.975) for i in eachindex(x)]

f2_mean = [mean(f2_density[:, i]) for i in eachindex(x)]
f2_025perc = [quantile!(f2_density[:, i], 0.025) for i in eachindex(x)]
f2_975perc = [quantile!(f2_density[:, i], 0.975) for i in eachindex(x)]

f3_mean = [mean(f3_density[:, i]) for i in eachindex(x)]
f3_025perc = [quantile!(f3_density[:, i], 0.025) for i in eachindex(x)]
f3_975perc = [quantile!(f3_density[:, i], 0.975) for i in eachindex(x)]


fig, axs = subplots(1, 3, figsize=(15, 4), sharey=true)
# figsize controls compactness: width ~ 3×height

# -----------------
# Panel 1: MDPM
# -----------------
ax = axs[1]
ax.hist(samples, bins=60, color="0.82", edgecolor="0.25",
    linewidth=0.8, density=true)
ax.plot(x, f1_mean, color="red", label="state 1")
ax.fill_between(x, f1_025perc, f1_975perc,
    alpha=0.5, facecolor="cornflowerblue")
ax.plot(x, f2_mean, color="blue", label="state 2")
ax.fill_between(x, f2_025perc, f2_975perc,
    alpha=0.5, facecolor="cornflowerblue")
ax.plot(x, f3_mean, color="orange", label="state 3")
ax.fill_between(x, f3_025perc, f3_975perc,
    alpha=0.5, facecolor="cornflowerblue",
    label="95% CI")
ax.set_xlim(-5.5, -1)
ax.set_title("MDPM")
ax.set_xlabel("Observations (lODBA)")
ax.set_ylabel("Density")
ax.legend(fontsize=9)

# -----------------
# Panel 2: adSP
# -----------------
ax = axs[2]
ax.hist(samples, bins=60, color="0.82", edgecolor="0.25",
    linewidth=0.8, density=true)
ax.plot(adsp[:, 1], adsp[:, 2], color="red", label="state 1")
ax.fill_between(adsp[:, 1], adsp[:, 5], adsp[:, 8],
    alpha=0.5, facecolor="cornflowerblue")
ax.plot(adsp[:, 1], adsp[:, 3], color="blue", label="state 2")
ax.fill_between(adsp[:, 1], adsp[:, 6], adsp[:, 9],
    alpha=0.5, facecolor="cornflowerblue")
ax.plot(adsp[:, 1], adsp[:, 4], color="orange", label="state 3")
ax.fill_between(adsp[:, 1], adsp[:, 7], adsp[:, 10],
    alpha=0.5, facecolor="cornflowerblue", label="95% CI")
ax.set_xlim(-5.5, -1)
ax.set_title("adSP")
ax.set_xlabel("Observations (lODBA)")
ax.legend(fontsize=9)

# -----------------
# Panel 3: fpSP
# -----------------
ax = axs[3]
ax.hist(samples, bins=60, color="0.82", edgecolor="0.25",
    linewidth=0.8, density=true)
ax.plot(fpsp[:, 1], fpsp[:, 2], color="red", label="state 1")
ax.plot(fpsp[:, 1], fpsp[:, 3], color="blue", label="state 2")
ax.plot(fpsp[:, 1], fpsp[:, 4], color="orange", label="state 3")
ax.set_xlim(-5.5, -1)
ax.set_title("fpSP")
ax.set_xlabel("Observations (lODBA)")
ax.legend(fontsize=9)


# Tight spacing
fig.tight_layout(rect=[0, 0, 1, 0.90])
savefig("../results/fig6.pdf")
