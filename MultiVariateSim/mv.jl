
include("../MvMixDP.jl")
pygui(true)

burnin = 115000
iteration = 120000
multithreads = false

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
β_all, ϕ_all, Σ_all, ind_all = MvMDP(samples; inits=inits, burnin=burnin, iteration=iteration,
                multithreads=multithreads, σ0=2.0, P0=PDMat(Matrix{Float64}(sigma_P0 * I, 2, 2)), seed=37284)

### make plots

T = iteration - burnin
f1s = [mv_comp_density_f(t, ind_all[1, :], β_all[1], ϕ_all[1], Σ_all[1]) for t in 1:T]
f2s = [mv_comp_density_f(t, ind_all[2, :], β_all[2], ϕ_all[2], Σ_all[2]) for t in 1:T]

r = 7.0
xygrid = collect.(Iterators.product(-r:0.05:r, -r:0.05:r))
f1s_pdf = Array{Float64}(undef, size(xygrid,1), size(xygrid, 2), T)
f2s_pdf = Array{Float64}(undef, size(xygrid, 1), size(xygrid, 2), T)
fs_pdf = Array{Float64}(undef, size(xygrid, 1), size(xygrid, 2), T)

for t in 1:T
    f1s_pdf[:, :, t] = pdf(f1s[t], xygrid)
    f2s_pdf[:, :, t] = pdf(f2s[t], xygrid)
    fs_pdf[:, :, t] = w_all[1, t] .* f1s_pdf[:, :, t] .+ w_all[2, t] .* f2s_pdf[:, :, t]
end


wf1s_pdf = Array{Float64}(undef, size(f1s_pdf, 1), size(f1s_pdf, 2), size(f1s_pdf, 3))
wf2s_pdf = Array{Float64}(undef, size(f2s_pdf, 1), size(f2s_pdf, 2), size(f2s_pdf, 3))
for t in 1:T
    wf1s_pdf[:, :, t] = w_all[1, t] .* f1s_pdf[:, :, t]
    wf2s_pdf[:, :, t] = w_all[2, t] .* f2s_pdf[:, :, t]
end

f1_mean = mean(f1s_pdf, dims=3)
f2_mean = mean(f2s_pdf, dims=3)
f_mean = mean(fs_pdf, dims=3)
wf1_mean = mean(wf1s_pdf, dims=3)
wf2_mean = mean(wf2s_pdf, dims=3)


### plot true density and fitted density
using PyCall
v = (elev=61.46284841009288, azim=-110.63222422866524, roll=4.724742314259799,
    xlim=(-8.020833333333334, 8.020833333333334),
    ylim=(-8.020833333333334, 8.020833333333334),
    zlim=(-0.0016509075310692665, 0.08089446903656264),
    aspect=[1.1904761932899772, 1.1904761932899772, 0.892857144967483])

function set_view!(ax, v)
    if v.roll === nothing
        ax.view_init(v.elev, v.azim)
    else
        ax.view_init(v.elev, v.azim, roll=v.roll)
    end
    ax.set_xlim3d(v.xlim)
    ax.set_ylim3d(v.ylim)
    ax.set_zlim3d(v.zlim)
    if v.aspect !== nothing && pyhasattr(ax, "set_box_aspect")
        ax.set_box_aspect(v.aspect)
    end
    return ax
end

if !isdefined(Main, :pyhasattr)
    pyhasattr(obj, name::AbstractString) = pybuiltin(:hasattr)(obj, name)
end

fig = figure(1, figsize=(8, 6.5)); clf();
ax = fig.add_subplot(111, projection="3d")
true_f = pdf(true_gmm, xygrid)
surf(-r:0.05:r, -r:0.05:r, transpose(true_f), cmap=ColorMap("terrain"))
mesh(-r:0.05:r, -r:0.05:r, transpose(true_f), linewidths=0.2)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax.grid(false)
ax.set_axis_off()
set_view!(ax, v)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_position([-0.29, -0.1, 1.5, 1.52])
fig.canvas.draw() 
bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
savefig("true_f.png", transparent=true, dpi=300, bbox_inches=bbox, pad_inches=0)

####### wf1
fig2 = figure(2, figsize=(8, 6.5)); clf();
ax2 = fig2.add_subplot(111, projection="3d")
surf(-r:0.05:r, -r:0.05:r, transpose(wf1_mean[:,:,1]), cmap=ColorMap("terrain"))
mesh(-r:0.05:r, -r:0.05:r, transpose(wf1_mean[:,:,1]), linewidths=0.2)
fig2.patch.set_alpha(0.0)
ax2.patch.set_alpha(0.0)
ax2.grid(false)
ax2.set_axis_off()
set_view!(ax2, v)
fig2.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax2.set_position([-0.29, -0.1, 1.5, 1.52])
fig2.canvas.draw() 
bbox = ax2.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
savefig("wf1.png", transparent=true, dpi=300, bbox_inches=bbox, pad_inches=0)


####### wf2
fig3 = figure(3, figsize=(8, 6.5)); clf();
ax3 = fig3.add_subplot(111, projection="3d")
surf(-r:0.05:r, -r:0.05:r, transpose(wf2_mean[:, :, 1]), cmap=ColorMap("terrain"))
mesh(-r:0.05:r, -r:0.05:r, transpose(wf2_mean[:, :, 1]), linewidths=0.2)
fig3.patch.set_alpha(0.0)
ax3.patch.set_alpha(0.0)
ax3.grid(false)
ax3.set_axis_off()
set_view!(ax3, v)
fig3.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax3.set_position([-0.29, -0.1, 1.5, 1.52])
fig3.canvas.draw()
bbox = ax3.get_window_extent().transformed(fig3.dpi_scale_trans.inverted())
savefig("wf2.png", transparent=true, dpi=300, bbox_inches=bbox, pad_inches=0)


img1 = imread("true_f.png")
img2 = imread("wf1.png")
img3 = imread("wf2.png")

fig = figure(figsize=(12, 4))

# Exact thirds
w = 1 / 3
ht = 0.92      # leave space at bottom for labels
y0 = 0.08

ax1 = fig.add_axes([0.000, y0, w, ht])
ax2 = fig.add_axes([w, y0, w, ht])
ax3 = fig.add_axes([2w, y0, w, ht])

axs = (ax1, ax2, ax3)
imgs = (img1, img2, img3)
labels = ("True density",
    "Fitted component 1 (weighted)",
    "Fitted component 2 (weighted)")

for (ax, img, lab) in zip(axs, imgs, labels)
    ax.imshow(img)
    ax.axis("off")

    # label BELOW the image
    ax.text(0.5, -0.06, lab,
        transform=ax.transAxes,
        ha="center", va="top", fontsize=17.5)
end

fig.patch.set_alpha(0.0)
savefig("three_in_row_labeled.png", dpi=300, transparent=true, pad_inches=0)