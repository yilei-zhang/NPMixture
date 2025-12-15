include("../MvMixDP_fixReg_noise.jl")
pygui(true)


burnin = 15000
iteration = 25000
thin = 1
sigma_P = 4.0
multithreads = true

dat = CSV.read("XMM.csv", DataFrame)
n = 758869
xy_dat = Matrix{Float64}(undef, n, 2)
xy_dat[:, 1] = dat[:, 1]
xy_dat[:, 2] = dat[:, 2]

w = [3 / 10, 1 / 2, 1 / 5]
c1 = [121.0 124.2]
r1 = [3.0 2.2]
c2 = [121.0 128.2]
r2 = [3.0 1.72]
c = [c1; c2]
r = [r1; r2]


R = (14.68466 + 14.6579) / 2 * (13.1059 + 13.03028) / 2
σ0=1.2
P0=PDMat(Matrix{Float64}(sigma_P * I, 2, 2))
seed=3793

#### running MCMC 
@time w_all, g_result, β_all,
ϕ_all, Σ_all, ind_all = MvMDP_fixReg_noise(xy_dat; w=w, c=c, r=r, R=R, σ0=σ0, P0=P0,
    burnin=burnin, iteration=iteration, thin=thin, multithreads=multithreads, seed=seed)

#### making 2d density plot
R = (14.68466 + 14.6579) / 2 * (13.1059 + 13.03028) / 2
T = size(w_all, 2)
iteration=T
f1s = [mv_comp_density_f(t, ind_all[1, :], β_all[1], ϕ_all[1], Σ_all[1]) for t in (iteration-T+1):iteration]
f2s = [mv_comp_density_f(t, ind_all[2, :], β_all[2], ϕ_all[2], Σ_all[2]) for t in (iteration-T+1):iteration]

f1s_thinned = similar(f1s, 1000)
f2s_thinned = similar(f2s, 1000)
w_thinned = similar(w_all, 3, 1000)
ind = 1
for t in 1:T
    if t % 10 == 1
        f1s_thinned[ind] = f1s[t]
        f2s_thinned[ind] = f2s[t]
        w_thinned[:, ind] = w_all[:, t]
        ind += 1
    end
end

wl = quantile.(eachrow(w_thinned), 0.16)
wu = quantile.(eachrow(w_thinned), 0.84)
w_mean = mean(w_thinned, dims=2)

xs = 115:0.1:127.5
ys = 120:0.1:132.5
xygrid = collect.(Iterators.product(xs, ys))
f1s_pdf = Array{Float64}(undef, size(xygrid,1), size(xygrid, 2), 1000)
f2s_pdf = Array{Float64}(undef, size(xygrid, 1), size(xygrid, 2), 1000)
fs_pdf = Array{Float64}(undef, size(xygrid, 1), size(xygrid, 2), 1000)
for t in 1:1000
    f1s_pdf[:, :, t] = pdf(f1s_thinned[t], xygrid)
    f2s_pdf[:, :, t] = pdf(f2s_thinned[t], xygrid)
    fs_pdf[:, :, t] = w_thinned[1, t] .* f1s_pdf[:, :, t] .+ w_thinned[2, t] .* f2s_pdf[:, :, t] .+ w_thinned[3, t] ./ R
end

f1_mean = mean(f1s_pdf, dims=3)
f2_mean = mean(f2s_pdf, dims=3)
f_mean = mean(fs_pdf, dims=3)


function fit_kde2d(x::AbstractVector, y::AbstractVector; bandwidth=:scott)
    n = length(x)
    X = collect(float.(x))
    Y = collect(float.(y))
    if bandwidth === :scott
        hx = (std(X) > 0 ? std(X) : (maximum(X) - minimum(X) + 1e-9)) * n^(-1 / 6)
        hy = (std(Y) > 0 ? std(Y) : (maximum(Y) - minimum(Y) + 1e-9)) * n^(-1 / 6)
    elseif bandwidth isa Number
        hx = hy = float(bandwidth)
    else
        hx, hy = float.(bandwidth)
    end
    invnorm = 1 / (2π * hx * hy * n)

    pdfgrid = function (xs::AbstractVector, ys::AbstractVector)
        xs = collect(float.(xs))
        ys = collect(float.(ys))
        Z = zeros(length(xs), length(ys))
        sx2 = [@. ((xv - X) / hx)^2 for xv in xs]
        sy2 = [@. ((yv - Y) / hy)^2 for yv in ys]
        for i in eachindex(xs), j in eachindex(ys)
            Z[i, j] = invnorm * @views sum(@. exp(-0.5 * (sx2[i] + sy2[j])))
        end
        return Z
    end
    return (bandwidth=(hx, hy), pdfgrid=pdfgrid)
end

function fd(xc, yc, ϵ, θ, d0, η)
    d = sqrt.((xc .* cos(θ) + yc .* sin(θ)) .^ 2 .+ (yc .* cos(θ) - xc .* sin(θ)) .^ 2 ./ (1 - ϵ)^2)
    fd = 1 ./ (1 .+ (d ./ d0) .^ 2) .^ η
    return fd
end

function King_profile_density(x, y, μx, μy, ϵ, θ, d0, η)
    xc = x .- μx
    yc = y .- μy

    f(x, y) = fd(x, y, ϵ, θ, d0, η)
    inner(x) = quadgk(y -> f(x, y), -Inf, Inf)[1]
    I = quadgk(x -> inner(x), -Inf, Inf)[1]

    return f(xc, yc) ./ I

end

function mix_King(x, y, w1, w2, μ1x, μ1y, μ2x, μ2y, w0, R, ϵ, θ, d0, η)
    return w1 .* King_profile_density(x, y, μ1x, μ1y, ϵ, θ, d0, η) .+
           w2 .* King_profile_density(x, y, μ2x, μ2y, ϵ, θ, d0, η) .+
           w0 ./ R
end

ϵ = 0.00574
θ = 0
d0 = 0.6
η = 1.5
R = (14.68466 + 14.6579) / 2 * (13.1059 + 13.03028) / 2

idx_lr = argmin(xy_dat[:, 2])
lrp = xy_dat[idx_lr, :]
idx_tr = argmax(xy_dat[:, 1])
trp = xy_dat[idx_tr, :]
idx_ll = argmin(xy_dat[:, 1])
llp = xy_dat[idx_ll, :]
idx_tl = argmax(xy_dat[:, 2])
tlp = xy_dat[idx_tl, :]

upedge = norm(tlp - trp)
rightedge = norm(trp - lrp)
leftedge = norm(tlp - llp)
bottomedge = norm(llp - lrp)
quad = [tlp, trp, lrp, llp]
xs, ys = first.(quad), last.(quad)
function polygon_area(xs, ys)
    n = length(xs)
    s = 0.0
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        s += xs[i] * ys[j] - xs[j] * ys[i]
    end
    return 0.5 * abs(s)
end
A = polygon_area(xs, ys)
adjust_ratio = (0.102 + (1.0 - 0.95)) / 0.102
f_fit(xy) = mix_King(xy[1], xy[2], 0.717, 0.182, 120.974, 124.873, 121.396, 127.319, 0.102, A/adjust_ratio , ϵ, θ, d0, η)

levs = 0:0.005:0.4
xgrid = 115:0.1:127.5
ygrid = 120:0.1:132.5
xygrid = collect.(Iterators.product(xgrid, ygrid))
king_fitted = f_fit.(xygrid)
kde_model = fit_kde2d(xy_dat[:, 1], xy_dat[:, 2])  
kde_fitted = kde_model.pdfgrid(xgrid, ygrid)

## rect1
c1 = [121.0 124.2]
r1 = [3.0 2.2]
x1 = c1[1] - r1[1]
y1 = c1[2] - r1[2]   
width, height = r1[1] * 2, r1[2] * 2
rect1 = PyPlot.matplotlib[:patches][:Rectangle](
    (x1, y1), width, height,
    edgecolor="red",
    facecolor="none",   
    linestyle="--",
    linewidth=2.0)
## rect2
c2 = [121.0 128.2]
r2 = [3.0 1.72]
x1 = c2[1] - r2[1]
y1 = c2[2] - r2[2]    
width, height = r2[1] * 2, r2[2] * 2
rect2 = PyPlot.matplotlib[:patches][:Rectangle](
    (x1, y1), width, height,
    edgecolor="red",
    facecolor="none",   
    linewidth=2.0)


fig, axs = subplots(1, 3, figsize=(18, 5), constrained_layout=true)
cm = ColorMap("viridis")
h1 = axs[1].contour(xgrid, ygrid, f_mean[:,:,1]', levels=levs, cmap=cm)
h2 = axs[2].contour(xgrid, ygrid, kde_fitted', levels=levs, cmap=cm)
h3 = axs[3].contour(xgrid, ygrid, king_fitted', levels=levs, cmap=cm)
for ax in axs
    ax.set_facecolor("white")
    ax.grid(true, color="grey", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_xlabel("X")
end
axs[1].set_ylabel("Y")
cbar = fig.colorbar(h3, ax=axs, orientation="vertical", location="right", fraction=0.02, pad=0.02)
cbar.set_label("Density", rotation=270, labelpad=12)
axs[1].set_title("MDPM")
axs[2].set_title("KDE")
axs[3].set_title("King's Profile")
axs[1].add_patch(rect1)
axs[1].add_patch(rect2)
savefig("astro_3fig.png", dpi=300, bbox_inches="tight", pad_inches=0.05)


#### CDF comparison

@inline function cross2d(a, b)
    return a[1] * b[2] - a[2] * b[1]
end

function point_in_convex_quad(p, q1, q2, q3, q4)
    # p, q1..q4 are the vertex of the quadrilaterals in clockwise order
    qs = (q1, q2, q3, q4)
    for i in 1:4
        a = qs[i]
        b = qs[mod1(i + 1, 4)]
        edge = (b .- a)
        to_p = (p .- a)
        if cross2d(edge, to_p) > 0   # assuming CCW order
            return false
        end
    end
    return true
end

function fmean(xy, f1s, f2s, w_all, R, q1, q2, q3, q4)
    T = size(w_all, 2)
    # or (x,y) if you don't want StaticArrays
    s = 0.0
    if point_in_convex_quad(xy, q1, q2, q3, q4)
        for t in 1:T
            s += w_all[1, t] * pdf(f1s[t], xy) +
                 w_all[2, t] * pdf(f2s[t], xy) + w_all[3, t] / R
        end
        return s / T
    else
        for t in 1:T
            s += w_all[1, t] * pdf(f1s[t], xy) +
                 w_all[2, t] * pdf(f2s[t], xy)
        end
        return s / T
    end
end

function fking(xy, w0, R, q1, q2, q3, q4)
    if point_in_convex_quad(xy, q1, q2, q3, q4)
        return f_fit(xy)
    else
        return f_fit(xy) - w0 / R
    end
end


Nx = 500
Ny = 500
xgrid = range(minimum(xy_dat[:,1])-0.1, maximum(xy_dat[:,1])+0.1; length=Nx)
ygrid = range(minimum(xy_dat[:,2])-0.1, maximum(xy_dat[:,2])+0.1; length=Ny)
xygrid = collect.(Iterators.product(xgrid, ygrid))
dx = step(xgrid)
dy = step(ygrid)
R = (14.68466 + 14.6579) / 2 * (13.1059 + 13.03028) / 2

C1 = zeros(Float64, Ny, Nx)
@time for j in 1:Ny
    for i in 1:Nx
        value = fmean([xgrid[i], ygrid[j]], f1s_thinned, f2s_thinned, w_thinned, R, tlp, trp, lrp, llp)
        cj = j > 1 ? C1[j-1, i] : 0.0
        ci = i > 1 ? C1[j, i-1] : 0.0
        cij = (j > 1 && i > 1) ? C1[j-1, i-1] : 0.0
        C1[j, i] = ci + cj - cij + value * dx * dy
    end
end
C1[:, end]

w0 = 0.102
C2 = zeros(Float64, Ny, Nx)
@time for j in 1:Ny
    for i in 1:Nx
        value = fking([xgrid[i], ygrid[j]], w0, A / adjust_ratio, tlp, trp, lrp, llp)
        cj = j > 1 ? C2[j-1, i] : 0.0
        ci = i > 1 ? C2[j, i-1] : 0.0
        cij = (j > 1 && i > 1) ? C2[j-1, i-1] : 0.0
        C2[j, i] = ci + cj - cij + value * dx * dy
    end
end
C2[:, end]

xloc = [119, 120.5, 121., 121.5, 123.0]
ix = [searchsortedlast(xgrid, loc) for loc in xloc]

C3 = zeros(Float64, Ny, length(xloc))
@time for (i,idx) in enumerate(ix)
    for j in 1:Ny
        C3[j, i] = sum(xy_dat[:,1] .<= xgrid[idx] .&& xy_dat[:,2] .<= ygrid[j]) / n
    end
end
C3

C4 = zeros(Float64, Ny, Nx)
@time for i in 1:Nx
    for j in 1:Ny
        C4[j, i] = sum(xy_dat[:, 1] .<= xgrid[i] .&& xy_dat[:, 2] .<= ygrid[j]) / n
    end
end
C4[:, end]


blue = "#1f77b4"
orange = "#ff7f0e"
green = "#2ca02c"
red = "#d62728"

Nx = 500
Ny = 500
xgrid = range(minimum(xy_dat[:, 1]) - 0.1, maximum(xy_dat[:, 1]) + 0.1; length=Nx)
ygrid = range(minimum(xy_dat[:, 2]) - 0.1, maximum(xy_dat[:, 2]) + 0.1; length=Ny)


fig, axs = subplots(1, 3, figsize=(15, 4))
fig.subplots_adjust(bottom=0.20, wspace=0.3, right=0.9)
ax0 = axs[1]
ax1 = axs[2]
ax2 = axs[3]
for i in 1:length(xloc)
    if i == 1
        ax0.plot(ygrid, C1[:, ix[i]], label="MDPM", alpha=0.6, linestyle="--", linewidth=2, color=blue)
        ax0.plot(ygrid, C2[:, ix[i]], label="King's Profile", linestyle=":", alpha=1, linewidth=2, color=orange)
        ax0.plot(ygrid, C3[:, i], label="Empirical CDF", linestyle="-", alpha=0.6, linewidth=2, color=green)
    else
        ax0.plot(ygrid, C1[:, ix[i]], alpha=0.6, linestyle="--", linewidth=2, color=blue)
        ax0.plot(ygrid, C2[:, ix[i]], linestyle=":", alpha=1, linewidth=2, color=orange)
        ax0.plot(ygrid, C3[:, i], linestyle="-", alpha=0.6, linewidth=2, color=green)
    end
end
ax0.legend(frameon=false)
ax0.text(132.5, 0.82, "X=$(xloc[5])", fontsize=10, color="black")
ax0.text(132.5, 0.61, "X=$(xloc[4])", fontsize=10, color="black")
ax0.text(132.5, 0.41, "X=$(xloc[3])", fontsize=10, color="black")
ax0.text(132.5, 0.19, "X=$(xloc[2])", fontsize=10, color="black")
ax0.text(132.5, 0.02, "X=$(xloc[1])", fontsize=10, color="black")
ax0.set_xlabel("Y")
ax0.set_ylabel("CDF")
ax0.set_title("CDFs along vertical slices")

Cdiff_MDPM = C1 - C4
Cdiff_King = C2 - C4
vmin = min(minimum(Cdiff_MDPM), minimum(Cdiff_King))
vmax = max(maximum(Cdiff_MDPM), maximum(Cdiff_King))
idx = [1, 100, 200, 300, 400, 500]
xvec = round.(xgrid[idx], digits=2)
yvec = round.(ygrid[idx], digits=2)

norm = matplotlib.colors.TwoSlopeNorm(vcenter=0,
    vmin=vmin,
    vmax=vmax)
im1 = ax1.imshow(Cdiff_MDPM, origin="lower", aspect="auto",
    cmap="PiYG", norm=norm)
ax1.set_xticks(idx .- 1)
ax1.set_xticklabels(string.(xvec))
ax1.set_yticks(idx .- 1)
ax1.set_yticklabels(string.(yvec))
ax1.set_title("MDPM")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

im2 = ax2.imshow(Cdiff_King, origin="lower", aspect="auto",
    cmap="PiYG", norm=norm)
ax2.set_xticks(idx)
ax2.set_xticklabels(xvec)
ax2.set_yticks(idx)
ax2.set_yticklabels(yvec)
ax2.set_title("King's Profile")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")

cbar = fig.colorbar(im1, ax=[ax1, ax2], fraction=0.046, pad=0.02)
cbar.set_label("diff", rotation=270, labelpad=12)

fig.text(0.64, 0.05, "Differences between model CDF and Empirical CDF",
    ha="center", va="center", fontsize=12)
fig.text(0.23, 0.05, "CDF Comparison at five X locations",
    ha="center", va="center", fontsize=12)
savefig("astro_cdfs_red.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
