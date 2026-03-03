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

function polygon_area(xs, ys)
    n = length(xs)
    s = 0.0
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        s += xs[i] * ys[j] - xs[j] * ys[i]
    end
    return 0.5 * abs(s)
end

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