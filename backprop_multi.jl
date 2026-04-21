using LinearAlgebra
using DifferentialEquations
using Plots
using Printf
ENV["GKS_ENCODING"] = "utf-8"
gr()

include("ks_hamiltonian.jl")
include("fun_ks_trans.jl")

# ===== パラメータ =====
mu     = 0.01215058426994  # Earth-Moon system
C      = 2.9               # ヤコビ定数
t_fin  = 10pi              # 積分終端（仮想時間）
n_save = 10000             # 出力点数

# ===== エネルギー拘束から w の大きさ =====
w_mag = sqrt(8 * mu)

# ===== w0[4] の候補値（網羅的）=====
# w4 = ±w_mag は r3=0 の退化ケースのため除外
# θ = 30°, 60°, 90°, 120°, 150° に対応する5値
w4_vals = [cos(θ) for θ in [30, 60, 90, 120, 150] .* (π/180)] .* w_mag

# ===== w1-w2-w3 球面上の14方向（正規化済み）=====
DIRS_14 = vcat(
    # 軸方向 6 点
    [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
     [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
    # 対角線方向 8 点
    vec([Float64[a, b, c] / sqrt(3.0)
         for a in [1.0, -1.0], b in [1.0, -1.0], c in [1.0, -1.0]]),
)

moon_x = 1 - mu

# ===== ユーティリティ =====
AXIS_MAX      = 5.0
AXIS_MIN_RANGE = 1e-6

function axis_lim(vals)
    lo = max(minimum(vals), -AXIS_MAX)
    hi = min(maximum(vals),  AXIS_MAX)
    if hi - lo < AXIS_MIN_RANGE
        center = (lo + hi) / 2
        lo = center - 0.5
        hi = center + 0.5
    end
    return lo, hi
end

function nicer_ticks(lo, hi; n=5)
    lo ≈ hi && return [lo]
    step  = (hi - lo) / (n - 1)
    mag   = 10.0 ^ floor(log10(abs(step)))
    step  = ceil(step / mag) * mag
    step == 0.0 && return [lo]
    start = ceil(lo / step) * step
    return start:step:hi
end

# ===== プロットスタイル =====
default(
    fontfamily       = "sans-serif",
    guidefontsize    = 10,
    tickfontsize     = 7,
    legendfontsize   = 9,
    titlefontsize    = 10,
    linewidth        = 0.8,
    framestyle       = :box,
    background_color = :white,
    foreground_color = :black,
    grid             = true,
    gridcolor        = :gray80,
    gridlinewidth    = 0.4,
    tick_direction   = :out,
    right_margin     = 4Plots.mm,
    bottom_margin    = 4Plots.mm,
)

let
    C_str = @sprintf("%.1f", C)

    for w4 in w4_vals
        r3 = sqrt(max(w_mag^2 - w4^2, 0.0))

        all_w0 = [vcat(r3 * d, w4) for d in DIRS_14]

        w4_pct   = round(Int, w4 / w_mag * 100)
        w4_sign  = w4_pct >= 0 ? "p" : "m"
        w4_label = w4_sign * @sprintf("%03d", abs(w4_pct))
        outdir   = "results/pdf/backprop_multi/C_$(C_str)/w4_$(w4_label)"
        mkpath(outdir)

        @printf("\n===== w4 = %+.3f (= %+d%% |w|)  r3 = %.4f =====\n",
                w4, w4_pct, r3)

        for (k, w0) in enumerate(all_w0)
            u0    = vcat(zeros(4), w0)
            tspan = (0.0, t_fin)

            prob = ODEProblem{true, DifferentialEquations.SciMLBase.FullSpecialize}(
                ks_hamiltonian, u0, tspan, (mu, C))
            sol = solve(prob, Vern9(), abstol=1e-12, reltol=1e-12,
                        saveat=range(tspan[1], tspan[2], length=n_save))

            num_pts  = length(sol.t)
            xyz_traj = zeros(3, num_pts)
            for j in 1:num_pts
                xyz_traj[:, j] = fun_ks_to_physical(sol[j], mu)[1:3]
            end

            xlim = axis_lim(xyz_traj[1, :])
            ylim = axis_lim(xyz_traj[2, :])
            zlim = axis_lim(xyz_traj[3, :])

            # ===== 軌道図（3D）: traj_<k>_orbit.pdf =====
            plt_orbit = plot3d(
                xyz_traj[1,:], xyz_traj[2,:], xyz_traj[3,:],
                lw=0.8, color=:royalblue,
                xlabel="x [-]", ylabel="y [-]", zlabel="z [-]",
                title="Orbit  C=$C_str  w4=$(w4_label)  k=$k",
                legend=false,
                xlims=xlim, ylims=ylim, zlims=zlim,
                xticks=nicer_ticks(xlim...), yticks=nicer_ticks(ylim...),
                zticks=nicer_ticks(zlim...),
                size=(520, 460), dpi=200,
            )
            plot3d!(plt_orbit, [moon_x], [0.0], [0.0],
                seriestype=:scatter, markersize=4,
                color=:black, markershape=:circle)

            savefig(plt_orbit, "$outdir/traj_$(lpad(k,2,'0'))_orbit.pdf")

            # ===== 球面図（3D）: traj_<k>_sphere.pdf =====
            lim3 = max(r3 * 1.3, 0.01)
            φ_c  = range(0, 2π, length=60)
            θ_c  = range(0, π,  length=30)

            plt_sphere = plot3d(
                title="Sphere  C=$C_str  w4=$(w4_label)  k=$k",
                xlabel="w₁", ylabel="w₂", zlabel="w₃",
                legend=false,
                xlims=(-lim3, lim3), ylims=(-lim3, lim3), zlims=(-lim3, lim3),
                size=(520, 460), dpi=200,
            )
            # 経線 8本
            for φ in range(0, 2π, length=9)[1:end-1]
                plot3d!(plt_sphere,
                    r3 .* sin.(θ_c) .* cos(φ),
                    r3 .* sin.(θ_c) .* sin(φ),
                    r3 .* cos.(θ_c),
                    lw=0.4, color=:lightgray)
            end
            # 緯線 5本
            for θ in range(0, π, length=7)[2:end-1]
                plot3d!(plt_sphere,
                    r3 .* sin(θ) .* cos.(φ_c),
                    r3 .* sin(θ) .* sin.(φ_c),
                    fill(r3 * cos(θ), length(φ_c)),
                    lw=0.4, color=:lightgray)
            end
            # 座標軸
            for (ax, ay, az) in [
                    ([-lim3, lim3], [0.0, 0.0], [0.0, 0.0]),
                    ([0.0, 0.0], [-lim3, lim3], [0.0, 0.0]),
                    ([0.0, 0.0], [0.0, 0.0], [-lim3, lim3])]
                plot3d!(plt_sphere, ax, ay, az, lw=1.0, color=:gray50)
            end
            # 初期点
            plot3d!(plt_sphere, [w0[1]], [w0[2]], [w0[3]],
                seriestype=:scatter, markersize=6,
                color=:red, markershape=:circle)

            savefig(plt_sphere, "$outdir/traj_$(lpad(k,2,'0'))_sphere.pdf")

            @printf("  [%2d/14] → traj_%02d_{orbit,sphere}.pdf\n", k, k)
        end
    end

    println("\n全ケース完了。")
end  # let
