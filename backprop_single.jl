using LinearAlgebra
using DifferentialEquations
using Plots
using Printf
ENV["GKS_ENCODING"] = "utf-8"
gr()

include("ks_hamiltonian.jl")
include("fun_ks_trans.jl")

# ===== パラメータ =====
mu   = 0.01215058426994  # Earth-Moon system
C    = 3.00               # ヤコビ定数
t_fin = 20pi              # 積分終端（仮想時間）

# ===== 初期条件（特異点 = 月、KS 空間原点） =====
# u = [0,0,0,0]：月の位置（特異点）
# w の大きさは エネルギー拘束から |w| = sqrt(8*mu)
# w の方向を指定して 1 本の軌道を決める

w_dir = [1.0, 1.0, 0.0, 1.0]          # w の方向ベクトル（適宜変更）
w_dir = w_dir / norm(w_dir)
w0    = sqrt(8 * mu) * w_dir

u0 = vcat([0.0, 0.0, 0.0, 0.0], w0)   # KS 初期状態ベクトル

@printf("初期条件 (w):\n  w = [%.6f, %.6f, %.6f, %.6f]\n  |w| = %.6f  (ref: sqrt(8μ) = %.6f)\n",
        w0[1], w0[2], w0[3], w0[4], norm(w0), sqrt(8*mu))

# ===== ODE 求解（特異点 → 逆伝播） =====
# tspan を (t_fin, 0.0) にすることで仮想時間を逆方向に積分
tspan = (0.0, t_fin)

n_save = 10000  # 出力点数
prob = ODEProblem{true, DifferentialEquations.SciMLBase.FullSpecialize}(ks_hamiltonian, u0, tspan, (mu, C))
sol = solve(prob, Vern9(), abstol=1e-12, reltol=1e-12, saveat=range(tspan[1], tspan[2], length=n_save))

println("積分ステップ数: $(length(sol.t))")

# ===== KS → 物理座標へ変換 =====
num_points = length(sol.t)
xyz_traj   = zeros(3, num_points)
for j in 1:num_points
    xyz_traj[:, j] = fun_ks_to_physical(sol[j], mu)[1:3]
end

moon_x = 1 - mu  # 月の x 座標

# ===== 軸範囲ユーティリティ =====
AXIS_MAX = 2.0
function axis_lim(vals)
    lo = max(minimum(vals), -AXIS_MAX)
    hi = min(maximum(vals),  AXIS_MAX)
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

xlim = axis_lim(xyz_traj[1, :])
ylim = axis_lim(xyz_traj[2, :])
zlim = axis_lim(xyz_traj[3, :])

# ===== プロットスタイル =====
default(
    fontfamily       = "sans-serif",
    guidefontsize    = 10,
    tickfontsize     = 7,
    legendfontsize   = 9,
    titlefontsize    = 11,
    linewidth        = 0.8,
    framestyle       = :box,
    background_color = :white,
    foreground_color = :black,
    grid             = true,
    gridcolor        = :gray70,
    gridlinewidth    = 0.5,
    tick_direction   = :out,
    right_margin     = 4Plots.mm,
    bottom_margin    = 4Plots.mm,
)

# === 3D 図 ===
plt_3d = plot3d(
    xyz_traj[1, :], xyz_traj[2, :], xyz_traj[3, :],
    lw=1.0, color=:royalblue,
    xlabel="x [-]", ylabel="y [-]", zlabel="z [-]",
    title="(a) 3D",
    legend=false,
    xlims=xlim, ylims=ylim, zlims=zlim,
    xticks=nicer_ticks(xlim...), yticks=nicer_ticks(ylim...), zticks=nicer_ticks(zlim...),
)
plot3d!(plt_3d, [moon_x], [0.0], [0.0],
    seriestype=:scatter, markersize=4, color=:black, markershape=:circle)

# === x-y 平面 ===
plt_xy = plot(
    xyz_traj[1, :], xyz_traj[2, :],
    lw=1.0, color=:royalblue,
    xlabel="x [-]", ylabel="y [-]",
    title="(b) x-y plane",
    legend=false, aspect_ratio=:equal,
    xlims=xlim, ylims=ylim,
    xticks=nicer_ticks(xlim...), yticks=nicer_ticks(ylim...),
)
scatter!(plt_xy, [moon_x], [0.0], markersize=4, color=:black)

# === y-z 平面 ===
plt_yz = plot(
    xyz_traj[2, :], xyz_traj[3, :],
    lw=1.0, color=:royalblue,
    xlabel="y [-]", ylabel="z [-]",
    title="(c) y-z plane",
    legend=false,
    xlims=ylim, ylims=zlim,
    xticks=nicer_ticks(ylim...), yticks=nicer_ticks(zlim...),
)
scatter!(plt_yz, [0.0], [0.0], markersize=4, color=:black)

# === z-x 平面 ===
plt_zx = plot(
    xyz_traj[3, :], xyz_traj[1, :],
    lw=1.0, color=:royalblue,
    xlabel="z [-]", ylabel="x [-]",
    title="(d) z-x plane",
    legend=false,
    xlims=zlim, ylims=xlim,
    xticks=nicer_ticks(zlim...), yticks=nicer_ticks(xlim...),
)
scatter!(plt_zx, [0.0], [moon_x], markersize=4, color=:black)

# === 4 パネル結合図 ===
plt_combined = plot(plt_3d, plt_xy, plt_yz, plt_zx,
    layout=(2, 2),
    size=(800, 700),
    dpi=300,
    plot_title="Backprop from Singularity  (C = $C)",
    plot_titlefontsize=12,
)

# ===== 保存 =====
C_str  = @sprintf("%.1f", C)
outdir = "results/pdf/backprop_C_$(C_str)"
mkpath(outdir)

savefig(plt_combined, "$outdir/combined.pdf")
savefig(plt_3d,       "$outdir/3d.pdf")
savefig(plt_xy,       "$outdir/xy.pdf")
savefig(plt_yz,       "$outdir/yz.pdf")
savefig(plt_zx,       "$outdir/zx.pdf")
println("PDF 保存完了 → $outdir/")

display(plt_combined)
println("完了。")
