using LinearAlgebra
using DifferentialEquations
using Plots
using Printf
ENV["GKS_ENCODING"] = "utf-8"  # マイナス記号などの文字化け防止
gr()  # PDF出力対応バックエンド

include("ks_hamiltonian.jl")
include("fun_ks_trans.jl")
# ===== パラメータ =====
num_trj = 10
mu      = 0.01215058426994  # Earth-Moon system
t_fin   = 4pi
C_list  = [2.8, 3.0, 3.2]  # 試すヤコビ定数のリスト

# 論文向けスタイル設定（共通）
default(
    fontfamily       = "sans-serif",  # GRで確実にレンダリングされるフォント
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
    tick_direction   = :out,          # 目盛りを外向きにして重なりを減らす
    right_margin     = 4Plots.mm,
    bottom_margin    = 4Plots.mm,
)

clrs   = palette(:tab10)
moon_x = 1 - mu  # 月の x 座標（y, z = 0）

# ===== ヤコビ定数ごとにループ =====
for C in C_list
    println("\n========================================")
    println("  C = $C")
    println("========================================")

    local tspan = (t_fin, 0.0)

    # 初期条件生成
    local u0_list = Vector{Vector{Float64}}(undef, num_trj)
    for i in 1:num_trj
        v            = randn(4)
        v_normalized = v / norm(v)
        P            = sqrt(8 * mu) * v_normalized
        u0_list[i]   = [0.0, 0.0, 0.0, 0.0, P[1], P[2], P[3], P[4]]
    end

    # ODE 求解 & KS → 変換
    local solutions_xyz = Vector{Any}(undef, num_trj)
    println("Solving ODEs for $num_trj trajectories...")
    for i in 1:num_trj
        println("  Trajectory $i/$num_trj")
        local prob = ODEProblem{true, DifferentialEquations.SciMLBase.FullSpecialize}(ks_hamiltonian, u0_list[i], tspan, (mu, C))
        local sol = solve(prob, Vern9(), abstol=1e-12, reltol=1e-12)

        num_points = length(sol.t)
        xyz_traj   = zeros(3, num_points)
        for j in 1:num_points
            xyz_traj[:, j] = fun_ks_to_physical(sol[j], mu)[1:3]
        end
        solutions_xyz[i] = xyz_traj
    end
    println("Done!")

    # 軸範囲：データ範囲と上限[-1.5, 1.5]の小さい方を採用
    AXIS_MAX = 2.0
    function axis_lim(vals)
        lo = max(minimum(vals), -AXIS_MAX)
        hi = min(maximum(vals),  AXIS_MAX)
        return lo, hi
    end

    # 目盛り：軸範囲内でキリのよい間隔を約5本
    function nicer_ticks(lo, hi; n=5)
        step = (hi - lo) / (n - 1)
        mag  = 10.0 ^ floor(log10(abs(step)))
        step = ceil(step / mag) * mag
        start = ceil(lo / step) * step
        return start:step:hi
    end

    all_x = vcat([solutions_xyz[i][1, :] for i in 1:num_trj]...)
    all_y = vcat([solutions_xyz[i][2, :] for i in 1:num_trj]...)
    all_z = vcat([solutions_xyz[i][3, :] for i in 1:num_trj]...)

    xlim = axis_lim(all_x)
    ylim = axis_lim(all_y)
    zlim = axis_lim(all_z)

    # === 3D 図 ===
    local plt_3d = plot3d(
        solutions_xyz[1][1, :], solutions_xyz[1][2, :], solutions_xyz[1][3, :],
        lw=0.8, color=clrs[1],
        xlabel="x [-]", ylabel="y [-]", zlabel="z [-]",
        title="(a) 3D",
        legend=false,
        xlims=xlim, ylims=ylim, zlims=zlim,
        xticks=nicer_ticks(xlim...), yticks=nicer_ticks(ylim...), zticks=nicer_ticks(zlim...),
        grid=true, xgrid=true, ygrid=true, zgrid=true,
    )
    for i in 2:num_trj
        plot3d!(plt_3d,
            solutions_xyz[i][1, :], solutions_xyz[i][2, :], solutions_xyz[i][3, :],
            lw=0.8, color=clrs[mod1(i, 10)])
    end
    plot3d!(plt_3d, [moon_x], [0.0], [0.0],
        seriestype=:scatter, markersize=4, color=:black, markershape=:circle)

    # === x-y 平面 ===
    local plt_xy = plot(
        solutions_xyz[1][1, :], solutions_xyz[1][2, :],
        lw=0.8, color=clrs[1],
        xlabel="x [-]", ylabel="y [-]",
        title="(b) x-y plane",
        legend=false, aspect_ratio=:equal,
        xlims=xlim, ylims=ylim,
        xticks=nicer_ticks(xlim...), yticks=nicer_ticks(ylim...),
        grid=true,
    )
    for i in 2:num_trj
        plot!(plt_xy, solutions_xyz[i][1, :], solutions_xyz[i][2, :],
              lw=0.8, color=clrs[mod1(i, 10)])
    end
    scatter!(plt_xy, [moon_x], [0.0], markersize=4, color=:black)

    # === y-z 平面 ===
    local plt_yz = plot(
        solutions_xyz[1][2, :], solutions_xyz[1][3, :],
        lw=0.8, color=clrs[1],
        xlabel="y [-]", ylabel="z [-]",
        title="(c) y-z plane",
        legend=false,
        xlims=ylim, ylims=zlim,
        xticks=nicer_ticks(ylim...), yticks=nicer_ticks(zlim...),
        grid=true,
    )
    for i in 2:num_trj
        plot!(plt_yz, solutions_xyz[i][2, :], solutions_xyz[i][3, :],
              lw=0.8, color=clrs[mod1(i, 10)])
    end
    scatter!(plt_yz, [0.0], [0.0], markersize=4, color=:black)

    # === z-x 平面 ===
    local plt_zx = plot(
        solutions_xyz[1][3, :], solutions_xyz[1][1, :],
        lw=0.8, color=clrs[1],
        xlabel="z [-]", ylabel="x [-]",
        title="(d) z-x plane",
        legend=false,
        xlims=zlim, ylims=xlim,
        xticks=nicer_ticks(zlim...), yticks=nicer_ticks(xlim...),
        grid=true,
    )
    for i in 2:num_trj
        plot!(plt_zx, solutions_xyz[i][3, :], solutions_xyz[i][1, :],
              lw=0.8, color=clrs[mod1(i, 10)])
    end
    scatter!(plt_zx, [0.0], [moon_x], markersize=4, color=:black)

    # === 4パネル結合図 ===
    local plt_combined = plot(plt_3d, plt_xy, plt_yz, plt_zx,
        layout=(2, 2),
        size=(800, 700),
        dpi=300,
        plot_title="Collision Orbits (C = $C)",
        plot_titlefontsize=12,
    )

    # PDF 保存：pdf_results/C_<値>/ フォルダに格納
    C_str  = @sprintf("%.1f", C)
    outdir = "results/pdf/C_$(C_str)"
    mkpath(outdir)

    savefig(plt_combined, "$outdir/combined.pdf")
    savefig(plt_3d,       "$outdir/3d.pdf")
    savefig(plt_xy,       "$outdir/xy.pdf")
    savefig(plt_yz,       "$outdir/yz.pdf")
    savefig(plt_zx,       "$outdir/zx.pdf")
    println("PDF 保存完了 → $outdir/")

    display(plt_combined)

    # 初期値を表形式で表示
    println("\n=== Initial Conditions (P components)  C = $C ===")
    println("  Traj |     P1      |     P2      |     P3      |     P4      |")
    println("-------|-------------|-------------|-------------|-------------|")
    for i in 1:num_trj
        P = u0_list[i][5:8]
        @printf("  %3d  | %+.6f | %+.6f | %+.6f | %+.6f |\n", i, P[1], P[2], P[3], P[4])
    end
end

println("\n全ケース完了。")
