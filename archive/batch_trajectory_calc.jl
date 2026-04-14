using Distributed
using LinearAlgebra
using DifferentialEquations
using Plots
using Printf

plotlyjs()

include("src/ks_eom_auto.jl")

# ===== Parameters =====
num_trj = 10                         # 各Cでの軌道数
mu = 0.01215058426994                # Earth-Moon system
t_fin = 20pi
tspan = (t_fin, 0.0)

# 計算するヤコビ定数のリスト
C_list = [2.0, 2.5, 3.0, 3.5]



# 出力ディレクトリ
output_dir = "trajectory_results"
mkpath(output_dir)

# ===== 各ヤコビ定数で計算 =====
println("=" ^ 50)
println("Batch Trajectory Calculation")
println("=" ^ 50)
println("Number of trajectories per C: $num_trj")
println("Jacobi constants: $C_list")
println("Output directory: $output_dir")
println("=" ^ 50)

for C in C_list
    println("\n>>> Computing C = $C ...")

    # 初期条件生成
    u0_list = Vector{Vector{Float64}}(undef, num_trj)
    for i in 1:num_trj
        v = randn(4)
        v_normalized = v / norm(v)
        P = sqrt(8*mu) * v_normalized
        u0_list[i] = [0.0, 0.0, 0.0, 0.0, P[1], P[2], P[3], P[4]]
    end

    # ODE計算
    solutions = Vector{Any}(undef, num_trj)
    solutions_xyz = Vector{Any}(undef, num_trj)
    solutions_vel = Vector{Any}(undef, num_trj)
    solutions_t = Vector{Any}(undef, num_trj)
    xyz_initial = Vector{Vector{Float64}}(undef, num_trj)

    # uの値が大きくなりすぎたら停止するコールバック
    condition(u, t, integrator) = norm(u) > 2
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!)

    for i in 1:num_trj
        prob = ODEProblem{true, DifferentialEquations.SciMLBase.FullSpecialize}(ks_eom_auto, u0_list[i], tspan, (mu, C))
        sol = solve(prob, Vern9(), abstol=1e-12, reltol=1e-12, callback=cb)
        solutions[i] = sol

        # KS → Cartesian 変換（位置と速度）
        num_points = length(sol.t)
        xyz_traj = zeros(3, num_points)
        vel_traj = zeros(3, num_points)
        for j in 1:num_points
            u = sol[j]
            Q = u[1:4]
            P = u[5:8]
            Au = hcat(
                [u[1], u[2], u[3]],
                [-u[2],  u[1],  u[4]],
                [-u[3], -u[4],  u[1]],
                [u[4], -u[3],  u[2]],
            )
            X = Au * Q
            xyz_traj[:, j] = X .+ [1 - mu; 0.0; 0.0]

            # 速度計算: p = A(Q) * P / (2 * r_M)
            r_M = Q[1]^2 + Q[2]^2 + Q[3]^2 + Q[4]^2
            if r_M > 1e-14
                vel_traj[:, j] = Au * P / (2 * r_M)
            else
                vel_traj[:, j] = [0.0, 0.0, 0.0]
            end
        end
        solutions_xyz[i] = xyz_traj
        solutions_vel[i] = vel_traj
        solutions_t[i] = sol.t
        xyz_initial[i] = xyz_traj[:, end]  # 逆時間なので最後が初期位置
    end

    # プロット作成
    plt = plot3d(
        solutions_xyz[1][1, :], solutions_xyz[1][2, :], solutions_xyz[1][3, :],
        lw=0.5,
        title="KS Trajectories (C = $C)",
        xlabel="X", ylabel="Y", zlabel="Z",
        legend=false
    )

    for i in 2:num_trj
        plot3d!(plt, solutions_xyz[i][1, :], solutions_xyz[i][2, :], solutions_xyz[i][3, :], lw=0.5)
    end

    # 月の位置
    scatter3d!(plt, [1 - mu], [0.0], [0.0], markersize=0.5, color=:gray, label="Moon")

    # プロットをHTMLとして一時保存
    plot_filename = joinpath(output_dir, "plot_C_$(C).html")
    savefig(plt, plot_filename)

    # 時間履歴プロットを作成（要素ごとに全軌道を表示）
    element_names = ["x", "y", "z", "vx", "vy", "vz"]

    for (elem_idx, elem_name) in enumerate(element_names)
        if elem_idx <= 3
            # 位置成分
            plt_elem = plot(
                xlabel="t", ylabel=elem_name,
                title="$elem_name(t) - All Trajectories (C = $C)",
                legend=false
            )
            for i in 1:num_trj
                plot!(plt_elem, solutions_t[i], solutions_xyz[i][elem_idx, :], lw=0.8)
            end
        else
            # 速度成分
            vel_idx = elem_idx - 3
            plt_elem = plot(
                xlabel="t", ylabel=elem_name,
                title="$elem_name(t) - All Trajectories (C = $C)",
                legend=false
            )
            for i in 1:num_trj
                plot!(plt_elem, solutions_t[i], solutions_vel[i][vel_idx, :], lw=0.8)
            end
        end

        elem_plot_filename = joinpath(output_dir, "time_history_C_$(C)_$(elem_name).html")
        savefig(plt_elem, elem_plot_filename)
    end

    # メインHTMLページを作成（プロット埋め込み + データテーブル）
    main_html = joinpath(output_dir, "trajectories_C_$(C).html")
    open(main_html, "w") do f
        println(f, """
<!DOCTYPE html>
<html>
<head>
    <title>Trajectories C = $C</title>
    <style>
        body { font-family: 'Courier New', monospace; margin: 20px; background: #f5f5f5; }
        h1, h2, h3 { color: #333; }
        .container { max-width: 1200px; margin: 0 auto; }
        .plot-container { width: 100%; height: 600px; border: 1px solid #ccc; margin: 20px 0; }
        table { border-collapse: collapse; margin: 20px 0; background: white; }
        th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }
        th { background: #4CAF50; color: white; }
        tr:nth-child(even) { background: #f9f9f9; }
        .section { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        a { color: #0066cc; }
        .download-links { margin: 10px 0; }
        .download-links a { margin-right: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>KS Trajectories (C = $C)</h1>
        <p><a href="index.html">← Back to Index</a></p>

        <div class="section">
            <h2>3D Plot (Interactive)</h2>
            <iframe src="plot_C_$(C).html" class="plot-container" frameborder="0"></iframe>
        </div>

        <div class="section">
            <h2>Initial Conditions</h2>
            <h3>KS Coordinates (Q, P)</h3>
            <table>
                <tr>
                    <th>Traj</th>
                    <th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th>
                    <th>P1</th><th>P2</th><th>P3</th><th>P4</th>
                </tr>
""")
        for i in 1:num_trj
            Q = u0_list[i][1:4]
            P = u0_list[i][5:8]
            @printf(f, "                <tr><td>%d</td>", i)
            @printf(f, "<td>%.6f</td><td>%.6f</td><td>%.6f</td><td>%.6f</td>", Q[1], Q[2], Q[3], Q[4])
            @printf(f, "<td>%+.6f</td><td>%+.6f</td><td>%+.6f</td><td>%+.6f</td></tr>\n", P[1], P[2], P[3], P[4])
        end
        println(f, """
            </table>

            <h3>Cartesian Coordinates (Initial XYZ)</h3>
            <table>
                <tr>
                    <th>Traj</th>
                    <th>X</th><th>Y</th><th>Z</th>
                </tr>
""")
        for i in 1:num_trj
            x, y, z = xyz_initial[i]
            @printf(f, "                <tr><td>%d</td><td>%.8f</td><td>%.8f</td><td>%.8f</td></tr>\n", i, x, y, z)
        end
        println(f, """
            </table>
        </div>

        <div class="section">
            <h2>Time History (Position)</h2>
            <h3>x(t)</h3>
            <iframe src="time_history_C_$(C)_x.html" class="plot-container" style="height:400px;" frameborder="0"></iframe>
            <h3>y(t)</h3>
            <iframe src="time_history_C_$(C)_y.html" class="plot-container" style="height:400px;" frameborder="0"></iframe>
            <h3>z(t)</h3>
            <iframe src="time_history_C_$(C)_z.html" class="plot-container" style="height:400px;" frameborder="0"></iframe>
        </div>

        <div class="section">
            <h2>Time History (Velocity)</h2>
            <h3>vx(t)</h3>
            <iframe src="time_history_C_$(C)_vx.html" class="plot-container" style="height:400px;" frameborder="0"></iframe>
            <h3>vy(t)</h3>
            <iframe src="time_history_C_$(C)_vy.html" class="plot-container" style="height:400px;" frameborder="0"></iframe>
            <h3>vz(t)</h3>
            <iframe src="time_history_C_$(C)_vz.html" class="plot-container" style="height:400px;" frameborder="0"></iframe>
        </div>

    </div>
</body>
</html>
""")
    end
    println("    Saved: $main_html")
end

# ===== 一覧HTMLを作成 =====
index_html = joinpath(output_dir, "index.html")
open(index_html, "w") do f
    println(f, """
<!DOCTYPE html>
<html>
<head>
    <title>Trajectory Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        h1 { color: #333; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        ul { list-style-type: none; padding: 0; }
        li { margin: 15px 0; padding: 15px; background: #f0f0f0; border-radius: 5px; }
        a { color: #0066cc; text-decoration: none; font-size: 20px; font-weight: bold; }
        a:hover { text-decoration: underline; }
        .info { color: #666; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>KS Trajectory Results</h1>
        <p class="info">Number of trajectories per C: $num_trj<br>
        μ = $mu (Earth-Moon system)</p>
        <h2>Select Jacobi Constant</h2>
        <ul>
""")
    for C in C_list
        println(f, "            <li><a href=\"trajectories_C_$(C).html\">C = $C</a></li>")
    end
    println(f, """
        </ul>
    </div>
</body>
</html>
""")
end

println("\n" * "=" ^ 50)
println("All computations completed!")
println("Open $index_html in browser to access all results.")
println("=" ^ 50)

# ブラウザで開く
run(`open $index_html`)
