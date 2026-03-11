using Distributed
using LinearAlgebra
using DifferentialEquations
using Plots
using Printf
plotlyjs()  # インタラクティブなバックエンド（回転・ズーム・パン可能）

include("ks_eom_auto.jl")

# Parameters
num_trj = 2
mu = 0.01215058426994 # earth-moon system
C = 2.0 # Jacobi constant
t_fin = 10pi 
tspan = (t_fin, 0.0)

# Generate initial conditions with constraints
u0_list = Vector{Vector{Float64}}(undef, num_trj)

for i in 1:num_trj
    # Pのランダム方向（4次元超球面上で均一分布）
    v = randn(4)
    v_normalized = v / norm(v)
    P = (mu/2) * v_normalized

    # Q = 0（特異点ぴったりから出発）
    u0_list[i] = [0.0, 0.0, 0.0, 0.0, P[1], P[2], P[3], P[4]]
end

# Solve ODEs for all initial conditions
solutions = Vector{Any}(undef, num_trj)
solutions_xyz = Vector{Any}(undef, num_trj)

println("Solving ODEs for $num_trj trajectories...")
for i in 1:num_trj
    println("  Trajectory $i/$num_trj")
    prob = ODEProblem{true, DifferentialEquations.SciMLBase.FullSpecialize}(
        ks_eom_auto, u0_list[i], tspan, (mu, C))
    solutions[i] = solve(prob, Vern9(), abstol=1e-12, reltol=1e-12)

    # Convert KS to Cartesian coordinates for plotting
    num_points = length(solutions[i].t)
    xyz_traj = zeros(3, num_points)
    for j in 1:num_points
        u = solutions[i][j]
        q = u[1:4]
        Au = hcat(
                [u[1], u[2], u[3]],
                [-u[2],  u[1],  u[4]],
                [-u[3], -u[4],  u[1]],
                [u[4], -u[3],  u[2]],
            )
        X = Au * q
        xyz_traj[:, j] = X .+ [1 - mu; 0.0; 0.0]  # Shift to Earth-centered frame
    end
    solutions_xyz[i] = xyz_traj
end
println("Done!")

# Plot all trajectories on the same interactive figure
plt = plot3d(
    solutions_xyz[1][1, :], solutions_xyz[1][2, :], solutions_xyz[1][3, :],
    lw=0.5,
    title="KS Trajectories (C = $C)",
    xlabel="X", ylabel="Y", zlabel="Z",
    legend=false
)

for i in 2:num_trj
    plot3d!(plt, solutions_xyz[i][1, :], solutions_xyz[i][2, :], solutions_xyz[i][3, :],
            lw=0.5)
end

plot3d!(plt, [1 - mu], [0.0], [0.0], seriestype=:scatter, markersize=5, color=:gray)

display(plt)

# 初期値を表形式で表示
println("\n=== Initial Conditions (P components) ===")
println("  Traj |     P1      |     P2      |     P3      |     P4      |")
println("-------|-------------|-------------|-------------|-------------|")
for i in 1:num_trj
    P = u0_list[i][5:8]
    @printf("  %3d  | %+.6f | %+.6f | %+.6f | %+.6f |\n", i, P[1], P[2], P[3], P[4])
end


plotlyjs()  # インタラクティブモードに戻す
