using Distributed
using LinearAlgebra
using DifferentialEquations
using Plots
using Printf
plotlyjs()  

include("LC_eom.jl")
include("LC_eom_auto.jl")

# Parameters
num_trj = 1000
mu = 0.01215058426994 # earth-moon system
C = 2.8 # Jacobi constant
t_fin = 2.0pi 
tspan = (t_fin, 0.0)

# Generate initial conditions with constraints
u0_list = Vector{Vector{Float64}}(undef, num_trj)

for i in 1:num_trj
    # Pのランダム方向（円周上で均一分布）
    v = randn(2)
    v_normalized = v / norm(v)
    P = sqrt(8*mu) * v_normalized
    # Q = 0（特異点ぴったりから出発）
    u0_list[i] = [0.0, 0.0, P[1], P[2]]
end

# Solve ODEs for all initial conditions
solutions = Vector{Any}(undef, num_trj)
solutions_xy = Vector{Any}(undef, num_trj)

println("Solving ODEs for $num_trj trajectories...")
for i in 1:num_trj
    println("  Trajectory $i/$num_trj")
    # prob = ODEProblem{true, DifferentialEquations.SciMLBase.FullSpecialize}(LC_eom, u0_list[i], tspan, (mu, C))
    prob = ODEProblem{true, DifferentialEquations.SciMLBase.FullSpecialize}(LC_eom_auto, u0_list[i], tspan, (mu, C))
    solutions[i] = solve(prob, Vern9(), abstol=1e-12, reltol=1e-12)

    # Convert KS to Cartesian coordinates for plotting
    num_points = length(solutions[i].t)
    xy_traj = zeros(2, num_points)
    for j in 1:num_points
        u = solutions[i][j]
        q = u[1:2]
        Au = hcat(
                [u[1], u[2]],
                [-u[2],  u[1]],
            )
        X = Au * q
        xy_traj[:, j] = X .+ [1 - mu; 0.0]  # Shift to Earth-centered frame
    end
    solutions_xy[i] = xy_traj
end
println("Done!")

# Plot all trajectories on the same interactive figure
plt = plot(
    solutions_xy[1][1, :], solutions_xy[1][2, :],
    lw=0.5,
    title="KS Trajectories (C = $C)",
    xlabel="X", ylabel="Y",
    legend=false
)

for i in 2:num_trj
    plot!(plt, solutions_xy[i][1, :], solutions_xy[i][2, :],
          lw=0.5)
end

plot!(plt, [1 - mu], [0.0], seriestype=:scatter, markersize=5, color=:gray)
display(plt)