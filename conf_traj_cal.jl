using Distributed
using LinearAlgebra
using DifferentialEquations
using Plots

include("ks_eom_hamilton.jl")

# Parameters
num_trj = 10
mu = 0.01215058426994 # earth-moon system
C = 2.95 # Jacobi constant
t_fin = 2.0pi 
tspan = (t_fin, 0.0)

# Generate initial conditions with constraints
u0_list = Vector{Vector{Float64}}(undef, num_trj)

for i in 1:num_trj
    theta_1 = rand() * pi
    theta_2 = rand() * pi
    theta_3 = rand() * 2.0pi

    U1 = 8*mu * cos(theta_1)
    U2 = 8*mu * sin(theta_1) * cos(theta_2)
    U3 = 8*mu * sin(theta_1) * sin(theta_2) * cos(theta_3)
    U4 = 8*mu * sin(theta_1) * sin(theta_2) * sin(theta_3)
    

    u0_list[i] = [0.0, 0.0, 0.0, 0.0,
                  U1, U2, U3, U4]  
end

# Solve ODEs for all initial conditions
solutions = Vector{Any}(undef, num_trj)
solutions_xyz = Vector{Any}(undef, num_trj)

println("Solving ODEs for $num_trj trajectories...")
for i in 1:num_trj
    println("  Trajectory $i/$num_trj")
    prob = ODEProblem{true, DifferentialEquations.SciMLBase.FullSpecialize}(
        ks_eom_hamilton, u0_list[i], tspan, (mu, C))
    solutions[i] = solve(prob, Vern9(), abstol=1e-12, reltol=1e-12)

    # Convert KS to Cartesian coordinates for plotting
    num_points = length(solutions[i].t)
    xyz_traj = zeros(3, num_points)
    for j in 1:num_points
        u = solutions[i][j]
        q = u[1:4]
        Au = hcat(
            [u[1], -u[2], -u[3]],
            [u[2],  u[1],  u[4]],
            [u[3], -u[4],  u[1]],
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
    lw=0.5, label="Trajectory 1",
    title="KS Trajectories in Earth-Moon System",
    xlabel="X", ylabel="Y", zlabel="Z",
    legend=:outertopright
)

for i in 2:num_trj
    plot3d!(plt, solutions_xyz[i][1, :], solutions_xyz[i][2, :], solutions_xyz[i][3, :],
            lw=0.5, label="Trajectory $i")
end

xlims!(0.7, 1.3)
ylims!(-0.3, 0.3)
zlims!(-0.3, 0.3)

display(plt)
