using Distributed
using LinearAlgebra
using DifferentialEquations
using Plots
include("ks_eom_hamilton.jl")

num_trj = 10
mu = 0.01215058426994 # earth-moon system
C = 3.12 # Jacobi constant
t_fin = 2.0pi * 100
u0_temp = [0.0, 0.8, 0.0, 0.0,
           sqrt(8mu), 0.0, 0.0, 0.0]
tspan = (t_fin, 0.0)
prob = ODEProblem(ks_eom_hamilton, u0_temp, tspan, (mu, C))
sol = solve(prob, Vern9(), abstol=1e-12, reltol=1e-12)

plot3d(sol[1, :], sol[2, :], sol[3, :], lw=0.5, label="Trajectory", title="KS Trajectory in Earth-Moon System", xlabel="X", ylabel="Y", zlabel="Z")

