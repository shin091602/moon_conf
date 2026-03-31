using LinearAlgebra
using DifferentialEquations
using Plots
using Printf
using CSV

include("cr3bp_eom_newton.jl")
include("eom_hamiltonian_normal.jl")
include("ks_auto_hamilton.jl")
include("ks_hamiltonian.jl")

mu = 0.01215058426994  # Earth-Moon system
# 天体の位置 (CR3BP: M1=-mu, M2=1-mu)
x_M1 = -mu
x_M2 = 1 - mu

function load_periodic_orbit(filepath::String)
    data = CSV.File(filepath; comment="#", header=true)
    row = data[1]  # Get first data row
    # Extract state vector: [x0, y0, z0, vx0, vy0, vz0]
    x0 = [
        row[Symbol("x0 (LU) ")],
        row[Symbol("y0 (LU) ")],
        row[Symbol("z0 (LU) ")],
        row[Symbol("vx0 (LU/TU) ")],
        row[Symbol("vy0 (LU/TU) ")],
        row[Symbol("vz0 (LU/TU) ")]
    ]
    # Extract period
    T = row[Symbol("Period (TU) ")]

    return x0, T
end

function fun_Jacobi_const(x,mu)
    r1 = sqrt((mu + x[1])^2 + x[2]^2 + x[3]^2)
    r2 = sqrt((1 - mu - x[1])^2 + x[2]^2 + x[3]^2)

    U = (x[1]^2 + x[2]^2) / 2 + (1 - mu) / r1 + mu / r2
    v = sqrt(x[4]^2 + x[5]^2 + x[6]^2)
    C = 2*U - v^2

    return C
end

function fun_physical_to_ks(z, mu)
    q = z[1:3]
    p = z[4:6]

    r2 = sqrt((q[1] - 1 + mu)^2 + q[2]^2 + q[3]^2)

    u_1 = sqrt((q[1] + r2 + mu - 1) / 2)
    u = [u_1, q[2] / (2*u_1), q[3] / (2*u_1), 0]

    A = hcat(
        [u[1],  u[2],  u[3]],
        [-u[2], u[1],  u[4]],
        [-u[3], -u[4], u[1]],
        [u[4],  -u[3], u[2]],
    )
    w = 2 * A' * p

    zeta = vcat(u, w)

    return zeta
end

function fun_ks_to_physical(zeta, mu)
    u = zeta[1:4]
    w = zeta[5:8]

    A = hcat(
        [u[1],  u[2],  u[3]],
        [-u[2], u[1],  u[4]],
        [-u[3], -u[4], u[1]],
        [u[4],  -u[3], u[2]],
    )
    rM = u[1]^2 + u[2]^2 + u[3]^2 + u[4]^2
    p = (1/(2*rM)) * A * w

    q = A * u + [1-mu, 0.0, 0.0]

    z = vcat(q, p)

    return z
    
end



x0_1, T_1 = load_periodic_orbit("./input/periodic_orbits_L1.csv")
C = fun_Jacobi_const(x0_1, mu)
z0_1 = [x0_1[1],x0_1[2],x0_1[3], x0_1[4]-x0_1[2], x0_1[5]+x0_1[1], x0_1[6]]
zeta0_1 = fun_physical_to_ks(z0_1, mu)


# newton形式で運動方程式を記述・計算・プロット
prob_newton = ODEProblem{true, DifferentialEquations.SciMLBase.FullSpecialize}(cr3bp_eom_newton, x0_1, (0.0, T_1), mu)
sol_newton = solve(prob_newton, Vern9(); reltol=1e-12, abstol=1e-12)
# 軌道データ抽出
xs = [u[1] for u in sol_newton.u]
ys = [u[2] for u in sol_newton.u]
zs = [u[3] for u in sol_newton.u]

# プロット (2x2レイアウト: 3D + xy, yz, zx)
p3d = plot3d(xs, ys, zs; label="orbit", lc=:blue, xlabel="x", ylabel="y", zlabel="z", title="3D")
scatter!(p3d, [x_M2], [0.0], [0.0]; label="Moon",  ms=4, mc=:gray)

pxy = plot(xs, ys; label="orbit", lc=:blue, xlabel="x", ylabel="y", title="xy", aspect_ratio=:equal)
scatter!(pxy, [x_M2], [0.0]; label="Moon",  ms=4, mc=:gray)

pyz = plot(ys, zs; label="orbit", lc=:blue, xlabel="y", ylabel="z", title="yz", aspect_ratio=:equal)

pzx = plot(zs, xs; label="orbit", lc=:blue, xlabel="z", ylabel="x", title="zx", aspect_ratio=:equal)

fig = plot(p3d, pxy, pyz, pzx; layout=(2, 2), size=(900, 800), plot_title="Halo Orbit with Newton Equation", titlefontsize=10, titlefontcolor=:black, titlefontfamily="sans-serif", titlefontweight=:bold)
display(fig)
mkpath("results/confirmation")
savefig(fig, "results/confirmation/orbit_newton.png")


# ハミルトン形式で運動方程式を記述・計算・プロット
prob_hamilton = ODEProblem{true, DifferentialEquations.SciMLBase.FullSpecialize}(eom_hamiltonian_normal, z0_1, (0.0, T_1), mu)
sol_hamilton = solve(prob_hamilton, Vern9(); reltol=1e-12, abstol=1e-12)
# 軌道データ抽出
xs_h = [u[1] for u in sol_hamilton.u]
ys_h = [u[2] for u in sol_hamilton.u]
zs_h = [u[3] for u in sol_hamilton.u]
# プロット (2x2レイアウト: 3D + xy, yz, zx)
p3d_h = plot3d(xs_h, ys_h, zs_h; label="orbit", lc=:red, xlabel="x", ylabel="y", zlabel="z", title="3D")
scatter!(p3d_h, [x_M2], [0.0], [0.0]; label="Moon",  ms=4, mc=:gray)
pxy_h = plot(xs_h, ys_h; label="orbit", lc=:red, xlabel="x", ylabel="y", title="xy", aspect_ratio=:equal)
scatter!(pxy_h, [x_M2], [0.0]; label="Moon",  ms=4, mc=:gray)
pyz_h = plot(ys_h, zs_h; label="orbit", lc=:red, xlabel="y", ylabel="z", title="yz", aspect_ratio=:equal)
pzx_h = plot(zs_h, xs_h; label="orbit", lc=:red, xlabel="z", ylabel="x", title="zx", aspect_ratio=:equal)
fig_h = plot(p3d_h, pxy_h, pyz_h, pzx_h; layout=(2, 2), size=(900, 800), plot_title="Halo Orbit with Hamiltonian Equation", titlefontsize=10, titlefontcolor=:black, titlefontfamily="sans-serif", titlefontweight=:bold)
display(fig_h)
savefig(fig_h, "results/confirmation/orbit_hamilton.png")

# KS座標におけるハミルトン形式で運動方程式を記述・数値微分・計算・プロット
prob_extended_h_auto = ODEProblem{true, DifferentialEquations.SciMLBase.FullSpecialize}(ks_auto_hamilton, zeta0_1, (0.0, 8T_1), [mu, C])
sol_extended_h_auto = solve(prob_extended_h_auto, Vern9(); reltol=1e-12, abstol=1e-12)
# 軌道データ抽出
xs_ext = [fun_ks_to_physical(u, mu)[1] for u in sol_extended_h_auto.u]
ys_ext = [fun_ks_to_physical(u, mu)[2] for u in sol_extended_h_auto.u]
zs_ext = [fun_ks_to_physical(u, mu)[3] for u in sol_extended_h_auto.u]
# プロット (2x2レイアウト: 3D + xy, yz, zx)
p3d_ext = plot3d(xs_ext, ys_ext, zs_ext; label="orbit", lc=:green, xlabel="x", ylabel="y", zlabel="z", title="3D")
scatter!(p3d_ext, [x_M2], [0.0], [0.0]; label="Moon",  ms=4, mc=:gray)
pxy_ext = plot(xs_ext, ys_ext; label="orbit", lc=:green, xlabel="x", ylabel="y", title="xy", aspect_ratio=:equal)
scatter!(pxy_ext, [x_M2], [0.0]; label="Moon",  ms=4, mc=:gray)
pyz_ext = plot(ys_ext, zs_ext; label="orbit", lc=:green, xlabel="y", ylabel="z", title="yz", aspect_ratio=:equal)
pzx_ext = plot(zs_ext, xs_ext; label="orbit", lc=:green, xlabel="z", ylabel="x", title="zx", aspect_ratio=:equal)
fig_ext = plot(p3d_ext, pxy_ext, pyz_ext, pzx_ext; layout=(2, 2), size=(900, 800), plot_title="Halo Orbit with Extended Hamiltonian Equation", titlefontsize=10, titlefontcolor=:black, titlefontfamily="sans-serif", titlefontweight=:bold)
display(fig_ext)
savefig(fig_ext, "results/confirmation/orbit_extended_h_auto.png")

# KS座標におけるハミルトン形式を解析的に微分して運動方程式を記述・計算・プロット
prob_extended_h = ODEProblem{true, DifferentialEquations.SciMLBase.FullSpecialize}(ks_hamiltonian, zeta0_1, (0.0, 8T_1), [mu, C])
sol_extended_h = solve(prob_extended_h, Vern9(); reltol=1e-12, abstol=1e-12)
# 軌道データ抽出
xs_ext_ana = [fun_ks_to_physical(u, mu)[1] for u in sol_extended_h.u]
ys_ext_ana = [fun_ks_to_physical(u, mu)[2] for u in sol_extended_h.u]
zs_ext_ana = [fun_ks_to_physical(u, mu)[3] for u in sol_extended_h.u]
# プロット (2x2レイアウト: 3D + xy, yz, zx)
p3d_ext_ana = plot3d(xs_ext_ana, ys_ext_ana, zs_ext_ana; label="orbit", lc=:orange, xlabel="x", ylabel="y", zlabel="z", title="3D")
scatter!(p3d_ext_ana, [x_M2], [0.0], [0.0]; label="Moon",  ms=4, mc=:gray)
pxy_ext_ana = plot(xs_ext_ana, ys_ext_ana; label="orbit", lc=:orange, xlabel="x", ylabel="y", title="xy", aspect_ratio=:equal)
scatter!(pxy_ext_ana, [x_M2], [0.0]; label="Moon",  ms=4, mc=:gray)
pyz_ext_ana = plot(ys_ext_ana, zs_ext_ana; label="orbit", lc=:orange, xlabel="y", ylabel="z", title="yz", aspect_ratio=:equal)
pzx_ext_ana = plot(zs_ext_ana, xs_ext_ana; label="orbit", lc=:orange, xlabel="z", ylabel="x", title="zx", aspect_ratio=:equal)
fig_ext_ana = plot(p3d_ext_ana, pxy_ext_ana, pyz_ext_ana, pzx_ext_ana; layout=(2, 2), size=(900, 800), plot_title="Halo Orbit with Extended Hamiltonian Equation (Analytical)", titlefontsize=10, titlefontcolor=:black, titlefontfamily="sans-serif", titlefontweight=:bold)
display(fig_ext_ana)
savefig(fig_ext_ana, "results/confirmation/orbit_extended_h_ana.png")  