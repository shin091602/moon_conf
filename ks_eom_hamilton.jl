using LinearAlgebra

"""
u = [q; U] (長さ8ベクトル)
  q = KS座標 (u1..u4)
  U = 共役運動量 (U1..U4)

params = [mu, C]
  mu = 月の質量比
  C  = Jacobi定数

(X, p_vec) は Moon-centered rotating frame:
  X = 月中心位置ベクトル (3次元)
  p_vec = Xに共役な運動量 (3次元)

正則化ハミルトニアン:
  K(u,U;C) = r_M(u) * (H(X,p) + C/2)

正準方程式:
  q' = ∂K/∂U
  U' = -∂K/∂q
"""
function ks_eom_hamilton(du, u, params, t)
    # パラメータ
    mu = params[1]
    C  = params[2]

    # KS 位置・運動量
    q = u[1:4]   # (u1..u4)
    U = u[5:8]   # (U1..U4)

    # KS 行列 A(u) : 3×4
    Au = hcat(
        [u[1], -u[2], -u[3]],
        [u[2],  u[1],  u[4]],
        [u[3], -u[4],  u[1]],
        [u[4], -u[3],  u[2]],
    )
    AuT = transpose(Au)   # 4×3

    # 月中心位置 X = A(u) q
    X = Au * q
    r_M = sqrt(X[1]^2 + X[2]^2 + X[3]^2)
    r_E = sqrt((X[1] + 1)^2 + X[2]^2 + X[3]^2)

    # 月中心判定用の閾値
    epsilon = 1e-10

    if r_M < epsilon
        # ====================================
        # 月中心付近の運動方程式
        du[1] = 0.25 * U[1]
        du[2] = 0.25 * U[2]
        du[3] = 0.25 * U[3]
        du[4] = 0.25 * U[4]
        du[5:8] .= 0.0
        return nothing
    end

    # 運動量 p = (1/(2 r_M)) A(u) U  (正準KS)
    p_vec = (1/(2*r_M)) * Au * U

    # dK/dU = (1/2) A(u)^T * dH/dp
    g_p = vcat(
        p_vec[1] + X[2],
        p_vec[2] - X[1] - 1 + mu,
        p_vec[3],
    )
    dK_U = 0.5 * AuT * g_p   # 4×1

    # Hamiltonian H(X,p)
    H = 0.5 * dot(p_vec, p_vec) +
        p_vec[1]*X[2] - p_vec[2]*(X[1] + 1 - mu) -
        (1 - mu)/r_E - mu/r_M

    # dH/dX = 3×1
    dH_X = vcat(
        -p_vec[2] + (1 - mu)*(X[1] + 1)/r_E^3 + mu*X[1]/r_M^3,
         p_vec[1] + (1 - mu)*X[2]/r_E^3 + mu*X[2]/r_M^3,
        (1 - mu)*X[3]/r_E^3 + mu*X[3]/r_M^3,
    )

    # dX/du : 3×4
    dX_u = hcat(
        [2*u[1],  2*u[2], 2*u[3]],
        [-2*u[2], 2*u[1], 2*u[4]],
        [-2*u[3], -2*u[4],2*u[1]],
        [2*u[4], -2*u[3], -2*u[2]],
    )

    # d(Au*U)/du : 3×4
    dAuU_u = hcat(
        [U[1],  U[2],  U[3]],
        [-U[2], U[1],  U[4]],
        [-U[3], -U[4], U[1]],
        [U[4], -U[3],  U[2]],
    )

    # dp/du : 3×4
    AuU = Au * U             # 3×1
    dp_u = (-1/r_M^2) * (AuU * q') + (1/(2*r_M)) * dAuU_u

    # dK/du = 2u(H+C/2) + r_M (X_u^T dH_X + p_u^T dH_dp)
    dK_u = 2 .* q .* (H + C/2) .+
           r_M * (
               transpose(dX_u) * dH_X +
               transpose(dp_u) * g_p
           )

    # 正準方程式
    du[1:4] .= dK_U
    du[5:8] .= -dK_u
    return nothing
end
