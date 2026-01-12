using LinearAlgebra
"""
u = (u, U)
Uは正準協約運動量
(u,U) → (X,p)
Xは月中心座標系での位置ベクトル
pはXに対する運動量ベクトル
mu = 月の質量比
C = ヤコビ定数
正準方程式を記述する
"""

function ks_eom_hamilton(du, u, p, t)
    # パラメータの展開
    mu = p[1]
    C = p[2]
    Au = hcat(
        [u[1], u[2], u[3]],
        [-u[2], u[1], u[4]],
        [-u[3], -u[4], u[1]],
        [u[4], -u[3], u[2]],
    )
    AuT = transpose(Au)
    
    X = Au * vcat(u[1], u[2], u[3], u[4])
    r_M = sqrt(X[1]^2 + X[2]^2 + X[3]^2)
    r_E = sqrt((X[1] + 1)^2 + X[2]^2 + X[3]^2)
    p = (1/r_M) * Au * vcat(u[5], u[6], u[7], u[8])

    dK_U = 0.5 * AuT * vcat(
        p[1] + X[2],
        p[2] - X[1] - 1 + mu,
        p[3],
    )
    
   H = 0.5*(dot(p,p)) + p[1]*X[2] - p[2]*(X[1] + 1 - mu) - (1 - mu)/r_E - mu/r_M

   dH_X = vcat(
        -p[2] + (1 - mu)*(X[1] + 1)/r_E^3 + mu*X[1]/r_M^3,
        p[1] + (1 - mu)*X[2]/r_E^3 + mu*X[2]/r_M^3,
        (1 - mu)*X[3]/r_E^3 + mu*X[3]/r_M^3,
    )
    
    dX_u = hcat(
        [2*u[1], 2*u[2], 2*u[3]],
        [-2*u[2], 2*u[1], -2*u[4]],
        [-2*u[3], -2*u[4], 2*u[1]],
        [2*u[4], -2*u[3], -2*u[2]],
    )

    # rM_u = transpose(dX_u) * vcat(X[1]/r_M, X[2]/r_M, X[3]/r_M)
    dAuU_u = hcat(
        [u[5], u[6], u[7]],
        [-u[6], u[5], u[8]],
        [-u[7], -u[8], u[5]],
        [u[8], -u[7], u[6]],
    )

    # AuU = Au * vcat(u[5], u[6], u[7], u[8])
    # ut = vcat(u[1], u[2], u[3], u[4])
    # dp_u = (-1/(r_M^2)) * AuU * ut + (1/(2*r_M)) * dAuU_u 

    U   = u[5:8]          # 共役運動量
    q   = u[1:4]          # KS 位置
    AuU = Au * U          # 3×1 = A(u)U

    dp_u = (-1/r_M^2) * (AuU * q') + (1/(2*r_M)) * dAuU_u
#                          ↑ ここで transpose して 1×4 にする


    dK_u = 2 * sqrt(r_M) * (H + C/2) + r_M * (
        transpose(dX_u) * dH_X +
        transpose(dp_u) * vcat(p[1] + X[2], p[2] - X[1] - 1 + mu, p[3])
    )

    du[1] = dK_U[1]
    du[2] = dK_U[2]
    du[3] = dK_U[3]
    du[4] = dK_U[4]
    du[5] = -dK_u[1]
    du[6] = -dK_u[2]
    du[7] = -dK_u[3]
    du[8] = -dK_u[4]

end
