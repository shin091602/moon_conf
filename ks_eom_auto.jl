using LinearAlgebra
using ForwardDiff

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
  Q' = ∂K/∂P
  P' = -∂K/∂Q
"""

"""
正則化ハミルトニアン K(Q, P; mu, C) をスカラーとして計算
自動微分のため純粋関数として定義

KS変換の性質 |A(Q)*P|² = |Q|² * |P|² を利用して、
r_M = 0 でも0除算が発生しない形で記述。

K = r_M * (H + C/2) を展開すると：
K = |P|²/8 + r_M * (q_x[2]*p_x[1] - (q_x[1] + 1 - mu)*p_x[2] - (1 -mu)/r_E + C/2) - mu
"""
function K_hamiltonian(Q, P, mu, C)
    # KS 行列 A(Q) : 3×4
    A = hcat(
        [Q[1],  Q[2],  Q[3]],
        [-Q[2], Q[1],  Q[4]],
        [-Q[3], -Q[4], Q[1]],
        [Q[4],  -Q[3], Q[2]],
    )

    # 月中心位置 q_x = A(Q) * Q
    q_x = A * Q

    # r_M = |Q|² （0除算を避けるため |q| ではなく |Q|² を使用）
    r_M = Q[1]^2 + Q[2]^2 + Q[3]^2 + Q[4]^2

    # r_E = |q_x + [1,0,0]|
    r_E = sqrt((q_x[1] + 1)^2 + q_x[2]^2 + q_x[3]^2)

    # p_x = A(Q) * P （3次元ベクトル）
    p_x = A * P

    # 正則化ハミルトニアン（r_M での除算なし）
    # K = r_M * (H + C/2) を展開して、r_M での除算を避ける形に変形
    # K = (P[1]^2 + P[2]^2 + P[3]^2 + P[4]^2) / 8 + r_M * (q_x[2]*p_x[1] - (q_x[1] + 1 - mu)*p_x[2] - (1 -mu)/r_E + C/2) - mu

    K = (P[1]^2 + P[2]^2 + P[3]^2 + P[4]^2) / 2 + C*r_M/8 + (r_M/4) * (q_x[2]*p_x[1] - (q_x[1] + 1 - mu)*p_x[2] - (1 -mu)/r_E ) - mu/4

    return K
end

function ks_eom_auto(du, u, params, t)
    # パラメータ
    mu = params[1]
    C  = params[2]

    # KS 位置・運動量
    Q = u[1:4]
    P = u[5:8]

    # 自動微分で偏微分を計算
    dK_dP = ForwardDiff.gradient(p -> K_hamiltonian(Q, p, mu, C), P)
    dK_dQ = ForwardDiff.gradient(q -> K_hamiltonian(q, P, mu, C), Q)

    # 正準方程式
    du[1:4] = dK_dP      # dQ/dt = ∂K/∂P
    du[5:8] = -dK_dQ     # dP/dt = -∂K/∂Q

    return nothing
end