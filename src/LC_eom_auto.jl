using LinearAlgebra
using ForwardDiff

function K_hamiltonian(q, p, mu, C)
    x1, x2 = q[1], q[2]
    p1, p2 = p[1], p[2]

    return (1/8) * (p1^2 + p2^2) +
           (C/2) * (x1^2 + x2^2) +
           x1*x2 * (x1*p1 - x2*p2) -
           (1/2) * (x1^2 - x2^2 + 1 - mu) * (x2*p1 + x1*p2) -
           (x1^2 + x2^2) * (1 - mu) / sqrt((1 + x1^2 - x2^2)^2 + (2*x1*x2)^2) - mu
end

function LC_eom_auto(du, u, params, t)
    # パラメータ
    mu = params[1]
    C  = params[2]

    # KS 位置・運動量
    q = u[1:2]
    p = u[3:4]

    # 自動微分で偏微分を計算
    dK_dP = ForwardDiff.gradient(p_var -> K_hamiltonian(q, p_var, mu, C), p)
    dK_dQ = ForwardDiff.gradient(q_var -> K_hamiltonian(q_var, p, mu, C), q)
    # 正準方程式
    du[1:2] = dK_dP      # dQ/dt = ∂K/∂P
    du[3:4] = -dK_dQ     # dP/dt = -∂K/∂Q

    return nothing
end