using LinearAlgebra
using ForwardDiff

function K_hamiltonian(q, p, mu, C)
    return (p[1]^2 + p[2]^2) / 8 + C*(q[1]^2 + q[2]^2)/2 + (q[1]^2 + q[2]^2) * (q[2]*p[1] - (q[1] + 1 - mu)*p[2] - (1 - mu)/sqrt((1 + q[1]^2 - q[2]^2)^2 + 4*(q[1]^2)*(q[2]^2))) - mu
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