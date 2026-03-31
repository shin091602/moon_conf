using LinearAlgebra
using ForwardDiff

function K_hamiltonian(u, w, mu, C)
    T = (1/8)*(w[1]^2 + w[2]^2 + w[3]^2 + w[4]^2)
    V1 = (1/2)*(-1 + mu - u[1]^2 + u[2]^2 + u[3]^2 - u[4]^2) * (u[2]*w[1] + u[1]*w[2] - u[4]*w[3] - u[3]*w[4])
    V2 = (u[1]*u[2]-u[3]*u[4]) * (u[1]*w[1] - u[2]*w[2] - u[3]*w[3] + u[4]*w[4])
    V3 = -(1 - mu)*(u[1]^2 + u[2]^2 + u[3]^2 + u[4]^2)/sqrt((u[1]^2 - u[2]^2 - u[3]^2 + u[4]^2 + 1)^2 + (2u[1]*u[3] + 2u[2]*u[4])^2 + (2u[1]*u[2] - 2u[3]*u[4])^2)
    E = -mu + (1/2) * C * (u[1]^2 + u[2]^2 + u[3]^2 + u[4]^2)
    K = T + V1 + V2 + V3 + E
    return K
end


function ks_auto_hamilton(dzeta, zeta, params, t)
    mu = params[1]
    C = params[2]
    u = zeta[1:4]
    w = zeta[5:8]
    dK_du = ForwardDiff.gradient(u -> K_hamiltonian(u, w, mu, C), u)
    dK_dw = ForwardDiff.gradient(w -> K_hamiltonian(u, w, mu, C), w)
    dzeta[1:4] = dK_dw
    dzeta[5:8] = -dK_du
end