using LinearAlgebra

function LC_eom(du, u, params, t)
    mu = params[1]
    C  = params[2]
    
    # r_E
    r_E = sqrt((1 + u[1]^2 - u[2]^2)^2 + 4*(u[1]^2)*(u[2]^2))

    # 共通項
    r2 = u[1]^2 + u[2]^2

    # x̄'
    du[1] = u[3]/4 + r2*u[2]/2
    du[2] = u[4]/4 - r2*u[1]/2

    # p̄'
    du[3] = -C*u[1] 
            - u[3]*u[1]*u[2] 
            + u[4]*(3*u[1]^2 + u[2]^2)/2 
            + 2*u[1]*(1-mu)/r_E 
            - 2*u[1]*r2*(1 + r2)*(1-mu)/r_E^3

    du[4] = -C*u[2] 
            - u[3]*(u[1]^2 + 3*u[2]^2)/2 
            + u[4]*u[1]*u[2] 
            + 2*u[2]*(1-mu)/r_E 
            + 2*u[2]*r2*(1 - r2)*(1-mu)/r_E^3
end