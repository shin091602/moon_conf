function cr3bp_eom_newton(dx, x, mu, t)

    r1 = sqrt((mu + x[1])^2 + x[2]^2 + x[3]^2)
    r2 = sqrt((x[1] - 1 + mu)^2 + x[2]^2 + x[3]^2)

    dx[1] = x[4]
    dx[2] = x[5]
    dx[3] = x[6]
    dx[4] =  2*x[5] +x[1] -(1-mu)*(x[1]+mu)/r1^3 -mu*(x[1]-1+mu)/r2^3
    dx[5] = -2*x[4] +x[2] -(1-mu)*x[2]/r1^3      -mu*x[2]/r2^3
    dx[6] =               -(1-mu)*x[3]/r1^3      -mu*x[3]/r2^3
end