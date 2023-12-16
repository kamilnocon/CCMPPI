function BicycleModelDynamics(x, u)
    L_f = 0.1
    L_r = 0.1
    beta = atan(L_r / (L_r + L_f) * tan(x[5]))
    
    dx = [x[4] * cos(x[3] + beta),
            x[4] * sin(x[3] + beta),
            x[4] / L_r * sin(beta),
            u[1],
            u[2]
    ]
    return x + dx*dt
end

function BicycleModelDynamics_direct_steer(x, u)
    L_f = 0.1
    L_r = 0.1
    beta = atan(L_r / (L_r + L_f) * tan(u[2]))
    
    dx = [x[4] * cos(u[2] + beta),
            x[4] * sin(u[2] + beta),
            x[4] / L_r * sin(beta),
            u[1]
    ]
    return x + dx*dt
end