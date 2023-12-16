using Plots
using LinearAlgebra
using Distributions
using Statistics
using DelimitedFiles
include("VehicleModels.jl")
include("utils.jl")

function mppi(x0, U_ref, x_dim, u_dim, N, sigma_control, mu, M, path, barriers, obstacles, cc, opt_model, sigma_xf, En, sigma_control_bar, Q_bar, R_bar)
    # Keep track of all control squences and their costs for MPPI 
    Sm_list = zeros(Float64, M)
    Um_list = zeros(Float64, M, N, u_dim)

    # Keep track of all trajectories (states) for plotting purposes
    Xm_list = zeros(Float64, M, N+1, x_dim)

    # set inital reference trajectory
    X_ref = zeros(Float64, N+1, x_dim)
    X_ref[1,:] = x0

    # roll out dynamics
    for k in 1:N
        X_ref[k+1,:] = BicycleModelDynamics(X_ref[k,:], U_ref[k,:])
    end

    if cc
        A, B, K = CovarianceControl(X_ref, U_ref, path, barriers, opt_model, sigma_xf, En, sigma_control_bar, Q_bar, R_bar)
    end

    for m in 1:M
        Sm = 0
        eps = rand(MvNormal(mu, sigma_control), N)
        Um = copy(U_ref)
        Xm = copy(X_ref)
        yk = zeros(Float64, x_dim)
        for k in 1:N
            if cc
                Kk = K[(k-1)*u_dim + 1:k*u_dim , (k)*x_dim + 1:(k+1)*x_dim]
                Um[k, :] += Kk * yk
                yk = A[k,:,:] * yk + B[k,:,:] * eps[:, k]
            end
            Um[k, :] += eps[:, k]
            Sm += cost(Xm[k, :], Um[k, :], eps, R, barriers, obstacles)
            Xm[k+1, :] = BicycleModelDynamics(Xm[k,:], Um[k,:])
        end
        Sm += terminal_cost(Xm, path, barriers, obstacles)
        Sm_list[m] = Sm
        Um_list[m, :, :] = Um
        Xm_list[m, :, :] = Xm
    end

    # Calculate Optimal Control
    V = optimal_control(Sm_list, Um_list, lambda, M)

    # Calculate optimal trajectory
    X_optimal = copy(X_ref)
    for k in 1:N
        X_optimal[k+1,:] = BicycleModelDynamics(X_optimal[k,:], V[k,:])
    end

    return V, X_optimal, Xm_list, Sm_list
end