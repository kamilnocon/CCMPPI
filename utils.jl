using LinearAlgebra
using JuMP
using SCS
using Mosek
using ForwardDiff
using QuadGK
using DelimitedFiles

function collision(x, barrier)
    if x[1] > barrier[1,1] && x[1] < barrier[2,1] && x[2] > barrier[1,2] && x[2] < barrier[2,2]
        return 1
    else
        return 0
    end
end

function cost(x, u, eps, R, barriers, obstacles)
    state_cost = 0

    # penalize outside of the bounds of the map
    if (x[1] < 0 || x[2] < 0 || x[1] > 13 || x[2] > 16)
        state_cost += 2000
    end

    # penalize being inside of barriers
    for barrier in barriers
        state_cost += 2000 * collision(x, barrier)
    end

    # penalize obstacles
    for obstacle in obstacles
        dist = sqrt((obstacle[1] - x[1])^2 + (obstacle[2] - x[2])^2) 
        if dist < obstacle_radius
            state_cost += 10
        end
    end

    # sampling cost for MPPI algorithm
    sampling_cost = (((1-nu^-1)/2)*transpose(eps)*R*eps)[1]+(transpose(u)*R*eps)[1] + (1/2)*(transpose(u)*R*u)[1]
    cost = state_cost + sampling_cost
    return cost
end

function closestPoint(x, path)
    # find the closest point to the track centerline for the start and end points in
    # the trajectory
    # return the which point along the path (what number) and the distance away from that point

    # can this be optimized using min norm solution, pseudo-inverse ???
    min_dist = sqrt((x[1] - path[1,1])^2 + (x[2] - path[1,2])^2)
    min_point = 1
    for i in 1:size(path)[1]
        dist = sqrt((x[1] - path[i,1])^2 + (x[2] - path[i,2])^2)
        if dist < min_dist
            min_dist = dist
            min_point = i
        end
    end
    return min_point, min_dist
end

function terminal_cost(X, path, barriers, obstacles)
    p1, d1 = closestPoint(X[1,:], path)
    p2, d2 = closestPoint(X[N+1,:], path)
    
    # s is the distance between the current vehicle state 
    # and the terminal state of the sample trajectory 
    # along the track centerline
    
    # for the wrap around case
    if p1 > size(path)[1] * 3/4 && p2 < size(path)[1] /4
        p2 += size(path)[1]
    end

    # normalize progress along track
    s = (p2 - p1)/size(path)[1]

    # if it's going backwards, make progress = 0 (trying to avoid negative when calculating the cost)
    if s < 0
        s = 0
    end

    # e is the lateral deviation from the track centerline
    e = d2

    term_cost = 2*(1/(s+0.000000001)) + e^2

    # penalize being inside of barriers
    for barrier in barriers
        term_cost += 2000 * collision(X[N+1,:], barrier)
    end
    # penalize outside of the bounds of the map
    if (X[N+1,1] < 0 || X[N+1,2] < 0 || X[N+1,1] > 13 || X[N+1,1] > 16)
        term_cost += 2000
    end

    # penalize obstacles
    for obstacle in obstacles
        dist = sqrt((obstacle[1] - X[N+1, 1])^2 + (obstacle[2] - X[N+1, 2])^2) 
        if dist < obstacle_radius
            term_cost += 10
        end
    end

    return term_cost
end

function state_cost(x, barriers)
    state_cost = 0

    # penalize being inside of barriers
    for barrier in barriers
        state_cost += 2000 * collision(x, barrier)
    end

    # penalize obstacles
    for obstacle in obstacles
        dist = sqrt((obstacle[1] - x[1])^2 + (obstacle[2] - x[2])^2) 
        if dist < obstacle_radius
            state_cost += 200
        end
    end

    return state_cost
end

function optimal_control(Sm_list, Um_list, lambda, M)
    b = minimum(Sm_list)
    V = zero(Um_list[1,:,:])
    sum = 0
    for m in 1:M
        wm = exp((-1/lambda)*(Sm_list[m]-b))
        V += wm.*Um_list[m, :, :]
        sum += wm
    end
    V = V ./ sum
    return V
end

function transfrom_K_var(K_var)
    if eltype(K_var) == VariableRef
        K = zeros(AffExpr, N*u_dim, (N+1)*x_dim)
    elseif eltype(K_var) == Float64
        K = zeros(Float64, N*u_dim, (N+1)*x_dim)
    end
    for i in 1:N
        for j in 1:N+1
            if (i+1 == j)
                K[(i-1)*u_dim+1:i*u_dim, (j-1)*x_dim+1:j*x_dim] = K_var[:,:,i]
            end
        end
    end
    return K
end

function CovarianceControl(x_ref, u_ref, path, barriers, opt_model, sigma_xf, En, sigma_control_bar, Q_bar, R_bar)
    A = zeros(Float64, (N, x_dim, x_dim))
    B = zeros(Float64, (N, x_dim, u_dim))

    # println("linearizing dyamics")
    for k in 1:N
        local Ak = ForwardDiff.jacobian((x) -> BicycleModelDynamics(x, u_ref[k,:]), x_ref[k,:])
        local Bk = ForwardDiff.jacobian((u) -> BicycleModelDynamics(x_ref[k,:], u), u_ref[k,:])
        A[k,:,:] = Ak
        B[k,:,:] = Bk
    end

    beta = zeros(Float64, ((N+1)*x_dim, (N)*u_dim))

    for k in 1:N
        for j in 1:k
            local Bk1k0 = B[j,:,:]
            for z in j+1:k
                Bk1k0 = A[z,:,:] * Bk1k0
            end
            beta[k*x_dim+1:(k+1)*x_dim, (j-1)*u_dim+1:(j)*u_dim] = Bk1k0
        end
    end
    # println("setting constraint")
    K = transfrom_K_var(K_var)
    @constraint(opt_model, con, [sigma_xf En*(I+beta*K)*beta*sqrt(sigma_control_bar); sqrt(sigma_control_bar)*beta'*(I+beta*K)'*En' I] >= 0, PSDCone())
    # println("computing obj")
    # flush(stdout)
    obj = tr(((I + beta*K)' * Q_bar * (I + beta*K) + K'*R_bar*K)*beta*sigma_control_bar*beta')
    @objective(opt_model, Min, obj)
    # println("optimizing")
    optimize!(opt_model)
    K_soln = transfrom_K_var(value.(K_var))
    delete(opt_model, con)
    unregister(opt_model, :con)
    # println("done")
    return A, B, K_soln
end