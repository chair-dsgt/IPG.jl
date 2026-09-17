"""
Multiplayer quadratic game (qIPG).

Based on Dragotto and Scatamacchia (2023) description. For further details, see Section 5.4
of:
    [*G. Dragotto, R. Scatamacchia, "The Zero Regrets Algorithm: Optimizing over Pure Nash Equilibria via Integer Programming". 2023. INFORMS Journal on Computing*](https://doi.org/10.1287/ijoc.2022.0282)

# Notes
- SCIP can be replaced with other solvers, such as Gurobi or CPLEX. It is only used for the
  best-response problems. SGM solves each sampled game through NormalGames.jl, which builds
  Gurobi models internally, so a Gurobi licence is needed to run this script end to end.
"""

using IPG, SGM, SCIP
using Random, LinearAlgebra
import IPG.JuMP  # imported, not `using`, so that JuMP's macros do not shadow IPG's

"""
Generate a random qIPG game.

Based on the code repository accompanying the original paper. The original can be found in
https://github.com/gdragotto/ZeroRegretsAlgorithm

Returns the `IPGModel` together with `x`, where `x[p]` is the vector of player `p`'s
variables.
"""
function generate_random_instance(n::Int, m::Int, lower_bnd::Int, upper_bnd::Int; i_type="H")
    factor = i_type == "H" ? 0.1 : 0.01
    RQ = 5

    # Generate positive semidefinite matrix M
    M = zeros(Float64, (n*m, n*m))

    max_iter = 1000
    iter = 0
    while ~isposdef(M) && iter < max_iter
        M = rand(Float64, (n*m, n*m))
        M = (M .* 2 .- 1) .* RQ  # scaling
        M = M * M'
        iter += 1
    end
    if ~isposdef(M)
        error("Failed to generate a positive definite matrix after $max_iter iterations.")
    end

    M_max = maximum(M)
    for i in 1:n
        for j in ((i-1) * m + 1):(i * m)
            for k in (i * m + 1):(size(M, 2))
                vjk = (rand() * 2 - 1) * factor * M_max
                vjk = round(vjk, digits=1)
                M[j, k] += vjk
                M[k, j] -= vjk
            end
        end
    end

    # build the model and the players' strategy spaces
    model = IPGModel(n)
    x = Vector{Vector{JuMP.VariableRef}}(undef, n)
    for p in 1:n
        x[p] = @variable(model, p, [1:m], Int, base_name="x_$p", lower_bound=lower_bnd, upper_bound=upper_bnd)
    end

    # add payoffs
    for p in 1:n
        cp = rand(-RQ:RQ, m)

        # build payoff matrix
        Qp = Vector{Matrix{Float64}}()
        for k in 1:n
            push!(Qp, M[((p-1) * m + 1):(p * m), ((k-1) * m + 1):(k * m)])
        end

        # build payoff expression
        Πp = cp'*x[p] + 0.5 * x[p]' * Qp[p] * x[p] + sum(x[k]' * Qp[k] * x[p] for k in 1:n if k != p)

        @payoff(model, p, Max, Πp)
    end

    return model, x
end

Random.seed!(0)  # only so that the run below is reproducible
model, x = generate_random_instance(3, 3, -500, 500)

# The players' variables carry no `start` value, so SGM initializes the sampled game by
# solving one feasibility problem per player. This is what `initialize_strategies_feasibility`
# used to select; it is now the behaviour of `SGM.initialize_strategies` whenever start
# values are missing.

set_solver!(model, SGM.EquilibriumSolver(SCIP.Optimizer))

find_equilibrium!(model; max_iter=25)

# ==== Results ====
println("termination status: ", termination_status(model))

if termination_status(model) == IPG.EQUILIBRIUM_FOUND
    for p in 1:length(model.players)
        println("player $p: x = ", [value(model, x[p][i]) for i in eachindex(x[p])],
                "   payoff = ", payoff_value(model, p))
    end
end
