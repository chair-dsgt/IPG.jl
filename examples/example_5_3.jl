"""
Computing an equilibrium with SGM (Example 5.3, CLP, 2022)

Implementation of Example 5.3 from Carvalho, Lodi, and Pedroso (2022). For futher details,
    see:
    [*M. Carvalho, A. Lodi, J. P. Pedroso, "Computing equilibria for integer programming games". 2022. European Journal of Operational Research*](https://www.sciencedirect.com/science/article/pii/S0377221722002727)
    [*M. Carvalho, A. Lodi, J. P. Pedroso, "Computing Nash equilibria for integer programming games". 2020. arXiv:2012.07082*](https://arxiv.org/abs/2012.07082)

Notes
- SCIP can be replaced with other solvers, such as Gurobi or CPLEX. It is only used for the
  best-response problems. SGM solves each sampled game through NormalGames.jl, which builds
  Gurobi models internally, so a Gurobi licence is needed to run this script end to end.
"""

using IPG, SGM, SCIP

# ==== Model Definition ====
# A single `IPGModel` holds one optimization problem per player. Players are addressed by
# their id here; `IPGModel(["P1", "P2"])` would let them be addressed by name instead.
model = IPGModel(2)

@variable(model, 1, x1, start=10.0)
@constraint(model, 1, x1 >= 0)

@variable(model, 2, x2, start=10.0)
@constraint(model, 2, x2 >= 0)

# Payoffs are JuMP expressions and may mix the variables of several players. Unlike
# `set_payoff!`, `@payoff` takes an explicit optimization sense.
@payoff(model, 1, Max, -x1^2 + x1*x2)
@payoff(model, 2, Max, -x2^2 + x1*x2)

# ==== SGM Algorithm Subroutines ====
# `SGM.get_player_order` is the one SGM subroutine that is a reassignable binding, so it can
# still be swapped from the outside. The options shipped by SGM.jl are
# `get_player_order_by_last_deviation` (the default, set below), `get_player_order_fixed_ascending`,
# `get_player_order_fixed_descending` and `get_player_order_random`.
SGM.get_player_order = SGM.get_player_order_by_last_deviation

# A custom subroutine can be passed as well. The following is a re-implementation of
# `SGM.get_player_order_random` for demonstration. Note the signature: the subroutine now
# receives the number of players and the past payoff improvements as `(player_id, improvement)`
# pairs, rather than the players themselves and the sampled profiles.
using Random
Random.seed!(0)  # only so that the run below is reproducible
function get_player_random_order(num_players::Int, payoff_improvements::Vector{Tuple{Int,Float64}})::Vector{Int}
    return shuffle(collect(1:num_players))
end
SGM.get_player_order = get_player_random_order

# The initialization of the sampled game, the sampled-game solve and the deviation test are
# no longer swappable by assignment: `SGM.initialize_strategies`, `SGM.solve` and
# `SGM.find_deviation` are ordinary functions of SGM.jl. `SGM.solve` dispatches on the type
# of the sampled game and calls `solve_PNS` for polymatrix games, and `find_deviation` is
# the best-response test that used to be `find_deviation_best_response`. Strategies are
# initialized from the `start` values above when every variable has one, and by solving a
# feasibility problem otherwise.

# ==== SGM ====
# The equilibrium solver can also be passed to the constructor, as `IPGModel(2, solver)`.
set_solver!(model, SGM.EquilibriumSolver(SCIP.Optimizer))

find_equilibrium!(model; max_iter=100, dev_tol=1e-3)

# ==== Results ====
println("termination status: ", termination_status(model))

if termination_status(model) == IPG.EQUILIBRIUM_FOUND
    println("x1 = ", value(model, x1), "   payoff 1 = ", payoff_value(model, 1))
    println("x2 = ", value(model, x2), "   payoff 2 = ", payoff_value(model, 2))
    println()
    println(equilibrium(model))
end
