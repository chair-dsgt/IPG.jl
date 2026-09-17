"""
Best response of a single player.

Minimal example showing how to compute one player's best response to a fixed strategy of the
other players, without running an equilibrium search. The game is the two-player quadratic
game of Example 5.3 of Carvalho, Lodi, and Pedroso (2022); see `example_5_3.jl`.

Notes
- SCIP can be replaced with other solvers, such as Gurobi or CPLEX.
"""

using IPG, SCIP

model = IPGModel(2)

@variable(model, 1, x1, start=10.0)
@constraint(model, 1, x1 >= 0)

@variable(model, 2, x2, start=10.0)
@constraint(model, 2, x2 >= 0)

@payoff(model, 1, Max, -x1*x1 + x1*x2)
@payoff(model, 2, Max, -x2*x2 + x1*x2)

# The strategy the other players are held at. `@payoff` replaced `x2` by a parameter of
# player 1's problem, so fixing player 2's strategy is done by handing player 1 a profile of
# the *other* players rather than by substituting into the payoff expression by hand.
profile_others = Strategy[PureStrategy(2, Dict(x2 => 5.0))]

# Player 1's best response to player 2 playing x2 = 5.0. `best_response` builds the problem,
# optimizes it and returns the resulting pure strategy.
best_response_1 = best_response(model, 1, profile_others, SCIP.Optimizer)

println("x1 = ", value(best_response_1, x1))  # max of -x1^2 + 5 x1 over x1 >= 0, i.e. 2.5

# The best-response problem can also be inspected before being solved. It is a copy of
# player 1's problem, so its variables are reached through the returned reference map.
best_response_model, reference_map = get_best_response_model(model, 1, profile_others, SCIP.Optimizer)

println()
println(best_response_model)

IPG.JuMP.set_silent(best_response_model)
IPG.JuMP.optimize!(best_response_model)

println("x1 = ", IPG.JuMP.value(reference_map[x1]))
