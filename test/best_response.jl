# Covers: get_best_response_model and best_response (src/best_response.jl),
# both the pure-profile branch and the mixed-profile expected-payoff branch
# (src/best_response.jl:31-55).
#
# Lives in IPG.jl: a best response is a single-player optimization problem built
# from the model, with no reference to any equilibrium algorithm. Assertions are
# against closed-form optima rather than against a solver round-trip.
#
# Solver note. HiGHS is used to keep the modeling layer free of a commercial
# dependency, and that constrains which problems can be solved here:
#
#   * The pure-profile branch sets parameter values but leaves the parameters in
#     the objective, so a payoff that is bilinear in own x opponent variables
#     stays a ScalarQuadraticFunction. HiGHS rejects a quadratic objective whose
#     quadratic terms involve parameters, even when fixing them would make the
#     problem linear. Rock-paper-scissors is exactly that shape, so its
#     pure-profile solve is exercised in SGM.jl under Gurobi instead.
#   * The mixed-profile branch calls replace_param_with_val, which substitutes
#     the values out, so the objective HiGHS receives is parameter-free. That
#     branch is solvable here for every fixture.
#
# See "Objective shape differs between the pure and mixed branches" below, which
# pins the asymmetry down without needing a solver at all.

# Covers: get_best_response_model (src/best_response.jl:14) - that it copies rather
# than mutates, and that profile values reach the copy's parameters.
@testitem "get_best_response_model copies the player problem" setup = [IPGFixtures, HiGHSOptimizer] begin
    game = get_example_two_player_game()
    x2 = only(IPG.get_player_variables(game.players[2]))

    profile = IPG.PureStrategy[IPG.PureStrategy(2, Dict(x2 => 4.0))]

    best_response_model, reference_map = IPG.get_best_response_model(game, 1, profile, highs_optimizer)

    @test best_response_model !== game.players[1]
    @test JuMP.solver_name(best_response_model) == "HiGHS"

    # The opponent's value landed on the copy's parameter ...
    copied_parameter = reference_map[game.var_to_param_maps[1][x2]]
    @test JuMP.parameter_value(copied_parameter) == 4.0

    # ... and the original model was left untouched.
    @test JuMP.parameter_value(game.var_to_param_maps[1][x2]) == 0.0
    @test JuMP.solver_name(game.players[1]) == "No optimizer attached."

    # The copy carries the player's own variable and constraint.
    @test length(IPG.get_player_variables(best_response_model)) == 1
    @test JuMP.num_constraints(best_response_model; count_variable_in_set_constraints = false) == 1
    @test JuMP.objective_sense(best_response_model) == JuMP.MAX_SENSE
end

# Pins down the pure/mixed asymmetry described in the solver note above. No
# solve, so this holds regardless of which optimizer is available.
@testitem "Objective shape differs between the pure and mixed branches" setup = [IPGFixtures] begin
    game = get_rock_paper_scissors_game()
    r2, p2, s2 = IPG.get_player_variables(game.players[2])

    rock = IPG.PureStrategy(2, Dict(r2 => 1.0, p2 => 0.0, s2 => 0.0))

    pure_model, _ = IPG.get_best_response_model(game, 1, IPG.PureStrategy[rock])
    pure_objective = JuMP.objective_function(pure_model)

    # The pure branch keeps the parameters, so the objective stays quadratic.
    @test pure_objective isa JuMP.QuadExpr
    @test !isempty(pure_objective.terms)
    @test any(JuMP.is_parameter, JuMP.all_variables(pure_model))

    # The mixed branch substitutes parameter values out. Against a degenerate
    # mixed strategy placing all mass on rock, player 1's payoff collapses to
    # p1 - s1: paper beats rock, scissors loses to it.
    mixed_rock = IPG.MixedStrategy(2, [1.0], [rock])
    mixed_model, reference_map = IPG.get_best_response_model(game, 1, IPG.Strategy[mixed_rock])
    mixed_objective = JuMP.objective_function(mixed_model)

    @test isempty(mixed_objective.terms) # no surviving quadratic terms
    r1, p1, s1 = IPG.get_player_variables(game.players[1])
    @test JuMP.coefficient(mixed_objective, reference_map[p1]) ≈ 1.0
    @test JuMP.coefficient(mixed_objective, reference_map[s1]) ≈ -1.0
    @test JuMP.coefficient(mixed_objective, reference_map[r1]) ≈ 0.0
end

# Covers: best_response on pure profiles (src/best_response.jl:77), checked against
# the analytic best-response map and its fixed point.
@testitem "best_response matches the closed-form optimum" setup = [IPGFixtures, HiGHSOptimizer] begin
    game = get_example_two_player_game()
    x1 = only(IPG.get_player_variables(game.players[1]))
    x2 = only(IPG.get_player_variables(game.players[2]))

    # Player 1 maximizes -x1^2 + x1 * x2 over x1 >= 0, so the best response to
    # x2 is x2 / 2, clipped at 0.
    for x2_value in [0.0, 4.0, 10.0]
        profile = IPG.PureStrategy[IPG.PureStrategy(2, Dict(x2 => x2_value))]
        response = IPG.best_response(game, 1, profile, highs_optimizer)

        @test response isa IPG.PureStrategy
        @test response.id == 1
        @test Set(keys(response.variable_values)) == Set([x1])
        @test IPG.value(response, x1) ≈ x2_value / 2 atol = 1e-5
    end

    # The equilibrium (0, 0) is a fixed point of the best-response map: no
    # player can improve by deviating from it.
    equilibrium_profile = IPG.PureStrategy[
        IPG.PureStrategy(1, Dict(x1 => 0.0)),
        IPG.PureStrategy(2, Dict(x2 => 0.0)),
    ]
    for player_id in 1:2
        others = IPG.Strategy[s for s in equilibrium_profile if s.id != player_id]
        response = IPG.best_response(game, player_id, others, highs_optimizer)
        deviating = IPG.Strategy[s.id == player_id ? response : s for s in equilibrium_profile]

        @test IPG.payoff(game, player_id, deviating) ≈ IPG.payoff(game, player_id, equilibrium_profile) atol = 1e-5
    end
end

# Covers: the mixed-profile branch (src/best_response.jl:31-55) - that it maximizes
# the expectation over the opponent's support.
@testitem "best_response against a mixed profile uses the expected payoff" setup = [IPGFixtures, HiGHSOptimizer] begin
    game = get_example_two_player_game()
    x1 = only(IPG.get_player_variables(game.players[1]))
    x2 = only(IPG.get_player_variables(game.players[2]))

    # Player 2 mixes uniformly over x2 = 0 and x2 = 4, so player 1 maximizes
    # -x1^2 + x1 * E[x2] = -x1^2 + 2 * x1 and should respond with x1 = 1.
    mixed = IPG.MixedStrategy(
        2,
        [0.5, 0.5],
        [IPG.PureStrategy(2, Dict(x2 => 0.0)), IPG.PureStrategy(2, Dict(x2 => 4.0))],
    )

    response = IPG.best_response(game, 1, IPG.Strategy[mixed], highs_optimizer)

    @test IPG.value(response, x1) ≈ 1.0 atol = 1e-5
end

# Covers: best_response over binary variables, where the substituted objective is a
# MILP. Uses degenerate mixed strategies; see the solver note above.
@testitem "best_response on a binary game" setup = [IPGFixtures, HiGHSOptimizer] begin
    game = get_rock_paper_scissors_game()
    r1, p1, s1 = IPG.get_player_variables(game.players[1])
    r2, p2, s2 = IPG.get_player_variables(game.players[2])

    pure(r, p, s) = IPG.PureStrategy(2, Dict(r2 => r, p2 => p, s2 => s))
    # Degenerate mixed strategies, to reach the branch HiGHS can solve. The
    # pure-profile equivalents are covered in SGM.jl under Gurobi.
    certainly(strategy) = IPG.Strategy[IPG.MixedStrategy(2, [1.0], [strategy])]

    # Paper beats rock, scissors beats paper, rock beats scissors.
    response = IPG.best_response(game, 1, certainly(pure(1.0, 0.0, 0.0)), highs_optimizer)
    @test (IPG.value(response, r1), IPG.value(response, p1), IPG.value(response, s1)) == (0.0, 1.0, 0.0)

    response = IPG.best_response(game, 1, certainly(pure(0.0, 1.0, 0.0)), highs_optimizer)
    @test (IPG.value(response, r1), IPG.value(response, p1), IPG.value(response, s1)) == (0.0, 0.0, 1.0)

    response = IPG.best_response(game, 1, certainly(pure(0.0, 0.0, 1.0)), highs_optimizer)
    @test (IPG.value(response, r1), IPG.value(response, p1), IPG.value(response, s1)) == (1.0, 0.0, 0.0)

    # Against the uniform mixed strategy every pure action ties, so any feasible
    # response is optimal and its expected payoff is zero.
    uniform = IPG.MixedStrategy(
        2,
        [1/3, 1/3, 1/3],
        [pure(1.0, 0.0, 0.0), pure(0.0, 1.0, 0.0), pure(0.0, 0.0, 1.0)],
    )
    response = IPG.best_response(game, 1, IPG.Strategy[uniform], highs_optimizer)

    @test sum(IPG.value(response, v) for v in (r1, p1, s1)) ≈ 1.0 atol = 1e-5
    @test IPG.payoff(game, 1, IPG.Strategy[response, uniform]) ≈ 0.0 atol = 1e-5
end

# Covers: the Union{Int, String} player argument of best_response.
@testitem "best_response accepts a player name" setup = [IPGFixtures, HiGHSOptimizer] begin
    game = get_example_two_player_game_with_names()
    x1 = only(IPG.get_player_variables(game.players[1]))
    x2 = only(IPG.get_player_variables(game.players[2]))

    profile = IPG.PureStrategy[IPG.PureStrategy(2, Dict(x2 => 4.0))]

    by_name = IPG.best_response(game, "Player1", profile, highs_optimizer)
    by_id = IPG.best_response(game, 1, profile, highs_optimizer)

    @test by_name.id == by_id.id == 1
    @test IPG.value(by_name, x1) ≈ IPG.value(by_id, x1) atol = 1e-5
    @test_throws ArgumentError IPG.best_response(game, "Nobody", profile, highs_optimizer)
end
