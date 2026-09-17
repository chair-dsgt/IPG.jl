# Covers: get_best_response_model and best_response (src/best_response.jl),
# both the pure-profile branch and the mixed-profile expected-payoff branch
# (src/best_response.jl:31-55).
#
# Lives in IPG.jl: a best response is a single-player optimization problem built
# from the model, with no reference to any equilibrium algorithm. Assertions are
# against closed-form optima rather than against a solver round-trip.

# Covers: get_best_response_model (src/best_response.jl:14) - that it copies rather
# than mutates, and that profile values reach the copy's parameters.
@testitem "get_best_response_model copies the player problem" setup = [IPGFixtures, SCIPOptimizer] begin
    game = get_example_two_player_game()
    x2 = only(IPG.get_player_variables(game.players[2]))

    profile = IPG.PureStrategy[IPG.PureStrategy(2, Dict(x2 => 4.0))]

    best_response_model, reference_map = IPG.get_best_response_model(game, 1, profile, scip_optimizer)

    @test best_response_model !== game.players[1]
    @test JuMP.solver_name(best_response_model) == "SCIP"

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

# Covers: the pure/mixed asymmetry described in the TODO above, at both the model
# level (which objective gets built) and the solver level (what a solver without
# non-convex quadratic support can do with it).
@testitem "Pure and mixed branches disagree on solver portability" setup = [IPGFixtures, HiGHSOptimizer] begin
    game = get_rock_paper_scissors_game()
    r1, p1, s1 = IPG.get_player_variables(game.players[1])
    r2, p2, s2 = IPG.get_player_variables(game.players[2])

    rock = IPG.PureStrategy(2, Dict(r2 => 1.0, p2 => 0.0, s2 => 0.0))
    # The same opponent strategy, expressed so that the mixed branch is taken.
    mixed_rock = IPG.MixedStrategy(2, [1.0], [rock])

    # Model level: the pure branch keeps the parameters, so the objective stays
    # quadratic; the mixed branch substitutes them out and it collapses to the
    # linear p1 - s1 (paper beats rock, scissors loses to it).
    pure_model, _ = IPG.get_best_response_model(game, 1, IPG.PureStrategy[rock])
    mixed_model, reference_map = IPG.get_best_response_model(game, 1, IPG.Strategy[mixed_rock])

    @test !isempty(JuMP.objective_function(pure_model).terms)
    @test any(JuMP.is_parameter, JuMP.all_variables(pure_model))

    mixed_objective = JuMP.objective_function(mixed_model)
    @test isempty(mixed_objective.terms)
    @test JuMP.coefficient(mixed_objective, reference_map[p1]) ≈ 1.0
    @test JuMP.coefficient(mixed_objective, reference_map[s1]) ≈ -1.0
    @test JuMP.coefficient(mixed_objective, reference_map[r1]) ≈ 0.0

    # Solver level: HiGHS solves the substituted model and answers paper ...
    response = IPG.best_response(game, 1, IPG.Strategy[mixed_rock], highs_optimizer)
    @test (IPG.value(response, r1), IPG.value(response, p1), IPG.value(response, s1)) == (0.0, 1.0, 0.0)

    # ... but cannot solve the identical problem posed through the pure branch,
    # because it arrives as a non-convex MIQP.
    #
    # This assertion is EXPECTED TO FAIL until the TODO in src/best_response.jl
    # is addressed. It is deliberately not marked @test_broken: a failing test is
    # the reminder that the pure branch still skips parameter substitution. Once
    # the branch substitutes, this passes and the whole test item can go.
    @test IPG.value(
        IPG.best_response(game, 1, IPG.PureStrategy[rock], highs_optimizer),
        p1,
    ) == 1.0
end

# Covers: best_response on pure profiles (src/best_response.jl:77), checked against
# the analytic best-response map and its fixed point.
@testitem "best_response matches the closed-form optimum" setup = [IPGFixtures, SCIPOptimizer] begin
    game = get_example_two_player_game()
    x1 = only(IPG.get_player_variables(game.players[1]))
    x2 = only(IPG.get_player_variables(game.players[2]))

    # Player 1 maximizes -x1^2 + x1 * x2 over x1 >= 0, so the best response to
    # x2 is x2 / 2, clipped at 0.
    for x2_value in [0.0, 4.0, 10.0]
        profile = IPG.PureStrategy[IPG.PureStrategy(2, Dict(x2 => x2_value))]
        response = IPG.best_response(game, 1, profile, scip_optimizer)

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
        response = IPG.best_response(game, player_id, others, scip_optimizer)
        deviating = IPG.Strategy[s.id == player_id ? response : s for s in equilibrium_profile]

        @test IPG.payoff(game, player_id, deviating) ≈ IPG.payoff(game, player_id, equilibrium_profile) atol = 1e-5
    end
end

# Covers: the mixed-profile branch (src/best_response.jl:31-55) - that it maximizes
# the expectation over the opponent's support.
@testitem "best_response against a mixed profile uses the expected payoff" setup = [IPGFixtures, SCIPOptimizer] begin
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

    response = IPG.best_response(game, 1, IPG.Strategy[mixed], scip_optimizer)

    @test IPG.value(response, x1) ≈ 1.0 atol = 1e-5
end

# Covers: best_response over binary variables, through both branches. SCIP solves
# the pure-branch models directly, so no degenerate mixed strategy is needed.
@testitem "best_response on a binary game" setup = [IPGFixtures, SCIPOptimizer] begin
    game = get_rock_paper_scissors_game()
    r1, p1, s1 = IPG.get_player_variables(game.players[1])
    r2, p2, s2 = IPG.get_player_variables(game.players[2])

    pure(r, p, s) = IPG.PureStrategy[IPG.PureStrategy(2, Dict(r2 => r, p2 => p, s2 => s))]
    played(response) = round.((IPG.value(response, r1), IPG.value(response, p1), IPG.value(response, s1)))

    # Paper beats rock, scissors beats paper, rock beats scissors.
    @test played(IPG.best_response(game, 1, pure(1.0, 0.0, 0.0), scip_optimizer)) == (0.0, 1.0, 0.0)
    @test played(IPG.best_response(game, 1, pure(0.0, 1.0, 0.0), scip_optimizer)) == (0.0, 0.0, 1.0)
    @test played(IPG.best_response(game, 1, pure(0.0, 0.0, 1.0), scip_optimizer)) == (1.0, 0.0, 0.0)

    # Against the uniform mixed strategy every pure action ties, so any feasible
    # response is optimal and its expected payoff is zero.
    uniform = IPG.MixedStrategy(
        2,
        [1/3, 1/3, 1/3],
        [only(pure(1.0, 0.0, 0.0)), only(pure(0.0, 1.0, 0.0)), only(pure(0.0, 0.0, 1.0))],
    )
    response = IPG.best_response(game, 1, IPG.Strategy[uniform], scip_optimizer)

    @test sum(IPG.value(response, v) for v in (r1, p1, s1)) ≈ 1.0 atol = 1e-5
    @test IPG.payoff(game, 1, IPG.Strategy[response, uniform]) ≈ 0.0 atol = 1e-5
end

# Covers: best_response on a mixed-integer quadratic problem - a convex quadratic
# in a continuous variable coupled to a binary. Needs MIQP support, which is the
# second reason SCIP is the default optimizer here.
@testitem "best_response on a mixed-integer game" setup = [IPGFixtures, SCIPOptimizer] begin
    game = get_example_three_player_game()
    y1, z1 = IPG.get_player_variables(game.players[1])
    y2, z2 = IPG.get_player_variables(game.players[2])
    y3, z3 = IPG.get_player_variables(game.players[3])

    # Against y3 = 2, z3 = 1, player 1 minimizes y1^2 - 2*y1 - 5*z1 subject to
    # -5*z1 <= y1 <= 5*z1. Taking z1 = 1 frees y1 to reach the unconstrained
    # minimizer y1 = 1, for objective 1 - 2 - 5 = -6; z1 = 0 forces y1 = 0 and
    # gives 0. So the best response is y1 = 1, z1 = 1.
    profile = IPG.PureStrategy[
        IPG.PureStrategy(2, Dict(y2 => 0.0, z2 => 0.0)),
        IPG.PureStrategy(3, Dict(y3 => 2.0, z3 => 1.0)),
    ]

    response = IPG.best_response(game, 1, profile, scip_optimizer)

    @test IPG.value(response, y1) ≈ 1.0 atol = 1e-5
    @test IPG.value(response, z1) ≈ 1.0 atol = 1e-5
    @test IPG.payoff(game, 1, IPG.Strategy[response, profile[1], profile[2]]) ≈ -6.0 atol = 1e-5
end

# Covers: the Union{Int, String} player argument of best_response.
@testitem "best_response accepts a player name" setup = [IPGFixtures, SCIPOptimizer] begin
    game = get_example_two_player_game_with_names()
    x1 = only(IPG.get_player_variables(game.players[1]))
    x2 = only(IPG.get_player_variables(game.players[2]))

    profile = IPG.PureStrategy[IPG.PureStrategy(2, Dict(x2 => 4.0))]

    by_name = IPG.best_response(game, "Player1", profile, scip_optimizer)
    by_id = IPG.best_response(game, 1, profile, scip_optimizer)

    @test by_name.id == by_id.id == 1
    @test IPG.value(by_name, x1) ≈ IPG.value(by_id, x1) atol = 1e-5
    @test_throws ArgumentError IPG.best_response(game, "Nobody", profile, scip_optimizer)
end
