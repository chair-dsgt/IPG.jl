# Covers: delegation from IPGModel to its attached equilibrium solver
# (src/find.jl), the Int/String equivalence of the accessors, and the value()
# family (src/variables.jl:68-101).
#
# Lives in IPG.jl: a MockEquilibriumSolver stands in for a real solver, so these
# assertions hold with SGM.jl absent. They test the interface contract, not any
# algorithm.

# Covers: the no-solver guards in src/find.jl (lines 9, 27, 57, 94) and the fact
# that termination_status alone is defined without one (src/find.jl:115).
@testitem "Accessors require an equilibrium solver" setup = [IPGFixtures] begin
    game = get_example_two_player_game()

    @test game.equilibrium_solver === nothing
    # termination_status is the one accessor that is defined without a solver.
    @test IPG.termination_status(game) == IPG.SOLVE_NOT_CALLED

    @test_throws ArgumentError IPG.find_equilibrium!(game)
    @test_throws ArgumentError IPG.find_all_equilibria!(game)
    @test_throws ArgumentError IPG.equilibrium(game)
    @test_throws ArgumentError IPG.equilibrium(game, 1)
    @test_throws ArgumentError IPG.equilibria(game)
end

# Covers: set_solver! (src/model.jl:92) and termination_status delegating to the
# attached solver's get_termination_status.
@testitem "set_solver! attaches a solver and routes termination_status" setup = [IPGFixtures, MockSolver] begin
    game = get_example_two_player_game()
    solver = MockEquilibriumSolver([example_two_player_equilibrium(game)])

    IPG.set_solver!(game, solver)

    @test game.equilibrium_solver === solver
    @test IPG.termination_status(game) == IPG.SOLVE_NOT_CALLED

    IPG.find_equilibrium!(game)

    @test IPG.termination_status(game) == IPG.EQUILIBRIUM_FOUND
end

# Covers: find_equilibrium! / find_all_equilibria! forwarding to search! with the
# correct `enumeration` flag (src/find.jl:13, 31).
@testitem "find_equilibrium! and find_all_equilibria! set the enumeration flag" setup = [IPGFixtures, MockSolver] begin
    game = get_example_two_player_game()
    solver = MockEquilibriumSolver([example_two_player_equilibrium(game)])
    IPG.set_solver!(game, solver)

    @test IPG.find_equilibrium!(game) === nothing
    @test solver.search_calls == [false]
    @test IPG.termination_status(game) == IPG.EQUILIBRIUM_FOUND

    @test IPG.find_all_equilibria!(game) === nothing
    @test solver.search_calls == [false, true]
    @test IPG.termination_status(game) == IPG.ENUMERATION_COMPLETE

    @test IPG.supports_enumeration(solver)
end

# Covers: equilibrium(model), equilibrium(model, i) and equilibria(model)
# (src/find.jl:56-99), including out-of-bounds rejection.
@testitem "equilibrium and equilibria read through to the solver" setup = [IPGFixtures, MockSolver] begin
    game = get_example_two_player_game()
    eq = example_two_player_equilibrium(game)
    solver = MockEquilibriumSolver([eq, eq])
    IPG.set_solver!(game, solver)

    IPG.find_all_equilibria!(game)

    @test IPG.equilibrium(game) == eq
    @test IPG.equilibrium(game, 1) == eq
    @test IPG.equilibrium(game, 2) == eq
    @test IPG.equilibria(game) == [eq, eq]

    @test_throws ArgumentError IPG.equilibrium(game, 0)
    @test_throws ArgumentError IPG.equilibrium(game, 3)
end

# Ported from SGM.jl/test/SGM.jl:190-193. Asserting that the String and Int
# forms of the accessors agree is IPG's own dispatch contract, and it needs no
# real solver - only a profile to read.
@testitem "Named and indexed accessors agree" setup = [IPGFixtures, MockSolver] begin
    game = get_example_two_player_game_with_names()
    eq = example_two_player_equilibrium(game)
    IPG.set_solver!(game, MockEquilibriumSolver([eq]))

    IPG.find_equilibrium!(game)

    @test IPG.equilibrium(game; player = "Player1") == eq[1]
    @test IPG.equilibrium(game; player = "Player2") == eq[2]
    @test IPG.equilibrium(game; player = "Player1") == IPG.equilibrium(game; player = 1)
    @test IPG.equilibrium(game; player = "Player2") == IPG.equilibrium(game; player = 2)

    @test IPG.equilibrium(game, 1; player = "Player1") == eq[1]

    @test IPG.payoff(game, "Player1", eq) ≈ IPG.payoff(game, 1, eq)
    @test IPG.payoff(game, "Player2", eq) ≈ IPG.payoff(game, 2, eq)
    @test IPG.payoff_value(game, "Player1") ≈ IPG.payoff_value(game, 1)
    @test IPG.payoff_value(game, "Player2") ≈ IPG.payoff_value(game, 2)

    @test_throws ArgumentError IPG.equilibrium(game; player = "Nobody")
    @test_throws ArgumentError IPG.payoff(game, "Nobody", eq)
end

# Covers: the one-strategy-per-player check in payoff (src/payoff.jl:210).
@testitem "payoff rejects malformed profiles" setup = [IPGFixtures, MockSolver] begin
    game = get_example_two_player_game()
    eq = example_two_player_equilibrium(game)

    @test IPG.payoff(game, 1, eq) ≈ 0.0
    @test IPG.payoff(game, 2, eq) ≈ 0.0

    # A profile must carry exactly one strategy per player (src/payoff.jl:210).
    @test_throws ArgumentError IPG.payoff(game, 1, IPG.PureStrategy[eq[1]])
    @test_throws ArgumentError IPG.payoff(game, 1, IPG.PureStrategy[eq[1], eq[2], eq[1]])
end

# Covers: compute_payoff_at_profile on a pure profile (src/payoff.jl:121), against
# hand-computed payoff values.
@testitem "payoff evaluates a pure profile against the player's objective" setup = [IPGFixtures] begin
    game = get_example_two_player_game()
    x1 = only(IPG.get_player_variables(game.players[1]))
    x2 = only(IPG.get_player_variables(game.players[2]))

    # Payoff of player i is -x_i^2 + x_i * x_j.
    profile = IPG.PureStrategy[
        IPG.PureStrategy(1, Dict(x1 => 3.0)),
        IPG.PureStrategy(2, Dict(x2 => 4.0)),
    ]

    @test IPG.payoff(game, 1, profile) ≈ -9.0 + 12.0
    @test IPG.payoff(game, 2, profile) ≈ -16.0 + 12.0
end

# Covers: every value() form (src/variables.jl:68-101), including the mixed-strategy
# expectation and the model-level lookup through the solver. Untested before the split.
@testitem "value reads pure, mixed and model-level strategies" setup = [IPGFixtures, MockSolver] begin
    game = get_example_two_player_game()
    x1 = only(IPG.get_player_variables(game.players[1]))
    x2 = only(IPG.get_player_variables(game.players[2]))

    pure_1 = IPG.PureStrategy(1, Dict(x1 => 2.0))
    pure_2 = IPG.PureStrategy(2, Dict(x2 => 5.0))

    @test IPG.value(pure_1, x1) == 2.0
    @test_throws ArgumentError IPG.value(pure_1, x2)

    # A mixed strategy returns the expectation over its support.
    mixed_1 = IPG.MixedStrategy(1, [0.25, 0.75], [pure_1, IPG.PureStrategy(1, Dict(x1 => 6.0))])
    @test IPG.value(mixed_1, x1) ≈ 0.25 * 2.0 + 0.75 * 6.0

    profile = IPG.Strategy[mixed_1, pure_2]
    @test IPG.value(profile, x1) ≈ 5.0
    @test IPG.value(profile, x2) == 5.0

    # value(model, ...) resolves through the attached solver's equilibrium.
    IPG.set_solver!(game, MockEquilibriumSolver([profile, IPG.Strategy[pure_1, pure_2]]))
    IPG.find_all_equilibria!(game)

    @test IPG.value(game, x1) ≈ 5.0
    @test IPG.value(game, x2) == 5.0
    @test IPG.value(game, x1, 2) == 2.0
end
