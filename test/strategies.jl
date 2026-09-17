# Covers: the MixedStrategy constructor and its validation (src/strategies.jl:6-24).
#
# Lives in IPG.jl: the strategy types are part of the modeling layer and involve
# no solver at all.
#
# Adapted from the "DiscreteMixedStrategy" test item of the earlier IPG.jl
# (github.com/brunompacheco/IPG.jl, test/game.jl). The constructor's invariants
# were previously untested here.

# Covers: MixedStrategy accepted inputs, the expectation it induces, and the
# fact that zero-probability support points are dropped at construction.
@testitem "MixedStrategy construction" setup = [IPGFixtures] begin
    game = get_example_two_player_game()
    x1 = only(IPG.get_player_variables(game.players[1]))

    zero_strategy = IPG.PureStrategy(1, Dict(x1 => 0.0))
    ten_strategy = IPG.PureStrategy(1, Dict(x1 => 10.0))

    mixed = IPG.MixedStrategy(1, [0.25, 0.75], [zero_strategy, ten_strategy])

    @test mixed.id == 1
    @test length(mixed.probs) == length(mixed.supp) == 2
    @test sum(mixed.probs) ≈ 1.0
    @test IPG.value(mixed, x1) ≈ 0.25 * 0.0 + 0.75 * 10.0

    # A pure strategy is the degenerate mixed strategy.
    @test IPG.value(IPG.MixedStrategy(1, [1.0], [ten_strategy]), x1) == 10.0

    # Support points carrying no probability are dropped by the constructor, so
    # the stored support is not always the one that was passed in.
    trimmed = IPG.MixedStrategy(1, [0.0, 1.0], [zero_strategy, ten_strategy])

    @test trimmed.probs == [1.0]
    @test length(trimmed.supp) == 1
    @test only(trimmed.supp) === ten_strategy
    @test IPG.value(trimmed, x1) == 10.0
end

# Covers: the three error paths of the MixedStrategy constructor.
@testitem "MixedStrategy rejects invalid distributions" setup = [IPGFixtures] begin
    game = get_example_two_player_game()
    x1 = only(IPG.get_player_variables(game.players[1]))

    a = IPG.PureStrategy(1, Dict(x1 => 0.0))
    b = IPG.PureStrategy(1, Dict(x1 => 10.0))

    # One probability per support point.
    @test_throws ErrorException IPG.MixedStrategy(1, [1.0], [a, b])
    @test_throws ErrorException IPG.MixedStrategy(1, [0.3, 0.3, 0.4], [a, b])

    # Probabilities must sum to one ...
    @test_throws ErrorException IPG.MixedStrategy(1, [0.5, 0.2], [a, b])
    @test_throws ErrorException IPG.MixedStrategy(1, [0.5, 0.7], [a, b])

    # ... and be non-negative. Note this case sums to one, so it is the sign
    # check that must reject it.
    @test_throws ErrorException IPG.MixedStrategy(1, [1.5, -0.5], [a, b])

    # The sum is compared with ≈, so ordinary floating-point error is accepted.
    @test IPG.MixedStrategy(1, [1/3, 1/3, 1/3], [a, b, a]) isa IPG.MixedStrategy
end

# Covers: how strategies compare. IPG defines neither == nor isapprox for them,
# so equality falls back to egal and two separately built strategies with
# identical contents are *not* equal. The earlier IPG.jl defined
# Base.:(==)(::DiscreteMixedStrategy, ::DiscreteMixedStrategy); this test records
# what the current package actually does, so the difference is not discovered by
# accident in a comparison that silently returns false.
@testitem "Strategy equality falls back to identity" setup = [IPGFixtures] begin
    game = get_example_two_player_game()
    x1 = only(IPG.get_player_variables(game.players[1]))

    a = IPG.PureStrategy(1, Dict(x1 => 0.0))
    b = IPG.PureStrategy(1, Dict(x1 => 10.0))

    first_mixed = IPG.MixedStrategy(1, [0.5, 0.5], [a, b])
    second_mixed = IPG.MixedStrategy(1, [0.5, 0.5], [a, b])

    # Same object compares equal, a structurally identical one does not.
    @test first_mixed == first_mixed
    @test first_mixed != second_mixed

    # Comparing field by field is what actually works today.
    @test first_mixed.id == second_mixed.id
    @test first_mixed.probs == second_mixed.probs
    @test first_mixed.supp == second_mixed.supp

    # The same applies to pure strategies.
    @test IPG.PureStrategy(1, Dict(x1 => 0.0)) != IPG.PureStrategy(1, Dict(x1 => 0.0))
    @test IPG.PureStrategy(1, Dict(x1 => 0.0)).variable_values == a.variable_values
end
