# Covers: IPGModel construction and validation (src/model.jl:28-90), player id
# resolution (src/model.jl:111-126), get_player_variables (src/model.jl:102),
# and the variable -> parameter machinery driven by @payoff (src/payoff.jl:97).
#
# Lives in IPG.jl: none of this involves an equilibrium solver.

# Covers: IPGModel(n) (src/model.jl:28) - player vector, empty maps, no solver.
@testitem "IPGModel construction by player count" setup = [IPGFixtures] begin
    model = IPG.IPGModel(3)

    @test length(model.players) == 3
    @test length(model.var_to_param_maps) == 3
    @test all(isempty, model.var_to_param_maps)
    @test isempty(model.name_to_id_map)
    @test model.equilibrium_solver === nothing
    @test all(p -> p isa JuMP.Model, model.players)
end

# Covers: IPGModel(player_names) (src/model.jl:61) - the name -> id map it builds.
@testitem "IPGModel construction by player name" setup = [IPGFixtures] begin
    model = IPG.IPGModel(["Player1", "Player2"])

    @test length(model.players) == 2
    @test length(model.var_to_param_maps) == 2
    @test model.name_to_id_map == Dict("Player1" => 1, "Player2" => 2)
    @test model.equilibrium_solver === nothing
end

# Covers: the ArgumentError guards in both constructors (src/model.jl:29, 64-69).
@testitem "IPGModel constructor rejects invalid player specifications" setup = [IPGFixtures] begin
    @test_throws ArgumentError IPG.IPGModel(0)
    @test_throws ArgumentError IPG.IPGModel(-1)
    @test_throws ArgumentError IPG.IPGModel(String[])
    @test_throws ArgumentError IPG.IPGModel(["Player1", "Player1"])
end

# Covers: get_player_id for Int and String, including all three rejection paths
# (src/model.jl:111-126). Untested before the split.
@testitem "get_player_id resolves ids and names" setup = [IPGFixtures] begin
    unnamed = IPG.IPGModel(2)

    @test IPG.get_player_id(unnamed, 1) == 1
    @test IPG.get_player_id(unnamed, 2) == 2
    @test_throws ArgumentError IPG.get_player_id(unnamed, 0)
    @test_throws ArgumentError IPG.get_player_id(unnamed, 3)
    # A model with unnamed players cannot be queried by name.
    @test_throws ArgumentError IPG.get_player_id(unnamed, "Player1")

    named = IPG.IPGModel(["Player1", "Player2"])

    @test IPG.get_player_id(named, "Player1") == 1
    @test IPG.get_player_id(named, "Player2") == 2
    @test IPG.get_player_id(named, 2) == 2
    @test_throws ArgumentError IPG.get_player_id(named, "Nobody")
    @test_throws ArgumentError IPG.get_player_id(named, 3)
end

# Covers: IPG.@variable / IPG.@constraint routing to the right player's JuMP model
# (src/variables.jl:33, src/constraints.jl:37), and start-value propagation.
@testitem "@variable and @constraint attach to the named player" setup = [IPGFixtures] begin
    game = get_example_two_player_game_with_names()

    x1 = only(IPG.get_player_variables(game.players[1]))
    x2 = only(IPG.get_player_variables(game.players[2]))

    @test JuMP.owner_model(x1) === game.players[1]
    @test JuMP.owner_model(x2) === game.players[2]
    @test JuMP.start_value(x1) == 10.0
    @test JuMP.start_value(x2) == 10.0
    @test JuMP.num_constraints(game.players[1]; count_variable_in_set_constraints = false) == 1
    @test JuMP.num_constraints(game.players[2]; count_variable_in_set_constraints = false) == 1
end

# Covers: that naming players is purely an addressing convenience - a named model
# and an id-addressed model build the same game.
#
# This is where the question "does giving players names change anything?" is
# answered, and it is IPG's to answer: SGM never reads name_to_id_map and
# addresses players only by integer id, so a named model is invisible to it.
@testitem "Naming players does not change the constructed model" setup = [IPGFixtures] begin
    unnamed = get_example_two_player_game()
    named = get_example_two_player_game_with_names()

    # The name map is the only field that differs.
    @test isempty(unnamed.name_to_id_map)
    @test named.name_to_id_map == Dict("Player1" => 1, "Player2" => 2)
    @test length(unnamed.players) == length(named.players)

    for player_id in 1:2
        unnamed_player = unnamed.players[player_id]
        named_player = named.players[player_id]

        unnamed_variables = IPG.get_player_variables(unnamed_player)
        named_variables = IPG.get_player_variables(named_player)

        @test JuMP.name.(unnamed_variables) == JuMP.name.(named_variables)
        @test JuMP.start_value.(unnamed_variables) == JuMP.start_value.(named_variables)
        @test JuMP.num_constraints(unnamed_player; count_variable_in_set_constraints = false) ==
              JuMP.num_constraints(named_player; count_variable_in_set_constraints = false)

        # The same parameters, standing in for the same opponent variables.
        unnamed_map = unnamed.var_to_param_maps[player_id]
        named_map = named.var_to_param_maps[player_id]

        @test Set(JuMP.name(v) for v in keys(unnamed_map)) == Set(JuMP.name(v) for v in keys(named_map))
        @test Set(JuMP.name(p) for p in values(unnamed_map)) == Set(JuMP.name(p) for p in values(named_map))

        # The same payoff, up to the variable references it is written over.
        @test JuMP.objective_sense(unnamed_player) == JuMP.objective_sense(named_player)
        @test string(JuMP.objective_function(unnamed_player)) == string(JuMP.objective_function(named_player))
    end

    # And the payoffs agree numerically at the same point, whichever way the
    # player is addressed.
    at(game, v1, v2) = IPG.PureStrategy[
        IPG.PureStrategy(1, Dict(only(IPG.get_player_variables(game.players[1])) => v1)),
        IPG.PureStrategy(2, Dict(only(IPG.get_player_variables(game.players[2])) => v2)),
    ]
    unnamed_profile = at(unnamed, 3.0, 4.0)
    named_profile = at(named, 3.0, 4.0)

    for player_id in 1:2
        @test IPG.payoff(unnamed, player_id, unnamed_profile) ≈ IPG.payoff(named, player_id, named_profile)
    end
    @test IPG.payoff(named, "Player1", named_profile) ≈ IPG.payoff(unnamed, 1, unnamed_profile)
    @test IPG.payoff(named, "Player2", named_profile) ≈ IPG.payoff(unnamed, 2, unnamed_profile)
end

# Covers: get_player_variables filtering out parameter variables (src/model.jl:102).
@testitem "get_player_variables excludes parameters" setup = [IPGFixtures] begin
    game = get_example_two_player_game()

    # @payoff added one parameter per player, standing in for the opponent's variable.
    @test length(JuMP.all_variables(game.players[1])) == 2
    @test length(JuMP.all_variables(game.players[2])) == 2
    @test length(IPG.get_player_variables(game.players[1])) == 1
    @test length(IPG.get_player_variables(game.players[2])) == 1
    @test all(!JuMP.is_parameter, IPG.get_player_variables(game.players[1]))
    @test all(!JuMP.is_parameter, IPG.get_player_variables(game.players[2]))
end

# This replaces the "Variable to parameter mapping" test that used to live in
# SGM.jl/test/IPG.jl. That version asserted against a var_to_param_map the test
# fixture had built by hand, so it exercised no IPG code. Here the map is built
# by @payoff -> prepare_payoff -> get_parameter!, and we assert the contract
# that get_parameter! documents.
@testitem "@payoff builds the variable-to-parameter map" setup = [IPGFixtures] begin
    game = get_example_two_player_game()

    x1 = only(IPG.get_player_variables(game.players[1]))
    x2 = only(IPG.get_player_variables(game.players[2]))

    # Each player holds a parameter for exactly the opponent's variable.
    @test Set(keys(game.var_to_param_maps[1])) == Set([x2])
    @test Set(keys(game.var_to_param_maps[2])) == Set([x1])

    for player_id in 1:2
        other_id = 3 - player_id
        other_variables = IPG.get_player_variables(game.players[other_id])

        for (variable, parameter) in game.var_to_param_maps[player_id]
            # Keys are real variables owned by another player ...
            @test !JuMP.is_parameter(variable)
            @test variable in other_variables
            @test JuMP.owner_model(variable) === game.players[other_id]

            # ... values are parameters living in this player's own model.
            @test JuMP.is_parameter(parameter)
            @test JuMP.owner_model(parameter) === game.players[player_id]
        end
    end

    parameter_for_x2 = game.var_to_param_maps[1][x2]

    # get_parameter! names the parameter after the variable it stands in for and
    # initialises it to zero. Note this differs from a hand-built map: the macro
    # path appends "_param" rather than reusing the bare variable name.
    @test JuMP.name(parameter_for_x2) == "$(JuMP.name(x2))_param"
    @test JuMP.parameter_value(parameter_for_x2) == 0.0

    # Idempotent: a repeat call returns the existing parameter and does not grow
    # the map or the player's model.
    variables_before = length(JuMP.all_variables(game.players[1]))
    @test IPG.get_parameter!(game, 1, x2) == parameter_for_x2
    @test length(game.var_to_param_maps[1]) == 1
    @test length(JuMP.all_variables(game.players[1])) == variables_before

    # A player's own variable is returned unchanged and never enters the map.
    @test IPG.get_parameter!(game, 1, x1) == x1
    @test !haskey(game.var_to_param_maps[1], x1)
    @test length(JuMP.all_variables(game.players[1])) == variables_before
end

# Covers: get_parameter! reached once per distinct opponent variable in a payoff,
# on a game where each player depends on three of them.
@testitem "Variable-to-parameter map covers every shared variable" setup = [IPGFixtures] begin
    game = get_rock_paper_scissors_game()

    player_1_variables = IPG.get_player_variables(game.players[1])
    player_2_variables = IPG.get_player_variables(game.players[2])

    @test length(player_1_variables) == 3
    @test length(player_2_variables) == 3
    @test JuMP.start_value.(player_1_variables) == [1.0, 0.0, 0.0]
    @test JuMP.start_value.(player_2_variables) == [1.0, 0.0, 0.0]
    @test all(JuMP.is_binary, player_1_variables)
    @test all(JuMP.is_binary, player_2_variables)

    # Every one of the opponent's three variables appears in each player's payoff,
    # so each map holds three parameters.
    @test Set(keys(game.var_to_param_maps[1])) == Set(player_2_variables)
    @test Set(keys(game.var_to_param_maps[2])) == Set(player_1_variables)
    @test all(JuMP.is_parameter, values(game.var_to_param_maps[1]))
    @test all(JuMP.is_parameter, values(game.var_to_param_maps[2]))
end

# Covers: parameter wiring when players depend on different subsets of each other,
# plus variable bounds and integrality set through IPG.@variable.
@testitem "Three-player model wires parameters across all players" setup = [IPGFixtures] begin
    game = get_example_three_player_game()

    @test length(game.players) == 3
    @test length(game.var_to_param_maps) == 3

    y1, z1 = IPG.get_player_variables(game.players[1])
    y2, z2 = IPG.get_player_variables(game.players[2])
    y3, z3 = IPG.get_player_variables(game.players[3])

    # Players 1 and 2 depend on player 3's variables only; player 3 depends on
    # the binaries of players 1 and 2 only.
    @test Set(keys(game.var_to_param_maps[1])) == Set([y3, z3])
    @test Set(keys(game.var_to_param_maps[2])) == Set([y3, z3])
    @test Set(keys(game.var_to_param_maps[3])) == Set([z1, z2])

    @test JuMP.has_lower_bound(y3) && JuMP.lower_bound(y3) == -5
    @test JuMP.has_upper_bound(y3) && JuMP.upper_bound(y3) == 4
    @test JuMP.is_binary(z1) && JuMP.is_binary(z2) && JuMP.is_binary(z3)
end
