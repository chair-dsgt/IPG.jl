using IPG
using TestItems

# Game fixtures, all built through the IPG macros.
#
# These deliberately default to *no* equilibrium solver: IPG.jl is the modeling
# layer and must be testable with no solver package present. Tests that need a
# solved model attach the `MockSolver` snippet instead.
@testsnippet IPGFixtures begin
    using JuMP
    using IPG

    # Example 5.3 from "Computing equilibria for integer programming games".
    #
    # Payoff of player i is -x_i^2 + x_i * x_j, maximized over x_i >= 0, so the
    # best response to x_j is x_j / 2 and the unique equilibrium is (0, 0).
    function get_example_two_player_game(; equilibrium_solver = nothing)
        model = IPG.IPGModel(2)
        equilibrium_solver === nothing || IPG.set_solver!(model, equilibrium_solver)

        IPG.@variable(model, 1, x1, start = 10.0)
        IPG.@constraint(model, 1, x1 >= 0)

        IPG.@variable(model, 2, x2, start = 10.0)
        IPG.@constraint(model, 2, x2 >= 0)

        function player_payoff(x_self, x_other)
            return -x_self * x_self + x_self * x_other
        end

        IPG.@payoff(model, 1, Max, player_payoff(x1, x2))
        IPG.@payoff(model, 2, Max, player_payoff(x2, x1))

        return model
    end

    # Same game, with named players, to exercise the String player API.
    function get_example_two_player_game_with_names(; equilibrium_solver = nothing)
        model = IPG.IPGModel(["Player1", "Player2"])
        equilibrium_solver === nothing || IPG.set_solver!(model, equilibrium_solver)

        IPG.@variable(model, "Player1", x1, start = 10.0)
        IPG.@constraint(model, "Player1", x1 >= 0)

        IPG.@variable(model, "Player2", x2, start = 10.0)
        IPG.@constraint(model, "Player2", x2 >= 0)

        function player_payoff(x_self, x_other)
            return -x_self * x_self + x_self * x_other
        end

        IPG.@payoff(model, "Player1", Max, player_payoff(x1, x2))
        IPG.@payoff(model, "Player2", Max, player_payoff(x2, x1))

        return model
    end

    # Rock-paper-scissors. Once the opponent's strategy is fixed, each player's
    # best-response problem is a pure MILP over the three binaries.
    function get_rock_paper_scissors_game(; equilibrium_solver = nothing)
        model = IPG.IPGModel(2)
        equilibrium_solver === nothing || IPG.set_solver!(model, equilibrium_solver)

        IPG.@variable(model, 1, r1, Bin, start = 1.0)
        IPG.@variable(model, 1, p1, Bin, start = 0.0)
        IPG.@variable(model, 1, s1, Bin, start = 0.0)
        IPG.@constraint(model, 1, r1 + p1 + s1 == 1)

        IPG.@variable(model, 2, r2, Bin, start = 1.0)
        IPG.@variable(model, 2, p2, Bin, start = 0.0)
        IPG.@variable(model, 2, s2, Bin, start = 0.0)
        IPG.@constraint(model, 2, r2 + p2 + s2 == 1)

        IPG.@payoff(model, 1, Max, -r1 * p2 + r1 * s2 + p1 * r2 - p1 * s2 - s1 * r2 + s1 * p2)
        IPG.@payoff(model, 2, Max, r1 * p2 - r1 * s2 - p1 * r2 + p1 * s2 + s1 * r2 - s1 * p2)

        return model
    end

    # From "When do Mixed-Integer Games Admit Rational Equilibria?".
    function get_example_three_player_game(; equilibrium_solver = nothing)
        model = IPG.IPGModel(3)
        equilibrium_solver === nothing || IPG.set_solver!(model, equilibrium_solver)

        IPG.@variable(model, 1, y1)
        IPG.@variable(model, 1, z1, Bin)
        IPG.@constraint(model, 1, -5 * z1 <= y1)
        IPG.@constraint(model, 1, y1 <= 5 * z1)

        IPG.@variable(model, 2, y2)
        IPG.@variable(model, 2, z2, Bin)
        IPG.@constraint(model, 2, -5 * z2 <= y2)
        IPG.@constraint(model, 2, y2 <= 5 * z2)

        IPG.@variable(model, 3, -5 <= y3 <= 4)
        IPG.@variable(model, 3, z3, Bin)

        IPG.@payoff(model, 1, Min, y1^2 - y3 * y1 + y3 * z1 - (8 * z3 - 1) * z1)
        IPG.@payoff(model, 2, Min, y2^2 - y3 * y2 + y3 * z2 + (30 * z3 + 1) * z2)
        IPG.@payoff(model, 3, Min, (z1 - z2 - 1/2) * z3)

        return model
    end
end

# A stand-in `IPG.EquilibriumSolver` that returns a profile handed to it at
# construction. It lets the accessor and delegation layer of IPG.jl be tested
# without depending on SGM.jl or on any optimizer.
@testsnippet MockSolver begin
    using IPG

    mutable struct MockEquilibriumSolver <: IPG.EquilibriumSolver
        equilibria::Vector{IPG.Profile}
        termination_status::IPG.TerminationStatusCode
        enumeration_supported::Bool
        search_calls::Vector{Bool} # records the `enumeration` flag of each call

        function MockEquilibriumSolver(equilibria::Vector{<:IPG.Profile}; enumeration_supported::Bool = true)
            return new(
                collect(IPG.Profile, equilibria),
                IPG.SOLVE_NOT_CALLED,
                enumeration_supported,
                Bool[],
            )
        end
    end

    function IPG.search!(solver::MockEquilibriumSolver, model::IPG.IPGModel; enumeration::Bool = false, kwargs...)::Nothing
        push!(solver.search_calls, enumeration)
        solver.termination_status = enumeration ? IPG.ENUMERATION_COMPLETE : IPG.EQUILIBRIUM_FOUND
        return nothing
    end

    function IPG.get_equilibrium(solver::MockEquilibriumSolver)::IPG.Profile
        isempty(solver.equilibria) && error("No equilibrium is available")
        return solver.equilibria[1]
    end
    function IPG.get_equilibrium(solver::MockEquilibriumSolver, i::Int)::IPG.Profile
        if i < 1 || i > length(solver.equilibria)
            throw(ArgumentError("i is out of bounds"))
        end
        return solver.equilibria[i]
    end

    function IPG.get_equilibria(solver::MockEquilibriumSolver)::Vector{IPG.Profile}
        isempty(solver.equilibria) && error("No equilibria are available.")
        return solver.equilibria
    end

    function IPG.get_termination_status(solver::MockEquilibriumSolver)::IPG.TerminationStatusCode
        return solver.termination_status
    end

    function IPG.supports_enumeration(solver::MockEquilibriumSolver)::Bool
        return solver.enumeration_supported
    end

    # Build the (0, 0) equilibrium of the Example 5.3 game for `model`.
    function example_two_player_equilibrium(model::IPG.IPGModel)::IPG.Profile
        x1 = only(IPG.get_player_variables(model.players[1]))
        x2 = only(IPG.get_player_variables(model.players[2]))
        return IPG.PureStrategy[
            IPG.PureStrategy(1, Dict(x1 => 0.0)),
            IPG.PureStrategy(2, Dict(x2 => 0.0)),
        ]
    end
end

# SCIP is the default optimizer for the best-response tests: it is the only free
# solver here that covers every fixture, because it solves MIQP and non-convex
# quadratic objectives globally. HiGHS handles neither.
@testsnippet SCIPOptimizer begin
    using JuMP
    using SCIP

    scip_optimizer = JuMP.optimizer_with_attributes(
        SCIP.Optimizer,
        "display/verblevel" => 0,
        "limits/gap" => 0.0,
    )
end

# HiGHS is kept only to pin the pure/mixed asymmetry described in the TODO at the
# top of best_response.jl. It is not the general-purpose test optimizer.
@testsnippet HiGHSOptimizer begin
    using JuMP
    using HiGHS

    highs_optimizer = JuMP.optimizer_with_attributes(
        HiGHS.Optimizer,
        "output_flag" => false,
        "mip_rel_gap" => 0.0,
    )
end
