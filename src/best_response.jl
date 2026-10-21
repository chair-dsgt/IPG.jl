"""
    get_best_response_model(model::IPGModel, player::Union{Int, String}, profile::Profile[, optimizer_factory])::Tuple{JuMP.Model, JuMP.ReferenceMap}

Build the best-response optimization problem of the player `player` from the IPG model `model`, with respect to `profile`.

Note that the resulting model will be a copy of `player`'s optimization problem, with the same variable names, but different references.

# Arguments
- `model`: the IPG model representing the game.
- `player::Union{Int, String}`: the player whose best-response problem is built.
- `profile::Profile`: should contain the strategies of all other players in the game to which the player belongs.
- `optimizer_factory`: if specified, sets the optimizer of the best response model.
"""
function get_best_response_model(model::IPGModel, player::Union{Int, String}, profile::Profile, optimizer_factory = nothing)::Tuple{JuMP.Model, JuMP.ReferenceMap}
    player_id = get_player_id(model, player)
    player_model = model.players[player_id]
    var_to_param_map = model.var_to_param_maps[player_id]
    best_response_model, reference_map = JuMP.copy_model(player_model)

    # TODO: substitute the parameter values here, as the mixed branch below does
    # via replace_param_with_val, instead of only setting them.
    #
    # Leaving the parameters in the objective keeps a payoff that is bilinear in
    # own x opponent variables as a quadratic function over a merely pinned
    # variable. Nothing downstream repairs that: MOI's only parameter bridge,
    # ParameterToEqualToBridge, maps `x in Parameter(v)` to a real variable with
    # `x == v`, and folding the value into a quadratic coefficient would be
    # presolve rather than a type reformulation. The solver is therefore handed a
    # non-convex MIQP to express what is a plain MILP once the parameters are
    # fixed, which restricts IPG to solvers accepting non-convex quadratics
    # (Gurobi, SCIP) and rules out HiGHS, Cbc and GLPK.
    #
    # The mixed branch already handles PureStrategy entries, so this guard looks
    # removable outright rather than needing its own substitution pass.
    if all(strategy -> strategy isa PureStrategy, profile) # Pure profile
        for strategy in profile
            for (var, val) in strategy.variable_values
                if haskey(var_to_param_map, var)
                    param = var_to_param_map[var]
                    best_response_param = reference_map[param]
    
                    JuMP.set_parameter_value(best_response_param, val)
                end
            end
        end
    else # Profile containing atleast one mixed strategy
        payoff_function = JuMP.objective_function(player_model, JuMP.QuadExpr)
        
        expected_payoff = zero(JuMP.QuadExpr)
        
        supports = [strategy isa PureStrategy ? [strategy] : strategy.supp for strategy in profile]
        probabilities = [strategy isa PureStrategy ? [1.0] : strategy.probs for strategy in profile]
        
        for (probs, strats) in zip(Iterators.product(probabilities...), Iterators.product(supports...))
            prob = prod(probs)

            parameter_values = Dict{JuMP.VariableRef, Float64}()
            for strat in strats
                for (var, val) in strat.variable_values
                    if haskey(var_to_param_map, var)
                        param = var_to_param_map[var]
                        parameter_values[param] = val
                    end
                end
            end

            expected_payoff += prob * replace_param_with_val(payoff_function, parameter_values, reference_map)
        end
        JuMP.set_objective_function(best_response_model, expected_payoff)
    end

    if optimizer_factory !== nothing
        JuMP.set_optimizer(best_response_model, optimizer_factory)
    end

    return best_response_model, reference_map
end

"""
    best_response(model::IPGModel, player::Union{Int, String}, profile::Profile, optimizer_factory)::PureStrategy

Return the best response strategy of the player `player` from the IPG model `model`, with respect to `profile`.

This function builds the best-response model, calls `optimize!` on it, and returns the resulting strategy.

# Arguments
- `model`: the IPG model representing the game.
- `player::Union{Int, String}`: the player that will respond optimally to profile.
- `profile::Profile`: should contain the strategies of all other players in the game to which the player belongs.
- `optimizer_factory`: sets the optimizer of the best response model.
"""
function best_response(model::IPGModel, player::Union{Int, String}, profile::Profile, optimizer_factory; timelimit::Float64=Inf)::PureStrategy
    player_id = get_player_id(model, player)
    player_model = model.players[player_id]

    best_response_model, reference_map = get_best_response_model(model, player_id, profile, optimizer_factory)
    JuMP.set_silent(best_response_model)
    if isfinite(timelimit) JuMP.set_time_limit_sec(best_response_model, timelimit) end
    JuMP.optimize!(best_response_model)

    variable_values = Dict{JuMP.VariableRef, Float64}()
    for var in get_player_variables(player_model)
        best_response_var = reference_map[var]
        variable_values[var] = JuMP.value(best_response_var)
    end

    return PureStrategy(player_id, variable_values)
end