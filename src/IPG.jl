module IPG

using JuMP

mutable struct IPGModel
    players::Vector{JuMP.Model} 
    # Each player has a dictionary that maps another player's variable to the parameter representing it in this player's model.
    var_to_param_maps::Vector{Dict{JuMP.VariableRef, JuMP.VariableRef}} 

    optimizer::IPGOptimizer
end

struct PureStrategy
    id::Int # Player id
    variable_values::Dict{JuMP.VariableRef, Float64}
end

struct MixedStrategy
    id::Int # Player id
    probs::Vector{Float64} # Probability vector
    supp::Vector{PureStrategy} # Vector of pure strategies (support)
    function MixedStrategy(id::Int, probs::Vector{Float64}, supp::Vector{PureStrategy})
        if length(probs) != size(supp, 1)
            error("There must be as many probabilities as strategies in the support.")
        end
        if ~(sum(probs) ≈ 1)
            error("Probabilities must sum to 1.")
        end
        if any(probs .< 0)
            error("Probabilities must be non-negative.")
        end

        # remove strategies with zero probability
        return new(id, probs[probs .> 0], supp[probs .> 0])
    end
end

const Strategy = Union{PureStrategy, MixedStrategy}

const Profile = Vector{T} where T <: Strategy

@enum TerminationStatusCode begin
    SOLVE_NOT_CALLED
    EQUILIBRIUM_FOUND
    ENUMERATION_COMPLETE
    NO_EQUILIBRIUM
    INFEASIBLE
    ITERATION_LIMIT
    TIME_LIMIT
    NODE_LIMIT
    MEMORY_LIMIT
    OTHER_LIMIT
    SLOW_PROGRESS
    NUMERICAL_ERROR
    INVALID_MODEL
    INVALID_OPTION
    INTERRUPTED
    OTHER_ERROR
end

"""
    compute_payoff_at_profile(game::IPGModel, player_id::Int, profile::Vector{<:PureStrategy})::Float64

Compute player `player_id`'s payoff at the pure profile `profile`.
"""
function compute_payoff_at_profile(game::IPGModel, player_id::Int, profile::Vector{<:PureStrategy})::Float64
    player_model = game.players[player_id]
    var_to_param_map = game.var_to_param_maps[player_id]
    payoff_function = JuMP.objective_function(player_model)

    assignments = copy(profile[player_id].variable_values)
    for strategy in profile
        if strategy.id != player_id
            for (variable, value) in strategy.variable_values
                if haskey(var_to_param_map, variable)
                    parameter = var_to_param_map[variable]
                    assignments[parameter] = value
                end
            end
        end
    end

    function get_assigned_value(variable::JuMP.VariableRef) return assignments[variable] end

    payoff_value = JuMP.value(get_assigned_value, payoff_function)
    return Float64(payoff_value)
end

"""
    compute_payoff_at_profile(game::IPGModel, player_id::Int, profile::Profile)::Float64

Compute player `player_id`'s expected payoff at the profile `profile` if `profile` has atleast one mixed strategy.
Otherwise, the `compute_payoff_at_profile` for pure profiles is called.
"""
function compute_payoff_at_profile(game::IPGModel, player_id::Int, profile::Profile)::Float64
    if all(strategy -> strategy isa PureStrategy, profile) # Pure profile
        pure_profile = PureStrategy[strategy for strategy in profile]
        return compute_payoff_at_profile(game, player_id, pure_profile)
    end

    # Expected payoff:
    expectation = 0.0
    supports = [strategy isa PureStrategy ? [strategy] : strategy.supp for strategy in profile]
    probabilities = [strategy isa PureStrategy ? [1.0] : strategy.probs for strategy in profile]

    for (probs, strats) in zip(Iterators.product(probabilities...), Iterators.product(supports...))
        prob = prod(probs)
        pure_profile = PureStrategy[strats...]

        expectation += prob * compute_payoff_at_profile(game, player_id, pure_profile)
    end
    
    return expectation
end

"""
    empty_strategy_sample(num_players::Int)

Create an empty strategy sample for `num_players` players.

The returned vector contains an empty `PureStrategy` vector per player.
"""
function empty_strategy_sample(num_players::Int)::Vector{Vector{PureStrategy}}
    [PureStrategy[] for _ in 1:num_players]
end

"""
    get_player_variables(model::JuMP.Model)::Vector{JuMP.VariableRef}

Get all variables from `model`, excluding parameters.
"""
function get_player_variables(model::JuMP.Model)::Vector{JuMP.VariableRef}
    return filter((var) -> !JuMP.is_parameter(var), JuMP.all_variables(model))
end

"""
    replace_param_with_val(payoff::JuMP.QuadExpr, parameter_values::Dict{JuMP.VariableRef, Float64}, reference_map::JuMP.ReferenceMap)::JuMP.QuadExpr

Replace all parameters in `payoff` by their associated value in `parameter_values`.

Variables that are not parameters stay variables in the returned payoff.
"""
function replace_param_with_val(payoff::JuMP.QuadExpr, parameter_values::Dict{JuMP.VariableRef, Float64}, reference_map::JuMP.ReferenceMap)::JuMP.QuadExpr
    new_payoff = zero(JuMP.QuadExpr)
    new_payoff += JuMP.constant(payoff) # Constant term

    # Linear terms
    for (coefficient, variable) in JuMP.linear_terms(payoff) 
        if haskey(parameter_values, variable)
            new_payoff += coefficient * parameter_values[variable] # Replace parameter with value
        else
            new_payoff += coefficient * reference_map[variable] # Keep variable
        end
    end

    # Quadratic terms
    for (coefficient, variable1, variable2) in JuMP.quad_terms(payoff)
        if haskey(parameter_values, variable1)
            factor1 = parameter_values[variable1] # Replace parameter with value
        else
            factor1 = reference_map[variable1] # Keep variable
        end
        if haskey(parameter_values, variable2)
            factor2 = parameter_values[variable2] # Replace parameter with value
        else
            factor2 = reference_map[variable2] # Keep variable
        end 
        new_payoff += coefficient * factor1 * factor2
    end

    return new_payoff
end

end
