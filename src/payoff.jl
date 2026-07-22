"""
    @payoff(model::IPGModel, player::Union{Int, String}, sense, func)

Set the player `player`'s payoff sense to `sense` and payoff function to `func`, in the IPG model `model`.

This macro uses the same syntax as JuMP's [`@objective`](https://jump.dev/JuMP.jl/stable/api/JuMP/#@objective) macro. 
The only difference is that the player whose problem the payoff belongs to must be specified by the argument `player`.

The return value is the payoff expression returned by JuMP's `@objective` macro.

# Example
```julia
julia> model = IPGModel(1);

julia> @variable(model, 1, x)
x

julia> @payoff(model, 1, Max, 2x - 1)
2 x - 1
```
"""
macro payoff(model, player, sense, func)
    id = gensym(:player_id)
    player_model = :($(model).players[$id])

    prepared_payoff = gensym(:prepared_payoff)

    objective_call = Expr(
        :macrocall,
        GlobalRef(JuMP, Symbol("@objective")),
        LineNumberNode(@__LINE__, @__FILE__),
        player_model,
        sense,
        prepared_payoff,
    )

    return esc(quote
        local $id = IPG.get_player_id($(model), $(player))

        local $prepared_payoff = IPG.prepare_payoff($(model), $(id), $(func))

        $objective_call
    end)
end

"""
    prepare_payoff(model::IPGModel, player::Int, payoff)

Prepare `player`'s payoff function `payoff` by replacing other players' variables with parameters.
"""
function prepare_payoff(model::IPGModel, player::Int, payoff)
    if payoff isa Real
        return payoff

    elseif payoff isa JuMP.VariableRef
        return get_parameter!(model, player, payoff)

    elseif payoff isa JuMP.AffExpr
        new_payoff = JuMP.AffExpr()
        new_payoff += JuMP.constant(payoff)

        for (coef, var) in JuMP.linear_terms(payoff)
            new_var = get_parameter!(model, player, var)
            new_payoff += coef * new_var
        end

        return new_payoff

    elseif payoff isa JuMP.QuadExpr
        new_payoff = JuMP.QuadExpr()
        new_payoff += JuMP.constant(payoff)

        for (coef, var) in JuMP.linear_terms(payoff)
            new_var = get_parameter!(model, player, var)
            new_payoff += coef * new_var
        end

        for (coef, var1, var2) in JuMP.quad_terms(payoff)
            new_var1 = get_parameter!(model, player, var1)
            new_var2 = get_parameter!(model, player, var2)
            new_payoff += coef * new_var1 * new_var2
        end

        return new_payoff

    end

    throw(ArgumentError("Payoff type $(typeof(payoff)) is not supported"))
end

"""
    get_parameter!(model::IPGModel, player::Int, variable::JuMP.VariableRef)::JuMP.VariableRef

Return `variable` if it belongs to `player`. 
Return the parameter associated with `variable` in `player`'s `JuMP` model, if it already exists.
Create and return a new parameter associated with `variable` if it didn't already exist.
"""
function get_parameter!(model::IPGModel, player::Int, variable::JuMP.VariableRef)::JuMP.VariableRef
    player_model = model.players[player]
    var_to_param_map = model.var_to_param_maps[player]

    if variable in get_player_variables(player_model) # Variable belongs to the player
        return variable
    end

    if haskey(var_to_param_map, variable) # Parameter already exists
        return var_to_param_map[variable]
    end

    # Create parameter
    parameter = JuMP.@variable(player_model, set = Parameter(0.0), base_name = "$(JuMP.name(variable))_param")
    var_to_param_map[variable] = parameter
    return parameter
end

"""
    compute_payoff_at_profile(game::IPGModel, player::Union{Int, String}, profile::Vector{<:PureStrategy})::Float64

Compute `player`'s payoff at the pure profile `profile`.
"""
function compute_payoff_at_profile(game::IPGModel, player::Union{Int, String}, profile::Vector{<:PureStrategy})::Float64
    player_id = get_player_id(game, player)
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
    compute_payoff_at_profile(game::IPGModel, player::Union{Int, String}, profile::Profile)::Float64

Compute `player`'s expected payoff at the profile `profile` if `profile` has atleast one mixed strategy.
Otherwise, the `compute_payoff_at_profile` for pure profiles is called.
"""
function compute_payoff_at_profile(game::IPGModel, player::Union{Int, String}, profile::Profile)::Float64
    player_id = get_player_id(game, player)

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
    payoff_value(model::IPGModel, player::Union{Int, String}[, i::Int])::Float64

Return the payoff value of the player `player` from the IPG model `model`.

## Supported forms:

* `payoff_value(model, player)` returns the payoff value of `player` at the equilibrium stored in the optimizer attached to the IPG model `model`. 
If multiple equilibria are stored, the first one is used.
* `payoff_value(model, player, i)` returns the payoff value of `player` at the `i`-th equilibrium stored in the optimizer attached to the IPG model `model`. 
An error is thrown if `i` is out of bounds.

If the equilibrium is mixed, the returned value is the player's expected payoff.
"""
function payoff_value(model::IPGModel, player::Union{Int, String})::Float64
    player_id = get_player_id(model, player)

    eq = equilibrium(model)

    return compute_payoff_at_profile(model, player_id, eq)
end
function payoff_value(model::IPGModel, player::Union{Int, String}, i::Int)::Float64
    player_id = get_player_id(model, player)

    eq = equilibrium(model, i)

    return compute_payoff_at_profile(model, player_id, eq)
end

"""
    payoff(model::IPGModel, player::Union{Int, String}, profile::Profile)::Float64

Return the payoff value of the player `player` from the IPG model `model` at the strategy profile `profile`.

If `profile` is mixed, the returned value is the player's expected payoff.
"""
function payoff(model::IPGModel, player::Union{Int, String}, profile::Profile)::Float64
    if length(profile) != length(model.players)
        throw(ArgumentError("A profile should contain one strategy per player"))
    end

    player_id = get_player_id(model, player)
    
    return compute_payoff_at_profile(model, player_id, profile)
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

"""
    comparable_payoff_value(model::IPGModel, player::Union{Int, String}, profile::Profile)::Float64
    comparable_payoff_value(model::IPGModel, player::Union{Int, String}, payoff_val::Float64)::Float64

Return the payoff value of `player` from `model` at the strategy profile `profile` or the payoff `payoff_val`, converted so that larger values are always preferred, 
no matter the objective sense.

If the player `player` has a maximisation payoff, the returned payoff is unchanged.
If the player `player` has a minimisation payoff, this returns the payoff value multiplied by -1.

This function is useful for comparing payoffs without having to consider the objective sense.
"""
function comparable_payoff_value(model::IPGModel, player::Union{Int, String}, payoff_val::Float64)::Float64
    player_id = get_player_id(model, player)
    player_model = model.players[player_id]

    if JuMP.objective_sense(player_model) == JuMP.MAX_SENSE
        return payoff_val
    elseif JuMP.objective_sense(player_model) == JuMP.MIN_SENSE
        return -payoff_val
    else
        throw(ArgumentError("Payoff sense must be MIN or MAX"))
    end
end
function comparable_payoff_value(model::IPGModel, player::Union{Int, String}, profile::Profile)::Float64
    return comparable_payoff_value(model, player, payoff(model, player, profile))
end