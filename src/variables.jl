"""
    @variable(model::IPGModel, player::Union{Int, String}, expr, args..., kw_args...)

Add a variable described by the expression `expr`, the positional arguments `args` and the keyword arguments `kw_args` to the player `player` from the IPG model `model`.

The return type is Union{VariableRef,AbstractArray{<:VariableRef}}.

This macro uses the same syntax as JuMP's [`@variable`](https://jump.dev/JuMP.jl/stable/api/JuMP/#@variable) macro. The
only difference is that the player controlling the variable must be specified by the argument `player`.

# Example

The following example creates a variable `x` controlled by player 1:
```julia
julia> model = IPGModel(1);

julia> @variable(model, 1, x)
x
```

Other example: 

```julia
julia> model = IPGModel(2);

julia> @variable(model, 1, x >= 0)
x

julia> @variable(model, 2, y, Int)
y
```
"""
macro variable(model, player, args...)
    id = gensym(:player_id)
    player_model = :($(model).players[$id])

    variable_call = Expr(
        :macrocall,
        GlobalRef(JuMP, Symbol("@variable")),
        LineNumberNode(@__LINE__, @__FILE__),
        player_model,
        args...,
    )

    return esc(quote
        local $id = IPG.get_player_id($(model), $(player))

        $variable_call
    end)
end

"""
    value(model::IPGModel, variable::JuMP.VariableRef)::Float64
    value(model::IPGModel, variable::JuMP.VariableRef, i::Int)::Float64
    value(strategy::Strategy, variable::JuMP.VariableRef)::Float64
    value(profile::Profile, variable::JuMP.VariableRef)::Float64

Return the value of the variable `variable` at a given profile or strategy.

## Supported forms:
* `value(model, variable)` returns the value of `variable` at the equilibrium stored in the equilibrium solver attached to `model`. If multiple equilibria are stored, the first one is used.
* `value(model, variable, i)` returns the value of `variable` at the `i`-th equilibrium stored in the equilibrium solver attached to `model`. An error is thrown if `i` is out of bounds.
* `value(strategy, variable)` returns the value of `variable` at the strategy `strategy`. The strategy must belong to the player that controls `variable`.
* `value(profile, variable)` returns the value of `variable` at the profile `profile`. The value is read from the strategy in `profile` of the player that controls `variable`.

If a strategy or profile is mixed, the returned value is the expected value of `variable` under the probability distribution over the pure strategies.
"""
function value(strategy::PureStrategy, variable::JuMP.VariableRef)::Float64
    if haskey(strategy.variable_values, variable)
        return strategy.variable_values[variable]
    end

    throw(ArgumentError("Variable is not part of this strategy"))
end
function value(strategy::MixedStrategy, variable::JuMP.VariableRef)::Float64
    expected_value = 0.0

    for (prob, pure_strat) in zip(strategy.probs, strategy.supp)
        expected_value += prob * value(pure_strat, variable)
    end

    return expected_value
end
function value(profile::Profile, variable::JuMP.VariableRef)::Float64
    for strategy in profile
        try 
            return value(strategy, variable)
        catch err
            if !(err isa ArgumentError)
                rethrow(err)
            end
        end
    end

    throw(ArgumentError("Variable is not part of this profile"))
end
function value(model::IPGModel, variable::JuMP.VariableRef)::Float64
    return value(equilibrium(model), variable)
end
function value(model::IPGModel, variable::JuMP.VariableRef, i::Int)::Float64
    return value(equilibrium(model, i), variable)
end