"""
    @variable(model::IPGModel, player::Int, expr, args..., kw_args...)

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
    player_model = :($(model).players[$(player)])

    variable_call = Expr(
        :macrocall,
        GlobalRef(JuMP, Symbol("@variable")),
        LineNumberNode(@__LINE__, @__FILE__),
        player_model,
        args...,
    )

    return esc(quote
        if $(player) < 1 || $(player) > length($(model).players)
            throw(ArgumentError("Player is invalid"))
        end

        $variable_call
    end)
end