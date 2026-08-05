"""
    @constraint(model::IPGModel, player::Union{Int, String}, expr, args...; kwargs...)
    @constraint(model::IPGModel, player::Union{Int, String}, [index_sets...], expr, args...; kwargs...)
    @constraint(model::IPGModel, player::Union{Int, String}, name, expr, args...; kwargs...)
    @constraint(model::IPGModel, player::Union{Int, String}, name[index_sets...], expr, args...; kwargs...)

Add a constraint described by the expression `expr` to the player `player` from the IPG model `model`.

The return type is Union{ConstraintRef,AbstractArray{<:ConstraintRef}}.

This macro uses the same syntax as JuMP's [`@constraint`](https://jump.dev/JuMP.jl/stable/api/JuMP/#@constraint) macro. 
The only difference is that the player whose problem the constraint belongs to must be specified by the argument `player`. 

A constraint assigned to `player` may only contain variables controlled by `player`.

# Example

```julia
julia> model = IPGModel(2);

julia> @variable(model, 1, x)
x

julia> @variable(model, 2, y)
y

julia> @variable(model, 2, z)
z

julia> @constraint(model, 1, x >= 9)
x ≥ 9

julia> @constraint(model, 2, y + z == 40)
y + z = 40
```
"""
macro constraint(model, player, args...)
    id = gensym(:player_id)
    player_model = :($(model).players[$id])

    constraint_call = Expr(
        :macrocall,
        GlobalRef(JuMP, Symbol("@constraint")),
        LineNumberNode(@__LINE__, @__FILE__),
        player_model,
        args...,
    )

    return esc(quote
        local $id = IPG.get_player_id($(model), $(player))

        constraint = $constraint_call

        try
            IPG.check_variables_in_constraint($(model), $id, constraint) # Check if all variables in constraint belong to player
        catch err
            IPG.remove_constraint!($(model).players[$id], constraint)
            rethrow()
        end

        constraint
    end)
end

"""
    check_variables_in_constraint(model::IPGModel, player::Int, constraint::JuMP.ConstraintRef)::Nothing

Check if all variables in the constraint `constraint` belong to player `player`. An error is thrown if that isn't the case.
"""
function check_variables_in_constraint(model::IPGModel, player::Int, constraint::JuMP.ConstraintRef)::Nothing
    player_variables = get_player_variables(model.players[player])
    constraint_object = JuMP.constraint_object(constraint)
    constraint_function = constraint_object.func

    for var in all_variables_in_constraint(constraint_function)
        if var in player_variables
            continue
        else
            throw(ArgumentError("Constraints for player $player can only contain variables that belong to player $player"))
        end
    end
    return nothing
end
"""
    check_variables_in_constraint(model::IPGModel, player::Int, constraints)::Nothing

Check if all variables in the constraints from the container of constraints `constraints` belong to player `player`. An error is thrown if that isn't the case.
"""
function check_variables_in_constraint(model::IPGModel, player::Int, constraints)::Nothing
    for constraint in constraints
        check_variables_in_constraint(model, player, constraint)
    end
    return nothing
end

function all_variables_in_constraint(constraint)::Vector{JuMP.VariableRef}
    if constraint isa JuMP.VariableRef
        return [constraint]

    elseif constraint isa JuMP.AffExpr
        return [var for (_, var) in JuMP.linear_terms(constraint)]

    elseif constraint isa JuMP.QuadExpr
        variables = JuMP.VariableRef[]

        for (_, var) in JuMP.linear_terms(constraint)
            push!(variables, var)
        end

        for (_, var1, var2) in JuMP.quad_terms(constraint)
            push!(variables, var1)
            push!(variables, var2)
        end

        return variables
    end

    throw(ArgumentError("Constraint type $(typeof(constraint)) unsupported"))
end

function remove_constraint!(model::JuMP.Model, constraint::JuMP.ConstraintRef)::Nothing
    if JuMP.is_valid(model, constraint)
        JuMP.delete(model, constraint)
    end
    return nothing
end
function remove_constraint!(model::JuMP.Model, constraints)::Nothing
    for constraint in constraints
        remove_constraint!(model, constraint)
    end
    return nothing
end