"""
    find_equilibrium!(model::IPGModel; kwargs...)::Nothing

Find an equilibrium profile of the IPG model `model`.

If an equilibrium is found, it is stored in the equilibrium solver attached to `model` and can be queried using the equilibrium access functions. 
"""
function find_equilibrium!(model::IPGModel; kwargs...)::Nothing
    if model.equilibrium_solver === nothing
        throw(ArgumentError("Equilibrium solver must have been set before calling find_equilibrium!"))
    end

    search!(model.equilibrium_solver, model; enumeration = false, kwargs...)
    return nothing
end

"""
    find_all_equilibria!(model::IPGModel; kwargs...)::Nothing

Find all equilibrium profiles of the IPG model `model`.

The model's equilibrium solver must support equilibria enumeration; otherwise an error is thrown.

If at least one equilibrium is found, the equilibria are stored in the equilibrium solver attached to `model` and can be queried using the equilibrium access functions. 
"""
function find_all_equilibria!(model::IPGModel; kwargs...)::Nothing
    if model.equilibrium_solver === nothing
        throw(ArgumentError("Equilibrium solver must have been set before calling find_all_equilibria!"))
    end
    
    search!(model.equilibrium_solver, model; enumeration = true, kwargs...)
    return nothing
end

"""
    equilibrium(model::IPGModel)::Profile
    equilibrium(model::IPGModel, i::Int)::Profile
    equilibrium(model::IPGModel; player::Union{Int, String})::Strategy
    equilibrium(model::IPGModel, i::Int; player::Union{Int, String})::Strategy
    

Return an equilibrium profile or player strategy.

Use `termination_status` to check whether a result exists before asking for equilibrium values. A solution method, 
such as `find_equilibrium!` or `find_all_equilibria!`, must have been called before querying an equilibrium.

## Supported forms:

* `equilibrium(model)` returns the equilibrium stored in the equilibrium solver attached to the IPG model `model`. If multiple equilibria are stored, the first one is returned.
* `equilibrium(model, i)` returns the `i`-th equilibrium stored in the equilibrium solver attached to the IPG model `model`. An error is thrown if `i` is out of bounds. 
* `equilibrium(model; player)` returns the strategy of `player` at the equilibrium stored in the equilibrium solver attached to the IPG model `model`. 
If multiple equilibria are stored, the first one is used.
* `equilibrium(model, i; player)` returns the strategy of `player` at the `i`-th equilibrium stored in the equilibrium solver attached to the IPG model 
`model`. An error is thrown if `i` is out of bounds. 
"""
function equilibrium(model::IPGModel; player::Union{Nothing, Int, String} = nothing)::Union{Profile, Strategy}
    if model.equilibrium_solver === nothing
        throw(ArgumentError("Equilibrium solver must have been set before calling equilibrium"))
    end

    eq = get_equilibrium(model.equilibrium_solver)

    if player === nothing
        return eq
    else
        player_id = get_player_id(model, player)
        return strategy(eq, player_id)
    end
end
function equilibrium(model::IPGModel, i::Int; player::Union{Nothing, Int, String} = nothing)::Union{Profile, Strategy}
    if model.equilibrium_solver === nothing
        throw(ArgumentError("Equilibrium solver must have been set before calling equilibrium"))
    end

    eq = get_equilibrium(model.equilibrium_solver, i)

    if player === nothing
        return eq
    else
        player_id = get_player_id(model, player)

        return strategy(eq, player_id)
    end
end

"""
    equilibria(model::IPGModel)::Vector{Profile}

Return a list of equilibrium profiles stored in the equilibrium solver attached to the IPG model `model`.

Use `termination_status` to check if a result exists before asking for values. `find_all_equilibria!` must have been called before querying the equilibria.
"""
function equilibria(model::IPGModel)::Vector{Profile}
    if model.equilibrium_solver === nothing
        throw(ArgumentError("Equilibrium solver must have been set before calling equilibria"))
    end

    return get_equilibria(model.equilibrium_solver)
end

"""
    termination_status(model::IPGModel)::TerminationStatusCode

Return a `TerminationStatusCode` describing why the IPG solver stopped.

# Example

```julia
julia> model = IPGModel(1);

julia> termination_status(model)
SOLVE_NOT_CALLED::TerminationStatusCode = 0
```
"""
function termination_status(model::IPGModel)::TerminationStatusCode
    if model.equilibrium_solver === nothing
        return SOLVE_NOT_CALLED
    end

    return get_termination_status(model.equilibrium_solver)
end