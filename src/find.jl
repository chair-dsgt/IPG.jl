"""
    find_equilibrium!(model::IPGModel)::Nothing

Find an equilibrium profile of the IPG model `model`.

If an equilibrium is found, it is stored in the optimizer attached to `model` and can be queried using the equilibrium access functions. 
"""
function find_equilibrium!(model::IPGModel; kwargs...)::Nothing
    if model.optimizer === nothing
        throw(ArgumentError("Optimizer must have been set before calling find_equilibrium!"))
    end

    search!(model.optimizer, model; enumeration = false, kwargs...)
    return nothing
end

"""
    find_all_equilibria!(model::IPGModel)::Nothing

Find all equilibrium profiles of the IPG model `model`.

The model's optimizer must support equilibria enumeration; otherwise an error is thrown.

If at least one equilibrium is found, the equilibria are stored in the optimizer attached to `model` and can be queried using the equilibrium access functions. 
"""
function find_all_equilibria!(model::IPGModel; kwargs...)::Nothing
    if model.optimizer === nothing
        throw(ArgumentError("Optimizer must have been set before calling find_all_equilibria!"))
    end
    
    search!(model.optimizer, model; enumeration = true, kwargs...)
    return nothing
end

"""
    equilibrium(model::IPGModel)::Profile
    equilibrium(model::IPGModel, i::Int)::Profile
    equilibrium(model::IPGModel; player::Int)::Strategy
    equilibrium(model::IPGModel, i::Int; player::Int)::Strategy
    

Return an equilibrium profile or player strategy.

Use `termination_status` to check whether a result exists before asking for equilibrium values. A solution method, 
such as `find_equilibrium!` or `find_all_equilibria!`, must have been called before querying an equilibrium.

## Supported forms:

* `equilibrium(model)` returns the equilibrium stored in the optimizer attached to the IPG model `model`. If multiple equilibria are stored, the first one is returned.
* `equilibrium(model, i)` returns the `i`-th equilibrium stored in the optimizer attached to the IPG model `model`. An error is thrown if `i` is out of bounds. 
* `equilibrium(model; player = n)` returns the strategy of player `n` at the equilibrium stored in the optimizer attached to the IPG model `model`. 
If multiple equilibria are stored, the first one is used.
* `equilibrium(model, i; player = n)` returns the strategy of player `n` at the `i`-th equilibrium stored in the optimizer attached to the IPG model 
`model`. An error is thrown if `i` is out of bounds. 
"""
function equilibrium(model::IPGModel; player::Union{Nothing, Int} = nothing)::Union{Profile, Strategy}
    if model.optimizer === nothing
        throw(ArgumentError("Optimizer must have been set before calling equilibrium"))
    end

    eq = get_equilibrium(model.optimizer)

    if player === nothing
        return eq
    else
        if player < 1 || player > length(model.players)
            throw(ArgumentError("Player is invalid"))
        end

        return strategy(eq, player)
    end
end
function equilibrium(model::IPGModel, i::Int; player::Union{Nothing, Int} = nothing)::Union{Profile, Strategy}
    if model.optimizer === nothing
        throw(ArgumentError("Optimizer must have been set before calling equilibrium"))
    end

    eq = get_equilibrium(model.optimizer, i)

    if player === nothing
        return eq
    else
        if player < 1 || player > length(model.players)
            throw(ArgumentError("Player is invalid"))
        end

        return strategy(eq, player)
    end
end

"""
    equilibria(model::IPGModel)::Vector{Profile}

Return a list of equilibrium profiles stored in the optimizer attached to the IPG model `model`.

Use `termination_status` to check if a result exists before asking for values. `find_all_equilibria!` must have been called before querying the equilibria.
"""
function equilibria(model::IPGModel)::Vector{Profile}
    if model.optimizer === nothing
        throw(ArgumentError("Optimizer must have been set before calling equilibria"))
    end

    return get_equilibria(model.optimizer)
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
    if model.optimizer === nothing
        return SOLVE_NOT_CALLED
    end

    return get_termination_status(model.optimizer)
end