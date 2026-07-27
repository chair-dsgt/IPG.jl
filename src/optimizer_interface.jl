"""
    search!(optimizer::IPGOptimizer, model::IPGModel; enumeration::Bool = false, kwargs...)::Nothing

Run the IPG solution algorithm implemented by `optimizer` on the IPG model `model`.

If `enumeration` is `false`, the optimizer searches for one equilibrium. If `enumeration` is `true`, the optimizer 
searches for all equilibria. In that case, the optimizer must support equilibria enumeration. 

The result is stored in the optimizer and can be retrieved using `get_equilibrium` or `get_equilibria`.
"""
function search!(optimizer::IPGOptimizer, model::IPGModel; enumeration::Bool = false, kwargs...)::Nothing
    error("search! must be implemented by the IPG optimizer")
end

"""
    get_equilibrium(optimizer::IPGOptimizer)::Profile
    get_equilibrium(optimizer::IPGOptimizer, i::Int)::Profile

Return an equilibrium stored in `optimizer`.

If multiple equilibria are stored, `get_equilibrium(optimizer)` returns the first one. The function `get_equilibrium(optimizer, i)` 
returns the `i`-th equilibrium stored in the optimizer. An error is returned if `i` is out of bounds.

This function can only be called after `search!` has been called.
"""
function get_equilibrium(optimizer::IPGOptimizer)::Profile
    error("get_equilibrium must be implemented by the IPG optimizer")
end 
function get_equilibrium(optimizer::IPGOptimizer, i::Int)::Profile
    error("get_equilibrium must be implemented by the IPG optimizer")
end

"""
    get_equilibria(optimizer::IPGOptimizer)::Vector{Profile}

Return the list of equilibria stored in `optimizer`.

Can only be called after `search!(optimizer, model, enumeration = true)` has been called.
"""
function get_equilibria(optimizer::IPGOptimizer)::Vector{Profile}
    error("get_equilibria must be implemented by the IPG optimizer")
end

"""
    get_termination_status(optimizer::IPGOptimizer)::TerminationStatusCode

Return a `TerminationStatusCode` describing why the IPG solver stopped.

The optimizer must translate its internal termination status to a `TerminationStatusCode`.
"""
function get_termination_status(optimizer::IPGOptimizer)::TerminationStatusCode
    error("get_termination_status must be implemented by the IPG optimizer")
end

"""
    supports_enumeration(optimizer::IPGOptimizer)::Bool

Return `true` if `optimizer` supports enumeration and return `false` otherwise.
"""
function supports_enumeration(optimizer::IPGOptimizer)::Bool
    error("supports_enumeration must be implemented by the IPG optimizer")
end