"""
    search!(equilibrium_solver::EquilibriumSolver, model::IPGModel; enumeration::Bool = false, kwargs...)::Nothing

Run the IPG solution algorithm implemented by `equilibrium_solver` on the IPG model `model`.

If `enumeration` is `false`, the equilibrium solver searches for one equilibrium. If `enumeration` is `true`, the equilibrium solver 
searches for all equilibria. In that case, the equilibrium solver must support equilibria enumeration. 

The result is stored in the equilibrium solver and can be retrieved using `get_equilibrium` or `get_equilibria`.
"""
function search!(equilibrium_solver::EquilibriumSolver, model::IPGModel; enumeration::Bool = false, kwargs...)::Nothing
    error("search! must be implemented by the IPG equilibrium_solver")
end

"""
    get_equilibrium(equilibrium_solver::EquilibriumSolver)::Profile
    get_equilibrium(equilibrium_solver::EquilibriumSolver, i::Int)::Profile

Return an equilibrium stored in `equilibrium_solver`.

If multiple equilibria are stored, `get_equilibrium(equilibrium_solver)` returns the first one. The function `get_equilibrium(equilibrium_solver, i)` 
returns the `i`-th equilibrium stored in the equilibrium solver. An error is returned if `i` is out of bounds.

This function can only be called after `search!` has been called.
"""
function get_equilibrium(equilibrium_solver::EquilibriumSolver)::Profile
    error("get_equilibrium must be implemented by the IPG equilibrium_solver")
end 
function get_equilibrium(equilibrium_solver::EquilibriumSolver, i::Int)::Profile
    error("get_equilibrium must be implemented by the IPG equilibrium_solver")
end

"""
    get_equilibria(equilibrium_solver::EquilibriumSolver)::Vector{Profile}

Return the list of equilibria stored in `equilibrium_solver`.

Can only be called after `search!(equilibrium_solver, model, enumeration = true)` has been called.
"""
function get_equilibria(equilibrium_solver::EquilibriumSolver)::Vector{Profile}
    error("get_equilibria must be implemented by the IPG equilibrium_solver")
end

"""
    get_termination_status(equilibrium_solver::EquilibriumSolver)::TerminationStatusCode

Return a `TerminationStatusCode` describing why the equilibrium solver stopped.

The equilibrium solver must translate its internal termination status to a `TerminationStatusCode`.
"""
function get_termination_status(equilibrium_solver::EquilibriumSolver)::TerminationStatusCode
    error("get_termination_status must be implemented by the IPG equilibrium_solver")
end

"""
    supports_enumeration(equilibrium_solver::EquilibriumSolver)::Bool

Return `true` if `equilibrium_solver` supports enumeration and return `false` otherwise.
"""
function supports_enumeration(equilibrium_solver::EquilibriumSolver)::Bool
    error("supports_enumeration must be implemented by the IPG equilibrium_solver")
end