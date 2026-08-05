mutable struct IPGModel
    players::Vector{JuMP.Model} 
    name_to_id_map::Dict{String, Int} # Maps a player's name to its id. Empty if players are unnamed.
    # Each player has a dictionary that maps another player's variable to the parameter representing it in this player's model.
    var_to_param_maps::Vector{Dict{JuMP.VariableRef, JuMP.VariableRef}} 

    equilibrium_solver::Union{Nothing, EquilibriumSolver}
end

"""
    IPGModel(n::Int[, equilibrium_solver::EquilibriumSolver])::IPGModel

Create a new instance of an integer programming game model with `n` players.

`n` must be a positive integer.

Players are unnamed and are referenced by their id.

`equilibrium_solver` must be one of the forms:

* Omitted, like `IPGModel(n)`, which creates an IPG model without an attached equilibrium solver
* An IPG equilibrium solver, like `IPGModel(n, SGM.EquilibriumSolver(HiGHS.Optimizer))`, where 
`SGM.EquilibriumSolver` is the main IPG solution algorithm and `HiGHS.Optimizer` is used to solve the best-response 
problems

If `equilibrium_solver` is not provided, use `set_solver!` to set the equilibrium solver before calling `find_equilibrium!` or `find_all_equilibria!`.
"""
function IPGModel(n::Int)::IPGModel
    if n <= 0
        throw(ArgumentError("The number of players must be a positive integer"))
    end

    players = [JuMP.Model() for _ in 1:n]
    name_to_id_map = Dict{String, Int}()
    var_to_param_maps = [Dict{JuMP.VariableRef, JuMP.VariableRef}() for _ in 1:n]

    return IPGModel(players, name_to_id_map, var_to_param_maps, nothing)
end
function IPGModel(n::Int, equilibrium_solver::EquilibriumSolver)::IPGModel
    model = IPGModel(n)
    set_solver!(model, equilibrium_solver)
    return model
end

"""
    IPGModel(player_names::Vector{String}[, equilibrium_solver::EquilibriumSolver])::IPGModel

Create a new instance of an integer programming game model whose players are named by `player_names`.

`player_names` must be nonempty and contain unique names.

`equilibrium_solver` must be one of the forms:

* Omitted, like `IPGModel(player_names)`, which creates an IPG model without an attached equilibrium solver
* An IPG equilibrium solver, like `IPGModel(player_names, SGM.EquilibriumSolver(HiGHS.Optimizer))`, where 
`SGM.EquilibriumSolver` is the main IPG solution algorithm and `HiGHS.Optimizer` is used to solve the best-response 
problems

If `equilibrium_solver` is not provided, use `set_solver!` to set the equilibrium solver before calling `find_equilibrium!` or `find_all_equilibria!`.
"""
function IPGModel(player_names::Vector{String})::IPGModel
    n = length(player_names)
    if n <= 0
        throw(ArgumentError("player_names must be nonempty"))
    end
    if length(unique(player_names)) != n
        throw(ArgumentError("Player names must be unique"))
    end

    players = [JuMP.Model() for _ in 1:n]

    name_to_id_map = Dict{String, Int}()
    for i in 1:n
        name_to_id_map[player_names[i]] = i
    end

    var_to_param_maps = [Dict{JuMP.VariableRef, JuMP.VariableRef}() for _ in 1:n]

    return IPGModel(players, name_to_id_map, var_to_param_maps, nothing)
end
function IPGModel(player_names::Vector{String}, equilibrium_solver::EquilibriumSolver)::IPGModel
    model = IPGModel(player_names)
    set_solver!(model, equilibrium_solver)
    return model
end

"""
    set_solver!(model::IPGModel, equilibrium_solver::EquilibriumSolver)::Nothing

Set the IPG equilibrium solver of the model `model` to `equilibrium_solver`.
"""
function set_solver!(model::IPGModel, equilibrium_solver::EquilibriumSolver)::Nothing
    model.equilibrium_solver = equilibrium_solver
    return nothing
end

"""
    get_player_variables(model::JuMP.Model)::Vector{JuMP.VariableRef}

Get all variables from `model`, excluding parameters.
"""
function get_player_variables(model::JuMP.Model)::Vector{JuMP.VariableRef}
    return filter((var) -> !JuMP.is_parameter(var), JuMP.all_variables(model))
end

"""
    get_player_id(model::IPGModel, player::Union{String, Int})::Int

Return the player id associated with the player `player` from the IPGModel `model`.
"""
function get_player_id(model::IPGModel, player::String)::Int
    if isempty(model.name_to_id_map)
        throw(ArgumentError("This model does not have named players"))
    end
    if !haskey(model.name_to_id_map, player)
        throw(ArgumentError("$(player) is an unknown player name"))
    end

    return model.name_to_id_map[player]
end
function get_player_id(model::IPGModel, player::Int)::Int
    if player < 1 || player > length(model.players)
        throw(ArgumentError("Player is invalid"))
    end

    return player
end