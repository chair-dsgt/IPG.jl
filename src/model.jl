mutable struct IPGModel
    players::Vector{JuMP.Model} 
    # Each player has a dictionary that maps another player's variable to the parameter representing it in this player's model.
    var_to_param_maps::Vector{Dict{JuMP.VariableRef, JuMP.VariableRef}} 

    optimizer::Union{Nothing, IPGOptimizer}
end

"""
    IPGModel(n::Int[, optimizer::IPGOptimizer])::IPGModel

Create a new instance of an integer programming game model with `n` players.

`n` must be a positive integer.

`optimizer` must be one of the forms:

* Omitted, like `IPGModel(n)`, which creates an IPG model without an attached optimizer
* An IPG optimizer, like `IPGModel(n, SGM.Optimizer(HiGHS.Optimizer, Gurobi.Optimizer))`, where 
`SGM.Optimizer` is the main IPG solution algorithm, `HiGHS.Optimizer` is used to solve the best-response 
problems, and `Gurobi.Optimizer` is used to compute equilibria for the sampled games

If `optimizer` is not provided, use `set_optimizer!` to set the optimizer before calling `find_equilibrium!` or `find_all_equilibria!`.
"""
function IPGModel(n::Int)::IPGModel
    if n <= 0
        throw(ArgumentError("The number of players must be a positive integer"))
    end

    players = [JuMP.Model() for _ in 1:n]
    var_to_param_maps = [Dict{JuMP.VariableRef, JuMP.VariableRef}() for _ in 1:n]

    return IPGModel(players, var_to_param_maps, nothing)
end
function IPGModel(n::Int, optimizer::IPGOptimizer)::IPGModel
    model = IPGModel(n)
    set_optimizer!(model, optimizer)
    return model
end

"""
    set_optimizer!(model::IPGModel, optimizer::IPGOptimizer)::Nothing

Set the IPG optimizer of the model `model` to `optimizer`.
"""
function set_optimizer!(model::IPGModel, optimizer::IPGOptimizer)::Nothing
    model.optimizer = optimizer
    return nothing
end

"""
    get_player_variables(model::JuMP.Model)::Vector{JuMP.VariableRef}

Get all variables from `model`, excluding parameters.
"""
function get_player_variables(model::JuMP.Model)::Vector{JuMP.VariableRef}
    return filter((var) -> !JuMP.is_parameter(var), JuMP.all_variables(model))
end