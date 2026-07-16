mutable struct IPGModel
    players::Vector{JuMP.Model} 
    # Each player has a dictionary that maps another player's variable to the parameter representing it in this player's model.
    var_to_param_maps::Vector{Dict{JuMP.VariableRef, JuMP.VariableRef}} 

    optimizer::IPGOptimizer
end

"""
    get_player_variables(model::JuMP.Model)::Vector{JuMP.VariableRef}

Get all variables from `model`, excluding parameters.
"""
function get_player_variables(model::JuMP.Model)::Vector{JuMP.VariableRef}
    return filter((var) -> !JuMP.is_parameter(var), JuMP.all_variables(model))
end