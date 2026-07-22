module IPG

using JuMP

include("optimizer.jl")
include("model.jl")
include("strategies.jl")
include("optimizer_interface.jl")
include("variables.jl")
include("constraints.jl")
include("payoff.jl")
include("best_response.jl")
include("find.jl")

export IPGModel, set_optimizer!, IPGOptimizer
export PureStrategy, MixedStrategy, Strategy, Profile
export @variable, @constraint, @payoff
export find_equilibrium!, find_all_equilibria!
export equilibrium, equilibria, termination_status
export payoff_value, payoff
export get_best_response_model, best_response
export value

end