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

end