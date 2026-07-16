module IPG

using JuMP

include("optimizer.jl")
include("model.jl")
include("strategies.jl")
include("optimizer_interface.jl")
include("payoff.jl")
include("variables.jl")
include("best_response.jl")

end