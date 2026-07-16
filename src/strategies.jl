struct PureStrategy
    id::Int # Player id
    variable_values::Dict{JuMP.VariableRef, Float64}
end

struct MixedStrategy
    id::Int # Player id
    probs::Vector{Float64} # Probability vector
    supp::Vector{PureStrategy} # Vector of pure strategies (support)
    function MixedStrategy(id::Int, probs::Vector{Float64}, supp::Vector{PureStrategy})
        if length(probs) != size(supp, 1)
            error("There must be as many probabilities as strategies in the support.")
        end
        if ~(sum(probs) ≈ 1)
            error("Probabilities must sum to 1.")
        end
        if any(probs .< 0)
            error("Probabilities must be non-negative.")
        end

        # remove strategies with zero probability
        return new(id, probs[probs .> 0], supp[probs .> 0])
    end
end

const Strategy = Union{PureStrategy, MixedStrategy}

const Profile = Vector{T} where T <: Strategy

"""
    empty_strategy_sample(num_players::Int)

Create an empty strategy sample for `num_players` players.

The returned vector contains an empty `PureStrategy` vector per player.
"""
function empty_strategy_sample(num_players::Int)::Vector{Vector{PureStrategy}}
    [PureStrategy[] for _ in 1:num_players]
end