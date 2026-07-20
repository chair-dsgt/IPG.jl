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
    strategy(profile::Profile, player::Int)::Strategy

Get the strategy of player `player` in the profile `profile`.
"""
function strategy(profile::Profile, player::Int)::Strategy
    for strat in profile
        if strat.id == player
            return strat
        end
    end
    throw(ArgumentError("No strategy found for player $(player)"))
end


"""
    empty_strategy_sample(num_players::Int)

Create an empty strategy sample for `num_players` players.

The returned vector contains an empty `PureStrategy` vector per player.
"""
function empty_strategy_sample(num_players::Int)::Vector{Vector{PureStrategy}}
    [PureStrategy[] for _ in 1:num_players]
end