abstract type IPGOptimizer 
end 

@enum TerminationStatusCode begin
    SOLVE_NOT_CALLED # No equilibrium search has been called yet, or the optimizer was reset before starting a search
    EQUILIBRIUM_FOUND # An equilibrium was found
    ENUMERATION_COMPLETE # The optimizer has finished enumerating equilibria
    NO_EQUILIBRIUM # There exists no equilibrium
    INFEASIBLE # Atleast one player's problem is infeasible or the game doesn't have any feasible strategies
    ITERATION_LIMIT # The optimizer stopped because it reached the maximum number of iterations
    TIME_LIMIT # The optimizer stopped because it reached the maximum time limit
    NODE_LIMIT # The optimizer stopped because a branch-and-bound node limit was reached
    MEMORY_LIMIT # The optimizer stopped because a memory limit was reached
    OTHER_LIMIT # The optimizer stopped because of another limit
    INVALID_MODEL # The model is not valid
    INVALID_OPTION # An invalid option or parameter was provided
    INTERRUPTED # The search was interrupted before completion
    OTHER_ERROR # The optimizer stopped because of another error
end