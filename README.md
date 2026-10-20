# Integer Programming Games

[![Build Status](https://github.com/chair-dsgt/IPG.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/chair-dsgt/IPG.jl/actions/workflows/CI.yml?query=branch%3Amain)

IPG.jl is a Julia package for modeling and solving integer programming games. It extends JuMP in order to provide a simple interface for defining player variables, constraints and payoffs. Equilibrium computation algorithms are integrated using the solver interface provided by this package.

## Requirements
IPG.jl requires Julia to be installed. All Julia package dependencies are automatically downloaded during the installation.

## Installation
```julia
import Pkg
Pkg.add(url = "https://github.com/chair-dsgt/IPG.jl.git")
```

## Modeling
An integer programming game is represented by an `IPGModel`. The number of players or a list of `String` player names must be provided when constructing the model. The equilibrium solver can be provided as well, but it can also be set later using `set_solver!`. 

To create a player's variables, constraints, and payoff function, the `@variable`, `@constraint`, and `@payoff` macros are used. These macros use the same syntax as JuMP's `@variable`, `@constraint`, and `@objective` macros, respectively, with an additional argument identifying the player number or name.

The following example is adapted from [Carvalho et al. (2022)](https://www.sciencedirect.com/science/article/abs/pii/S0377221722002727?via%3Dihub):
```julia
using IPG

model = IPGModel(2)

@variable(model, 1, x1)
@constraint(model, 1, x1 >= 0)

@variable(model, 2, x2)
@constraint(model, 2, x2 >= 0)

@payoff(model, 1, Max, -x1 * x1 + x1 * x2)
@payoff(model, 2, Max, -x2 * x2 + x2 * x1)
```

## Solving
To solve an `IPGModel`, an equilibrium solver must have been set beforehand. Equilibrium solvers must be subtypes of `EquilibriumSolver` and must implement the solver interface provided by this package. Once a solver has been provided, either when constructing the model or using `set_solver!`, the integer programming game can be solved using `find_equilibrium!` or `find_all_equilibria!`. The `find_equilibrium!` function searches for one equilibrium and the `find_all_equilibria!` function searches for all equilibria of the game. To use the latter, the equilibrium solver must support equilibria enumeration. 

After the search, its termination status can be accessed by calling `termination_status`. The computed equilibrium or equilibria list can be accessed by calling `equilibrium` or `equilibria`. 

The model defined above can be solved using the SGM.jl equilibrium solver:
```julia
using SGM
using Gurobi

set_solver!(model, SGM.EquilibriumSolver(Gurobi.Optimizer))

find_equilibrium!(model)
```

The results can then be queried:
```julia
julia> termination_status(model)
IPG.EQUILIBRIUM_FOUND

julia> equilibrium(model)
IPG.Strategy[
    IPG.MixedStrategy(1, [1.0],
        IPG.PureStrategy[
            IPG.PureStrategy(1, Dict(x1 => 0.001220703125))
        ],
    ),
    IPG.MixedStrategy(2, [1.0],
        IPG.PureStrategy[
            IPG.PureStrategy(2, Dict(x2 => 0.0006103515625))
        ],
    ),
]
```
The returned equilibrium profile contains one mixed strategy per player, where each player assigns probability 1.0 to a pure strategy who's variable value is near 0.

## Supported games
IPG.jl can represent integer programming games in which each player controls continuous, binary or integer variables. The constraints of each player can only contain variables controlled by that player. A player's payoff function can depend on any player variables and may be affine or quadratic functions.