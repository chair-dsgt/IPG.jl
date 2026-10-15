"""
Two-player Competitive Facility Location and Design (CFLD) problem.

Based on Crönert and Minner (2024). For further details, see Section 3.2 of:
    [*T. Crönert, S. Minner, "Equilibrium Identification and Selection in Finite Games". 2024. Operations Research*](https://doi.org/10.1287/opre.2022.2413)

# Status: this game cannot currently be expressed with IPG.jl

The players' payoffs are *rational* functions of the two players' variables (a market-share
ratio), not affine or quadratic ones. IPG.jl stores a payoff by routing it through
`prepare_payoff`, which accepts `Real`, `VariableRef`, `AffExpr` and `QuadExpr` only, so the
`@payoff` calls at the bottom of this file throw

    ArgumentError: Payoff type JuMP.NonlinearExpr is not supported

This is a restriction of the *modeling* layer, not of the algorithm. A sampled game is built
by *evaluating* payoffs at pure profiles — `SGM.get_polymatrix_twoplayers` calls
`IPG.comparable_payoff_value` — which needs no particular functional form, and this is how
the original implementation handled this instance. SGM.jl reaches that code path only after
`SGM.is_quad_payoff`, though, and sends anything that is not a `QuadExpr` to a normal-form
sampled game instead. Either way the search itself would run; what blocks it is that the
payoff cannot be stored on the model in the first place.

The file is kept as a faithful adaptation so that the gap is visible and so that the example
starts working as soon as IPG.jl accepts nonlinear payoffs. Everything above the payoff
section — the data, both players' variables and both budget/assignment constraints — builds
without error.

A second, smaller difference: the original selected the initialization subroutine
`IPG.initialize_strategies_player_alone`. SGM.jl no longer offers a choice: it exposes a
single `initialize_strategies`, which uses the variables' `start` values when every variable
has one and solves a feasibility problem otherwise.

# Notes
- SCIP can be replaced with other solvers, such as Gurobi or CPLEX.
"""

using IPG, SGM, SCIP

# ==== Problem Parameters ====
d_max = 20
β = 0.5
B = 40  # B = B^A = B^B

# ==== Problem Data ====
a = [
    3.00 7.31 14.35;
    3.00 7.14 10.41;
    3.00 4.43 6.25;
    3.00 7.21 12.57;
    3.00 4.16 11.50;
    3.00 4.36 6.77;
    3.00 6.63 13.63;
    3.00 4.89 6.16;
    3.00 6.93 12.90;
    3.00 5.95 9.82;
    3.00 5.22 8.19;
    3.00 4.65 13.12;
    3.00 5.90 13.17;
    3.00 5.50 7.50;
    3.00 7.74 9.30;
    3.00 5.01 6.05;
    3.00 5.35 10.35;
    3.00 4.76 7.15;
    3.00 5.65 9.19;
    3.00 5.61 14.32;
    3.00 8.97 12.93;
    3.00 5.86 13.04;
    3.00 5.98 14.29;
    3.00 4.60 11.48;
    3.00 4.28 9.89;
    3.00 4.26 5.31;
    3.00 6.01 11.17;
    3.00 7.14 9.95;
    3.00 5.98 9.19;
    3.00 4.76 13.25;
    3.00 6.11 8.19;
    3.00 5.93 9.93;
    3.00 5.34 13.93;
    3.00 4.94 8.32;
    3.00 4.37 8.58;
    3.00 4.51 8.58;
    3.00 5.62 8.16;
    3.00 4.56 10.33;
    3.00 6.56 7.92;
    3.00 7.92 16.48;
    3.00 6.98 12.84;
    3.00 5.81 11.57;
    3.00 7.40 12.45;
    3.00 4.80 11.05;
    3.00 6.62 8.61;
    3.00 8.93 11.76;
    3.00 7.38 14.59;
    3.00 8.66 11.27;
    3.00 5.49 8.06;
    3.00 5.23 11.40
]
f = [
    12.70 25.75 60.05;
    13.67 33.94 45.15;
    13.20 11.91 17.13;
    13.03 37.58 46.85;
    14.58 27.63 24.94;
    10.26 9.96 32.24;
    10.49 25.28 35.73;
    13.07 19.89 28.53;
    16.65 26.25 66.88;
    14.86 34.10 54.43;
    12.90 20.61 47.22;
    12.50 24.02 62.47;
    18.76 24.12 59.14;
    19.63 28.00 42.23;
    23.37 36.27 43.87;
    19.28 20.00 28.22;
    11.69 35.63 48.83;
    11.17 29.47 22.53;
    10.74 22.22 20.72;
    17.06 21.77 72.05;
    13.30 32.47 54.42;
    10.90 20.49 46.20;
    12.94 22.73 62.78;
    12.30 17.13 21.80;
    22.82 29.16 34.72;
    20.50 13.67 25.78;
    13.32 20.25 35.01;
    17.04 31.14 45.76;
    18.18 18.25 41.56;
    13.33 26.81 28.67;
    8.18 39.76 46.08;
    9.30 18.06 52.03;
    10.58 19.97 52.17;
    15.18 18.75 32.50;
    15.08 26.37 16.99;
    10.03 17.78 26.01;
    18.82 16.54 24.26;
    14.08 22.27 32.44;
    13.92 32.62 33.39;
    16.08 17.31 47.45;
    15.29 32.82 51.17;
    18.97 34.00 26.90;
    17.51 43.23 67.03;
    15.12 29.63 44.91;
    18.69 33.23 29.94;
    16.75 36.22 65.61;
    18.76 45.20 39.84;
    11.16 26.68 47.90;
    18.34 29.54 26.95;
    18.25 23.82 50.04
]
locs = [
    (81.423, 63.358),
    (23.986, 24.907),
    (58.360, 94.707),
    (70.994, 95.909),
    (16.525, 18.016),
    (33.446, 55.179),
    (35.339, 61.412),
    (82.131, 82.424),
    (17.120, 30.418),
    (89.266, 36.745),
    (22.486, 28.671),
    (50.849, 62.173),
    (66.774, 83.822),
    (34.897, 22.596),
    (63.452, 8.297),
    (3.999, 8.276),
    (25.426, 49.606),
    (85.620, 81.129),
    (60.199, 9.892),
    (88.603, 50.964),
    (19.818, 39.411),
    (89.291, 99.867),
    (91.544, 33.614),
    (35.724, 96.915),
    (40.176, 88.541),
    (43.333, 70.414),
    (37.977, 94.200),
    (80.580, 29.840),
    (43.137, 70.615),
    (67.976, 54.672),
    (12.704, 76.245),
    (38.914, 92.317),
    (60.930, 43.457),
    (35.754, 32.293),
    (98.959, 20.722),
    (39.608, 48.879),
    (16.806, 26.254),
    (57.243, 29.399),
    (93.555, 42.752),
    (13.555, 30.350),
    (46.719, 7.948),
    (23.052, 28.537),
    (56.692, 80.636),
    (38.770, 59.428),
    (27.251, 86.887),
    (2.071, 59.893),
    (68.009, 76.120),
    (41.762, 57.527),
    (9.797, 41.724),
    (95.959, 73.750)
]
w = [3, 9, 6, 4, 4, 3, 4, 9, 2, 6, 10, 6, 10, 8, 2, 7, 2, 3, 7, 5, 4, 4, 2, 2, 6, 8, 3, 7, 8, 4, 2, 6, 2, 9, 3, 4, 6, 8, 7, 5, 5, 2, 4, 1, 4, 3, 7, 6, 8, 4]

I = J = 1:size(a,1)
R = 1:size(a,2)

# It is not clear how the distance matrix is computed. In the source, they say that "The
# matrix d_ij is the Euclidean distance matrix", but that does not give me the info on the
# location of the facilities, as the .txt only has the location of the customers.

# I assume that the facilities' locations are the same as the customers
euclidean_distance(x,y) = sqrt((x[1]-y[1])^2 + (x[2]-y[2])^2)
d = [euclidean_distance(locs[i], locs[j]) for i in I, j in J]

# utility computation
CM_utility(i,j,r) = (d[i,j] <= d_max) ? a[i,r]/((d[i,j]+1)^β) : 0
u = [CM_utility(i,j,r) for i in I, j in J, r in R]

# ==== Model Definition ====
model = IPGModel(["Player A", "Player B"])

@variable(model, "Player A", x1[I,R], Bin)
@constraint(model, "Player A", sum(f[i,r] .* x1[i,r] for i in I for r in R) <= B)
@constraint(model, "Player A", [i in I], sum(x1[i,r] for r in R) <= 1)

@variable(model, "Player B", x2[I,R], Bin)
@constraint(model, "Player B", sum(f[i,r] .* x2[i,r] for i in I for r in R) <= B)
@constraint(model, "Player B", [i in I], sum(x2[i,r] for r in R) <= 1)

const ε = 1e-3  # small value to prevent division by zero
function cfld_payoff(x_self, x_other)
    self_costs = [sum(u[i,j,r] * x_self[i,r] for i in I for r in R) for j in J]
    others_costs = [sum(u[i,j,r] * x_other[i,r] for i in I for r in R) for j in J]
    return sum(
        w[j] * self_costs[j] / (self_costs[j] + others_costs[j] + ε)
        for j in J
    )
end

# Both calls below throw `ArgumentError: Payoff type JuMP.NonlinearExpr is not supported`.
# See the "Status" section of the docstring above.
@payoff(model, "Player A", Max, cfld_payoff(x1, x2))
@payoff(model, "Player B", Max, cfld_payoff(x2, x1))

# ==== SGM ====
set_solver!(model, SGM.EquilibriumSolver(SCIP.Optimizer))

find_equilibrium!(model; max_iter=5)

# ==== Results ====
println("termination status: ", termination_status(model))

if termination_status(model) == IPG.EQUILIBRIUM_FOUND
    println("payoff A = ", payoff_value(model, "Player A"))
    println("payoff B = ", payoff_value(model, "Player B"))
end
