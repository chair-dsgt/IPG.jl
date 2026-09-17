"""
Invasive-species surveillance game over a lake network (the "SELFISH" game).

Each county chooses which of its lakes to inspect, under a budget, and is credited for the
boater traffic on the arcs leaving it whose endpoints are covered.

# Adaptation note: the coupling constraint was projected out

The original formulation gives county `c` a second family of binary variables `y_a`, one per
arc `a` leaving `c`, tied to the inspection decisions by

    y_a <= x_tail(a) + x_head(a),     a in arcs_minus_c[c],

with payoff `sum(t[a] * n[a] * y_a)`. Since `tail(a)` is a lake of `c` and `head(a)` is a
lake of another county, that constraint contains a variable the player does not control.
IPG.jl does not allow this: each player is a separate `JuMP.Model`, so `@constraint` rejects
foreign variables (`VariableNotOwned`), and the variable-to-parameter mechanism that lets
`@payoff` mix players applies to payoffs only. The old implementation worked around it with
`IPG.internalize_expr`, which has no counterpart here.

The constraint can be removed without changing the game. Informally: `y` is a pure auxiliary
variable that the player always wants to push to its upper bound, and that bound is an OR of
two binaries, which is quadratic. Formally, fix the opponents' inspections `x_{-c}`. The
coefficients `t[a] * n[a]` are nonnegative, so an optimal `y` is
`y_a = min(1, x_tail(a) + x_head(a))`, and since `x` is binary that equals
`x_tail(a) + x_head(a) - x_tail(a) * x_head(a)`. County `c`'s optimal value is therefore

    max over feasible x_c of  sum_a t[a] * n[a] * (x_tail(a) + x_head(a) - x_tail(a) * x_head(a)),

with the same maximizers in `x_c`. Note that the original model is a *generalized* Nash
equilibrium problem — the coupling constraint makes `c`'s feasible set depend on `x_{-c}` —
but the coupling only bounds `y` from above and `y = 0` is always feasible, so the
best-response correspondence in `x_c` is exactly the one of the payoff above. The two games
therefore have the same equilibria in `x`.

The resulting payoff is quadratic, and each of its quadratic terms pairs one variable of `c`
with one variable of a single other player, so `SGM.is_pairwise_payoff` holds and the sampled
games stay polymatrix games.

# Notes
- This script needs the `EBMC_generated` dataset next to it, plus `CSV`, `DataFrames` and
  `PyCall` (with a working Python) to read it. Only the game-definition section below was
  changed; the data-loading section is the original one.
- SCIP can be replaced with other solvers, such as Gurobi or CPLEX. It is only used for the
  best-response problems. SGM solves each sampled game through NormalGames.jl, which builds
  Gurobi models internally, so a Gurobi licence is needed to run this script end to end.
"""

using CSV
using DataFrames
using PyCall

py"""
import pickle
 
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""

load_pickle = py"load_pickle"

# these parameters must match those in the csv file name, and the folder from which they are downloaded
type_dataset = "multi"
county_size = 5
num_lakes_per_county = 50
budget_ratio = 0.5

dirname = "EBMC_generated/$(type_dataset)_dataset/"
fname = "$(county_size)_$(num_lakes_per_county)_$(budget_ratio).csv"
df_edge = DataFrame(CSV.File(dirname * fname))

info_data = load_pickle(dirname * "info_data.pickle")

# === Unpack Experiment Settings === #
# extract the value list for the (county_size, num_lakes_per_county, budget_ratio) key
vals = info_data[(county_size, num_lakes_per_county, budget_ratio)]

# values come from Python (0-based originally) so use 1-based Julia indices
counties = String[string(c) for c in vals[1]]   # `IPGModel` names players with a Vector{String}
num_lakes_per_county = Int(vals[2])         # ensure it's an Int
infestation_status = vals[3]             # likely a Python dict
county_budget = vals[4]

# determine infested lakes (any value > 0 in the nested dict)
infested_lakes = String[]
for (key, infestation_vals) in infestation_status
    if any(v -> v > 0, values(infestation_vals))
        push!(infested_lakes, string(key))
    end
end

# === lakes and lake->county mapping === #
lakes = unique(vcat(df_edge[:, :dow_origin], df_edge[:, :dow_destination]))
lake_county = Dict(lake => lake[1:2] for lake in lakes)   # first two chars like Python's [:2]

# === Compute Lake Weights === #
w = Dict{String, Float64}(lake => 0.0 for lake in lakes)
for row in eachrow(df_edge)
    if row[:bij] != 0
        ori = string(row[:dow_origin])
        dst = string(row[:dow_destination])
        w[ori] += row[:bij] * row[:weight]
        w[dst] += row[:bij] * row[:weight]
    end
end

# === Set Model Parameters === #
I = lakes

I_c = Dict(county => [i for i in I if i[1:2] == county] for county in counties)
I_c_complement = Dict(county => [i for i in I if !(i in I_c[county])] for county in counties)

# build arc dictionaries n (weight) and t (bij)
n = Dict{Tuple{Any,Any}, Float64}()
t = Dict{Tuple{Any,Any}, Float64}()
for row in eachrow(df_edge)
    arc = (row[:dow_origin], row[:dow_destination])
    n[arc] = row[:weight]
    t[arc] = row[:bij]
end

arcs = collect(keys(n))

# arcs within, incoming to, and outgoing from each county
arcs_c = Dict{Any, Vector{Tuple{Any,Any}}}()
arcs_plus_c = Dict{Any, Vector{Tuple{Any,Any}}}()
arcs_minus_c = Dict{Any, Vector{Tuple{Any,Any}}}()

for county in counties
    arcs_c[county] = [arc for arc in arcs if (arc[1][1:2] == county) && (arc[2][1:2] == county)]
    arcs_plus_c[county] = [arc for arc in arcs if (arc[2][1:2] == county) && (arc[1][1:2] != county)]
    arcs_minus_c[county] = [arc for arc in arcs if (arc[1][1:2] == county) && (arc[2][1:2] != county)]
end

# === Define and Solve SELFISH Game using IPG.jl === #

using IPG, SGM, SCIP
import IPG.JuMP  # imported, not `using`, so that JuMP's macros do not shadow IPG's

# one player per county, addressed by name
model = IPGModel(counties)

# add variables
x_c = Dict{String, Any}()
for county in counties
    x_c[county] = @variable(model, county, [I_c[county]], Bin, base_name="x_$(county)_")
end

# a single lake -> variable lookup across all players, so that a county's payoff can name the
# lakes of the counties it sends boaters to
x = Dict{eltype(I), JuMP.VariableRef}()
for county in counties, i in I_c[county]
    x[i] = x_c[county][i]
end

for county in counties
    ### add constraints
    # Only the budget constraint is left: see the adaptation note above for why the
    # `y_a <= x_tail(a) + x_head(a)` family, which mixed players, could be projected out.
    @constraint(model, county, sum(x_c[county]) <= county_budget[county])

    ### set payoff
    # t[arc] * n[arc] * (x_tail + x_head - x_tail * x_head): the traffic on `arc` is credited
    # to `county` when either endpoint is inspected. `x[arc[2]]` belongs to another player,
    # which `@payoff` handles by substituting a parameter of this player's problem.
    @payoff(model, county, Max, sum(
        t[arc] * n[arc] * (x[arc[1]] + x[arc[2]] - x[arc[1]] * x[arc[2]])
        for arc in arcs_minus_c[county];
        init = zero(JuMP.QuadExpr)   # a county may have no outgoing arc
    ))
end

set_solver!(model, SGM.EquilibriumSolver(SCIP.Optimizer))

find_equilibrium!(model; max_iter=10)

# ==== Results ====
println("termination status: ", termination_status(model))

if termination_status(model) == IPG.EQUILIBRIUM_FOUND
    for county in counties
        inspected = [i for i in I_c[county] if value(model, x[i]) > 0.5]
        println("county $county: payoff = ", payoff_value(model, county),
                ", inspects ", length(inspected), " lakes: ", inspected)
    end
end
