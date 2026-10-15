"""
Invasive-species surveillance game over a lake network (the "SELFISH" game).

Each county chooses which of its lakes to inspect, under a budget, and is credited for the
boater traffic on the arcs entering it, and on arcs within it, whose endpoints are covered.

# Adaptation note: the coupling constraint was projected out

The original formulation gives county `c` a second family of binary variables `y_a`, one per
arc in its induced inbound arc set `arcs_minus_c[c]`, tied to the inspection decisions by

    y_a <= x_tail(a) + x_head(a),     a in arcs_minus_c[c],

with payoff `sum(t[a] * n[a] * y_a)`. For an incoming cross-county arc, its tail is controlled
by another county, so the constraint contains a variable the player does not control. IPG.jl
does not allow this: each player is a separate `JuMP.Model`, so `@constraint` rejects foreign
variables (`VariableNotOwned`), and the variable-to-parameter mechanism that lets `@payoff`
mix players applies to payoffs only. The old implementation worked around it with
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

The resulting payoff is quadratic. Internal-arc terms involve only variables of `c`, while
incoming cross-county terms pair one variable of `c` with one variable of a single other
player. Thus `SGM.is_pairwise_payoff` holds and the sampled games stay polymatrix games.

# Notes
- From the `IPG.jl/examples` directory, instantiate the example environment once:

      julia --project=. -e 'using Pkg; Pkg.instantiate()'

- Download the synthetic data used by the default settings from revision
  `3447789649de75037921d153da44868da328d9e6` of the authors' repository:

      mkdir -p EBMC_generated/multi_dataset
      curl -L -o EBMC_generated/multi_dataset/5_50_0.5.csv \
        https://raw.githubusercontent.com/HyunwooLee0429/Best-response-dynamics-IPG/3447789649de75037921d153da44868da328d9e6/BZR_EBMC/EBMC_generated/multi_dataset/5_50_0.5.csv
      curl -L -o EBMC_generated/multi_dataset/info_data.pickle \
        https://raw.githubusercontent.com/HyunwooLee0429/Best-response-dynamics-IPG/3447789649de75037921d153da44868da328d9e6/BZR_EBMC/EBMC_generated/multi_dataset/info_data.pickle

  By default, data is resolved relative to this file. Set `EBMC_DATA_DIR` to the directory
  containing the CSV and `info_data.pickle` to use data stored elsewhere.
- Run the example with:

      julia --project=. max-cov.jl

- The data loader uses `PyCall` and Python's standard-library `pickle` module. If PyCall has
  not been configured yet, set `ENV["PYTHON"]` to a working Python executable and run
  `Pkg.build("PyCall")` in the example environment.
- SCIP can be replaced with other solvers, such as Gurobi or CPLEX. It is only used for the
  best-response problems. SGM solves each sampled game through NormalGames.jl, which builds
  Gurobi models internally, so a working Gurobi installation and licence are needed to run
  this script end to end. Configure the licence as described by Gurobi (for example via
  `GRB_LICENSE_FILE`) before starting Julia.
- The instance and solve settings can be overridden with `EBMC_TYPE`, `EBMC_COUNTIES`,
  `EBMC_LAKES_PER_COUNTY`, `EBMC_BUDGET_RATIO`, `EBMC_MAX_ITER`, `EBMC_TIME_LIMIT`, and
  `EBMC_DEV_TOL`. The time limit is in seconds; its default is unlimited.
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

# These parameters must match a key in info_data.pickle and the corresponding CSV file name.
type_dataset = get(ENV, "EBMC_TYPE", "multi")
county_size = parse(Int, get(ENV, "EBMC_COUNTIES", "5"))
num_lakes_per_county = parse(Int, get(ENV, "EBMC_LAKES_PER_COUNTY", "50"))
budget_ratio = parse(Float64, get(ENV, "EBMC_BUDGET_RATIO", "0.5"))

max_iter = parse(Int, get(ENV, "EBMC_MAX_ITER", "10"))
time_limit = parse(Float64, get(ENV, "EBMC_TIME_LIMIT", "Inf"))
dev_tol = parse(Float64, get(ENV, "EBMC_DEV_TOL", "1e-3"))

default_data_dir = joinpath(@__DIR__, "EBMC_generated", "$(type_dataset)_dataset")
dirname = abspath(get(ENV, "EBMC_DATA_DIR", default_data_dir))
fname = "$(county_size)_$(num_lakes_per_county)_$(budget_ratio).csv"
csv_path = joinpath(dirname, fname)
info_path = joinpath(dirname, "info_data.pickle")

for path in (csv_path, info_path)
    isfile(path) || error(
        "Required EBMC data file not found: $path. " *
        "See the setup instructions at the top of $(abspath(@__FILE__)).",
    )
end

df_edge = DataFrame(CSV.File(csv_path))

info_data = load_pickle(info_path)

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

# Arc sets from Definition 1 of the paper:
# - arcs_c[c]       = A_c, arcs whose two endpoints are in county c;
# - arcs_minus_c[c] = A^-[I_c], all arcs whose terminal point is in county c;
# - delta_plus_c[c]  = δ^+(I_c), arcs leaving county c.
# Consequently, arcs_minus_c[c] contains A_c ∪ δ^-(I_c), and the union of
# arcs_minus_c[c] and delta_plus_c[c] is the induced neighborhood arc set A[I_c].
arcs_c = Dict{Any, Vector{Tuple{Any,Any}}}()
delta_plus_c = Dict{Any, Vector{Tuple{Any,Any}}}()
arcs_minus_c = Dict{Any, Vector{Tuple{Any,Any}}}()

for county in counties
    arcs_c[county] = [arc for arc in arcs if (arc[1][1:2] == county) && (arc[2][1:2] == county)]
    arcs_minus_c[county] = [arc for arc in arcs if arc[2][1:2] == county]
    delta_plus_c[county] = [arc for arc in arcs if (arc[1][1:2] == county) && (arc[2][1:2] != county)]
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

# A single lake -> variable lookup across all players, so that a county's payoff can name the
# origins in other counties that send boaters to its lakes.
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
    # t[arc] * n[arc] * (x_tail + x_head - x_tail * x_head): the traffic on an internal or
    # incoming arc is credited to `county` when either endpoint is inspected. On an incoming
    # cross-county arc, `x[arc[1]]` belongs to another player; `@payoff` substitutes a
    # parameter of this player's problem for it.
    @payoff(model, county, Max, sum(
        t[arc] * n[arc] * (x[arc[1]] + x[arc[2]] - x[arc[1]] * x[arc[2]])
        for arc in arcs_minus_c[county];
        init = zero(JuMP.QuadExpr)
    ))
end

set_solver!(model, SGM.EquilibriumSolver(SCIP.Optimizer))

println("solving with max_iter=$max_iter, time_limit=$time_limit, dev_tol=$dev_tol")
find_equilibrium!(model; max_iter=max_iter, time_limit=time_limit, dev_tol=dev_tol)

# ==== Results ====
status = termination_status(model)
println("termination status: ", status)

if status == IPG.EQUILIBRIUM_FOUND
    for county in counties
        county_strategy = equilibrium(model; player=county)
        println("county $county: expected payoff = ", payoff_value(model, county))

        if county_strategy isa PureStrategy
            inspected = [i for i in I_c[county] if value(county_strategy, x[i]) > 0.5]
            println("  pure strategy inspects ", length(inspected), " lakes: ", inspected)
        elseif length(county_strategy.supp) == 1 && only(county_strategy.probs) ≈ 1.0
            pure_strategy = only(county_strategy.supp)
            inspected = [i for i in I_c[county] if value(pure_strategy, x[i]) > 0.5]
            println(
                "  pure-equivalent strategy inspects ",
                length(inspected),
                " lakes: ",
                inspected,
            )
        else
            println("  mixed strategy with ", length(county_strategy.supp), " support points:")
            for (prob, pure_strategy) in zip(county_strategy.probs, county_strategy.supp)
                inspected = [i for i in I_c[county] if value(pure_strategy, x[i]) > 0.5]
                println("    probability $prob: inspects ", length(inspected), " lakes: ", inspected)
            end
        end
    end
    social_welfare = sum(payoff_value(model, county) for county in counties)
    println("social welfare: $social_welfare")
    println("social welfare (paper units, thousands): ", round(social_welfare / 1000; digits=1))
elseif status == IPG.ITERATION_LIMIT
    println(
        "No equilibrium was certified before EBMC_MAX_ITER=$max_iter. " *
        "Increase EBMC_MAX_ITER or set EBMC_TIME_LIMIT.",
    )
elseif status == IPG.TIME_LIMIT
    println("No equilibrium was certified before EBMC_TIME_LIMIT=$time_limit seconds.")
else
    println("The equilibrium search ended without a result (status $status).")
end
