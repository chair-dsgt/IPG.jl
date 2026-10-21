"""
Selfish EBMC game from Lee et al., *Random-Restart Best-Response Dynamics for Large-Scale
Integer Programming Games and Their Applications* (February 24, 2026):
https://optimization-online.org/wp-content/uploads/2025/06/IPG_preprint_OO.pdf

The model implements §5.2, Definition 1 and (11), with the quadratic IPG reformulation of
Appendix I. Full utilities retain the opponent-only term c_i(x^{-i}) for welfare reporting.
The default instance is Table A11's multi-AIS row n=5, BG=0.5 (50 lakes per county, §7.2).
Its reference PNE welfare is 557.6 in thousands, rounded to one decimal place (§7.1).

This example starts from the all-zero profile (§7.1) and uses SGM with SCIP best responses.
It compares equilibrium welfare with Table A11; it does not replicate §7.1's RRR-BRD
algorithm, Gurobi 11.0.3 / 16-thread setup,
reusable best-response models, or timing and round counts. A mixed result cannot be used
as a reproduction of that table's PNE result.

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
- SCIP solves best responses; NormalGames uses Gurobi for sampled games. A working Gurobi
  licence is required (for example, configured with `GRB_LICENSE_FILE`).
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

# PyCall converts the stored Python tuple to a Julia tuple (indexed from 1).
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
# Include isolated lakes recorded in the metadata, even if they have no risky CSV arcs.
lakes = unique(vcat(
    String.(df_edge[:, :dow_origin]), String.(df_edge[:, :dow_destination]),
    String.(collect(keys(infestation_status))),
))
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

# Check the synthetic-instance construction against §7.2 and the supplied metadata.
@assert length(counties) == county_size
@assert sum(length(I_c[c]) for c in counties) == length(I)
for county in counties
    @assert length(I_c[county]) == num_lakes_per_county
    expected_budget = floor(Int, budget_ratio * count(i -> i in infested_lakes, I_c[county]))
    @assert county_budget[county] == expected_budget
end

# build arc dictionaries n (weight) and t (bij)
n = Dict{Tuple{Any,Any}, Float64}()
t = Dict{Tuple{Any,Any}, Float64}()
for row in eachrow(df_edge)
    arc = (string(row[:dow_origin]), string(row[:dow_destination]))
    @assert !haskey(n, arc) "Duplicate origin-destination arc: $arc"
    @assert isfinite(row[:weight]) && row[:weight] >= 0
    origin_status = infestation_status[arc[1]]
    destination_status = infestation_status[arc[2]]
    risk = sum(origin_status[s] * (1 - destination_status[s]) for s in keys(origin_status))
    @assert row[:bij] == risk "Risk weight disagrees with infestation metadata: $arc"
    @assert string(row[Symbol("county_name.origin")]) == lake_county[arc[1]]
    @assert string(row[Symbol("county_name.destination")]) == lake_county[arc[2]]
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

# Initialize with the all-zero profile (0^N), as discussed in §7.1 and reported in Table A11.
x_c = Dict{String, Any}()
for county in counties
    x_c[county] = @variable(model, county, [I_c[county]], Bin, start=0.0, base_name="x_$(county)_")
end

# A single lake -> variable lookup across all players, so that a county's payoff can name the
# origins in other counties that send boaters to its lakes.
x = Dict{eltype(I), JuMP.VariableRef}()
for county in counties, i in I_c[county]
    x[i] = x_c[county][i]
end

for county in counties
    ### add constraints
    # Independent binary strategy set, (9c), (9e); coverage variables are eliminated
    # according to Appendix I.
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

    # Table A11 concerns pure equilibria. Validate the actual support before comparison.
    profile = equilibrium(model)
    is_pure = all(s -> s isa PureStrategy || length(s.supp) == 1, profile)
    if is_pure
        pure_profile = PureStrategy[s isa PureStrategy ? s : only(s.supp) for s in profile]
        selected = Set(i for i in I if value(pure_profile, x[i]) > 0.5)
        for county in counties
            @assert all(i -> isapprox(value(pure_profile, x[i]), round(value(pure_profile, x[i])); atol=1e-6), I_c[county])
            @assert count(i -> i in selected, I_c[county]) <= county_budget[county]
            covered_traffic = sum(n[a] * t[a] for a in arcs_minus_c[county] if a[1] in selected || a[2] in selected; init=0.0)
            @assert isapprox(covered_traffic, payoff_value(model, county); atol=dev_tol)
        end
        direct_welfare = sum(n[a] * t[a] for a in arcs if a[1] in selected || a[2] in selected; init=0.0)
        @assert isapprox(direct_welfare, social_welfare; atol=dev_tol)

        # Re-solve all deviations to optimality, independently of SGM's stopping test.
        regrets = Float64[]
        for (id, county) in enumerate(counties)
            others = PureStrategy[s for s in pure_profile if s.id != id]
            br, _ = get_best_response_model(model, county, others, SCIP.Optimizer)
            JuMP.set_silent(br)
            JuMP.set_optimizer_attribute(br, "limits/gap", 0.0)
            JuMP.set_optimizer_attribute(br, "limits/absgap", 0.0)
            JuMP.optimize!(br)
            @assert JuMP.termination_status(br) == JuMP.MOI.OPTIMAL
            push!(regrets, max(0.0, JuMP.objective_bound(br) - payoff_value(model, county)))
        end
        @assert maximum(regrets) <= dev_tol
        println("pure-equilibrium certificate: maximum unilateral gain <= ", maximum(regrets))
        if (type_dataset, county_size, num_lakes_per_county, budget_ratio) == ("multi", 5, 50, 0.5)
            matches = round(social_welfare / 1000; digits=1) == 557.6
            println("Table A11 (n=5, BG=0.5): reference 557.6; rounded welfare matches: ", matches)
        end
    else
        println("Mixed equilibrium: no comparison with Table A11's PNE column.")
    end
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
