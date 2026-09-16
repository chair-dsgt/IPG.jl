# IPG.jl test suite

## Where a test belongs

IPG.jl is the modeling layer; SGM.jl is one implementation of the
`IPG.EquilibriumSolver` interface, in the relationship JuMP.jl has with
Gurobi.jl. The boundary rule for the two suites is:

> A test belongs in IPG.jl **iff it stays meaningful with SGM.jl deleted.**

So IPG.jl owns model construction, the `@variable`/`@constraint`/`@payoff`
macros, the variable-to-parameter machinery, player-name resolution, `payoff`,
`best_response`, the strategy types, and the *delegation* behaviour of
`find_equilibrium!`/`equilibrium`/`termination_status`. Whether a particular
algorithm actually finds an equilibrium is SGM.jl's business and is tested there.

This suite must pass with no solver package installed beyond HiGHS, and with
SGM.jl absent from the environment. If a new test needs SGM, it is in the wrong
package.

The suite uses [TestItemRunner](https://github.com/julia-vscode/TestItemRunner):
each `@testitem` is independent and shared fixtures live in `test_utils.jl` as
`@testsnippet` blocks. Run it with `julia --project=. -e 'using Pkg; Pkg.test()'`.

## Fixtures (`test_utils.jl`)

| Snippet | Provides |
|---|---|
| `IPGFixtures` | The four game fixtures, all built through the IPG macros and defaulting to **no** equilibrium solver. |
| `MockSolver` | `MockEquilibriumSolver <: IPG.EquilibriumSolver`, returning a profile given at construction and recording the `enumeration` flag of each `search!`. Lets the accessor layer be tested with no algorithm and no optimizer. |
| `HiGHSOptimizer` | A silent HiGHS factory for the best-response solves. |

## What each test addresses

### `model.jl` — construction and the parameter map

| Test item | Covers |
|---|---|
| IPGModel construction by player count | `IPGModel(n)`: player vector, empty maps, no attached solver |
| IPGModel construction by player name | `IPGModel(player_names)` and the name → id map it builds |
| IPGModel constructor rejects invalid player specifications | The `ArgumentError` guards: `n <= 0`, empty and duplicate name vectors |
| get_player_id resolves ids and names | `get_player_id` for `Int` and `String`, plus all three rejection paths |
| @variable and @constraint attach to the named player | `IPG.@variable`/`IPG.@constraint` routing to the right player's JuMP model; start-value propagation |
| **Naming players does not change the constructed model** | That a named model and an id-addressed model build the same game — same variables, start values, constraints, parameter map and payoff — differing only in `name_to_id_map`. See "Naming is IPG's concern" below |
| get_player_variables excludes parameters | `get_player_variables` filtering parameters out of `all_variables` |
| **@payoff builds the variable-to-parameter map** | The `get_parameter!` contract: key ownership, parameter ownership, naming, initial value, idempotence, and that a player's own variable is returned unchanged |
| Variable-to-parameter map covers every shared variable | One parameter per distinct opponent variable, on a game with three per player |
| Three-player model wires parameters across all players | Parameter wiring when players depend on different subsets of each other; bounds and integrality |

### `solver_interface.jl` — delegation and accessors

| Test item | Covers |
|---|---|
| Accessors require an equilibrium solver | The no-solver `ArgumentError` guards, and that `termination_status` alone works without one |
| set_solver! attaches a solver and routes termination_status | `set_solver!` and status delegation |
| find_equilibrium! and find_all_equilibria! set the enumeration flag | That each forwards to `search!` with the right `enumeration` value |
| equilibrium and equilibria read through to the solver | `equilibrium(model)`, `equilibrium(model, i)`, `equilibria(model)`, and out-of-bounds rejection |
| Named and indexed accessors agree | The `Union{Int, String}` dispatch contract across `equilibrium`, `payoff` and `payoff_value` |
| payoff rejects malformed profiles | The one-strategy-per-player check |
| payoff evaluates a pure profile against the player's objective | `compute_payoff_at_profile` against hand-computed values |
| value reads pure, mixed and model-level strategies | Every `value()` form, including the mixed-strategy expectation |

### `best_response.jl` — the single-player problem

| Test item | Covers |
|---|---|
| get_best_response_model copies the player problem | That it copies rather than mutates, and that profile values reach the copy's parameters |
| Objective shape differs between the pure and mixed branches | The parameter-substitution asymmetry documented below; no solve, so it holds under any optimizer |
| best_response matches the closed-form optimum | `best_response` on pure profiles, against the analytic map `x_j / 2` and its fixed point |
| best_response against a mixed profile uses the expected payoff | The expected-payoff branch |
| best_response on a binary game | Best responses over binaries (rock → paper, and the uniform mix) |
| best_response accepts a player name | The `Union{Int, String}` player argument |

## Naming is IPG's concern

Whether players carry names is invisible to a solver: `SGM.jl/src` never
references `name_to_id_map`, `get_player_id`, or a `String` player, and addresses
players exclusively by integer id. A named model and an id-addressed model are
therefore the *same* input to any equilibrium algorithm, and re-running a solver
over a named game proves nothing about the solver.

So the whole question lives here, split across four test items:

- **Naming players does not change the constructed model** — the models are
  structurally and numerically identical apart from `name_to_id_map`.
- **get_player_id resolves ids and names** — the name → id mapping and its
  rejection paths.
- **Named and indexed accessors agree** — `equilibrium`, `payoff`, `payoff_value`
  give the same answer for `"Player1"` as for `1`.
- **best_response accepts a player name** — likewise for `best_response`.

## Known limitation: HiGHS and parameters in a quadratic objective

`get_best_response_model` handles the two profile kinds differently:

- **Pure profiles** set the copy's parameter *values* but leave the parameters in
  the objective. A payoff bilinear in own × opponent variables therefore stays a
  `ScalarQuadraticFunction`, and HiGHS rejects a quadratic objective whose
  quadratic terms involve parameters — even though fixing them would make the
  problem a plain MILP. Rock-paper-scissors is exactly that shape.
- **Mixed profiles** go through `replace_param_with_val`, which substitutes the
  values out, so the objective HiGHS receives is parameter-free and solvable.

Consequently the binary-game tests here reach the solver through degenerate
mixed strategies (probability 1 on a single pure strategy), and the
*pure-profile* binary best response is exercised in SGM.jl under Gurobi instead.
The "Objective shape differs between the pure and mixed branches" test pins the
asymmetry down directly so it cannot regress unnoticed.

This is a portability limit of the models IPG hands to solvers, not a property of
HiGHS alone: any solver that does not accept parameters inside a quadratic
objective is affected. Gurobi does accept them, which is why SGM.jl's suite has
never hit this.
