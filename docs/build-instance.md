# Building a Small Redispatch Example

## What are we solving?

At each time step, choose one output level for each generator. We want to meet demand, keep costs low, and avoid line overloads. Each choice must follow two operating rules.

This follows the **Optimization problem** section of `Redispatch_QAOA_OnePager.pdf`.

## Input data

The symbols below come from the **Optimization problem** section of the one-page document. The code uses `t` for time, `a` for generator, and `l` for line.

| Input | Code | One-page symbol | Meaning / use |
| --- | --- | --- | --- |
| Number of time steps | `len(demand) = 2` | `T` | Choose generator states at two times |
| Number of generators | `len(outputs) = 2` | `n` | Two controllable resources: A and B |
| Number of states | `len(outputs[0]) = 3` | `k` | Each resource has three operating states |
| Output levels | `outputs = [[0, 10, 20], [0, 10, 20]]` | `p[a,i]` | Output of generator a in state i; used to compute total generation P and line flow L |
| Unit production costs | `unit_costs = [1, 3]` | Used to calculate `C(z)`| Known input: A costs 1 example unit per MW per time step; B costs 3. |
| Demand | `demand = [20, 30]` | `tau[t]` | Target total generation in D: 20 MW first, then 30 MW |
| Line sensitivity | `sensitivity = [[1.0, 0.2]]` | `S[a,l]` | Contribution of each generator to a line's flow; here L = A output + 0.2 * B output |
| Line capacity | `limits = [[19, 19]]` | `M[t,l]` | Threshold used in O; the monitored line has a 19 MW limit at both times |

In mathematical notation, `p[a,i]`, `tau[t]`, `S[a,l]`, and `M[t,l]` mean p with subscripts a,i; tau with subscript t; S with subscripts a,l; and M with subscripts t,l.

The one-page equations become:

```text
P[t]   = sum over generators a of p[a, z[t,a]]
L[t,l] = sum over generators a of S[a,l] * p[a, z[t,a]]
D      = sum over times t of (P[t] - tau[t]) ** 2
O      = sum over times t and lines l of indicator(L[t,l] >= M[t,l])
```

`z[t,a]` is the state we choose, not an input. In the example, states 0, 1, and 2 mean outputs of 0, 10, and 20 MW; the one-page numbers states from 1 to k. The code stores sensitivity as `[line][generator]` and capacity as `[line][time]`, the reverse of the one-page index order.

## How is C(z) calculated?

C(z) is the total production cost for schedule z. For each generator and time step, multiply its selected output by its unit production cost, then add all costs:

```text
C(z) = sum over t and a of unit_costs[a] * outputs[a][z[t,a]]
```

For `z = [(2, 0), (2, 1)]`, the outputs are (20, 0) MW first and (20, 10) MW second. With `unit_costs = [1, 3]`:

```text
C(z) = (1 * 20 + 3 * 0) + (1 * 20 + 3 * 10) = 70
```

The objective terms, two constraints, and their existing QUBO functions are explained in [Objectives, Constraints, and QUBO Functions](objectives-and-qubo.md).

## Files and reusable inputs

| File | Role |
| --- | --- |
| `examples/instances/small.json` | Original two-time input data and weights |
| `examples/instances/three_steps.json` | Another example: three times and two lines |
| `examples/redispatch_instance.py` | Reusable `RedispatchInstance`: input validation, objective evaluation, and constraint checks |
| `examples/classical_solver.py` | Enumerates feasible schedules and finds weighted global minimal |
| `examples/small_instance.py` | Loads a JSON file and prints results |

`RedispatchInstance.evaluate(states)` returns C/W/O/D. `is_feasible(states)` checks valid state selections and ramping directly. Each state entry selects exactly one level, so the classical search does not need a binary one-hot conversion.

The example still does not call RedispatchModel, RedispatchQUBO, or QAOA.

## Run any small instance

From the repository root, provide a JSON input file (required):

```bash
python examples/small_instance.py examples/instances/small.json
```

Copy either JSON file to create your own input. Edit `outputs`, `unit_costs`, `demand`, `sensitivity`, `limits`, and the named `weights` cases. The input table above uses the same field names as the JSON files.

- Generators = number of rows in `outputs`; all generators must have the same number of states.
- Time steps = length of `demand`.
- Lines = number of rows in `sensitivity`; each row needs one coefficient per generator.
- `limits` needs one row per line and one value per time step.
- Each weight case has four values in C, W, O, D order.
- Optional `switch_factor` multiplies W; default is 1. Ramp remains at most one state per step.
- Outputs must be nonnegative and strictly increasing within each generator. This model has no fixed injections or time-dependent unit costs.

Only the standard library is needed. Exact enumeration grows exponentially; by default, the runner refuses more than 1,000,000 candidate schedules. The limit can be changed with `--max-schedules`.

Expected results:

| Priority | Weights (C, W, O, D) | States by time | C | W | O | D |
| --- | --- | --- | --- | --- | --- | --- |
| Cost | (1, 0, 0, 100) | [(2, 0), (2, 1)] | 70 | 10 | 2 | 0 |
| Avoid overloads | (1, 0, 100, 100) | [(1, 1), (1, 2)] | 110 | 10 | 0 | 0 |
| Switching | (1, 10, 100, 100) | [(1, 1), (1, 2)] | 110 | 10 | 0 | 0 |

The first two cases show the main trade-off: spending 40 more removes two overloads. Both meet demand exactly. The switching case returns the same schedule here; these weights do not produce a separate switching trade-off.


## Random three-generator SimBench example

Randomly select three active `gen` units and solve a two-time-step, five-state instance:

```bash
python examples/small_instance.py --simbench
```

Set a seed to repeat the same selection, and specify weights in C, W, O, D order:

```bash
python examples/small_instance.py --simbench --seed 7 --weights 1 0 100 100
```

Optional arguments:

- `--T 2`: number of time points, spaced two hours apart (default: 2). Alternatively use `--time-steps 0 8 16 24`; do not combine both options.
- `--generate-only`: build input without solving, useful when a larger T exceeds the enumeration limit.
- `--lines 2`: number of monitored physical lines.
- `--output examples/instances/sample.json`: save the input and metadata without overwriting an existing file.

Data loading is implemented in `RedispatchInstance.from_simbench()`. Output includes the selected generators, baseline outputs, demand, and optimal schedule.

**Data and model:** We use SimBench profiles, the paper repository's fitted S with its P/L data, and costs sampled from Table II. Fixed resource contributions are subtracted from line capacities as in Eq. (47). Profile ordering is checked, individual external-grid labels remain unverified.

This is a classical example without QUBO. Only selected lines are checked; demand mismatch may remain. Reactive power and the ramp from the initial operating state are not modeled.

The supplied fitted matrix is reused, not re-estimated by rerunning Algorithm 4. Its training RMSE against the saved L array is printed (about 16.11 MW); metadata includes maximum error and coefficient range. The fit is approximate, so low predicted overload does not prove AC network feasibility. Random three-generator selection, sampled costs still differ from the full paper experiment.

SimBench uses five output states following paper Eq. (42): if minimum output is positive, use 0 plus four equally spaced levels from minimum to maximum; if minimum is 0, use five equally spaced levels from 0 to maximum. For three generators and two times, this gives 15,625 candidate schedules and 2,197 ramp-feasible schedules. Ramp still permits at most one state change per step. Hand-written JSON files retain their own state counts.
