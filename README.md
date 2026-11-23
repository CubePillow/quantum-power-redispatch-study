# Constraint-Aware QAOA for Power System Redispatch

Student project based on Lee et al., *Multi-Objective Quantum Power System Redispatch* (code: <https://gitlab.com/lklee/quantum-power-redispatch>).
We reuse their redispatch model and QUBO, build **small SimBench instances with known exact optima**,
and compare constraint-aware QAOA variants on them.

## Tasks → code

| Task | Goal | Where |
|---|---|---|
| 1 · Reproduce | Model (C, W, O, D + one-hot / ramp), small instances, exact optimum | `examples/` (enumeration), `package/.../reduce.py` (instances), `redispatch_stats.py` (evaluation) |
| 2 · QAOA variants | Baseline (penalties), GM-QAOA, IF-QAOA | QUBO: `package/.../qubo.py`, `examples/qubo_adapter.py` · QAOA: *to do* |
| 3 · Multi-objective | Weights μ, non-dominated samples → Pareto front, Kotil et al. | *to do* (paper's weight search: `package/quantum_energy/moo.py`) |
| 4 · Benchmark | Quality, feasibility, hypervolume, resources vs. classical / α-expansion | *to do* (α-expansion: `package/quantum_energy/decompose.py`) |

## Structure

```
package/quantum_energy/          original library (Lee et al.)
  problem/modeller.py            RedispatchModel: levels, costs, tau, M from SimBench
  problem/simulator.py           SimBench power flow, time-point selection
  problem/redispatch_stats.py    evaluate C, W, O, D and feasibility (Z form)
  problem/qubo.py                QUBO matrices Q_C, Q_S, Q_O, Q_P, Q_H, Q_A
  problem/reduce.py              NEW: line-aware small instances (used by qubo_adapter --simbench)
  decompose.py                   alpha-expansion
  util.py                        x_invalids (one-hot + ramp check), helpers
experiments/
  create_models.py               build full models T2..T8
  experiments.py                 paper experiments (Pareto weights, transfer, D-Wave)
  data/sensitivity/S.npy         fitted sensitivity matrix S (lines x devices)
examples/                        NEW: QUBO from SimBench, exact check (see "How to run")
  legacy/                        Z-enumeration CLI and JSON examples (see docs/build-instance.md)
docs/                            notes on the small example and QUBO terms
paper/                           LaTeX source of the paper
```

## Setup

Python ≥ 3.10.
```bash
pip install numpy numba simbench pandapower dimod dwave-samplers dwave-hybrid
```

## How to run

Run from `examples/`.

### `qubo_adapter.py` – QUBO encoding, solved with exact / Tabu / alpha-expansion

```bash
python qubo_adapter.py --simbench --seed 1 --gens 2 --states 3 --T 2
python qubo_adapter.py --simbench --seed 1 --gens 2 --states 3 --T 2 --solver all
```

| Option | Meaning | Default |
|---|---|---|
| `--simbench` | build instance as in Lee et al.: SimBench power flow + `S.npy` → `reduce.py` (line first, then its most influential generators; all lines they can switch are monitored) → `RedispatchModel` | off |
| `--seed` | which switchable line is picked, and the costs | random |
| `--gens` | generators n | 2 |
| `--states` | states per generator k (≥ 3) | 3 |
| `--T` | time points, 2 h apart (1–24) | 2 |
| `--grid` | SimBench grid | `1-EHV-mixed--0-sw` |
| `--weights C W O D` | objective weights | `1 0 100 100` |
| `--output` | save as JSON (no overwrite) | – |
| `--solver` | `exact` (`dimod.ExactSolver`, all bit strings) · `tabu` (`TabuSampler` on the whole QUBO) · `aexp` (alpha-expansion + Tabu of the original repository, as in the paper) · `all` | `exact` |
| `--num-reads` | `tabu`: independent runs | 20 |
| `--sub-size` | `aexp`: subproblem size | 6 |
| `--max-bits` | `exact`: limit for n·k·T (enumerates all 2^(n·k·T) bit strings) | 20 |

### `print_data.py` – preview of the data used by the model

```bash
python print_data.py
python print_data.py --show flow instance --seed 2
```

| Option | Meaning | Default |
|---|---|---|
| `--show` | sections: `net flow levels costs capacity s instance` | all |
| `--rows` | rows per table | 3 |
| `--T` | time points, 2 h apart | 2 |
| `--seed`, `--gens`, `--states` | small instance shown in `instance` (same selection as `qubo_adapter.py`) | 1, 2, 3 |

## How the paper uses SimBench (and what we change)

1. Grid `1-EHV-mixed--0-sw`: 338 generators, 225 sgens, 397 loads/ext. grids, 849 lines.
2. `run_and_record_profiles`: power flow for 192 steps (15 min, 48 h), ~15 s.
3. `get_even_first_T_time_points(all_res, T, 2)`: one time point every 2 h.
4. `RedispatchModel(sim_res, k)` + `S.npy` → levels, costs, tau, M.

| | Paper | This project |
|---|---|---|
| Size | all 338 generators, k=5, T=2…8 (T·1690 bits) | 2–4 generators, k=3–5, T=1–3 (12–30 bits) |
| Solver | α-expansion + Tabu / D-Wave | exact enumeration, QAOA (statevector) |
| Reference | SimBench schedule (no exact optimum) | exact optimum / exact Pareto front |

## Changes to the original code
- `run_experiments.py`, `experiments/util.py`, `dwave_solver.py`: hard-coded API keys replaced by environment variables.

