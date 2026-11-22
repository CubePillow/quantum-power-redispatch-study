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
  problem/reduce.py              NEW: line-aware small instances (create_small_models)
  decompose.py                   alpha-expansion
  util.py                        x_invalids (one-hot + ramp check), helpers
experiments/
  create_models.py               build full models T2..T8
  experiments.py                 paper experiments (Pareto weights, transfer, D-Wave)
  data/sensitivity/S.npy         fitted sensitivity matrix S (lines x devices)
examples/                        NEW: small instances, exact solver, QUBO check (see "How to run")
docs/                            notes on the small example and QUBO terms
paper/                           LaTeX source of the paper
```

## Setup

Python ≥ 3.10.
```bash
pip install numpy numba simbench pandapower dimod
```

## How to run

Run from `examples/` (section 3 from `experiments/`).

### `small_instance.py` – exact solution by enumeration

```bash
python small_instance.py instances/small.json
python small_instance.py --simbench --seed 1 --gens 2 --states 3 --T 2
```

| Option | Meaning | Default |
|---|---|---|
| `input` | JSON instance file (or use `--simbench`) | – |
| `--simbench` | build instance from SimBench: line first, then its most influential generators; all lines they can switch are monitored | off |
| `--seed` | which switchable line is picked, and the costs | random |
| `--gens` | generators n | 3 |
| `--states` | states per generator k (≥ 3) | 5 |
| `--T` | time points, 2 h apart (1–24) | 2 |
| `--grid` | SimBench grid | `1-EHV-mixed--0-sw` |
| `--weights C W O D` | objective weights | from JSON / `1 0 100 100` |
| `--output` | save as JSON (no overwrite) | – |
| `--max-schedules` | enumeration limit (k^(n·T)) | 1 000 000 |

### `qubo_adapter.py` – QUBO vs. exact optimum

```bash
python qubo_adapter.py instances/small.json
```

| Option | Meaning |
|---|---|
| `input` | JSON instance file (max. n·k·T = 24 bits) |

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

