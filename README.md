# Standard QAOA baseline (X mixer + penalties)

Baseline for task 2 ("QAOA variants") of the project *Constraint-Aware QAOA for
Power System Redispatch*: plain QAOA with the transverse-field **X mixer**, where
the one-hot and ramp constraints are enforced only by **penalty terms** in the
QUBO. It is the reference that constraint-aware variants (GM-QAOA, IF-QAOA,
Prog-QAOA) have to beat.

## What is in this folder

| File | Purpose |
|---|---|
| `qaoa_baseline.py` | Main script. Uses the repository code (`qubo_adapter.build_qubo`, `RedispatchInstance`, `classical_solver`) and runs QAOA on the resulting QUBO. |
| `qaoa_xmixer_standalone.py` | Same QAOA on the hand-built example, evaluating the objective itself (numpy and scipy only). Use it as an independent cross-check. |
| `test_standalone.py` | Smoke test of the standalone script (feasible count, known optima, mixer norm). |
| `instances/hand_example.json` | The hand-built example from `docs/build-instance.md` in the adapter's JSON format. |
| `requirements.txt`, `.gitignore` | Dependencies; result files are not committed. |

## Quick start

Run from `examples/qaoa_xmixer/` (the script finds `examples/qubo_adapter.py` itself).

```bash
pip install numpy scipy matplotlib        # matplotlib only for --plot

python test_standalone.py                  # no repository code needed

python qaoa_baseline.py instances/hand_example.json --mode pscan --pmax 4
python qaoa_baseline.py instances/hand_example.json --mode rho --p 3 --rhos adapter 0.5 1 2 5 --plot
python qaoa_baseline.py instances/hand_example.json --no-ramp --p 3          # ablation
python qaoa_baseline.py --simbench --seed 1 --gens 2 --states 3 --T 2 --mode rho
```

## How it works

1. **Instance:** JSON file or `--simbench` (same options as `qubo_adapter.py`).
2. **QUBO:** `build_qubo(inst, weights, lam_hard)` returns
   `Q = w_C Q_C + w_W Q_S + w_O Q_O + w_D Q_P + lam_H Q_H + lam_A Q_A`.
3. **Cost vector:** `c[s] = x^T Q x` for all `2^N` bit strings (`N = n*k*T`).
   Bit `q` of `s` is qubit `q`, with `q = (t*n + a)*k + i` as in the adapter.
4. **QAOA:** start in `|+>^N`, then `p` rounds of the diagonal phase
   `exp(-i*gamma*c)` and the X mixer `Rx(2*beta)` on every qubit. COBYLA minimizes
   `<c>`; the first start uses a linear-ramp initialization, the others are random.
   The cost vector is standardized (mean 0, std 1), so constant offsets in `Q`
   (global phase) do not matter.
5. **Reference and metrics:** the exact optimum comes from the enumeration of all
   ramp-feasible schedules (`RedispatchInstance.evaluate`).

### Penalty weight `rho`

`--rho` scales the hard-constraint weight: `lam_hard = rho * (range of the weighted
objective over the feasible schedules)`; the ramp weight follows the adapter's rule
`lam_A = lam_hard / min positive entry of Q_A`. `--rho adapter` uses the adapter's own
default (`lam_hard = 1 + sum|Q_obj|`), which is a safe bound for an exact solver but
can be very large for QAOA, so always compare it with smaller values.

## Options

| Option | Meaning | Default |
|---|---|---|
| `input` / `--simbench` | JSON instance or SimBench instance (`--seed --gens --states --T --grid`) | – |
| `--case NAME` | weight case from the JSON | first case |
| `--weights C W O D` | objective weights (overrides the JSON) | JSON / `1 0 100 100` |
| `--mode single\|pscan\|rho` | one run, scan over depth `p`, or scan over `rho` | `single` |
| `--p`, `--pmax` | depth for `single`/`rho`, maximum depth for `pscan` | 3, 5 |
| `--rho`, `--rhos` | penalty factor(s), `adapter` allowed | `1.0` |
| `--no-ramp` | ablation: drop the ramp penalty (one-hot only) | off |
| `--starts`, `--maxiter`, `--seed` | optimizer restarts, COBYLA iterations, RNG seed | 10, 300, 0 |
| `--max-bits` | refuse instances with more than this many qubits | 20 |
| `--log`, `--plot` | JSON-lines log file, save a PNG | `results_baseline.jsonl`, off |

## Output

One line per run, also appended to the log file:

| Column | Meaning |
|---|---|
| `feasible` | probability mass on feasible schedules (one-hot and ramp satisfied) |
| `P(opt)` | probability of the exact optimum (exact objective) |
| `P(Qopt)` | probability of the global minimum of the QUBO |
| `ratio` | approximation ratio over feasible schedules: `(worst - <obj|feasible>) / (worst - best)`; 1 is optimal |
| `Q=exact`, `rank` | does the QUBO minimum (over feasible schedules) equal the exact optimum, and which rank has the exact optimum under the QUBO? |
| `std` | standard deviation of the optimized energy over the restarts |

`Q_O` only approximates the overload count, so the QUBO optimum can differ from the
exact optimum; that is why both `P(opt)` and `P(Qopt)` are reported.

## Instances

Included:
- `instances/hand_example.json`: 2 generators, 2 time steps, 3 states (12 qubits,
  4096 bit strings, 49 ramp-feasible schedules). It was reconstructed from
  `docs/build-instance.md`; compare it with `examples/legacy/instances/small.json`
  before relying on it. Expected exact optima (weights C, W, O, D):

  | Case | Weights | Schedule | C | W | O | D | Weighted score |
  |---|---|---|---|---|---|---|---|
  | Cost | 1 0 0 100 | `[(2,0),(2,1)]` | 70 | 10 | 2 | 0 | 70 |
  | Avoid overloads | 1 0 100 100 | `[(1,1),(1,2)]` | 110 | 10 | 0 | 0 | 110 |
  | Switching | 1 10 100 100 | `[(1,1),(1,2)]` | 110 | 10 | 0 | 0 | 210 |

  The three non-dominated schedules in (C, W, O) with demand met exactly are
  (70, 10, 2), (90, 10, 1) and (110, 10, 0); W is 10 for all of them, so this
  example has no separate switching trade-off.

Not included (on purpose):
- `examples/legacy/instances/three_steps.json`: its numbers are not documented.
- SimBench instances: they depend on the seed, the data and the fitted `S`; create
  them with `qubo_adapter.py --simbench --seed <s> --output <file>` or pass
  `--simbench --seed <s>` to `qaoa_baseline.py`.

Feasible fraction for the uniform start, by size (`N = n*k*T`):

| n, k, T | N | feasible | fraction | X-mixer simulation |
|---|---|---|---|---|
| 2, 3, 2 | 12 | 49 | 1.2 % | fast |
| 2, 3, 3 | 18 | 289 | 0.11 % | feasible |
| 3, 3, 2 | 18 | 343 | 0.13 % | feasible |
| 2, 5, 2 | 20 | 169 | 0.016 % | borderline |
| 4, 3, 2 | 24 | 2,401 | 0.014 % | impractically slow |
| 3, 5, 2 (default `--simbench`) | 30 | 2,197 | 0.0002 % | not possible (2^30 amplitudes) |

The statevector needs `2^N` complex amplitudes and the script a `2^N x N` bit matrix;
beyond roughly 20 qubits the run time explodes (rough estimate).

## Status and limitations

- `test_standalone.py` passes. `qaoa_baseline.py` was **tested end to end with a
  stand-in for `quantum_energy/problem/qubo.py`**, not with the original
  `RedispatchQUBO` and not with SimBench data. Before trusting results:
  1. run the adapter's `check_exact_terms` (Q_C, Q_S, Q_P against C, W, D);
  2. check that the printed number of feasible schedules matches the enumeration
     (hand example: 49);
  3. look at `Q=exact` and `rank` for your instances.
- No results are checked in. Fill in the table below from your own runs.
- COBYLA with few restarts; results vary with the start (see `std`). No noise model,
  no hardware.
- `Q_O` is the unbalanced-penalty (Taylor) approximation of the overload count;
  the fitted `S` has an RMSE of about 16 MW, so overloads compare algorithms, not
  network feasibility.

## Results (to fill in)

| Instance (n, k, T) | weights | rho | p | feasible | P(opt) | P(Qopt) | ratio | std | seed / starts |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

## Next steps

- GM-QAOA in the feasible subspace with the same metrics (the feasibility mask and
  the exact objective from `exact_reference` are exactly what it needs).
- Normalized objective weights and multi-objective sweeps (Pareto set,
  hypervolume).
- More instances (SimBench seeds) once the first runs are checked.

## References

- Lee et al., *Multi-Objective Quantum Power System Redispatch*, arXiv:2409.09857
  (model and QUBO).
- Bärtschi, Eidenbenz, *Grover Mixers for QAOA*, IEEE QCE 2020.
- Farhi, Goldstone, Gutmann, *A Quantum Approximate Optimization Algorithm*,
  arXiv:1411.4028.
- `Redispatch_QAOA_OnePager.pdf` (project description).
