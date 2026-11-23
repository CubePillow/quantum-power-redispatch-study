# Classical Baseline on SimBench: alpha-Expansion + Tabu Search

This reproduces the classical solver path of Lee et al. (2025) on the SimBench grid
`1-EHV-mixed--0-sw` (338 controllable resources, 5 states each), using the paper's
code in `package/` and `experiments/`.

## Setup

```bash
./scripts/setup_env.sh
```

The script creates the environment `env/` (Python 3.11, versions pinned in
`requirements-lock.txt`), compiles the C++ extensions of `package/`, runs the SimBench
power flow and writes the instances T2, T4, T6, T8 to `experiments/data/models/`
(about 130 MB each, not committed).

Changes compared with the paper's `requirements.txt` that were needed to run the code:
`pandas==2.0.3`, `simbench==1.4.0` and `pandapower<3` (pandapower 3 removed
`bus_geodata`); `qubolite` additionally needs `scikit-learn`, `igraph` and `portion`.

## Run

```bash
source env/bin/activate
python scripts/run_simbench_tabu.py --prob T2 --runs 3
```

What the script does (same steps as `a_exp_solver` in `experiments/experiments.py`):

1. **Weights.** It takes the best entry of the paper's Pareto front
   (`experiments/results/norm_pf.pkl`), selected with `filter_and_sort_pf` as in the paper:
   lambda_P = 10^4 (demand), mu_O = 0.75 (overloads), mu_C ≈ 0.117 (cost),
   mu_S ≈ 0.133 (switching). The Pareto front is **not** recomputed.
2. **QUBO.** Normalized QUBO from `get_qubo_obj_normed_nt`: 2 × 338 × 5 = 3380 binary variables.
3. **alpha-expansion.** Start with every resource in its lowest state; subproblems of
   128 variables (as in the paper; the code default is 150), each solved with
   `dwave.samplers.TabuSampler` (classical, CPU); stop after 5 sweeps without improvement.

Output: `results/simbench_tabu_T2.json` (statistics per time point and solution vectors).

## Results for T2 (3 runs, 23–48 s each)

| | Overloaded lines (t0 / t1) | Cost in M€ (t0 / t1) | Max. demand deviation | Switches |
|---|---|---|---|---|
| SimBench configuration | 4 / 6 | 1.233 / 1.105 | – | 11 |
| Run 1 | 4 / 5 | 1.465 / 1.309 | 0.012 % | 184 |
| Run 2 (lowest QUBO energy) | 5 / 6 | 1.358 / 1.181 | 0.005 % | 164 |
| Run 3 | 5 / 4 | 1.375 / 1.220 | 0.010 % | 176 |
| Paper, Table I (normalized) | 5 / 4 | 1.445 / 1.277 | – | 156 |

All solutions satisfy one-hot and ramping (maximum state change 1).

**Interpretation**
- The results are in the same range as Table I of the paper; run 3 matches its overload
  counts (5 / 4).
- Compared with the SimBench configuration: 9–11 overloads in total instead of 10, at
  6–18 % higher cost and far more switching. Overloads are not reduced at every time
  point, which matches Fig. 3 of the paper.
- The runs differ because Tabu Search is a heuristic. The lowest QUBO energy does not
  necessarily mean the fewest overloads, because the QUBO trades off all objectives.
  Whether any solution is optimal cannot be said from these runs; an exact solver
  (e.g. MILP) would be needed for that.
