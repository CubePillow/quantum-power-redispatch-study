"""Classical baseline from Lee et al. (2025): alpha-expansion + Tabu Search.

Solves a SimBench redispatch instance (default: T = 2 time points) with the
paper's code in ``package/`` and ``experiments/`` and compares the result with
the SimBench configuration (the reference schedule in the data set).

Run from the repository root after ``scripts/setup_env.sh``:
    python scripts/run_simbench_tabu.py --prob T2 --runs 3
"""

import argparse
import json
import os
import pickle
import time

import dwave.samplers as dws
import numpy as np
from dimod import BinaryQuadraticModel
from quantum_energy.decompose import dict_to_array, run_decomposer
from quantum_energy.moo import filter_and_sort_pf
from quantum_energy.problem.qubo import RedispatchQUBO
from quantum_energy.problem.redispatch_stats import RedispatchStats
from quantum_energy.solver.subsolvers import BetaExpansion
from quantum_energy.util import get_n_switches_mat, sol_to_x

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")


def tabu_solver(Q_ql):
    """Subproblem solver: classical Tabu Search (dwave-samplers, runs on the CPU)."""
    samp_set = dws.TabuSampler().sample(BinaryQuadraticModel(Q_ql.m, "BINARY"))
    return samp_set.first.sample


def alpha_expansion(Rd, Q, sub_size, max_runs, solver=tabu_solver):
    """alpha-expansion as in ``a_exp_solver`` (experiments/experiments.py).

    Start: every resource in its lowest state (satisfies one-hot and ramping).
    Stop: 5 sweeps without improvement, or ``max_runs``.
    """
    x_init = sol_to_x(np.zeros(Rd.T * Rd.n_resources), Rd.k)
    x_init = {i: x_init[i] for i in range(len(x_init))}
    subsolver = BetaExpansion(Rd.T, Rd.n_resources, Rd.k, sub_size, solver).subsolver_ql
    x, _ = run_decomposer(Q, subsolver, x_init, max_runs=max_runs, max_stale=5)
    return dict_to_array(x)


def simbench_reference(Rd):
    """Statistics of the SimBench configuration (as in nbs/apply_weights.ipynb)."""
    return {
        "powers": np.asarray(Rd.prod_target),
        "costs": Rd.prod_resos.T @ Rd.prod_cost_multi,
        "overloads": np.sum(
            Rd.S_resos @ Rd.sim_res.reso_prods
            + Rd.S_sgens @ Rd.sim_res.sgen_prods
            + Rd.S_loads @ Rd.sim_res.load_prods
            > Rd.max_load,
            axis=0,
        ),
        "switches": get_n_switches_mat(Rd.prod_resos.T),
    }


def max_ramp(Rd, x):
    """Largest state change of any resource between adjacent time points."""
    levels = x.reshape(Rd.T, Rd.n_resources, Rd.k).argmax(axis=2)
    return int(np.abs(np.diff(levels, axis=0)).max()) if Rd.T > 1 else 0


def to_jsonable(d):
    return {k: np.asarray(v).tolist() for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prob", default="T2", help="instance: T2, T4, T6 or T8")
    ap.add_argument("--runs", type=int, default=3, help="independent runs")
    ap.add_argument("--sub-size", type=int, default=128,
                    help="subproblem size (paper: 128, code default: 150)")
    ap.add_argument("--max-runs", type=int, default=10000)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Objective weights: best entry of the paper's Pareto front
    # (experiments/results/norm_pf.pkl), selected on T2 as in the paper.
    with open(os.path.join(EXP, "data/models/T2.pkl"), "rb") as f:
        Rd_T2 = pickle.load(f)
    with open(os.path.join(EXP, "results/norm_pf.pkl"), "rb") as f:
        pf = pickle.load(f)
    weights = filter_and_sort_pf(RedispatchStats(Rd_T2), pf)[0].weights
    print("weights (lambda_P, mu_O, mu_C, mu_S):", weights)

    with open(os.path.join(EXP, f"data/models/{args.prob}.pkl"), "rb") as f:
        Rd = pickle.load(f)
    stats = RedispatchStats(Rd)
    Q = RedispatchQUBO(Rd).get_qubo_obj_normed_nt(*weights)
    print(f"instance {args.prob}: T={Rd.T}, resources={Rd.n_resources}, "
          f"states={Rd.k}, binary variables={Q.shape[0]}")

    runs = []
    for r in range(args.runs):
        t0 = time.time()
        x = alpha_expansion(Rd, Q, args.sub_size, args.max_runs)
        seconds = time.time() - t0
        energy = float(x @ Q @ x)
        one_hot, ramp = bool(stats.x_is_valid(x)), max_ramp(Rd, x)
        print(f"run {r + 1}: {seconds:.0f} s, one-hot valid: {one_hot}, "
              f"max ramp: {ramp}, QUBO energy: {energy:.4f}")
        runs.append({
            "seconds": seconds, "one_hot_valid": one_hot, "max_ramp": ramp,
            "energy": energy, "stats_over_time": to_jsonable(stats.x_sol_to_stats_over_time(x)),
            "x": x.astype(int).tolist(),
        })

    out = {
        "prob": args.prob, "sub_size": args.sub_size,
        "weights": np.asarray(weights).tolist(),
        "simbench_reference": to_jsonable(simbench_reference(Rd)), "runs": runs,
    }
    path = os.path.join(args.out, f"simbench_tabu_{args.prob}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("saved:", path)


if __name__ == "__main__":
    main()
