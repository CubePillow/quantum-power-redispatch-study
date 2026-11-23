#!/usr/bin/env python3
"""QAOA baseline (standard X mixer + penalty terms) on the repository QUBO.

Location: examples/qaoa_xmixer/ (next to this file's parent: qubo_adapter.py,
classical_solver.py, redispatch_instance.py in examples/).
Needs numpy and scipy (matplotlib optional for --plot) plus whatever
qubo_adapter.py needs.

Pipeline:
  instance (JSON or --simbench)
    -> build_qubo() from qubo_adapter.py          (Q with hard-constraint weight)
    -> cost vector c[s] = x^T Q x for all 2^N bit strings
    -> QAOA statevector simulation with X mixer
    -> metrics against the exact enumeration (classical_solver / RedispatchInstance)

Bit convention as in the adapter: x[(t*n + a)*k + i]. Basis state s has bit
(s >> q) & 1 on qubit q.

Examples (run from examples/qaoa_xmixer/):
  python qaoa_baseline.py instances/hand_example.json --mode pscan --pmax 4
  python qaoa_baseline.py instances/hand_example.json --mode rho --p 3 \
      --rhos adapter 0.5 1 2 5 --plot
  python qaoa_baseline.py instances/hand_example.json --no-ramp --p 3
  python qaoa_baseline.py --simbench --seed 1 --gens 2 --states 3 --T 2 --mode rho
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

EXAMPLES = Path(__file__).resolve().parents[1]      # examples/
sys.path.insert(0, str(EXAMPLES))

from classical_solver import enumerate_feasible, weighted_score  # noqa: E402
from qubo_adapter import build_qubo, decode, encode  # noqa: E402
from redispatch_instance import RedispatchInstance  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Instance, reference and cost vector
# ---------------------------------------------------------------------------
def load_instance(args):
    """As in the adapter: JSON file or SimBench."""
    if args.simbench:
        inst, _ = RedispatchInstance.from_simbench(
            seed=args.seed, time_steps=[8 * t for t in range(args.T)],
            grid=args.grid, gens=args.gens, states=args.states)
        weights = [1, 0, 100, 100]
    else:
        data = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
        inst = RedispatchInstance(**data["instance"])
        cases = data.get("weights", {"Default": [1, 0, 100, 100]})
        weights = list(cases[args.case] if args.case else next(iter(cases.values())))
    if args.weights is not None:
        weights = list(args.weights)
    return inst, weights


def exact_reference(inst, weights):
    """Feasibility mask and exact objective (defined on feasible schedules only)."""
    N = inst.T * inst.n * inst.k
    feasible = np.zeros(2 ** N, dtype=bool)
    objective = np.full(2 ** N, np.nan)
    powers = 1 << np.arange(N)
    records = enumerate_feasible(inst.T, inst.n, inst.k,
                                 inst.evaluate, inst.is_feasible)
    for states, terms in records:
        s = int(encode(states, inst) @ powers)
        feasible[s] = True
        objective[s] = weighted_score(terms, weights)
    return feasible, objective


def all_bits(N):
    s = np.arange(2 ** N)
    return ((s[:, None] >> np.arange(N)) & 1).astype(float)


def energies(Q, bits, offset=0.0):
    """c[s] = x^T Q x (+ offset) for all bit strings."""
    return ((bits @ np.asarray(Q, dtype=float)) * bits).sum(axis=1) + offset


# ---------------------------------------------------------------------------
# 2. QAOA simulation (statevector, X mixer)
# ---------------------------------------------------------------------------
def apply_rx_all(psi, beta, N):
    """Mixer exp(-i beta sum_q X_q) = Rx(2 beta) on every qubit (in place)."""
    c, s = np.cos(beta), -1j * np.sin(beta)
    for q in range(N):
        v = psi.reshape(-1, 2, 1 << q)
        a0, a1 = v[:, 0, :].copy(), v[:, 1, :].copy()
        v[:, 0, :] = c * a0 + s * a1
        v[:, 1, :] = s * a0 + c * a1


def qaoa_state(params, cost, N):
    p = len(params) // 2
    psi = np.full(2 ** N, 2 ** (-N / 2), dtype=complex)         # |+>^N
    for g, b in zip(params[:p], params[p:]):
        psi *= np.exp(-1j * g * cost)                           # phase separator
        apply_rx_all(psi, b, N)                                 # X mixer
    return psi


def expectation(params, cost, N):
    psi = qaoa_state(params, cost, N)
    return float(np.real(np.sum(np.abs(psi) ** 2 * cost)))


def linear_ramp_init(p, dt=0.75):
    x = (np.arange(p) + 0.5) / p
    return np.concatenate([dt * x, dt * (1 - x)])


def run_qaoa(cost, N, p, n_starts=10, maxiter=300, seed=0):
    rng = np.random.default_rng(seed)
    best, restart_energies = None, []
    for r in range(n_starts):
        x0 = linear_ramp_init(p) if r == 0 else np.concatenate(
            [rng.uniform(0, 1.5, p), rng.uniform(0, np.pi / 2, p)])
        res = minimize(expectation, x0, args=(cost, N), method="COBYLA",
                       options={"maxiter": maxiter, "rhobeg": 0.3})
        restart_energies.append(res.fun)
        if best is None or res.fun < best.fun:
            best = res
    return best.x, best.fun, restart_energies


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------
def at_min(a, tol=1e-9):
    """Mask of the entries equal to the minimum (tight tolerance)."""
    m = np.nanmin(a)
    return a <= m + tol * max(1.0, abs(m))


def metrics(psi, cost, feasible, objective):
    probs = np.abs(psi) ** 2
    feas_prob = float(probs[feasible].sum())
    best = np.nanmin(objective[feasible])
    worst = np.nanmax(objective[feasible])
    exact_opt = feasible & at_min(objective)
    q_opt = at_min(cost)                          # global minimum of the QUBO
    cond = (float((probs[feasible] * objective[feasible]).sum() / feas_prob)
            if feas_prob > 0 else float("nan"))
    ratio = (worst - cond) / (worst - best) if worst > best else float("nan")
    return dict(feas_prob=feas_prob, p_opt=float(probs[exact_opt].sum()),
                p_qopt=float(probs[q_opt].sum()), approx_ratio=float(ratio),
                n_feasible=int(feasible.sum()), opt_value=float(best))


def qubo_vs_exact(cost, feasible, objective):
    """Does the QUBO minimum over feasible schedules equal the exact optimum?"""
    f = np.flatnonzero(feasible)
    exact_best = f[at_min(objective[f])]
    q_best = f[at_min(cost[f])]
    tol = 1e-9 * max(1.0, abs(cost[f].min()))
    rank = 1 + int((cost[f] < cost[exact_best[0]] - tol).sum())
    return bool(set(exact_best) == set(q_best)), rank, len(f)


def describe(s, N, inst):
    x = (s >> np.arange(N)) & 1
    states = decode(x, inst)
    if states is None:
        return "(invalid one-hot)"
    return [tuple(states[t * inst.n:(t + 1) * inst.n]) for t in range(inst.T)]


# ---------------------------------------------------------------------------
# 4. Runs
# ---------------------------------------------------------------------------
def one_run(inst, weights, base, bits, feasible, objective, rho, p, args):
    N = inst.T * inst.n * inst.k
    lam_hard = None if rho is None else rho * base
    Q, Q_obj, rq = build_qubo(inst, weights, lam_hard)
    if args.no_ramp:                                  # ablation: drop ramp penalty
        lam = lam_hard if lam_hard is not None else 1.0 + np.abs(Q_obj).sum()
        Q = Q_obj + lam * rq.qubo_one_hot
    cost = energies(Q, bits)
    same, rank, _ = qubo_vs_exact(cost, feasible, objective)
    # Standardize so that gamma lives in a handy range (constants drop out)
    c = (cost - cost.mean()) / (cost.std() if cost.std() > 0 else 1.0)
    t0 = time.time()
    params, _, restarts = run_qaoa(c, N, p, args.starts, args.maxiter, args.seed)
    psi = qaoa_state(params, c, N)
    row = dict(p=p, rho="adapter" if rho is None else rho, weights=list(weights),
               no_ramp=args.no_ramp, q_min_is_exact_opt=same,
               exact_opt_rank_in_Q=rank, restart_std=float(np.std(restarts)),
               seconds=round(time.time() - t0, 2), params=params.tolist(),
               **metrics(psi, cost, feasible, objective))
    with open(args.log, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return row, psi


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="JSON instance (as for qubo_adapter.py)")
    ap.add_argument("--case", help="name of the weight case in the JSON (default: first)")
    ap.add_argument("--simbench", action="store_true")
    ap.add_argument("--seed", type=int, help="SimBench: line choice and costs")
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--states", type=int, default=3)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--grid", default="1-EHV-mixed--0-sw")
    ap.add_argument("--weights", type=float, nargs=4, metavar=("C", "W", "O", "D"))
    ap.add_argument("--mode", choices=["single", "pscan", "rho"], default="single")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--pmax", type=int, default=5)
    ap.add_argument("--rho", default="1.0",
                    help="penalty factor or 'adapter' (the adapter's default weight)")
    ap.add_argument("--rhos", nargs="+",
                    default=["adapter", "0.1", "0.5", "1", "2", "5"])
    ap.add_argument("--no-ramp", action="store_true",
                    help="ablation: leave out the ramp penalty term")
    ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--max-bits", type=int, default=20)
    ap.add_argument("--log", default="results_baseline.jsonl")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if bool(args.input) == args.simbench:
        ap.error("Provide either a JSON path or --simbench")

    inst, weights = load_instance(args)
    N = inst.T * inst.n * inst.k
    if N > args.max_bits:
        ap.error(f"{N} bits > --max-bits {args.max_bits}")
    bits = all_bits(N)
    feasible, objective = exact_reference(inst, weights)
    f = objective[feasible]
    base = float(np.nanmax(f) - np.nanmin(f)) or 1.0    # range of feasible objective
    print(f"n={inst.n}, k={inst.k}, T={inst.T} -> N={N} qubits, {2 ** N} bit strings, "
          f"{int(feasible.sum())} feasible ({feasible.mean():.3%}); "
          f"weights (C,W,O,D)={weights}")

    def parse(r):
        return None if str(r) == "adapter" else float(r)

    if args.mode == "single":
        runs = [(parse(args.rho), args.p)]
    elif args.mode == "pscan":
        runs = [(parse(args.rho), p) for p in range(1, args.pmax + 1)]
    else:
        runs = [(parse(r), args.p) for r in args.rhos]

    rows = []
    print(f"{'rho':>8} {'p':>3} {'feasible':>9} {'P(opt)':>7} {'P(Qopt)':>8} "
          f"{'ratio':>6} {'Q=exact':>8} {'rank':>5} {'std':>7}")
    for rho, p in runs:
        row, psi = one_run(inst, weights, base, bits, feasible, objective,
                           rho, p, args)
        rows.append(row)
        print(f"{str(row['rho']):>8} {p:3d} {row['feas_prob']:9.3f} "
              f"{row['p_opt']:7.3f} {row['p_qopt']:8.3f} "
              f"{row['approx_ratio']:6.3f} {str(row['q_min_is_exact_opt']):>8} "
              f"{row['exact_opt_rank_in_Q']:5d} {row['restart_std']:7.3f}")
    probs = np.abs(psi) ** 2
    print("\nMost probable states of the last run:")
    for s in np.argsort(probs)[::-1][:3]:
        print(f"  s={int(s):>6}  P={probs[s]:.3f}  feasible={bool(feasible[s])}  "
              f"{describe(int(s), N, inst)}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = ([r["p"] for r in rows] if args.mode == "pscan"
              else [str(r["rho"]) for r in rows])
        plt.figure(figsize=(6, 4))
        for key, lab in [("feas_prob", "feasibility"),
                         ("p_opt", "P(exact optimum)"),
                         ("p_qopt", "P(QUBO optimum)"),
                         ("approx_ratio", "approximation ratio")]:
            plt.plot(xs, [r[key] for r in rows], marker="o", label=lab)
        plt.xlabel("p" if args.mode == "pscan" else "rho")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        out = f"baseline_{args.mode}.png"
        plt.savefig(out, dpi=150)
        print("Plot saved:", out)


if __name__ == "__main__":
    main()
