#!/usr/bin/env python3
"""Standalone baseline: standard QAOA (X mixer) with penalty terms.

Needs only numpy and scipy (matplotlib optional for --plot). It evaluates the
objective terms itself and does not need the repository code, so it can be used
to cross-check qaoa_baseline.py on the hand-built example.

Statevector simulation for small instances (N = T*n*k qubits, N <= ~20).
Bit convention: qubit q = t*(n*k) + a*k + i (time t, generator a, state i,
one-hot). Basis state s has bit (s >> q) & 1 on qubit q.

Examples:
  python qaoa_xmixer_standalone.py --mode single --p 3 --rho 1.0
  python qaoa_xmixer_standalone.py --mode pscan --pmax 5 --rho 1.0 --normalize --plot
  python qaoa_xmixer_standalone.py --mode rho --p 3 --rhos 0.1 0.5 1 2 5 --plot
  python qaoa_xmixer_standalone.py --instance instances/hand_example.json
"""
import argparse
import json
import time

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# 1. Instance (same field names as docs/build-instance.md)
# ---------------------------------------------------------------------------
DEFAULT_INSTANCE = {
    "outputs": [[0, 10, 20], [0, 10, 20]],   # p[a][i]
    "unit_costs": [1, 3],                    # cost per MW and generator
    "demand": [20, 30],                      # tau[t]
    "sensitivity": [[1.0, 0.2]],             # S[line][generator]
    "limits": [[19, 19]],                    # M[line][time]
}


def load_instance(path=None):
    """Load a JSON instance (plain fields or wrapped in an "instance" key)."""
    if path is None:
        return DEFAULT_INSTANCE
    with open(path, encoding="utf-8-sig") as f:
        raw = json.load(f)
    raw = raw.get("instance", raw)
    return {k: raw[k] for k in DEFAULT_INSTANCE}


# ---------------------------------------------------------------------------
# 2. Evaluate all 2^N bit strings: objective terms, penalties, feasibility
# ---------------------------------------------------------------------------
def build_tables(inst):
    p = np.array(inst["outputs"], dtype=float)            # (n, k)
    n, k = p.shape
    T = len(inst["demand"])
    N = T * n * k
    S_ = np.array(inst["sensitivity"], dtype=float)       # (L, n)
    M = np.array(inst["limits"], dtype=float).T           # (T, L)
    tau = np.array(inst["demand"], dtype=float)
    uc = np.array(inst["unit_costs"], dtype=float)

    s = np.arange(2 ** N)
    bits = ((s[:, None] >> np.arange(N)) & 1).astype(np.int8)   # (2^N, N)
    X = bits.reshape(-1, T, n, k).astype(float)                 # (2^N, T, n, k)

    P = np.einsum("stak,ak->sta", X, p)                         # power per (t, a)
    C = np.einsum("sta,a->s", P, uc)
    D = ((P.sum(axis=2) - tau) ** 2).sum(axis=1)

    # Switching cost as in Lee et al. Eq. 14: sum x[t,i] x[t+1,i'] |p_i - p_i'|
    dp = np.abs(p[:, :, None] - p[:, None, :])                  # (n, k, k)
    W = np.einsum("stai,staj,aij->s", X[:, :-1], X[:, 1:], dp)

    # Line load and overload
    L = np.einsum("sta,la->stl", P, S_)                         # (2^N, T, L)
    O_exact = (L >= M).sum(axis=(1, 2)).astype(float)
    h = M - L
    hmax = M - (S_ @ p[:, 0])[None, :]                          # Lee Eq. 20
    z = h / hmax
    O_taylor = (1 - z + 0.5 * z ** 2).sum(axis=(1, 2))          # Lee Eq. 21

    # Penalty terms
    onehot = ((X.sum(axis=3) - 1) ** 2).sum(axis=(1, 2))
    ramp_mat = (np.abs(np.arange(k)[:, None] - np.arange(k)[None, :]) > 1).astype(float)
    ramp = np.einsum("staj,staj->s", X[:, :-1] @ ramp_mat, X[:, 1:])
    feasible = (onehot == 0) & (ramp == 0)
    return dict(N=N, T=T, n=n, k=k, p=p, bits=bits, X=X,
                C=C, W=W, O_exact=O_exact, O_taylor=O_taylor, D=D,
                onehot=onehot, ramp=ramp, feasible=feasible)


def _norm(term, feasible):
    span = term[feasible].max() - term[feasible].min()
    return term / (span if span > 0 else 1.0)


def build_cost(tab, weights=(1, 0, 0, 100), rho=1.0, overload="exact",
               normalize=False):
    """Cost vector c[x] = objective + rho * base * (one-hot + ramp penalty).

    weights = (mu_C, mu_W, mu_O, lambda_D).
    normalize=True divides each term by its range over the feasible schedules.
    base = range of the objective over the feasible schedules, so rho does
    not depend on the scale of the instance.
    """
    f = tab["feasible"]
    terms = [tab["C"], tab["W"],
             tab["O_exact" if overload == "exact" else "O_taylor"], tab["D"]]
    if normalize:
        terms = [_norm(t, f) for t in terms]
    objective = sum(w * t for w, t in zip(weights, terms))
    base = objective[f].max() - objective[f].min()
    base = base if base > 0 else 1.0
    cost = objective + rho * base * (tab["onehot"] + tab["ramp"])
    return cost, objective


def cost_from_qubo(Q, bits, offset=0.0):
    """Hook for a QUBO matrix: c[x] = x^T Q x + offset for all bit strings."""
    b = bits.astype(float)
    return ((b @ np.asarray(Q, dtype=float)) * b).sum(axis=1) + offset


# ---------------------------------------------------------------------------
# 3. QAOA simulation (statevector, X mixer)
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
    gammas, betas = params[:p], params[p:]
    psi = np.full(2 ** N, 2 ** (-N / 2), dtype=complex)         # |+>^N
    for g, b in zip(gammas, betas):
        psi *= np.exp(-1j * g * cost)                           # phase separator
        apply_rx_all(psi, b, N)                                 # mixer
    return psi


def expectation(params, cost, N):
    psi = qaoa_state(params, cost, N)
    return float(np.real(np.sum(np.abs(psi) ** 2 * cost)))


def linear_ramp_init(p, dt=0.75):
    """Common starting values (linear ramp): gamma grows, beta shrinks."""
    x = (np.arange(p) + 0.5) / p
    return np.concatenate([dt * x, dt * (1 - x)])


def run_qaoa(cost, N, p, n_starts=10, maxiter=300, seed=0):
    rng = np.random.default_rng(seed)
    best, energies = None, []
    for r in range(n_starts):
        x0 = linear_ramp_init(p) if r == 0 else np.concatenate(
            [rng.uniform(0, 1.5, p), rng.uniform(0, np.pi / 2, p)])
        res = minimize(expectation, x0, args=(cost, N), method="COBYLA",
                       options={"maxiter": maxiter, "rhobeg": 0.3})
        energies.append(res.fun)
        if best is None or res.fun < best.fun:
            best = res
    return best.x, best.fun, energies


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
def metrics(psi, tab, objective):
    probs = np.abs(psi) ** 2
    f = tab["feasible"]
    feas_prob = float(probs[f].sum())
    best, worst = objective[f].min(), objective[f].max()
    opt_mask = f & (objective <= best + 1e-9 * max(1.0, abs(best)))
    p_opt = float(probs[opt_mask].sum())
    cond = (float((probs[f] * objective[f]).sum() / feas_prob)
            if feas_prob > 0 else float("nan"))
    ratio = (worst - cond) / (worst - best) if worst > best else float("nan")
    return dict(feas_prob=feas_prob, p_opt=p_opt, approx_ratio=float(ratio),
                n_feasible=int(f.sum()), opt_value=float(best))


def describe(s, tab):
    """Bit string -> schedule [(state of each generator) per time step]."""
    x = tab["X"][s]
    if not np.all(x.sum(axis=2) == 1):
        return "(invalid one-hot)"
    z = x.argmax(axis=2)
    return [tuple(int(v) for v in row) for row in z]


def top_states(psi, tab, k=3):
    probs = np.abs(psi) ** 2
    return [(int(s), float(probs[s]), bool(tab["feasible"][s]), describe(s, tab))
            for s in np.argsort(probs)[::-1][:k]]


# ---------------------------------------------------------------------------
# 5. Runs and logging
# ---------------------------------------------------------------------------
def one_run(tab, weights, rho, p, args, qubo=None, log=None):
    cost, objective = build_cost(tab, weights, rho, args.overload, args.normalize)
    if qubo is not None:
        cost = cost_from_qubo(qubo, tab["bits"], args.qubo_offset)
    # Scale to std 1 so that gamma lives in a handy range
    scale = cost.std() if cost.std() > 0 else 1.0
    c = (cost - cost.mean()) / scale
    t0 = time.time()
    params, energy, restarts = run_qaoa(c, tab["N"], p, args.starts,
                                        args.maxiter, args.seed)
    psi = qaoa_state(params, c, tab["N"])
    m = metrics(psi, tab, objective)
    row = dict(p=p, rho=rho, weights=list(weights), overload=args.overload,
               normalize=args.normalize, energy_scaled=energy,
               restart_std=float(np.std(restarts)),
               seconds=round(time.time() - t0, 2), params=params.tolist(), **m)
    if log:
        with open(log, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    return row, psi


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default=None, help="JSON instance")
    ap.add_argument("--weights", type=float, nargs=4, default=[1, 0, 0, 100],
                    metavar=("C", "W", "O", "D"))
    ap.add_argument("--overload", choices=["exact", "taylor"], default="exact")
    ap.add_argument("--normalize", action="store_true",
                    help="normalize each objective term to range 1")
    ap.add_argument("--mode", choices=["single", "pscan", "rho"], default="single")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--pmax", type=int, default=5)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.1, 0.5, 1, 2, 5])
    ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qubo", default=None,
                    help=".npy file with a QUBO matrix (replaces the cost vector)")
    ap.add_argument("--qubo-offset", type=float, default=0.0)
    ap.add_argument("--log", default="results.jsonl")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    tab = build_tables(load_instance(args.instance))
    qubo = np.load(args.qubo) if args.qubo else None
    print(f"N = {tab['N']} qubits, {2 ** tab['N']} bit strings, "
          f"{int(tab['feasible'].sum())} feasible ({tab['feasible'].mean():.2%})")

    weights = tuple(args.weights)
    if args.mode == "single":
        runs = [(args.rho, args.p)]
    elif args.mode == "pscan":
        runs = [(args.rho, p) for p in range(1, args.pmax + 1)]
    else:
        runs = [(r, args.p) for r in args.rhos]

    rows = []
    print(f"{'rho':>6} {'p':>3} {'feasible':>9} {'P(opt)':>8} {'ratio':>7} "
          f"{'std(starts)':>11}")
    for rho, p in runs:
        row, psi = one_run(tab, weights, rho, p, args, qubo, args.log)
        rows.append(row)
        print(f"{rho:6.2f} {p:3d} {row['feas_prob']:9.3f} {row['p_opt']:8.3f} "
              f"{row['approx_ratio']:7.3f} {row['restart_std']:11.4f}")
    print("\nMost probable states of the last run "
          "(bit string, P, feasible, schedule):")
    for s, pr, ok, desc in top_states(psi, tab):
        print(f"  {s:>5}  P={pr:.3f}  feasible={ok}  {desc}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = ([r["p"] for r in rows] if args.mode == "pscan"
              else [r["rho"] for r in rows])
        plt.figure(figsize=(6, 4))
        for key, lab in [("feas_prob", "feasibility"), ("p_opt", "P(optimum)"),
                         ("approx_ratio", "approximation ratio")]:
            plt.plot(xs, [r[key] for r in rows], marker="o", label=lab)
        plt.xlabel("p" if args.mode == "pscan" else "rho (penalty weight)")
        if args.mode == "rho":
            plt.xscale("log")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        out = f"baseline_{args.mode}.png"
        plt.savefig(out, dpi=150)
        print("Plot saved:", out)


if __name__ == "__main__":
    main()
