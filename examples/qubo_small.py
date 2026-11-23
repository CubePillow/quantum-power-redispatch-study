"""QUBO for a small redispatch instance (one-hot encoding), checked against enumeration.

Bit order: x[(t*n + a)*k + i] = 1  <=>  generator a is in state i at time t
(same order as the `states` list in RedispatchInstance).

Energy: E(x) = x^T Q x + offset, Q upper triangular.

Terms (following Lee et al., DSAA 2025):
  C  production cost          linear
  W  switching cost           quadratic between t and t+1 (exact if one-hot holds)
  D  (sum_a P_t,a - tau_t)^2  quadratic, constant tau^2 in offset
  O  overloads                Taylor approximation of the indicator (not exact!)
  H  one-hot penalty          (sum_i x - 1)^2
  A  ramp penalty             x_t,a,i * x_t+1,a,j for |i - j| > 1

Usage:
  python examples/qubo_small.py examples/instances/small.json
"""
import itertools
import json
import sys

import numpy as np

from redispatch_instance import RedispatchInstance


def idx(inst, t, a, i):
    return (t * inst.n + a) * inst.k + i


def add(Q, u, v, w):
    """Add w * x_u * x_v to an upper-triangular Q (x_u * x_u = x_u)."""
    if u > v:
        u, v = v, u
    Q[u, v] += w


def term_C(inst):
    N = inst.T * inst.n * inst.k
    Q = np.zeros((N, N))
    for t in range(inst.T):
        for a in range(inst.n):
            for i in range(inst.k):
                add(Q, idx(inst, t, a, i), idx(inst, t, a, i), inst.unit_costs[a] * inst.outputs[a][i])
    return Q, 0.0


def term_W(inst):
    N = inst.T * inst.n * inst.k
    Q = np.zeros((N, N))
    for t in range(inst.T - 1):
        for a in range(inst.n):
            for i in range(inst.k):
                for j in range(inst.k):
                    w = inst.switch_factor * abs(inst.outputs[a][j] - inst.outputs[a][i])
                    if w:
                        add(Q, idx(inst, t, a, i), idx(inst, t + 1, a, j), w)
    return Q, 0.0


def term_D(inst):
    N = inst.T * inst.n * inst.k
    Q = np.zeros((N, N))
    off = 0.0
    for t in range(inst.T):
        vars_t = [(idx(inst, t, a, i), inst.outputs[a][i]) for a in range(inst.n) for i in range(inst.k)]
        for (u, pu), (v, pv) in itertools.combinations_with_replacement(vars_t, 2):
            add(Q, u, v, pu * pv if u == v else 2 * pu * pv)
        for u, pu in vars_t:
            add(Q, u, u, -2 * inst.demand[t] * pu)
        off += inst.demand[t] ** 2
    return Q, off


def term_O_taylor(inst):
    """Lee et al.: 1[L >= M] ~ 1 - h + h^2/2 with h = (M - L) / h_max, h_max = M - L(all lowest)."""
    N = inst.T * inst.n * inst.k
    Q = np.zeros((N, N))
    off = 0.0
    for l in range(inst.L):
        for t in range(inst.T):
            M = inst.limits[l][t]
            base = sum(inst.sensitivity[l][a] * inst.outputs[a][0] for a in range(inst.n))
            hmax = M - base
            # h = (M - sum_u s_u x_u) / hmax = h0 - sum_u g_u x_u
            h0 = M / hmax
            g = {idx(inst, t, a, i): inst.sensitivity[l][a] * inst.outputs[a][i] / hmax
                 for a in range(inst.n) for i in range(inst.k)}
            # 1 - h + h^2/2
            off += 1 - h0 + h0 ** 2 / 2
            for u, gu in g.items():
                add(Q, u, u, gu - h0 * gu)          # -h  -> +g_u x_u ; h^2/2 -> -h0 g_u x_u
            for (u, gu), (v, gv) in itertools.combinations_with_replacement(g.items(), 2):
                add(Q, u, v, gu * gv / 2 if u == v else gu * gv)
    return Q, off


def term_onehot(inst):
    N = inst.T * inst.n * inst.k
    Q = np.zeros((N, N))
    off = 0.0
    for t in range(inst.T):
        for a in range(inst.n):
            us = [idx(inst, t, a, i) for i in range(inst.k)]
            for u in us:
                add(Q, u, u, -1.0)                   # x^2 - 2x = -x
            for u, v in itertools.combinations(us, 2):
                add(Q, u, v, 2.0)
            off += 1.0
    return Q, off


def term_ramp(inst):
    N = inst.T * inst.n * inst.k
    Q = np.zeros((N, N))
    for t in range(inst.T - 1):
        for a in range(inst.n):
            for i in range(inst.k):
                for j in range(inst.k):
                    if abs(i - j) > 1:
                        add(Q, idx(inst, t, a, i), idx(inst, t + 1, a, j), 1.0)
    return Q, 0.0


def build_qubo(inst, mu=(1, 0, 0), lam_D=100, lam_H=1000, lam_A=1000):
    """Total QUBO: mu_C C + mu_W W + mu_O O_taylor + lam_D D + lam_H onehot + lam_A ramp."""
    parts = [(mu[0], term_C(inst)), (mu[1], term_W(inst)), (mu[2], term_O_taylor(inst)),
             (lam_D, term_D(inst)), (lam_H, term_onehot(inst)), (lam_A, term_ramp(inst))]
    N = inst.T * inst.n * inst.k
    Q, off = np.zeros((N, N)), 0.0
    for w, (Qp, op) in parts:
        Q += w * Qp
        off += w * op
    return Q, off


def energy(Q, off, x):
    return float(x @ Q @ x + off)


def bits(inst, m):
    N = inst.T * inst.n * inst.k
    return np.array([(m >> (N - 1 - b)) & 1 for b in range(N)], dtype=float)


def decode(inst, x):
    """Bits -> states, or None if one-hot is violated."""
    states = []
    for t in range(inst.T):
        for a in range(inst.n):
            block = x[idx(inst, t, a, 0): idx(inst, t, a, 0) + inst.k]
            if block.sum() != 1:
                return None
            states.append(int(np.argmax(block)))
    return states


def check(inst):
    N = inst.T * inst.n * inst.k
    terms = {"C": term_C(inst), "W": term_W(inst), "D": term_D(inst), "O": term_O_taylor(inst),
             "onehot": term_onehot(inst), "ramp": term_ramp(inst)}
    n_onehot = n_feasible = 0
    max_err = {"C": 0.0, "W": 0.0, "D": 0.0}
    pen_ok = True
    o_pairs = []
    for m in range(2 ** N):
        x = bits(inst, m)
        st = decode(inst, x)
        oh = energy(*terms["onehot"], x)
        rp = energy(*terms["ramp"], x)
        if st is None:
            pen_ok &= oh > 0
            continue
        n_onehot += 1
        pen_ok &= abs(oh) < 1e-9
        feas = inst.is_feasible(st)
        n_feasible += feas
        pen_ok &= (abs(rp) < 1e-9) == feas
        c, w, o, d = inst.evaluate(st)
        for name, ref in (("C", c), ("W", w), ("D", d)):
            max_err[name] = max(max_err[name], abs(energy(*terms[name], x) - ref))
        o_pairs.append((o, energy(*terms["O"], x)))
    print(f"{N} qubits, {2 ** N} bitstrings: {n_onehot} one-hot, {n_feasible} feasible")
    print("max |QUBO - enumeration| on one-hot bitstrings:",
          ", ".join(f"{k} = {v:.2e}" for k, v in max_err.items()))
    print("one-hot / ramp penalties are 0 exactly on valid bitstrings and > 0 otherwise:", pen_ok)
    print("overload O: exact count vs Taylor value (min .. max over one-hot bitstrings)")
    for o in sorted({p[0] for p in o_pairs}):
        vals = [p[1] for p in o_pairs if p[0] == o]
        print(f"  O = {o}: Taylor {min(vals):.3f} .. {max(vals):.3f}  ({len(vals)} plans)")


def ground_state(inst, Q, off):
    N = inst.T * inst.n * inst.k
    best = min(range(2 ** N), key=lambda m: energy(Q, off, bits(inst, m)))
    x = bits(inst, best)
    st = decode(inst, x)
    return st, energy(Q, off, x), (inst.evaluate(st) if st is not None else None)


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    inst = RedispatchInstance(**data["instance"])
    check(inst)
    print()
    for name, w in data["weights"].items():
        Q, off = build_qubo(inst, mu=(w[0], w[1], w[2]), lam_D=w[3])
        st, e, cwod = ground_state(inst, Q, off)
        print(f"{name}: QUBO ground state {st}, C,W,O,D = {cwod}, energy = {e:.2f}")
