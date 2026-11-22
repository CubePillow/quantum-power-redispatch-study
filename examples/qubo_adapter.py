"""Build the original repository's QUBO from a JSON instance and check it.

JSON instance -> RedispatchQUBO (package/quantum_energy/problem/qubo.py)
-> brute force over all 2^(T*n*k) bit strings -> compare with the exact
enumeration of classical_solver.py.

Usage:
    python qubo_adapter.py instances/small.json
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'package'))
try:  # qubo.py imports qubolite but does not use it
    import qubolite  # noqa: F401
except ImportError:
    import types
    sys.modules['qubolite'] = types.ModuleType('qubolite')
from quantum_energy.problem.qubo import RedispatchQUBO  # noqa: E402

if __package__:
    from .classical_solver import enumerate_feasible, solve_exact, weighted_score
    from .redispatch_instance import RedispatchInstance
else:
    from classical_solver import enumerate_feasible, solve_exact, weighted_score
    from redispatch_instance import RedispatchInstance


def instance_to_model(inst: RedispatchInstance):
    """Duck-typed stand-in for RedispatchModel with exactly the fields
    RedispatchQUBO reads. Bit order: x[(t*n + a)*k + i]."""
    T, n, k = inst.T, inst.n, inst.k
    P = np.array(inst.outputs, dtype=float)                  # n x k
    PE = np.broadcast_to(P, (T, n, k)).copy()                # T x n x k
    cost = np.array(inst.unit_costs, dtype=float)
    cost_vals = PE * cost[None, :, None]                     # c_{t,a,i}
    # gamma*|p_i - p_i'|; the repository only charges adjacent levels,
    # which is identical for every ramp-feasible schedule
    diff = np.abs(P[:, :, None] - P[:, None, :])
    adj = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) == 1
    switch_cost = inst.switch_factor * diff * adj[None]
    return SimpleNamespace(
        T=T, n_resources=n, k=k,
        prod_vals=P, prod_vals_eff=PE,
        cost_vals=cost_vals, switch_cost=switch_cost,
        S_resos=np.array(inst.sensitivity, dtype=float),     # L x n
        M=np.array(inst.limits, dtype=float),                # L x T
        # the original RedispatchQUBO recomputes M from these three:
        max_load=np.array(inst.limits, dtype=float),
        S_loads=np.zeros((len(inst.sensitivity), 0)),
        prod_static=np.zeros((0, T)),
        S_sgens=np.zeros((len(inst.sensitivity), 0)),
        sgen_prods=np.zeros((0, T)),
        prod_target=np.array(inst.demand, dtype=float),
    )


def build_qubo(inst, weights, lam_hard=None):
    """Q = w_C*Q_C + w_W*Q_S + w_O*Q_O + w_D*Q_P + lam_H*Q_H + lam_A*Q_A.

    The repository's get_qubo_obj omits Q_H (one-hot) and Q_A (ramp)
    because alpha-expansion keeps them by construction; a plain QUBO
    solver (or baseline QAOA) needs them as penalties.
    """
    rq = RedispatchQUBO(instance_to_model(inst))
    w_c, w_w, w_o, w_d = weights
    Q_obj = rq.get_qubo_obj(lam_p=w_d, lam_o=w_o, lam_c=w_c, lam_s=w_w)
    if lam_hard is None:
        # any objective change is bounded by sum |Q_obj|; a single
        # constraint violation costs >= 1 (one-hot) or >= min entry (ramp)
        lam_hard = 1.0 + np.abs(Q_obj).sum()
    a_min = rq.qubo_adjacent[rq.qubo_adjacent > 0]
    lam_a = lam_hard / a_min.min() if a_min.size else 0.0
    Q = Q_obj + lam_hard * rq.qubo_one_hot + lam_a * rq.qubo_adjacent
    return Q, Q_obj, rq


def brute_force_qubo(Q, chunk=1 << 16):
    """Exact minimum of x^T Q x over all bit strings (N <= ~24)."""
    N = len(Q)
    if N > 24:
        raise ValueError(f'{N} bits is too many for brute force')
    best_e, best_x = np.inf, None
    bits = np.arange(N)[::-1]
    for start in range(0, 1 << N, chunk):
        idx = np.arange(start, min(start + chunk, 1 << N))
        X = ((idx[:, None] >> bits) & 1).astype(float)
        E = np.einsum('bi,ij,bj->b', X, Q, X)
        j = int(np.argmin(E))
        if E[j] < best_e:
            best_e, best_x = E[j], X[j].astype(int)
    return best_x, best_e


def decode(x, inst):
    """Bit string -> state tuple, or None if one-hot is violated."""
    blocks = np.asarray(x).reshape(inst.T * inst.n, inst.k)
    if not np.all(blocks.sum(axis=1) == 1):
        return None
    return tuple(int(i) for i in blocks.argmax(axis=1))


def encode(states, inst):
    return np.eye(inst.k, dtype=int)[list(states)].ravel()


def check_exact_terms(inst, rq, records):
    """Q_C, Q_S and Q_P must reproduce C, W, D exactly on every
    feasible schedule (Q_O is only an approximation of O)."""
    worst = 0.0
    for states, (c, w, o, d) in records:
        x = encode(states, inst)
        got = (x @ rq.qubo_cost @ x, x @ rq.qubo_switch @ x,
               x @ rq.qubo_power_e @ x)
        worst = max(worst, *(abs(g - e) / max(1.0, abs(e))
                             for g, e in zip(got, (c, w, d))))
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8-sig'))
    inst = RedispatchInstance(**data['instance'])
    cases = data.get('weights', {'Default': [1, 1, 100, 100]})
    N = inst.T * inst.n * inst.k
    print(f'{inst.n} generators, {inst.T} time steps, {inst.k} states '
          f'-> {N} binary variables (qubits)')

    records = enumerate_feasible(inst.T, inst.n, inst.k,
                                 inst.evaluate, inst.is_feasible)
    _, _, rq0 = build_qubo(inst, [1, 1, 1, 1])
    print(f'Check Q_C/Q_S/Q_P vs exact C/W/D on {len(records)} feasible '
          f'schedules: max rel. error {check_exact_terms(inst, rq0, records):.2e}')

    for name, weights in cases.items():
        ex_states, ex_terms, ex_score = solve_exact(records, weights)
        Q, _, _ = build_qubo(inst, weights)
        x, energy = brute_force_qubo(Q)
        states = decode(x, inst)
        print(f'\n{name}; weights (C,W,O,D)={weights}')
        print(f'  exact : states={ex_states} C,W,O,D={ex_terms} score={ex_score:g}')
        if states is None or not inst.is_feasible(states):
            print(f'  QUBO  : infeasible bit string {x.tolist()}')
            continue
        terms = inst.evaluate(states)
        score = weighted_score(terms, weights)
        rank = 1 + sum(weighted_score(t, weights) < score - 1e-9
                       for _, t in records)
        print(f'  QUBO  : states={states} C,W,O,D={terms} score={score:g}')
        print('  ->', 'same optimum' if abs(score - ex_score) < 1e-9 else
              f'differs (rank {rank} of {len(records)}; '
              'Q_O only approximates the overload count)')


if __name__ == '__main__':
    main()
