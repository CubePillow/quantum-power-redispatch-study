"""Encode an instance as the original repository's QUBO and solve it.

JSON or SimBench instance -> RedispatchQUBO (package/.../qubo.py) -> Q
-> QUBO solver -> decode -> C, W, O, D
-> compared with the exact enumeration of classical_solver.py.

QUBO solvers (--solver):
  exact  dimod.ExactSolver: all 2^(n*k*T) bit strings (exact)
  tabu   dwave.samplers.TabuSampler on the whole QUBO (heuristic)
  aexp   alpha-expansion of the original repository (decompose.py +
         solver/subsolvers.py) with TabuSampler as subsolver, as in
         Lee et al. and scripts/run_simbench_tabu.py (heuristic)

Usage:
    python qubo_adapter.py --simbench --seed 1 --gens 2 --states 3 --T 2 --solver all
"""
import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'package'))
try:
    import qubolite  # noqa: F401
except ImportError:
    # qubolite does not build on Windows. qubo.py only imports it; the
    # alpha-expansion code only uses ql.qubo(Q): .m (matrix) and Q(x)
    # (energy x^T Q x). This minimal stand-in provides exactly that.
    import types

    class _Qubo:
        def __init__(self, m):
            self.m = np.asarray(m, dtype=float)

        def __call__(self, x):
            x = np.asarray(x, dtype=float)
            return float(x @ self.m @ x)

    sys.modules['qubolite'] = types.ModuleType('qubolite')
    sys.modules['qubolite'].qubo = _Qubo
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


def solve_qubo_exact(Q, max_bits=20):
    """Exact minimum of x^T Q x with dimod.ExactSolver (all 2^N bit strings)."""
    import dimod
    N = len(Q)
    if N > max_bits:
        raise ValueError(f'{N} bits > {max_bits}; use a smaller instance')
    best = dimod.ExactSolver().sample(dimod.BinaryQuadraticModel(Q, 'BINARY')).first
    return np.array([best.sample[i] for i in range(N)]), best.energy


def solve_tabu(Q, num_reads=20, seed=0):
    """dwave.samplers.TabuSampler on the whole QUBO (with penalties)."""
    import dimod
    from dwave.samplers import TabuSampler
    best = TabuSampler().sample(dimod.BinaryQuadraticModel(Q, 'BINARY'),
                                num_reads=num_reads, seed=seed).first
    return np.array([best.sample[i] for i in range(len(Q))]), best.energy


def solve_aexp(Q_obj, inst, sub_size=6, max_runs=200):
    """alpha-expansion + Tabu from the original repository, as in
    a_exp_solver (experiments/experiments.py) and Radu's
    scripts/run_simbench_tabu.py: start with every generator off,
    only one-hot/ramp-preserving moves -> no penalty terms needed."""
    import contextlib
    import io
    import dimod
    from dwave.samplers import TabuSampler
    from quantum_energy.decompose import run_decomposer
    from quantum_energy.solver.subsolvers import BetaExpansion
    from quantum_energy.util import dict_to_array, sol_to_x

    def tabu_solver(Q_ql):
        return TabuSampler().sample(dimod.BinaryQuadraticModel(Q_ql.m, 'BINARY')).first.sample

    x0 = sol_to_x(np.zeros(inst.T * inst.n), inst.k)
    x0 = {i: x0[i] for i in range(len(x0))}
    sub = BetaExpansion(inst.T, inst.n, inst.k, sub_size, tabu_solver).subsolver_ql
    with contextlib.redirect_stdout(io.StringIO()):      # silence progress output
        x, _ = run_decomposer(Q_obj, sub, x0, max_runs=max_runs, max_stale=5)
    x = dict_to_array(x).astype(int)
    return x, float(x @ Q_obj @ x)


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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', nargs='?', type=Path, help='JSON instance file (or use --simbench)')
    parser.add_argument('--simbench', action='store_true',
                        help='build instance from SimBench: line first, then its most influential generators')
    parser.add_argument('--seed', type=int, help='which switchable line is picked, and the costs')
    parser.add_argument('--gens', type=int, default=2, help='generators n (default: 2)')
    parser.add_argument('--states', type=int, default=3, help='states per generator k, >= 3 (default: 3)')
    parser.add_argument('--T', type=int, default=2, help='time points, 2 h apart, 1-24 (default: 2)')
    parser.add_argument('--grid', default='1-EHV-mixed--0-sw', help='SimBench grid')
    parser.add_argument('--weights', type=float, nargs=4, metavar=('C', 'W', 'O', 'D'),
                        help='objective weights (default: from JSON / 1 0 100 100)')
    parser.add_argument('--output', type=Path, help='save instance as JSON (no overwrite)')
    parser.add_argument('--solver', choices=['exact', 'tabu', 'aexp', 'all'], default='exact',
                        help='QUBO solver (default: exact)')
    parser.add_argument('--num-reads', type=int, default=20, help='tabu: independent runs (default: 20)')
    parser.add_argument('--sub-size', type=int, default=6, help='aexp: subproblem size (default: 6)')
    parser.add_argument('--max-bits', type=int, default=20,
                        help='exact: limit for n*k*T; dimod.ExactSolver keeps all 2^(n*k*T) results in memory (default: 20)')
    args = parser.parse_args()
    if bool(args.input) == args.simbench:
        parser.error('Provide either a JSON path or --simbench')
    if args.output and args.output.exists():
        parser.error('Output already exists; choose a new path')
    if args.simbench:
        if not 1 <= args.T <= 24 or args.gens < 1 or args.states < 3:
            parser.error('Need --T in 1..24, --gens >= 1, --states >= 3')
        print('SimBench power flow + S.npy (as in Lee et al.)...', flush=True)
        inst, meta = RedispatchInstance.from_simbench(
            seed=args.seed, time_steps=[8 * t for t in range(args.T)],
            grid=args.grid, gens=args.gens, states=args.states)
        data = {'instance': asdict(inst), 'metadata': meta,
                'weights': {'Illustrative fixed weights': [1, 0, 100, 100]}}
        print('Seed:', meta['seed'])
        for a, gen in enumerate(meta['generators']):
            print(f"Generator {a}: {gen['name']} (id={gen['id']}), "
                  f"levels={inst.outputs[a]}, cost={inst.unit_costs[a]:.4f}")
        print('Demand MW:', inst.demand)
        print('Monitored line IDs:', meta['line_ids'])
    else:
        data = json.loads(args.input.read_text(encoding='utf-8-sig'))
        inst = RedispatchInstance(**data['instance'])
    if args.weights is not None:
        data['weights'] = {'Custom weights': args.weights}
    cases = data.get('weights', {'Default': [1, 1, 100, 100]})
    N = inst.T * inst.n * inst.k
    print(f'{inst.n} generators, {inst.T} time steps, {inst.k} states '
          f'-> {N} binary variables (qubits), 2^{N} bit strings')

    records = enumerate_feasible(inst.T, inst.n, inst.k,
                                 inst.evaluate, inst.is_feasible)
    _, _, rq0 = build_qubo(inst, [1, 1, 1, 1])
    print(f'Check Q_C/Q_S/Q_P vs exact C/W/D on {len(records)} feasible '
          f'schedules: max rel. error {check_exact_terms(inst, rq0, records):.2e}')

    solvers = {
        'exact': lambda Q, Q_obj: solve_qubo_exact(Q, max_bits=args.max_bits),
        'tabu': lambda Q, Q_obj: solve_tabu(Q, args.num_reads, args.seed or 0),
        'aexp': lambda Q, Q_obj: solve_aexp(Q_obj, inst, args.sub_size),
    }
    names = list(solvers) if args.solver == 'all' else [args.solver]
    for name, weights in cases.items():
        ex_states, ex_terms, ex_score = solve_exact(records, weights)
        Q, Q_obj, _ = build_qubo(inst, weights)
        print(f'\n{name}; weights (C,W,O,D)={weights}')
        print(f'  true optimum (Z enumeration): states={ex_states} C,W,O,D={ex_terms} score={ex_score:g}')
        for solver in names:
            try:
                x, energy = solvers[solver](Q, Q_obj)
            except ValueError as exc:
                print(f'  {solver:5}: skipped ({exc})')
                continue
            states = decode(x, inst)
            if states is None or not inst.is_feasible(states):
                print(f'  {solver:5}: infeasible bit string')
                continue
            terms = inst.evaluate(states)
            score = weighted_score(terms, weights)
            rank = 1 + sum(weighted_score(t, weights) < score - 1e-9 for _, t in records)
            verdict = ('= true optimum' if abs(score - ex_score) <= 1e-9 * max(1.0, abs(ex_score))
                       else f'rank {rank} of {len(records)}')
            print(f'  {solver:5}: states={states} C,W,O,D={terms} score={score:g} -> {verdict}')

    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2)
        print('Saved:', args.output)


if __name__ == '__main__':
    main()
