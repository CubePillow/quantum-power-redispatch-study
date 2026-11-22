"""Exact classical baseline as a mixed-integer program, solved with Gurobi.

Usage:
  python3 examples/milp_solver.py examples/instances/small.json
  python3 examples/milp_solver.py examples/instances/small.json --check
"""
import argparse
import json
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

if __package__:
    from .classical_solver import enumerate_feasible, solve_exact
    from .redispatch_instance import RedispatchInstance
else:
    from classical_solver import enumerate_feasible, solve_exact
    from redispatch_instance import RedispatchInstance


def build_model(inst, weights, eps=1e-6):
    """Return (model, x, terms) for weights ordered as C, W, O, D.

    eps separates 'L >= limit' (overloaded) from 'L < limit' (not overloaded).
    """
    T, n, k, L = inst.T, inst.n, inst.k, inst.L
    p = inst.outputs
    m = gp.Model('redispatch')
    m.Params.OutputFlag = 0

    x = m.addVars(T, n, k, vtype=GRB.BINARY, name='x')      # one-hot state choice
    w = m.addVars(T - 1, n, lb=0.0, name='w')               # |P[t+1,a] - P[t,a]|
    o = m.addVars(T, L, vtype=GRB.BINARY, name='o')         # line l overloaded at t

    power = {(t, a): gp.quicksum(p[a][i] * x[t, a, i] for i in range(k))
             for t in range(T) for a in range(n)}
    state = {(t, a): gp.quicksum(i * x[t, a, i] for i in range(k))
             for t in range(T) for a in range(n)}
    total = {t: gp.quicksum(power[t, a] for a in range(n)) for t in range(T)}
    flow = {(t, l): gp.quicksum(inst.sensitivity[l][a] * power[t, a] for a in range(n))
            for t in range(T) for l in range(L)}

    C = gp.quicksum(inst.unit_costs[a] * power[t, a] for t in range(T) for a in range(n))
    W = inst.switch_factor * w.sum()
    O = o.sum()
    D = gp.quicksum((total[t] - inst.demand[t]) * (total[t] - inst.demand[t])
                    for t in range(T))
    wC, wW, wO, wD = weights
    m.setObjective(wC * C + wW * W + wO * O + wD * D, GRB.MINIMIZE)

    # hard constraints
    m.addConstrs((x.sum(t, a, '*') == 1 for t in range(T) for a in range(n)), 'onehot')
    m.addConstrs((state[t + 1, a] - state[t, a] <= 1
                  for t in range(T - 1) for a in range(n)), 'ramp_up')
    m.addConstrs((state[t, a] - state[t + 1, a] <= 1
                  for t in range(T - 1) for a in range(n)), 'ramp_down')
    # w >= |P[t+1,a] - P[t,a]|  (tight because w is minimized with W >= 0)
    m.addConstrs((w[t, a] >= power[t + 1, a] - power[t, a]
                  for t in range(T - 1) for a in range(n)), 'abs_pos')
    m.addConstrs((w[t, a] >= power[t, a] - power[t + 1, a]
                  for t in range(T - 1) for a in range(n)), 'abs_neg')
    # o is exactly the indicator 1[flow >= limit], also when wO = 0:
    #   o = 0  =>  flow <  limit   (so flow >= limit forces o = 1)
    #   o = 1  =>  flow >= limit   (so o cannot be 1 without an overload)
    m.addConstrs(((o[t, l] == 0) >> (flow[t, l] <= inst.limits[l][t] - eps)
                  for t in range(T) for l in range(L)), 'overload_off')
    m.addConstrs(((o[t, l] == 1) >> (flow[t, l] >= inst.limits[l][t])
                  for t in range(T) for l in range(L)), 'overload_on')
    return m, x, (C, W, O, D)


def solve(inst, weights):
    """Return (states, terms, objective) in the same format as solve_exact."""
    m, x, terms = build_model(inst, weights)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise ValueError(f'Gurobi status {m.Status} (not optimal)')
    states = tuple(next(i for i in range(inst.k) if x[t, a, i].X > 0.5)
                   for t in range(inst.T) for a in range(inst.n))
    # Report C, W, O, D from the chosen states with the same code as the
    # enumeration. (An auxiliary variable such as w is only bounded from
    # below and may be loose when its weight is 0; the states are exact.)
    values = tuple(float(v) for v in inst.evaluate(states))
    objective = sum(wt * v for wt, v in zip(weights, values))
    return states, values, objective


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path, help='Path to a JSON instance file')
    parser.add_argument('--check', action='store_true',
                        help='Compare every weight case with exact enumeration')
    parser.add_argument('--max-schedules', type=int, default=1_000_000,
                        help='Largest instance that --check enumerates (default: 1000000)')
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding='utf-8-sig'))
    inst = RedispatchInstance(**data['instance'])
    cases = data.get('weights', {'Default': [1, 1, 100, 100]})
    print(f'{inst.n} generators, {inst.T} time steps, {inst.k} states, {inst.L} lines')
    records = None
    if args.check:
        count = inst.k ** (inst.T * inst.n)
        if count > args.max_schedules:
            print(f'--check skipped: {count} schedules exceed --max-schedules {args.max_schedules}')
        else:
            records = enumerate_feasible(inst.T, inst.n, inst.k, inst.evaluate, inst.is_feasible)

    for name, weights in cases.items():
        states, terms, objective = solve(inst, weights)
        schedule = [list(states[t * inst.n:(t + 1) * inst.n]) for t in range(inst.T)]
        print(f'\n{name}; weights (C,W,O,D)={weights}')
        print(f'  States by time: {schedule}')
        print(f'  C,W,O,D = {terms}; objective = {objective}')
        if records is not None:
            _, _, best = solve_exact(records, weights)
            status = 'OK' if abs(best - objective) <= 1e-6 * max(1.0, abs(best)) else 'MISMATCH'
            print(f'  enumeration objective = {best}  -> {status}')


if __name__ == '__main__':
    main()
