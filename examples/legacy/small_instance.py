"""Legacy: solve a JSON or SimBench instance by enumerating states Z (no QUBO).

The true optimum it prints is also used by ../qubo_adapter.py to check the QUBO.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import sys

# shared modules live one folder up (examples/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classical_solver import enumerate_feasible, solve_exact  # noqa: E402
from redispatch_instance import RedispatchInstance  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', type=Path, help='Path to a JSON instance file')
    parser.add_argument('--simbench', action='store_true', help='Build an instance from SimBench (line first, then generators)')
    parser.add_argument('--gens', type=int, default=3, help='SimBench: number of generators n (default: 3)')
    parser.add_argument('--states', type=int, default=5, help='SimBench: states per generator k, >= 3 (default: 5, as in the paper)')
    parser.add_argument('--weights', type=float, nargs=4, metavar=('C', 'W', 'O', 'D'),
                        help='Override objective weights in C W O D order')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--T', type=int, help='Number of SimBench times, two hours apart (default: 2)')
    parser.add_argument('--grid', default='1-EHV-mixed--0-sw')
    parser.add_argument('--output', type=Path, help='Save a new JSON file; will not overwrite')
    parser.add_argument('--max-schedules', type=int, default=1_000_000,
                        help='Refuse enumeration above this count (default: 1000000)')
    args = parser.parse_args()
    if bool(args.input) == args.simbench:
        parser.error('Provide either a JSON path or --simbench, not both')
    if args.output and args.output.exists():
        parser.error('Output already exists; choose a new path')
    if not args.simbench and args.T is not None:
        parser.error('--T applies only to --simbench')
    if args.T is not None and not 1 <= args.T <= 24:
        parser.error('--T must be 1 to 24 for the saved 192-step data at two-hour spacing')
    time_steps = [8 * t for t in range(args.T if args.T is not None else 2)]
    if not args.simbench and (args.gens != 3 or args.states != 5):
        parser.error('--gens and --states apply only to --simbench')
    if args.gens < 1 or args.states < 3:
        parser.error('--gens must be >= 1 and --states >= 3')
    if args.simbench:
        print(f'Size: n={args.gens} generators, k={args.states} states, '
              f'T={len(time_steps)} -> {args.states ** (args.gens * len(time_steps))} schedules, '
              f'{args.gens * args.states * len(time_steps)} QUBO bits', flush=True)
    if args.simbench and args.states ** (args.gens * len(time_steps)) > args.max_schedules:
        parser.error('Instance exceeds the enumeration limit (k^(n*T) schedules). '
                     'Use smaller --gens/--states/--T or raise --max-schedules.')
    try:
        if args.simbench:
            print('SimBench power flow + S.npy (as in Lee et al.)...', flush=True)
            instance, metadata = RedispatchInstance.from_simbench(
                seed=args.seed, time_steps=time_steps,
                grid=args.grid,
                gens=args.gens, states=args.states)
            data = {'instance': asdict(instance), 'metadata': metadata,
                    'weights': {'Illustrative fixed weights': [1, 0, 100, 100]}}
            print('Seed:', metadata['seed'])
            print('Profile time indices:', metadata['time_step_indices'])
            for a, gen in enumerate(metadata['generators']):
                print(f"Generator {a}: {gen['name']} (id={gen['id']}), "
                      f"levels={instance.outputs[a]}, cost={instance.unit_costs[a]:.4f}")
            print('Baseline outputs MW:', metadata['baseline_outputs_mw'])
            print('Demand MW:', instance.demand)
            print('Monitored physical line IDs:', metadata['line_ids'])
        else:
            data = json.loads(args.input.read_text(encoding='utf-8-sig'))
            instance = RedispatchInstance(**data['instance'])
        if args.weights is not None:
            data['weights'] = {'Custom weights': args.weights}
        cases = data.get('weights', {'Default': [1, 1, 100, 100]})
        if not isinstance(cases, dict) or not cases:
            raise ValueError('weights must be a nonempty object of named cases')
        from math import isfinite
        for weights in cases.values():
            if not isinstance(weights, list) or len(weights) != 4 or any(
                    not isinstance(w, (int, float)) or not isfinite(w) or w < 0
                    for w in weights):
                raise ValueError('Each weight case needs four nonnegative finite C/W/O/D weights')
        count = instance.k ** (instance.T * instance.n)
        if args.max_schedules < 1 or count > args.max_schedules:
            raise ValueError(f'{count} schedules exceeds enumeration limit; use a smaller instance or --max-schedules')
        records = enumerate_feasible(instance.T, instance.n, instance.k,
                                     instance.evaluate, instance.is_feasible)
        print(f'{instance.n} generators, {instance.T} time steps, '
              f'{instance.k} states, {instance.L} line constraint rows')
        print(f'{count} schedules; {len(records)} feasible schedules')
        for name, weights in cases.items():
            states, terms, objective = solve_exact(records, weights)
            schedule = [states[t * instance.n:(t + 1) * instance.n]
                        for t in range(instance.T)]
            print(f'\n{name}; weights (C,W,O,D)={weights}')
            print(f'  States by time: {schedule}')
            print(f'  C,W,O,D = {terms}; objective = {objective}')
        if args.output:
            with args.output.open('x', encoding='utf-8') as stream:
                json.dump(data, stream, indent=2)
            print('Saved:', args.output)
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()

