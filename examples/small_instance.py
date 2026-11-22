"""Solve an explicit JSON input or a random SimBench instance classically."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

if __package__:
    from .classical_solver import enumerate_feasible, solve_exact
    from .redispatch_instance import RedispatchInstance
else:
    from classical_solver import enumerate_feasible, solve_exact
    from redispatch_instance import RedispatchInstance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', type=Path, help='Path to a JSON instance file')
    parser.add_argument('--simbench', action='store_true', help='Generate three random gen units instead of reading JSON')
    parser.add_argument('--weights', type=float, nargs=4, metavar=('C', 'W', 'O', 'D'),
                        help='Override objective weights in C W O D order')
    parser.add_argument('--seed', type=int)
    times = parser.add_mutually_exclusive_group()
    times.add_argument('--T', type=int, help='Number of SimBench times, two hours apart (default: 2)')
    times.add_argument('--time-steps', type=int, nargs='+', help='Explicit increasing profile indices')
    parser.add_argument('--generate-only', action='store_true', help='Build and optionally save input without enumeration')
    parser.add_argument('--lines', type=int, default=2)
    parser.add_argument('--grid', default='1-EHV-mixed--0-sw')
    parser.add_argument('--output', type=Path, help='Save a new JSON file; will not overwrite')
    parser.add_argument('--max-schedules', type=int, default=1_000_000,
                        help='Refuse enumeration above this count (default: 1000000)')
    args = parser.parse_args()
    if bool(args.input) == args.simbench:
        parser.error('Provide either a JSON path or --simbench, not both')
    if args.output and args.output.exists():
        parser.error('Output already exists; choose a new path')
    if not args.simbench and (args.T is not None or args.time_steps is not None):
        parser.error('--T and --time-steps apply only to --simbench')
    if args.T is not None and not 1 <= args.T <= 24:
        parser.error('--T must be 1 to 24 for the saved 192-step data at two-hour spacing')
    time_steps = args.time_steps if args.time_steps is not None else [8 * t for t in range(args.T if args.T is not None else 2)]
    if args.simbench and (not time_steps or any(t < 0 or t > 191 for t in time_steps) or any(a >= b for a, b in zip(time_steps, time_steps[1:]))):
        parser.error('Time indices must increase and be in 0..191')
    if args.simbench and not args.generate_only and 5 ** (3 * len(time_steps)) > args.max_schedules:
        parser.error('Requested T exceeds the enumeration limit. Use --generate-only to build the input without solving, or a smaller T.')
    try:
        if args.simbench:
            print('Loading SimBench benchmark (paper fitted S; synthetic costs)...', flush=True)
            instance, metadata = RedispatchInstance.from_simbench(
                seed=args.seed, time_steps=time_steps,
                lines=args.lines, grid=args.grid)
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
            print('Fitted S training RMSE MW:', metadata['fit_rmse_mw'])
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
        if args.generate_only:
            print(f'Built {instance.T} time steps; optimization skipped.')
            if args.output:
                with args.output.open('x', encoding='utf-8') as stream:
                    json.dump(data, stream, indent=2)
                print('Saved:', args.output)
            return
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

