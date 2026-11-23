"""Run a required JSON redispatch input with the classical exact solver."""
import argparse
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
    parser.add_argument('input', type=Path, help='Path to a JSON instance file')
    parser.add_argument('--max-schedules', type=int, default=1_000_000,
                        help='Refuse enumeration above this count (default: 1000000)')
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text(encoding='utf-8-sig'))
        instance = RedispatchInstance(**data['instance'])
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
              f'{instance.k} states, {instance.L} lines')
        print(f'{count} schedules; {len(records)} feasible schedules')
        for name, weights in cases.items():
            states, terms, objective = solve_exact(records, weights)
            schedule = [states[t * instance.n:(t + 1) * instance.n]
                        for t in range(instance.T)]
            print(f'\n{name}; weights (C,W,O,D)={weights}')
            print(f'  States by time: {schedule}')
            print(f'  C,W,O,D = {terms}; objective = {objective}')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()

