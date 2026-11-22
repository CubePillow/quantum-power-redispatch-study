"""Print SimBench devices, output setpoints, and operating bounds."""
import argparse


def print_devices(rows=5, full=False):
    import simbench as sb

    net = sb.get_simbench_net('1-EHV-mixed--0-sw')
    print('SimBench setpoints, not live measurements. Only gen is adjusted by the original model.')
    for kind in ('gen', 'sgen', 'ext_grid', 'load'):
        table = net[kind]
        columns = [c for c in ('name', 'type', 'in_service', 'p_mw',
                               'min_p_mw', 'max_p_mw') if c in table.columns]
        view = table[columns].copy()
        view['selected_by_model'] = table.in_service if kind == 'gen' else False
        print(f'\n{kind}: {len(table)} devices')
        print((view if full else view.head(rows)).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=5)
    parser.add_argument('--full', action='store_true', help='Print every device')
    args = parser.parse_args()
    if args.rows < 1:
        parser.error('rows must be positive')
    try:
        print_devices(args.rows, args.full)
    except (ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
