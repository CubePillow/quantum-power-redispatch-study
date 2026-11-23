"""Preview the SimBench data used by the model, in slide order.

  1 net       element counts (what is controllable, what is fixed)
  2 flow      time profiles -> power flow -> injections -> demand tau
  3 levels    min/max power of gen -> levels p
  4 costs     plant type -> price per MWh (Lee et al., Tab. II)
  5 capacity  rated current and voltage -> V*I*sqrt(3)
  6 s         sensitivity S: flow = S * p
  7 instance  a small instance built from all of the above
"""
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID = '1-EHV-mixed--0-sw'
SECTIONS = ('net', 'flow', 'levels', 'costs', 'capacity', 's', 'instance')


def title(text):
    print(f'\n===== {text} =====')


def show_net(net):
    title('1 Netz: steuerbar vs. fest')
    print(f'gen       {len(net.gen):4}  controllable (decision variables)')
    print(f'sgen      {len(net.sgen):4}  fixed (wind, solar, biomass, ...)')
    print(f'load      {len(net.load):4}  fixed (consumption)')
    print(f'ext_grid  {len(net.ext_grid):4}  fixed (neighbouring grids)')
    print(f'line      {len(net.line):4}  overload checked')


def show_flow(net, res, steps, rows):
    import numpy as np
    import simbench as sb
    title('2 Zeitprofile + Lastfluss -> Einspeisung -> Bedarf tau')
    prof = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    print(f'profiles: 15 min steps; the model uses steps {steps} (2 h apart)')
    print(f'gen p_mw, first {rows} gens (rows = steps):')
    print(prof[('gen', 'p_mw')].iloc[steps, :rows].round(1).to_string())
    print(f'load p_mw, first {rows} loads:')
    print(prof[('load', 'p_mw')].iloc[steps, :rows].round(1).to_string())
    print('power flow result, sum over all elements (MW):')
    gen, sgen, rest = (res.reso_prods[:, steps].sum(0), res.sgen_prods[:, steps].sum(0),
                       res.load_prods[:, steps].sum(0))
    print(f'  gen {np.round(gen)} + sgen {np.round(sgen)} + load/ext_grid {np.round(rest)}'
          f' = {np.round(gen + sgen + rest)} (line losses)')
    print('-> generation = consumption + losses; with all other elements fixed, the selected')
    print('   generators must produce what they produced in the power flow: tau = their sum')


def show_levels(net, rows, k):
    from quantum_energy.problem.modeller import create_power_prod_levels
    title(f'3 Min/Max-Leistung -> Stufen p (k = {k}: off, min ... max)')
    view = net.gen[['name', 'min_p_mw', 'max_p_mw']].head(rows).copy()
    view['levels p (MW)'] = [[round(v, 1) for v in create_power_prod_levels(a, b, k)]
                             for a, b in zip(view.min_p_mw, view.max_p_mw)]
    print(view.to_string())


def show_costs(net):
    title('4 Kraftwerkstyp -> Kosten c = c* x p   (c* ~ U(min, max) EUR/MWh, Lee et al. Tab. II)')
    table = {'hard coal': (50, 90), 'gas': (40, 100), 'lignite': (40, 70), 'oil': (90, 160),
             'waste': (80, 110), 'imp0': (30, 100), 'imp1': (30, 100)}
    for typ, count in net.gen.type.value_counts().items():
        lo, hi = table.get(typ, ('?', '?'))
        print(f'  {typ:10} {count:4} gens   c* in [{lo}, {hi}] EUR/MWh')


def show_capacity(net, rows):
    title('5 Nennstrom & Spannung -> Kapazitaet V*I*sqrt(3)')
    view = net.line[['name', 'max_i_ka', 'max_mw_per_line']].head(rows).copy()
    view.insert(1, 'V (kV)', net.bus.vn_kv.loc[net.line.from_bus.head(rows)].to_numpy())
    view.columns = ['name', 'V (kV)', 'I (kA)', 'capacity (MW)']
    print(view.round(1).to_string())


def show_s(S, rows):
    import numpy as np
    title('6 Sensitivitaet S (aus Lee et al.): Fluss = S * p')
    print(f'{S.shape[0]} lines x {S.shape[1]} elements, values {S.min():.2f}..{S.max():.2f}')
    print(f'first {rows} lines x first {rows} gens  (0.96 = 96% of 1 MW of gen 0 flows over line 0):')
    print(np.round(S[:rows, :rows], 2))


def show_instance(res, S, seed, gens, states, T):
    import numpy as np
    from quantum_energy.problem.simulator import get_even_first_T_time_points
    from quantum_energy.problem.reduce import create_line_aware_subset
    Rd = create_line_aware_subset(get_even_first_T_time_points(res, T, 2), S, k=states,
                                  n_lines=None, n_gen=gens, seed=seed, line_seed=seed)
    title(f'7 Kleine Instanz (seed {seed}, n={gens}, k={states}, T={T} -> {gens * states * T} qubits)')
    print('generators:', list(Rd.net.gen.name), list(Rd.net.gen.type))
    print('levels p (MW):', np.round(Rd.prod_vals, 1).tolist())
    print('price c* (EUR/MWh):', np.round(Rd.prod_cost_multi, 2).tolist())
    print('power flow output (MW):', np.round(Rd.prod_resos[:, 0], 1).tolist(),
          '-> tau =', np.round(Rd.prod_target, 1).tolist())
    print('lines:', list(Rd.net.line.name))
    cap = Rd.net.line.max_mw_per_line.to_numpy()
    print('capacity (MW):', np.round(cap, 1).tolist(),
          ' fixed flow t0 (MW):', np.round(np.asarray(Rd.fixed_flow)[:, 0], 1).tolist(),
          ' -> M t0 (MW):', np.round(np.asarray(Rd.M[:, 0], float), 1).tolist())
    print('S (lines x selected gens):', np.round(np.asarray(Rd.S_resos, float), 3).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--show', nargs='+', choices=SECTIONS, default=list(SECTIONS),
                        help='sections to print (default: all)')
    parser.add_argument('--rows', type=int, default=3, help='rows per table (default: 3)')
    parser.add_argument('--T', type=int, default=2, help='time points, 2 h apart (default: 2)')
    parser.add_argument('--seed', type=int, default=1, help='instance: line and costs (default: 1)')
    parser.add_argument('--gens', type=int, default=2, help='instance: generators n (default: 2)')
    parser.add_argument('--states', type=int, default=3, help='states k (default: 3)')
    args = parser.parse_args()
    if args.rows < 1 or not 1 <= args.T <= 24 or args.states < 3:
        parser.error('--rows >= 1, --T in 1..24, --states >= 3')
    sys.path.insert(0, str(ROOT / 'package'))
    try:
        import numpy as np
        import simbench as sb
        from quantum_energy.problem.simulator import add_line_capacity_to_net, run_and_record_profiles
        net = sb.get_simbench_net(GRID)
        add_line_capacity_to_net(net)
        steps = [8 * t for t in range(args.T)]
        res = S = None
        if {'flow', 'instance'} & set(args.show):
            print('Running SimBench power flow (192 steps, ~15 s)...', flush=True)
            res = run_and_record_profiles(copy.deepcopy(net), 0)   # keep `net` unchanged
        if {'s', 'instance'} & set(args.show):
            S = np.load(ROOT / 'experiments/data/sensitivity/S.npy').astype('float32')
        shows = {
            'net': lambda: show_net(net),
            'flow': lambda: show_flow(net, res, steps, args.rows),
            'levels': lambda: show_levels(net, args.rows, args.states),
            'costs': lambda: show_costs(net),
            'capacity': lambda: show_capacity(net, args.rows),
            's': lambda: show_s(S, args.rows),
            'instance': lambda: show_instance(res, S, args.seed, args.gens, args.states, args.T),
        }
        for section in SECTIONS:
            if section in args.show:
                shows[section]()
    except (ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
