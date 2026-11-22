"""Redispatch data and objectives; optional SimBench data loader."""
from dataclasses import dataclass
from math import isfinite


COST_RANGES = {
    'hard coal': (50, 90), 'gas': (40, 100), 'solar': (30, 60),
    'nuclear': (80, 120), 'offshore wind': (70, 120),
    'onshore wind': (40, 80), 'waste': (80, 110),
    'lignite': (40, 70), 'oil': (90, 160),
    'imp0': (30, 100), 'imp1': (30, 100),
}


@dataclass
class RedispatchInstance:
    outputs: list  # generator, state; same state count for all generators
    unit_costs: list  # generator; illustrative cost per MW per time step
    demand: list  # time
    sensitivity: list  # line, generator
    limits: list  # line, time
    switch_factor: float = 1.0

    def __post_init__(self):
        self.n = len(self.outputs)
        self.T = len(self.demand)
        self.k = len(self.outputs[0]) if self.n else 0
        self.L = len(self.sensitivity)
        if min(self.n, self.T, self.k, self.L) < 1:
            raise ValueError('Need generators, time steps, states, and lines')
        if any(len(row) != self.k for row in self.outputs):
            raise ValueError('All generators must have the same state count')
        if len(self.unit_costs) != self.n:
            raise ValueError('unit_costs must have one value per generator')
        if any(len(row) != self.n for row in self.sensitivity):
            raise ValueError('sensitivity must have shape (lines, generators)')
        if len(self.limits) != self.L or any(
                len(row) != self.T for row in self.limits):
            raise ValueError('limits must have shape (lines, time steps)')
        values = (self.unit_costs + self.demand + [self.switch_factor]
                  + [v for rows in (self.outputs, self.sensitivity, self.limits)
                     for row in rows for v in row])
        if any(not isinstance(v, (int, float)) or not isfinite(v)
               for v in values):
            raise ValueError('Inputs must contain finite numbers')
        if any(v < 0 for v in self.unit_costs) or self.switch_factor < 0:
            raise ValueError('Cost coefficients must be nonnegative')
        if any(v < 0 for row in self.outputs for v in row):
            raise ValueError('Generator outputs must be nonnegative')
        if any(any(a >= b for a, b in zip(row, row[1:]))
               for row in self.outputs):
            raise ValueError('Output levels must be strictly increasing')

    @classmethod
    def from_simbench(cls, seed=None, time_steps=(0, 8),
                      grid="1-EHV-mixed--0-sw", gens=3, states=5, margin=0.1):
        """Return (instance, metadata), reusing the paper repository's fitted S.

        Selection is line first, then generators: the seed picks one line
        that `gens` generators can switch between overloaded and not
        overloaded; its `gens` most influential generators are selected and
        every line these generators can switch is monitored. All other
        resources stay fixed at their profile values.

        Saved P/L/S remain paired. gen/sgen/load ordering is checked against
        all saved profile times. Individual ext_grid labels are not verified.
        """
        import random
        from pathlib import Path
        import numpy as np
        import simbench as sb

        if grid != "1-EHV-mixed--0-sw":
            raise ValueError("Saved fitted S supports only 1-EHV-mixed--0-sw")
        time_steps = list(time_steps)
        if not time_steps or any(not isinstance(t, int) or t < 0 for t in time_steps) or any(a >= b for a, b in zip(time_steps, time_steps[1:])):
            raise ValueError("Provide increasing nonnegative time indices")
        if seed is None:
            seed = random.SystemRandom().randrange(2**32)
        rng = random.Random(seed)
        net = sb.get_simbench_net(grid)
        profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
        folder = Path(__file__).resolve().parents[1] / 'experiments/data/sensitivity'
        P, L, fitted_S = [np.load(folder / f'{name}.npy', allow_pickle=False)
                          for name in ('P', 'L', 'S')]
        counts = [len(net[name]) for name in ('gen', 'sgen', 'ext_grid', 'load')]
        if P.ndim != 2 or L.ndim != 2 or P.shape[0] != sum(counts) or L.shape != (len(net.line), P.shape[1]) or fitted_S.shape != (len(net.line), P.shape[0]):
            raise ValueError('Saved P/L/S dimensions do not match this grid')
        if time_steps[-1] >= P.shape[1]:
            raise ValueError(f'Saved data supports time indices 0 to {P.shape[1]-1}')
        if not all(np.isfinite(x).all() for x in (P, L, fitted_S)) or fitted_S.min() < 0 or fitted_S.max() > 1:
            raise ValueError('Expected finite data and fitted S in [0,1]')
        offset = 0
        for name, count in zip(('gen', 'sgen', 'ext_grid', 'load'), counts):
            if name != 'ext_grid':
                actual = profiles[(name, 'p_mw')].loc[
                    list(range(P.shape[1])), net[name].index].to_numpy().T
                if name == 'load':
                    actual = -actual
                if not np.allclose(actual, P[offset:offset+count], rtol=1e-9, atol=1e-8):
                    raise ValueError(f'Saved {name} row ordering/profile mismatch')
            offset += count
        eligible = net.gen[
            net.gen.in_service & (net.gen.min_p_mw >= 0)
            & (net.gen.max_p_mw > net.gen.min_p_mw)
            & net.gen.type.isin(COST_RANGES)]
        active = profiles[('gen', 'p_mw')].loc[time_steps, eligible.index].max(axis=0) > 0
        eligible = eligible.loc[active[active].index]
        if not isinstance(gens, int) or gens < 1:
            raise ValueError('gens must be a positive integer')
        if not isinstance(states, int) or states < 3:
            raise ValueError('states must be an integer >= 3 (off + min + max)')
        if len(eligible) < gens:
            raise ValueError(f'Only {len(eligible)} active generators with valid bounds and known cost types')
        capacities = (net.line.max_i_ka.to_numpy()
                      * net.line.from_bus.map(net.bus.vn_kv).to_numpy() * np.sqrt(3))
        valid = np.isfinite(capacities) & (capacities > 0) & net.line.in_service.to_numpy()
        candidates = np.flatnonzero(valid)
        elig_pos = net.gen.index.get_indexer(eligible.index)
        elig_max = eligible.max_p_mw.to_numpy(dtype=float)
        base_flow = fitted_S[:, :] @ P[:, time_steps]          # lines x T

        def switchable(line_ids, pos, max_p):
            """Lines whose remaining capacity lies strictly inside the flow
            range of the selected generators (all off = 0 ... all at max)."""
            S_sel = fitted_S[np.ix_(line_ids, pos)]
            limit = (capacities[line_ids, None] - base_flow[line_ids]
                     + S_sel @ P[pos][:, time_steps])
            span = (S_sel @ max_p)[:, None]
            return ((limit > margin * span) & (limit < (1 - margin) * span)).all(axis=1)

        def top_gens(line):
            order = np.argsort(-fitted_S[line, elig_pos], kind='stable')[:gens]
            return np.sort(order)

        options = [l for l in candidates
                   if switchable([l], elig_pos[top_gens(l)], elig_max[top_gens(l)])[0]]
        if not options:
            raise ValueError(f'No line can be switched by {gens} generators; try a larger --gens')
        first = rng.choice(options)
        pick = top_gens(first)
        ids = list(eligible.index[pick])
        positions = elig_pos[pick]
        chosen = candidates[switchable(candidates, positions, elig_max[pick])]
        selected = net.gen.loc[ids]
        # Paper Eq. (42): `states` levels, with zero included exactly once
        # (paper uses states=5).
        outputs = [([0.0] + np.linspace(float(row.min_p_mw),
                                       float(row.max_p_mw), states - 1).tolist()
                    if row.min_p_mw > 0 else
                    np.linspace(0.0, float(row.max_p_mw), states).tolist())
                   for _, row in selected.iterrows()]
        costs = [rng.uniform(*COST_RANGES[row['type']])
                 for _, row in selected.iterrows()]
        S = fitted_S[chosen][:, positions]
        base_outputs = P[positions][:, time_steps].T
        fixed = np.ones(P.shape[0], dtype=bool)
        fixed[positions] = False
        # Eq. (47), also fixing the unselected controllable generators.
        fixed_flow = fitted_S[chosen][:, fixed] @ P[fixed][:, time_steps]
        limits = capacities[chosen, None] - fixed_flow
        prediction = fitted_S @ P
        residual = prediction - L
        if prediction.min() < -1e-8:
            raise ValueError('Saved fitted flows violate nonnegative-flow condition')
        instance = cls(outputs=outputs, unit_costs=costs,
                       demand=base_outputs.sum(axis=1).tolist(),
                       sensitivity=S.tolist(), limits=limits.tolist())
        metadata = {
            'source': 'SimBench benchmark and original repository P/L/S arrays; not live measurements',
            'grid': grid, 'seed': seed, 'time_step_indices': time_steps,
            'profile_step_hours': 0.25,
            'generators': [{'id': int(i), 'name': str(row['name']),
                            'type': str(row['type']), 'min_p_mw': float(row.min_p_mw),
                            'max_p_mw': float(row.max_p_mw)}
                           for i, row in selected.iterrows()],
            'baseline_outputs_mw': base_outputs.tolist(),
            'line_ids': [int(net.line.index[i]) for i in chosen],
            'physical_line_count': int(len(chosen)),
            'selection': f'seed picks line {int(net.line.index[first])} among {len(options)} switchable lines; its {gens} most influential generators; all lines they can switch (margin {margin}) are monitored',
            'constraint_row_order': 'one row per physical line, paper nonnegative-flow model',
            'sensitivity_method': 'Precomputed fitted S.npy supplied by original paper repository (Appendix B)',
            'fit_rmse_mw': float(np.sqrt(np.mean(residual**2))),
            'fit_max_abs_error_mw': float(np.max(np.abs(residual))),
            'fitted_coefficient_range': [float(fitted_S.min()), float(fitted_S.max())],
            'minimum_fitted_flow_mw': float(prediction.min()),
            'row_validation': 'gen/sgen/load match all 192 saved profiles; seven ext_grid rows kept together as fixed contributions, individual labels unverified',
            'capacity_approximation': 'sqrt(3) * nominal kV * rated kA; ignores reactive power',
            'cost_source': 'Uniform samples from paper Table II; not measured costs',
            'demand_definition': 'Selected generator baseline total; all other resources fixed',
            'baseline_line_flows_mw': L[chosen][:, time_steps].T.tolist(),
            'fitted_baseline_line_flows_mw': prediction[chosen][:, time_steps].T.tolist(),
            'fixed_line_contribution_mw': fixed_flow.T.tolist(),
            'note': 'Uses supplied fit, does not rerun Algorithm 4. Only switchable lines scored; no initial-state ramp; D may be nonzero.',
        }
        return instance, metadata

    def valid_states(self, states):
        return len(states) == self.T * self.n and all(
            isinstance(s, int) and 0 <= s < self.k for s in states)

    def is_feasible(self, states):
        """State encoding selects one level implicitly; also enforce ramp."""
        return self.valid_states(states) and all(
            abs(states[(t + 1) * self.n + a] - states[t * self.n + a]) <= 1
            for t in range(self.T - 1) for a in range(self.n))

    def evaluate(self, states):
        """Raw C/W/O/D; >= overload; fixed flow offsets must be included in limits."""
        if not self.valid_states(states):
            raise ValueError('Invalid state schedule')
        power = [[self.outputs[a][states[t * self.n + a]]
                  for a in range(self.n)] for t in range(self.T)]
        c = sum(self.unit_costs[a] * power[t][a]
                for t in range(self.T) for a in range(self.n))
        w = self.switch_factor * sum(
            abs(power[t + 1][a] - power[t][a])
            for t in range(self.T - 1) for a in range(self.n))
        o = sum(sum(self.sensitivity[l][a] * power[t][a]
                    for a in range(self.n)) >= self.limits[l][t]
                for l in range(self.L) for t in range(self.T))
        d = sum((sum(power[t]) - self.demand[t]) ** 2
                for t in range(self.T))
        return c, w, o, d

