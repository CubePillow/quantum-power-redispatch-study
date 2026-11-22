"""Redispatch data and objectives; SimBench loader via the original repository."""
from dataclasses import dataclass
from math import isfinite




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
                      grid="1-EHV-mixed--0-sw", gens=3, states=5):
        """Return (instance, metadata) built with the original repository,
        as in Lee et al.'s experiments/create_models.py:

        SimBench power flow (run_and_record_profiles) + saved S.npy
        -> time points 2 h apart -> package/.../reduce.py: line first, then
        its `gens` most influential generators, every line they can switch
        is monitored, the rest is fixed -> RedispatchModel (levels, costs,
        tau, M). `seed` picks the line among all candidates and the costs.
        """
        import random
        import sys
        from pathlib import Path
        import numpy as np

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / 'package'))
        import simbench as sb
        from quantum_energy.problem.simulator import (
            add_line_capacity_to_net, run_and_record_profiles,
            get_even_first_T_time_points)
        from quantum_energy.problem.reduce import create_line_aware_subset

        time_steps = list(time_steps)
        if time_steps != [8 * t for t in range(len(time_steps))]:
            raise ValueError('time_steps must be 0, 8, 16, ... (2 h apart)')
        if not isinstance(gens, int) or gens < 1:
            raise ValueError('gens must be a positive integer')
        if not isinstance(states, int) or states < 3:
            raise ValueError('states must be an integer >= 3 (off + min + max)')
        if seed is None:
            seed = random.SystemRandom().randrange(2**32)
        net = sb.get_simbench_net(grid)
        add_line_capacity_to_net(net)
        all_res = run_and_record_profiles(net, 0)
        S = np.load(root / 'experiments/data/sensitivity/S.npy').astype('float32')
        sim_res = get_even_first_T_time_points(all_res, len(time_steps), 2)
        Rd = create_line_aware_subset(sim_res, S, k=states, n_lines=None,
                                      n_gen=gens, seed=seed % 2**32,
                                      line_seed=seed)
        instance = cls(outputs=Rd.prod_vals.tolist(),
                       unit_costs=Rd.prod_cost_multi.tolist(),
                       demand=Rd.prod_target.tolist(),
                       sensitivity=np.asarray(Rd.S_resos, float).tolist(),
                       limits=np.asarray(Rd.M, float).tolist())
        metadata = {
            'source': 'SimBench power flow (run_and_record_profiles) + S.npy of the original repository',
            'grid': grid, 'seed': seed, 'time_step_indices': time_steps,
            'profile_step_hours': 0.25,
            'generators': [{'id': int(i), 'name': str(row['name']),
                            'type': str(row['type']), 'min_p_mw': float(row.min_p_mw),
                            'max_p_mw': float(row.max_p_mw)}
                           for i, row in Rd.net.gen.iterrows()],
            'baseline_outputs_mw': Rd.prod_resos.T.tolist(),
            'line_ids': [int(i) for i in Rd.net.line.index],
            'physical_line_count': int(len(Rd.orig_line_ids)),
            'selection': f'seed picks line {int(net.line.index[Rd.orig_line_ids[0]])} '
                         f'among {Rd.n_line_options} switchable lines; its {gens} most '
                         'influential generators; all lines they can switch are monitored',
            'fixed_line_contribution_mw': np.asarray(Rd.fixed_flow).T.tolist(),
            'cost_source': 'RedispatchModel: uniform samples from paper Table II',
            'demand_definition': 'Selected generator output from the power flow; all other resources fixed',
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

