"""Reusable physical redispatch instance; no third-party dependencies."""
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

    def valid_states(self, states):
        return len(states) == self.T * self.n and all(
            isinstance(s, int) and 0 <= s < self.k for s in states)

    def is_feasible(self, states):
        """State encoding selects one level implicitly; also enforce ramp."""
        return self.valid_states(states) and all(
            abs(states[(t + 1) * self.n + a] - states[t * self.n + a]) <= 1
            for t in range(self.T - 1) for a in range(self.n))

    def evaluate(self, states):
        """Raw C/W/O/D; overload uses >=; no fixed injections included."""
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

