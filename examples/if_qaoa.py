"""IF-QAOA (indicator-function QAOA) for the redispatch problem.

Objective (minimised), with the exact overload count O:

    F(z) = mu_C*C + mu_W*W + mu_O*O + lambda_D*D

Inequality constraints are not turned into penalties or slack qubits.
They enter the cost operator as an indicator function
1(z) = 1 if z is feasible, else 0:

    ramp  |s_{t+1,a} - s_{t,a}| <= 1          (always)
    lines sum_a S_la P_{t,a} < M_lt  (O = 0)  (only with --hard-lines;
                                               otherwise O stays a soft term)

The cost operator is diagonal, so it is set by hand, classically:

    H_IF = diag( 1(z) * (F(z) - F_ub) ) / (F_ub - F_min)

F_ub is the largest F over feasible schedules. The shift F - F_ub is
needed because we minimise: if we multiplied 1(z) onto F >= 0 directly,
every infeasible z would get 0 and look *better* than every feasible
one. After the shift, feasible z have values in [-1, 0] and infeasible z
have 0, so the ground state is still the true optimum. This is the
maximisation form 1(z)*(F_ub - F(z)) of the original IF-QAOA, negated.

Ansatz (--ansatz):
  onehot  one-hot ansatz (default). Each (t, a) block of k qubits is kept
          in the one-hot subspace: start in a W state (uniform over the k
          levels), XY ring mixer inside each block. One-hot is therefore
          satisfied by construction; the indicator only checks the
          inequalities. Simulated in the k^(n*T) one-hot subspace.
  x       plain X mixer on all n*k*T qubits, start |+>^N. The indicator
          must then also check one-hot. Simulated on 2^(n*k*T) amplitudes.

Statevector simulation with numpy/scipy only; angles optimised with
L-BFGS-B, layer by layer (p = 1..P, warm start by interpolation).

Usage (from examples/):
    python if_qaoa.py legacy/instances/small.json --p 3
    python if_qaoa.py legacy/instances/small.json --ansatz x --p 3
    python if_qaoa.py legacy/instances/simbench_seed7_T2.json --p 4
    python if_qaoa.py --simbench --seed 1 --gens 2 --states 3 --T 2 --p 3 --hard-lines
"""
import argparse
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

if __package__:
    from .classical_solver import weighted_score
    from .redispatch_instance import RedispatchInstance
else:
    from classical_solver import weighted_score
    from redispatch_instance import RedispatchInstance


# --------------------------------------------------------------------------
# Classical part: cost diagonal with indicator function
# --------------------------------------------------------------------------
@dataclass
class IFDiagonal:
    states: np.ndarray     # (k^m, m) schedules, m = n*T, order (time, gen)
    F: np.ndarray          # (k^m,) objective F(z)
    feasible: np.ndarray   # (k^m,) indicator 1(z) for the inequalities
    h: np.ndarray          # (k^m,) normalised cost diagonal 1(z)*(F-F_ub)/(F_ub-F_min)
    F_min: float
    F_ub: float


def build_if_diagonal(inst, weights, hard_lines=False):
    """Enumerate all k^(n*T) one-hot schedules (classically) and build the
    IF cost diagonal. Index order equals numpy C order of shape (k,)*m."""
    m = inst.n * inst.T
    states = np.array(list(product(range(inst.k), repeat=m)), dtype=np.int64)
    F = np.empty(len(states))
    feasible = np.empty(len(states), dtype=bool)
    for idx, s in enumerate(states):
        s = tuple(int(v) for v in s)
        terms = inst.evaluate(s)
        F[idx] = weighted_score(terms, weights)
        feasible[idx] = inst.is_feasible(s) and (not hard_lines or terms[2] == 0)
    if not feasible.any():
        raise SystemExit('No feasible schedule: the indicator is 0 everywhere')
    F_min, F_ub = F[feasible].min(), F[feasible].max()
    scale = max(F_ub - F_min, 1e-12)
    h = np.where(feasible, (F - F_ub) / scale, 0.0)
    return IFDiagonal(states, F, feasible, h, F_min, F_ub)


def full_space_diagonal(inst, d: IFDiagonal):
    """Diagonal on all 2^(n*k*T) bit strings for the plain X-mixer ansatz.
    Indicator = one-hot AND inequalities; returns (h, map to schedule or -1)."""
    m, k = inst.n * inst.T, inst.k
    N = m * k
    bits = ((np.arange(2**N)[:, None] >> np.arange(N - 1, -1, -1)) & 1).astype(np.int8)
    blocks = bits.reshape(-1, m, k)
    onehot = (blocks.sum(axis=2) == 1).all(axis=1)
    levels = blocks.argmax(axis=2)
    sched_idx = np.ravel_multi_index(levels.T, (k,) * m)
    sched_idx = np.where(onehot, sched_idx, -1)
    h = np.where(onehot, d.h[np.maximum(sched_idx, 0)], 0.0)
    return h, sched_idx


# --------------------------------------------------------------------------
# Quantum part: statevector simulation
# --------------------------------------------------------------------------
def ring_xy_generator(k):
    """XY ring mixer restricted to the one-hot subspace of k qubits:
    sum_i (X_i X_{i+1} + Y_i Y_{i+1}) / 2 maps |e_i> <-> |e_{i+1}>,
    i.e. the adjacency matrix of a ring on the k levels."""
    A = np.zeros((k, k))
    for i in range(k if k > 2 else 1):
        j = (i + 1) % k
        A[i, j] = A[j, i] = 1.0
    return A


class QAOASim:
    def __init__(self, h, local_dim, n_sites, mixer_generator):
        self.h = h
        self.d = local_dim
        self.m = n_sites
        self.B = mixer_generator
        self.psi0 = np.full(local_dim**n_sites, local_dim ** (-n_sites / 2), dtype=complex)

    def _mix(self, psi, beta):
        U = expm(-1j * beta * self.B)
        psi = psi.reshape((self.d,) * self.m)
        for ax in range(self.m):   # same U on every site (blocks commute)
            psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [ax])), 0, ax)
        return psi.reshape(-1)

    def state(self, angles):
        p = len(angles) // 2
        gammas, betas = angles[:p], angles[p:]
        psi = self.psi0.copy()
        for g, b in zip(gammas, betas):
            psi = psi * np.exp(-1j * g * self.h)   # IF cost operator
            psi = self._mix(psi, b)
        return psi

    def energy(self, angles):
        return float(np.real(np.abs(self.state(angles)) ** 2 @ self.h))


def optimise(sim, p_max, restarts, rng):
    """Layer-wise optimisation: p = 1..p_max, INTERP warm start + random starts."""
    history, best = [], None
    for p in range(1, p_max + 1):
        starts = []
        if best is not None:   # interpolate previous optimum to p layers
            g, b = best[: p - 1], best[p - 1:]
            x_old = np.linspace(0, 1, p - 1)
            x_new = np.linspace(0, 1, p)
            starts.append(np.concatenate([np.interp(x_new, x_old, g) if p > 2 else np.repeat(g, p),
                                          np.interp(x_new, x_old, b) if p > 2 else np.repeat(b, p)]))
        starts.append(np.concatenate([np.linspace(0.1, 1.0, p) * np.pi,      # linear ramp
                                      np.linspace(1.0, 0.1, p) * np.pi / 4]))
        starts += [np.concatenate([rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)])
                   for _ in range(restarts)]
        results = [minimize(sim.energy, x0, method='L-BFGS-B') for x0 in starts]
        res = min(results, key=lambda r: r.fun)
        best = res.x
        history.append((p, res.fun, best))
    return history


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def report(p, probs_sched, p_invalid_onehot, d: IFDiagonal, inst, weights, rng, shots):
    feas = d.feasible
    opt = feas & np.isclose(d.F, d.F_min, rtol=1e-9, atol=1e-9)
    p_feas = probs_sched[feas].sum()
    p_opt = probs_sched[opt].sum()
    exp_F = (probs_sched[feas] @ d.F[feas]) / p_feas if p_feas > 0 else float('nan')
    ratio = (d.F_ub - exp_F) / max(d.F_ub - d.F_min, 1e-12)
    # sampling: best feasible schedule among `shots` measurements
    all_p = np.append(probs_sched, p_invalid_onehot)
    sample = rng.choice(len(all_p), size=shots, p=all_p / all_p.sum())
    sample = sample[sample < len(probs_sched)]
    sample = sample[feas[sample]]
    if sample.size:
        best = sample[np.argmin(d.F[sample])]
        best_txt = f'{tuple(int(v) for v in d.states[best])} F={d.F[best]:g}'
        best_txt += ' = optimum' if opt[best] else ''
    else:
        best_txt = 'no feasible sample'
    print(f'  p={p}: P(feasible)={p_feas:.3f}  P(optimum)={p_opt:.3f}  '
          f'E[F|feasible]={exp_F:.4g}  approx. ratio={ratio:.3f}  '
          f'best of {shots} shots: {best_txt}')
    return p_feas, p_opt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', nargs='?', type=Path, help='JSON instance (or --simbench)')
    ap.add_argument('--simbench', action='store_true', help='build instance from SimBench')
    ap.add_argument('--seed', type=int, help='SimBench: line/cost seed; also RNG seed')
    ap.add_argument('--gens', type=int, default=2)
    ap.add_argument('--states', type=int, default=3)
    ap.add_argument('--T', type=int, default=2)
    ap.add_argument('--weights', type=float, nargs=4, metavar=('C', 'W', 'O', 'D'))
    ap.add_argument('--ansatz', choices=['onehot', 'x'], default='onehot',
                    help='onehot: W states + XY ring mixer (default); x: plain X mixer')
    ap.add_argument('--hard-lines', action='store_true',
                    help='line limits as hard inequality in the indicator (O must be 0)')
    ap.add_argument('--p', type=int, default=3, help='max. QAOA depth (default 3)')
    ap.add_argument('--restarts', type=int, default=5, help='random starts per depth')
    ap.add_argument('--shots', type=int, default=1000)
    ap.add_argument('--max-dim', type=int, default=2**20,
                    help='limit for the simulated state dimension')
    args = ap.parse_args()
    if bool(args.input) == args.simbench:
        ap.error('Provide either a JSON path or --simbench')

    if args.simbench:
        inst, meta = RedispatchInstance.from_simbench(
            seed=args.seed, time_steps=[8 * t for t in range(args.T)],
            gens=args.gens, states=args.states)
        cases = {'Illustrative fixed weights': [1, 0, 100, 100]}
        print('SimBench seed:', meta['seed'])
    else:
        data = json.loads(args.input.read_text(encoding='utf-8-sig'))
        inst = RedispatchInstance(**data['instance'])
        cases = data.get('weights', {'Default': [1, 0, 100, 100]})
    if args.weights is not None:
        cases = {'Custom weights': args.weights}

    m, k = inst.n * inst.T, inst.k
    N = m * k
    dim = k**m if args.ansatz == 'onehot' else 2**N
    print(f'{inst.n} generators, {inst.T} time steps, {k} states -> {N} qubits; '
          f'ansatz={args.ansatz}, simulated dimension {dim}')
    if dim > args.max_dim:
        raise SystemExit(f'dimension {dim} > --max-dim {args.max_dim}')
    rng = np.random.default_rng(args.seed or 0)

    for name, weights in cases.items():
        print(f'\n{name}; weights (C,W,O,D)={weights}'
              + ('; line limits hard' if args.hard_lines else ''))
        d = build_if_diagonal(inst, weights, hard_lines=args.hard_lines)
        n_feas = int(d.feasible.sum())
        opt = np.flatnonzero(d.feasible & np.isclose(d.F, d.F_min))
        print(f'  indicator: {n_feas} of {len(d.F)} one-hot schedules feasible; '
              f'optimum F={d.F_min:g} at {[tuple(int(v) for v in d.states[i]) for i in opt]}')

        if args.ansatz == 'onehot':
            sim = QAOASim(d.h, k, m, ring_xy_generator(k))
            to_sched = lambda psi: (np.abs(psi) ** 2, 0.0)  # noqa: E731
        else:
            h_full, sched_idx = full_space_diagonal(inst, d)
            sim = QAOASim(h_full, 2, N, np.array([[0.0, 1.0], [1.0, 0.0]]))

            def to_sched(psi):
                pr = np.abs(psi) ** 2
                valid = sched_idx >= 0
                ps = np.bincount(sched_idx[valid], weights=pr[valid], minlength=len(d.F))
                return ps, pr[~valid].sum()

        ps, pin = to_sched(sim.psi0)
        report(0, ps, pin, d, inst, weights, rng, args.shots)
        for p, fun, angles in optimise(sim, args.p, args.restarts, rng):
            ps, pin = to_sched(sim.state(angles))
            report(p, ps, pin, d, inst, weights, rng, args.shots)
        if args.ansatz == 'x':
            print(f'  (probability outside one-hot at p={args.p}: {pin:.3f})')


if __name__ == '__main__':
    main()
