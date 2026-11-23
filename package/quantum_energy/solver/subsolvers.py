import itertools as it
import time

import hybrid as dwh
import qubolite as ql
from numba import njit

from quantum_energy.util import (
    QUBO_mat_to_BQM,
    array_to_dict,
    dict_to_array,
    gen_ji_iter,
    gen_tji_iter,
    get_idx,
    np,
    rand_sol_n,
    sol_to_x,
    x_to_sol,
)


# * Beta-Expansion
# ** Util Functions
# *** a-expand inner loop
@njit
def apply_subsol_swaps(x, subsol_swaps):
    x_new = np.zeros_like(x)
    for pos1, pos2 in subsol_swaps:
        if pos1 == pos2:
            break
        x_new[pos1] = x[pos2] - x[pos1]
        x_new[pos2] = x[pos1] - x[pos2]
    return x_new

@njit
def get_cycles(x, *cycle_sets):
    n_cycles = 0
    for cs in cycle_sets:
        for s in cs:
            # if s[0] != s[1]:
            n_cycles += 1
    cycles = np.zeros((n_cycles, len(x)))
    i = 0
    for cs in cycle_sets:
        for s in cs:
            # if s[0] != s[1]:
            cycles[i] = apply_subsol_swaps(x, s)
            i += 1
    return cycles


@njit
def cycles_to_a_expand_qubo(cycles, x, Q):
    Q_new_1 = cycles @ Q @ cycles.T
    Q_new_2 = np.diag(cycles @ Q @ x + x.T @ Q @ cycles.T)
    Q_new = Q_new_1 + Q_new_2
    return Q_new

@njit
def rectify_state_change(sol, bases, t, j, i):
    cycles = [((t, j, sol[get_idx(bases[0:2], (t, j))]), (t, j, i))]
    prev = i
    tl = t - 1
    si = get_idx(bases[0:2], (tl, j))
    while tl >= 0 and abs(prev - sol[si]) > 1:
        prev = prev - 1 if sol[si] < prev else prev + 1
        cycles.append(((tl, j, sol[si]), (tl, j, prev)))
        tl = tl - 1
        si = get_idx(bases[0:2], (tl, j))
    prev = i
    tr = t + 1
    si = get_idx(bases[0:2], (tr, j))
    while tr < bases[0] and abs(prev - sol[si]) > 1:
        prev = prev - 1 if sol[si] < prev else prev + 1
        cycles.append(((tr, j, sol[si]), (tr, j, prev)))
        tr = tr + 1
        si = get_idx(bases[0:2], (tr, j))
    return cycles


def apply_cycle_set(x, cycle_set):
    for pos1, pos2 in cycle_set:
        tmp = x[pos1]
        x[pos1] = x[pos2]
        x[pos2] = tmp
    return x


# ** Class
class BetaExpansion:
    def __init__(self, T, n, k, n_cycles, solver):
        self.rng = np.random.default_rng(0)
        self.n_cycles = n_cycles
        self.bases = (T, n, k)
        self.sublen = k  # hardcode sub_len to k
        self.solver = solver
        # Init things to visit
        self.all_tjis = {
            idx: list(np.arange(self.bases[-1]))
            for idx in gen_ji_iter(*self.bases[0:2])
        }
        # Reset to start
        self.done = False

    def get_random_cycle_sets(self, x, n_sets, i=0):
        k = self.bases[-1]
        sol = x_to_sol(x, self.bases[-1])
        # Create matrix of available (t,j)
        avail_jts = np.zeros(self.bases[0:2], dtype="bool").T
        for t, j in self.all_tjis.keys():
            avail_jts[j, t] = True
            assert (t, j) in self.all_tjis
        # Get random order to iterate resources with
        js = np.arange(self.bases[1])
        self.rng.shuffle(js)
        jdx = 0
        # Create cycle_sets for current inner loop iteration
        cycle_sets = []
        cycle_count = 0
        max_swap_idx = 0
        while (
            not np.all(~avail_jts)
            and cycle_count < n_sets
            and jdx < self.bases[1]
        ):
            j = js[jdx]
            # Skip this resource if all timepoints unavailable
            if np.all(~avail_jts[j]):
                jdx += 1
                continue
            t = self.rng.choice(np.flatnonzero(avail_jts[j]))
            sol_idx = get_idx(self.bases[0:2], (t, j))
            # Get random state to switch to
            i = self.rng.choice(self.all_tjis[(t, j)])
            self.all_tjis[(t, j)].remove(i)
            if len(self.all_tjis[(t, j)]) == 0:
                self.all_tjis.pop((t, j))
                avail_jts[j, t] = False
            if i == sol[sol_idx]:
                # print("already in this state, skipping...")
                continue
            # print("rectify and make modded tjs unavailable")
            swaps = rectify_state_change(sol, self.bases, t, j, i)
            t_min, t_max = (np.inf, -1)
            cycle_set = []
            # print("Handle swaps")
            swap_idx = 0
            for swap in swaps:
                t, j, i = swap[1]
                # remove state from states to visit
                if (t, j) in self.all_tjis:
                    if i in self.all_tjis[(t, j)]:
                        self.all_tjis[(t, j)].remove(i)
                    # if (t,j) has no more states, add it to used list
                    if len(self.all_tjis[(t, j)]) == 0:
                        self.all_tjis.pop((t, j))
                        avail_jts[j, t] = False
                t_min = max(0, min(t_min + k, t) - k)
                t_max = min(self.bases[0] - 1, max(t_max - k, t) + k)
                # add swap to cycle_sets
                cycle_set.append(
                    (
                        get_idx(self.bases, swap[0]),
                        get_idx(self.bases, swap[1]),
                    )
                )
                swap_idx += 1
            max_swap_idx = max(max_swap_idx, swap_idx)
            # make modded tjs unavailable
            avail_jts[j][t_min : t_max + 1] = np.zeros(
                (t_max - t_min) + 1, dtype="bool"
            )
            cycle_sets.append(tuple(cycle_set))
            cycle_count += 1
        # If we visited all states, end inner loop and reset
        if len(self.all_tjis) == 0:
            self.all_tjis = {
                idx: list(np.arange(self.bases[-1]))
                for idx in gen_ji_iter(*self.bases[0:2])
            }
            self.done = True
        all_cycles = np.zeros(
            (cycle_count, max_swap_idx, 2), dtype="int"
        )
        # Make into numpy array
        for si in range(len(cycle_sets)):
            s = cycle_sets[si]
            for ci in range(len(s)):
                all_cycles[si, ci] = np.array([s[ci][0], s[ci][1]])
        # return tuple(cycle_sets)
        return all_cycles

    def subsolver_ql(self, Q, x):
        x_new = dict_to_array(x)
        prev_e = np.inf
        # print("Creating sub-problem")
        run_i = 0
        print('')
        while not self.done:
            print("\rinner loop: ", run_i, end="—")
            run_i += 1
            xa = x_new.copy().astype('float64')
            cset = self.get_random_cycle_sets(xa, self.n_cycles, run_i)
            cycles = get_cycles(xa, cset)
            if len(cycles) == 0:
                continue
            Q_alpha = cycles_to_a_expand_qubo(cycles, xa, Q.m)
            alphas = self.solver(ql.qubo(Q_alpha))
            for i in range(len(alphas)):
                if alphas[i]:
                    xa = apply_cycle_set(xa, cset[i])
            e = Q(xa)
            if e <= prev_e:
                # print("new minimum")
                # print(prev_e, e)
                # print(np.where(xa != x_new))
                prev_e = e
                x_new = xa
                # print(np.where(dict_to_array(x) != x_new))
        print(f'Did {run_i} inner loops.')
        self.done = False
        # print("Done creating sub-problem")
        return array_to_dict(x_new)

    def subsolver(self, BQM, x):
        Q = ql.qubo.from_dict(BQM.to_qubo()[0], relabel=False)[0]
        return self.subsolver_ql(Q, x)
