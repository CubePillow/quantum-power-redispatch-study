import itertools as it
import time

import hybrid as dwh  # type: ignore [import-untyped]
import qubolite as ql  # type: ignore [import-untyped]
from numba import njit  # type: ignore [import-untyped]

from quantum_energy.solver.subsolvers import BetaExpansion
from quantum_energy.util import (
    QUBO_mat_to_BQM,
    dict_to_array,
    get_idx,
    np,
)


# * util
def get_conv_bar(past_e, min_e):
    # 58 character progress bar excluding [ and ]
    bar = np.full(52, "-")
    bar[0] = "["
    bar[-1] = "]"
    bar[-2] = "M"
    if len(np.unique(past_e)) < 2:
        return "".join(bar)
    lin_e = (
        np.hstack([[min_e], past_e[-10:]]) - np.max(past_e[-10:])
    ) * -1
    norm_e = (
        lin_e[1:] / np.max(lin_e) if np.max(lin_e) != 0 else lin_e[1:]
    )
    ids = (norm_e * 49).astype("int8")
    c = 9
    for i in ids:
        bar[i + 1] = str(c)
        c -= 1
    return "".join(bar)


# * subsolver
def ocean_subsolver_factory(ocean_branch):
    return lambda BQM, x: ocean_branch.next(
        dwh.State.from_sample(x, BQM)
    ).samples.first.sample


# ** a-expansion
# *** Util functions
@njit
def get_random_assignment(current_assignment, k):
    r = np.random.randint(0, k - 1)
    return r if r < current_assignment else r + 1


@njit
def get_cycle(bases, j, t, i1, i2):
    row1 = get_idx(bases, (t, j, i1))
    row2 = get_idx(bases, (t, j, i2))
    C = np.identity(np.product(bases))
    C[[row1, row2]] = C[[row2, row1]]
    return C


def get_random_cycle(bases, offset, x, t, j):
    k = bases[-1]
    xidx1 = offset + get_idx(bases, (t, j, 0))
    xidx2 = offset + get_idx(bases, (t, j, k))
    i1 = [x[i] for i in range(xidx1, xidx2)].index(1)
    i2 = get_random_assignment(i1, k)
    return (
        offset + get_idx(bases, (t, j, i1)),
        offset + get_idx(bases, (t, j, i2)),
    )


def get_random_cycles(reso_times, x):
    return np.array(
        [
            get_random_cycle((T, n, k), o, x, t, j)
            for (T, n, k, o, j, t) in reso_times
        ]
    )


def swap_pos(x, pos1, pos2):
    x_new = x.copy()
    tmp = x_new[pos1]
    x_new[pos1] = x_new[pos2]
    x_new[pos2] = tmp
    return x_new


def get_alpha_expand_qubo(cycles, x, Q):
    def E(v1, v2):
        return v1.T @ Q @ v2

    cycles_new = np.vstack(
        [swap_pos(x, c[0], c[1]) for c in cycles]
    ) - (np.identity(Q.shape[0]) @ x)
    Q_new_1 = cycles_new @ Q @ cycles_new.T
    Q_new_2 = np.diag(cycles_new @ Q @ x + x.T @ Q @ cycles_new.T)
    Q_new = Q_new_1 + Q_new_2
    return Q_new

# * run
def run_decomposer(
    Q,
    subsolver,
    x_init,
    bqm=None,
    max_time=np.inf,
    min_runs=10,
    max_runs=200,
    max_stale=10,
):
    # setup
    ql_Q = ql.qubo(Q)
    # initial states
    x = x_init.copy()
    # stats
    min_x = x
    min_e = ql_Q(dict_to_array(x))
    es = [min_e]
    # counters
    stale_count = 0
    run_count = 0

    def progress_printer(n_run, bar):
        return lambda m: print(f"run {n_run}: {bar} {m}")

    time_start = time.time()
    while (
        (stale_count < max_stale or run_count < min_runs)
        and time.time() - time_start < max_time
        and run_count < max_runs
    ):
        # housekeeping
        print(
            f"{run_count},{stale_count}: ",
            get_conv_bar(es, min_e),
            end="\r",
        )
        # solve subproblem
        x_new = subsolver(ql_Q, x) if bqm is None else subsolver(bqm, x)
        e = ql_Q(dict_to_array(x_new))
        if e < es[-1] or np.isclose(e, es[-1]):
            x = x_new
            es.append(e)
        # check for staleness
        if e < min_e:
            min_x = x
            min_e = e
            stale_count = 0
        elif np.isclose(e, min_e):
            min_x = x
            stale_count += 1
        else:
            stale_count += 1
        run_count += 1
    return min_x, es
