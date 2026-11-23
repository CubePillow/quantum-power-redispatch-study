import itertools
import random

import numpy as np
from numba import njit

from quantum_energy.problem.modeller import RedispatchModel
from quantum_energy.util import (
    gen_ji_iter,
    gen_tji_iter,
    get_constant_sol,
    get_idx,
    get_SPE,
    rand_x,
    sol_to_x,
)


# * Entry Normaliser
@njit
def get_norm_const(lb, T, n):
    return lb / (T * n)


@njit
def norm_ele(T, n, idx, jdx, lb, ub, ele):
    new_ele = ele - get_norm_const(lb, T, n) if idx == jdx else ele
    return new_ele / (ub - lb)


# * QUBO populators
# ** Hard Constraints
@njit
def get_hard_constraint_penalty(Q: np.ndarray):
    Q_pos = Q * (Q >= 0)
    Q_neg = Q * (Q < 0)
    x = np.ones(len(Q))
    return np.max(np.array((x.T @ Q_pos @ x, x.T @ Q_neg @ x)))


# *** Adjacent switching
@njit
def add_adjacent_lvls(Q, lam, T, n, k):
    """Create QUBO to contraint switching to only adjanect power levels.

    Creates a matrix with 1 in non-adjacent power levels for each
    controllable resources between T and T - 1

    Parameters
    ----------
    T : int
        Number of time points
    n : int
        Number of controllable resources
    k : int
        Number of power levels for each controllable resources

    Returns
    -------
    Q : ndarray
        QUBO for constraint

    """
    tji_iter = gen_tji_iter(T, n, k)
    bases = np.array((T, n, k))
    max_c = n * (T - 1) * (2 ** (k - 1))
    for t, j, i in tji_iter:
        idx1 = get_idx(bases, np.array((t, j, i)))
        for i2 in range(k):
            if i2 - 1 != i and i2 + 1 != i and i2 != i and t < T - 1:
                idx2 = get_idx(bases, np.array((t + 1, j, i2)))
                Q[idx1, idx2] += lam * (2 ** abs(i2 - i) / max_c)
    return Q


# *** One-Hot Constraint
@njit
def add_one_hot(Q, lam, T, n, k):
    tji_iter = gen_tji_iter(T, n, k)
    for t1, j1, i1 in tji_iter:
        idx = get_idx((T, n, k), (t1, j1, i1))
        for t2, j2, i2 in tji_iter:
            jdx = get_idx((T, n, k), (t2, j2, i2))
            if idx == jdx:
                Q[idx, jdx] += -1 * lam
            elif j1 == j2 and t1 == t2:
                Q[idx, jdx] += lam
    return Q


# ** Constant QUBO
@njit
def add_constant(Q, const: int, T: int, n: int, k: int):
    c_ele = const / (T * n)
    for t in range(T):
        for j, i in gen_ji_iter(n, k):
            idx = get_idx((T, n, k), (t, j, i))
            Q[idx, idx] += c_ele
    return Q


# ** Switching
@njit
def get_switch_bounds(SwC: np.ndarray, T: int):
    ub = 0
    for j in range(SwC.shape[0]):
        max_swc = np.max(SwC[j, :, :])
        ub += max_swc * (T - 1)
    return 0, ub


@njit
def add_switch_cost(Q, lam, SwC, T):
    """
    The cost of going between adjacent power levels between times T and
    T - 1 is the difference in generated power (MWh) of the levels.

    Parameters
    ----------
    SwC : (n x k x k) np.ndarray
        Switching Cost of each controllable resources

    """
    n, k, _ = SwC.shape
    lb, ub = get_switch_bounds(SwC, T)
    tji_iter = gen_tji_iter(T, n, k)
    bases = (T, n, k)
    for t, j, i in tji_iter:
        idx = get_idx(bases, (t, j, i))
        for t2, j2, i2 in tji_iter:
            jdx = get_idx(bases, (t2, j2, i2))
            if j == j2 and t2 == t + 1:
                Q[idx, jdx] += lam * norm_ele(
                    T, n, idx, jdx, lb, ub, SwC[j, i, i2]
                )
    return Q


# ** Cost
@njit
def get_cost_bounds(C: np.ndarray, SgPr=None):
    """
    C   = (T x n x k) matrix: Cost of resources at diff levels/time
    SgPr  = (m x T) matrix: Price of stopping static generators
    """
    return (np.sum(C[:, :, 0]), np.sum(C[:, :, -1]))


@njit
def add_cost(Q, lam, cost_vals):
    """cost_vals = (T x n x k) matrix
    switch_cost = n (k x k) matrices
    T = number of time points
    """
    T, n, k = cost_vals.shape
    lb, ub = get_cost_bounds(cost_vals)
    tji_iter = gen_tji_iter(T, n, k)
    for t, j, i in tji_iter:
        idx = get_idx((T, n, k), (t, j, i))
        Q[idx, idx] += lam * norm_ele(
            T, n, idx, idx, lb, ub, cost_vals[t, j, i]
        )
    return Q


# ** Unbalanced Penalties
# *** Overload
def get_overload_mins_maxs(M, S, PE):
    """M  = (L x T) matrix    : max load for each line at each timepoint
    S  = (L x n) matrix    : sensitivity matrix for resources
    PE = (T x n x k) matrix: Power levels of resources over time
    """
    min_load = S @ PE.min(axis=2).T
    max_load = S @ PE.max(axis=2).T
    return M - max_load, M - min_load


def get_overload_norm_bounds(M, S, PE):
    mins, maxs = get_overload_mins_maxs(M, S, PE)
    norms = maxs.astype("float32")
    mins_norm, maxs_norm = mins / norms, maxs / norms
    ubs = (mins_norm**2 / 2) - mins_norm + 1
    lbs = (maxs_norm**2 / 2) - maxs_norm + 1
    return norms, lbs.sum(), ubs.sum()


@njit
def add_overload_jit(Q, lam, M, S, PE, norms, lb, ub):
    """M  = (L x T) matrix    : max load for each line at each timepoint
    S  = (L x n) matrix    : sensitivity matrix for resources
    PE = (T x n x k) matrix: Power levels of resources over time
    norms = (L x T) matrix
    """
    T, n, k = PE.shape
    offset = (M.shape[0] * T) / (T * n)
    normS = norms.T @ S
    MnormS = (M * norms**2).T @ S
    m1 = (M * norms).sum() / (n * T)
    m2 = ((M * norms) ** 2).sum() / (n * T)
    ji_iter = gen_ji_iter(n, k)
    for t in range(T):
        SnormS = S.T @ np.diag(norms.T[t] ** 2) @ S
        for j1, i1 in ji_iter:
            idx = get_idx(PE.shape, (t, j1, i1))
            for j2, i2 in ji_iter:
                jdx = get_idx(PE.shape, (t, j2, i2))
                if idx != jdx:
                    Q[idx, jdx] += norm_ele(
                        T,
                        n,
                        idx,
                        jdx,
                        lb,
                        ub,
                        (PE[t, j1, i1] * PE[t, j2, i2] * SnormS[j1, j2])
                        / 2,
                    )
            Q[idx, idx] += norm_ele(
                T,
                n,
                idx,
                idx,
                lb,
                ub,
                offset
                - (m1 - PE[t, j1, i1] * normS[t, j1])
                + (
                    m2
                    + PE[t, j1, i1] * PE[t, j1, i1] * SnormS[j1, j1]
                    - 2 * PE[t, j1, i1] * MnormS[t, j1]
                )
                / 2,
            )
    return Q


def add_overload(Q, lam, M, S, PE):
    norms, lb, ub = get_overload_norm_bounds(M, S, PE)
    norms = 1 / norms
    return add_overload_jit(Q, lam, M, S, PE, norms, lb, ub)


# *** Power
def get_power_eq_bounds(PE, tau):
    """
    Tau = T vector          : Minimum target power
    PE = (T x n x k) matrix : Power levels of resources over time
    """
    return (
        0,
        max(
            np.sum(np.power(np.sum(PE[:, :, 0], 1) - tau, 2)),
            np.sum(np.power(np.sum(PE[:, :, -1], 1) - tau, 2)),
        ),
    )


@njit
def eles_power(Q, lam, Tau, PE, lb, ub):
    """Tau  = (T) vector
    PE = (T x n x k) matrix
    """
    T, n, k = PE.shape
    ji_iter = gen_ji_iter(n, k)
    tau = np.linalg.norm(Tau) ** 2 / (n * T)
    for t in range(T):
        for j1, i1 in ji_iter:
            idx = get_idx(PE.shape, (t, j1, i1))
            Q[idx, idx] += lam * norm_ele(
                T,
                n,
                idx,
                idx,
                lb,
                ub,
                PE[t, j1, i1] ** 2 + tau - 2 * Tau[t] * PE[t, j1, i1],
            )
            for j2, i2 in ji_iter:
                jdx = get_idx(PE.shape, (t, j2, i2))
                if idx != jdx:
                    Q[idx, jdx] += lam * norm_ele(
                        T,
                        n,
                        idx,
                        jdx,
                        lb,
                        ub,
                        PE[t, j1, i1] * PE[t, j2, i2],
                    )
    return Q


def add_power(Q, lam, Tau, PE):
    lb, ub = get_power_eq_bounds(PE, Tau)
    return eles_power(Q, lam, Tau, PE, lb, ub)
