import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
import qubolite as ql
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


# * Utility functions
# ** Unbalanced Penalty
@njit
def get_min_sol_for_S_rows(S, k):
    """S = (L x n) matrix: Sensitivity Matrix
    k: no power levels per resource
    """
    return (S < 0).astype("int8") * (k - 1)


@njit
def get_min_x_for_S_rows(S, k):
    """S = (L x n) matrix: Sensitivity Matrix
    k: no power levels per resource
    """
    L, n = S.shape
    min_sols = get_min_sol_for_S_rows(S, k)
    min_xs = np.zeros((L, n * k), dtype="int8")
    for i in range(L):
        min_xs[i] = sol_to_x(min_sols[i], k)
    return min_xs


# * QUBO generators
# ** Constant QUBO
@njit
def get_constant_qubo(c: int, T: int, n: int, k: int):
    c_ele = c / (T * n)
    N = T * n * k
    Q = np.zeros((N, N))
    for t in range(T):
        for j, i in gen_ji_iter(n, k):
            idx = get_idx((T, n, k), (t, j, i))
            Q[idx, idx] = c_ele
    return Q


# ** Switching QUBOs
# *** Switching Cost
@njit
def gen_qubo_switch_cost(SwC, T: int):
    """Create (T x n x k) switching cost matrix.

    The cost of going between adjacent power levels between times T and
    T - 1 is the difference in generated power (MWh) of the levels.

    Parameters
    ----------
    SwC : (n x k x k) np.ndarray
        Switching Cost of each controllable resources

    """
    n, k, _ = SwC.shape
    N = T * n * k

    Q = np.zeros((N, N))
    tji_iter = gen_tji_iter(T, n, k)
    bases = np.array((T, n, k))
    for t, j, i in tji_iter:
        idx1 = get_idx(bases, np.array((t, j, i)))
        for t2, j2, i2 in tji_iter:
            idx2 = get_idx(bases, np.array((t2, j2, i2)))
            if j == j2 and t2 == t + 1:
                Q[idx1, idx2] += SwC[j, i, i2]
    return Q


# *** Adjacent levels constraint
@njit
def gen_qubo_adjacent_lvls(T, n, k):
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
    N = T * n * k
    Q = np.zeros((N, N))
    tji_iter = gen_tji_iter(T, n, k)
    bases = np.array((T, n, k))
    max_c = n * (T - 1) * (2 ** (k - 1))
    for t, j, i in tji_iter:
        idx1 = get_idx(bases, np.array((t, j, i)))
        for i2 in range(k):
            if i2 - 1 != i and i2 + 1 != i and i2 != i and t < T - 1:
                idx2 = get_idx(bases, np.array((t + 1, j, i2)))
                Q[idx1, idx2] = 2 ** abs(i2 - i) / max_c
    return Q


# ** Cost QUBOs
@njit
def gen_qubo_cost(cost_vals):
    """cost_vals = (T x n x k) matrix
    switch_cost = n (k x k) matrices
    T = number of time points
    """
    T, n, k = cost_vals.shape
    N = n * k * T
    Q = np.zeros((N, N))
    tji_iter = gen_tji_iter(T, n, k)
    for t1, j1, i1 in tji_iter:
        idx = get_idx((T, n, k), (t1, j1, i1))
        Q[idx, idx] += cost_vals[t1, j1, i1]
    return Q


# ** Max load QUBOs
@njit
def gen_qubos_overload_unbalanced(M, S, PE, norms):
    """M  = (L x T) matrix    : max load for each line at each timepoint
    S  = (L x n) matrix    : sensitivity matrix for resources
    PE = (T x n x k) matrix: Power levels of resources over time
    norms = (L x T) matrix
    """
    T, n, k = PE.shape
    N = T * n * k
    Q1 = np.zeros((N, N))
    Q2 = np.zeros((N, N))
    normS = norms.T @ S
    MnormS = (M * norms**2).T @ S
    m1 = (M * norms).sum() / (n * T)
    m2 = ((M * norms) ** 2).sum() / (n * T)
    ji_iter = gen_ji_iter(n, k)
    MnormS = (M * norms**2).T @ S
    for t in range(T):
        SnormS = S.T @ np.diag(norms.T[t] ** 2) @ S
        for j1, i1 in ji_iter:
            idx = get_idx(PE.shape, (t, j1, i1))
            for j2, i2 in ji_iter:
                jdx = get_idx(PE.shape, (t, j2, i2))
                Q2[idx, jdx] += (
                    PE[t, j1, i1] * PE[t, j2, i2] * SnormS[j1, j2]
                )
            Q2[idx, idx] -= 2 * PE[t, j1, i1] * MnormS[t, j1]
            Q1[idx, idx] -= PE[t, j1, i1] * normS[t, j1]
            Q2[idx, idx] += m2
            Q1[idx, idx] += m1
    return (Q1, Q2)


# ** Power target QUBOs
@njit
def gen_qubos_power_unb_pen(Tau, PE, norms):
    """Tau  = (T) vector
    PE = (T x n x k) matrix
    """
    T, n, k = PE.shape
    N = T * n * k
    Q1 = np.zeros((N, N))
    Q2 = np.zeros((N, N))
    ji_iter = gen_ji_iter(n, k)
    tau2 = ((Tau * norms) ** 2).sum() / (T * n)
    tau1 = (Tau * norms).sum() / (T * n)
    for t in range(T):
        for j1, i1 in ji_iter:
            idx = get_idx(PE.shape, (t, j1, i1))
            for j2, i2 in ji_iter:
                jdx = get_idx(PE.shape, (t, j2, i2))
                Q2[idx, jdx] += (
                    PE[t, j1, i1] * PE[t, j2, i2] * norms[t] ** 2
                )
            Q2[idx, idx] -= 2 * PE[t, j1, i1] * Tau[t] * norms[t] ** 2
            Q1[idx, idx] += PE[t, j1, i1] * norms[t]
            Q2[idx, idx] += tau2
            Q1[idx, idx] -= tau1
    return (Q1, Q2)


# *** Equality
@njit
def gen_qubo_power_eq(PE, Tau):
    """PE  = (T x n x k) matrix: Power levels of resources over time
    Tau = T-length vector   : Target total power at each timepoint
    """
    T, n, k = PE.shape
    N = n * k * T
    Q = np.zeros((N, N))
    ji = gen_ji_iter(n, k)
    bases = np.array([T, n, k])
    tau = np.linalg.norm(Tau) ** 2 / (n * T)
    for t in range(T):
        for j1, i1 in ji:
            idx1 = get_idx(bases, np.array([t, j1, i1]))
            Q[idx1, idx1] -= 2 * Tau[t] * PE[t, j1, i1]
            Q[idx1, idx1] += tau
            for j2, i2 in ji:
                idx2 = get_idx(bases, np.array([t, j2, i2]))
                Q[idx1, idx2] += PE[t, j1, i1] * PE[t, j2, i2]
    return Q


# ** One-Hot Constraint
@njit
def gen_qubo_one_hot(T, n, k):
    N = T * n * k
    tji_iter = gen_tji_iter(T, n, k)
    A = np.zeros(N**2).reshape((N, N))
    for t1, j1, i1 in tji_iter:
        idx = get_idx((T, n, k), (t1, j1, i1))
        for t2, j2, i2 in tji_iter:
            jdx = get_idx((T, n, k), (t2, j2, i2))
            if idx == jdx:
                A[idx, jdx] = -1
            elif j1 == j2 and t1 == t2:
                A[idx, jdx] = 1
    return A


# * QUBO Normalisation
def norm_qubo(qubo, lb, ub, T, n, k):
    return (qubo - get_constant_qubo(lb, T, n, k)) / (ub - lb)

def norm_qubo_no_trans(qubo, lb, ub, T, n, k):
    return qubo / (ub - lb)


# * QUBO Penalties
# ** Hard Constarint Constant
@njit
def get_hard_constraint_penalty(Q: np.ndarray):
    Q_pos = Q * (Q >= 0)
    Q_neg = Q * (Q < 0)
    x = np.ones(len(Q))
    return np.max(np.array((x.T @ Q_pos @ x, x.T @ Q_neg @ x)))


# ** Unbalanced Penalty Norms
def unb_taylor(h):
    return ((h**2) / 2) - h + 1


# *** Power
def get_power_norm_bounds(Tau, PE, Pc=None):
    """
    Tau = T vector          : Minimum target power
    PE = (T x n x k) matrix : Power levels of resources over time
    Pc = (m x T) matrix     : Power of static gens at all timepoints
    """
    pb = [PE.min(2).sum(1), PE.max(2).sum(1)]
    if Pc is not None:
        pb[1] = pb[1] + Pc.sum(0)
    norms = (pb[1] - Tau).astype("float32")
    extrema_h_normed = ((pb[0] - Tau) / norms, (pb[1] - Tau) / norms)
    bs = (
        unb_taylor(extrema_h_normed[0]).sum(),
        unb_taylor(extrema_h_normed[1]).sum(),
    )
    return norms, min(bs), max(bs)


# *** Overload
def get_overload_mins_maxs(
    M: np.ndarray,
    S_resos: np.ndarray,
    prod_vals_eff: np.ndarray,
    S_sgens=None,
    sgen_prods=None,
):
    # (L x (n + m)) matrix
    S = S_resos
    if S_sgens is not None:
        S = np.hstack((S_resos, S_sgens))
    # L-length vector
    min_p = prod_vals_eff.min(axis=2).T
    max_p = prod_vals_eff.max(axis=2).T
    if sgen_prods is not None:
        min_p = np.vstack((min_p, np.zeros_like(sgen_prods)))
        max_p = np.vstack((max_p, sgen_prods))
    min_load = S @ min_p
    max_load = S @ max_p
    return M - max_load, M - min_load


def get_overload_norm_bounds(
    Rd: RedispatchModel, M: np.ndarray, use_sgens: bool = False
):
    mins, maxs = get_overload_mins_maxs(
        M,
        Rd.S_resos,
        Rd.prod_vals_eff,
        Rd.S_sgens if use_sgens else None,
        Rd.sgen_prods if use_sgens else None,
    )
    norms = maxs.astype("float64")
    mins_norm, maxs_norm = mins / norms, maxs / norms
    ubs = (mins_norm**2 / 2) - mins_norm + 1
    lbs = (maxs_norm**2 / 2) - maxs_norm + 1
    return norms, lbs.sum(), ubs.sum()


# ** Lower/Upper Bounds
def get_power_eq_bounds(PE, tau):
    """
    Tau = T vector          : Minimum target power
    PE = (T x n x k) matrix : Power levels of resources over time
    """
    return (
        0,
        max(
            np.sum(np.power(np.sum(PE[:, :, 0], 1) - tau, 2)),
            np.sum(np.power(np.sum(PE[:, :, -1], 1) - tau, 2))
        )
    )


@njit
def get_cost_bounds(C: np.ndarray, SgPr=None):
    """
    C   = (T x n x k) matrix: Cost of resources at diff levels/time
    SgPr  = (m x T) matrix: Price of stopping static generators
    """
    return (
        np.sum(C[:, :, 0]),
        np.sum(C[:, :, -1]) + (0 if SgPr is None else SgPr.sum()),
    )


@njit
def get_switch_bounds(SwC: np.ndarray, T: int):
    ub = 0
    for j in range(SwC.shape[0]):
        max_swc = np.max(SwC[j, :, :])
        ub += max_swc * (T - 1)
    return 0, ub


# * Qubo Classes
class RedispatchQUBO:
    def __init__(self, R: RedispatchModel):
        self.R = R
        self.M = (
            R.max_load
            - R.S_sgens @ R.sgen_prods
            - R.S_loads @ R.prod_static
        ).astype("float64")
        self.S = self.R.S_resos.astype('float64')
        self.tau = R.prod_target - R.sgen_prods.sum(axis=0)
        #self.tau = self.tau + (self.tau * 0.01)
        self.m = -1
        # print("generating cost qubo...")
        self.qubo_cost = gen_qubo_cost(R.cost_vals)
        self.bs_cost = get_cost_bounds(R.cost_vals)
        # print("generating switch qubo...")
        self.qubo_switch = gen_qubo_switch_cost(R.switch_cost, R.T)
        self.bs_switch = get_switch_bounds(R.switch_cost, R.T)
        # print("generating max_load qubo...")
        self.Cs_load, lb, ub = get_overload_norm_bounds(self.R, self.M)
        self.qubos_load = gen_qubos_overload_unbalanced(
            self.M, self.S, R.prod_vals_eff, 1 / self.Cs_load
        )
        self.bs_load = (lb, ub)
        self.qubo_load = self.get_unb_qubo(
            self.M.shape[0] * self.R.T, self.qubos_load
        )
        # print("generating power_eq qubo...")
        self.power_Cs, lb, ub = get_power_norm_bounds(
            self.tau, R.prod_vals_eff
        )
        self.bs_power = (lb, ub)
        self.qubos_power = gen_qubos_power_unb_pen(
            self.tau,
            R.prod_vals_eff,
            1 / self.power_Cs,
        )
        self.qubo_power = self.get_unb_qubo(self.R.T, self.qubos_power)
        # Power (Equality)
        self.qubo_power_e = gen_qubo_power_eq(R.prod_vals_eff, self.tau)
        self.bs_power_e = get_power_eq_bounds(R.prod_vals_eff, self.tau)
        # print("generating one-hot qubo")
        self.qubo_one_hot = gen_qubo_one_hot(R.T, R.n_resources, R.k)
        # print("generating adjacent qubo...")
        self.qubo_adjacent = gen_qubo_adjacent_lvls(
            R.T, R.n_resources, R.k
        )

    def norm_qubo(self, Q, lb, ub):
        return norm_qubo(
            Q, lb, ub, self.R.T, self.R.n_resources, self.R.k
        )

    def norm_qubo_nt(self, Q, lb, ub):
        return norm_qubo_no_trans(
            Q, lb, ub, self.R.T, self.R.n_resources, self.R.k
        )

    def get_unb_qubo(self, offset, qubos):
        return (
            get_constant_qubo(
                offset,
                self.R.T,
                self.R.n_resources,
                self.R.k,
            )
            - qubos[0]
            + qubos[1] / 2
        )

    def get_qubo_obj(self, lam_p=1, lam_o=1, lam_c=1, lam_s=1):
        return (
            lam_p * self.qubo_power_e
            + lam_o * self.qubo_load
            + lam_c * self.qubo_cost
            + lam_s * self.qubo_switch
        )

    def get_qubo_obj_normed(self, lam_p=1, lam_o=1, lam_c=1, lam_s=1):
        return (
            lam_p * self.norm_qubo(self.qubo_power_e, *self.bs_power_e)
            + lam_o * self.norm_qubo(self.qubo_load, *self.bs_load)
            + lam_c * self.norm_qubo(self.qubo_cost, *self.bs_cost)
            + lam_s * self.norm_qubo(self.qubo_switch, *self.bs_switch)
        )


    def get_qubo_obj_normed_nt(self, lam_p=1, lam_o=1, lam_c=1, lam_s=1):
        return (
            lam_p * self.norm_qubo_nt(self.qubo_power_e, *self.bs_power_e)
            + lam_o * self.norm_qubo_nt(self.qubo_load, *self.bs_load)
            + lam_c * self.norm_qubo_nt(self.qubo_cost, *self.bs_cost)
            + lam_s * self.norm_qubo_nt(self.qubo_switch, *self.bs_switch)
        )


    def get_qubo_obj_normed_ie(self, lam_p=1, lam_o=1, lam_c=1, lam_s=1):
        return (
            lam_p * self.norm_qubo(self.qubo_power, *self.bs_power)
            + lam_o * self.norm_qubo(self.qubo_load, *self.bs_load)
            + lam_c * self.norm_qubo(self.qubo_cost, *self.bs_cost)
            + lam_s * self.norm_qubo(self.qubo_switch, *self.bs_switch)
        )


class RedispatchResosQUBO(RedispatchQUBO):
    def __init__(self, R):
        super().__init__(R, False)


# * slack vars
def get_resos_load_inequalities_to_qubo(S, PE, M, K: int):
    T, n, k = PE.shape
    L, T = M.shape
    SPE = get_SPE(S, PE)
    # SPs[t] @ x[t] < Cs[t]
    assert (L, T, n, k) == SPE.shape
    N = T * n * k
    nk = n * k
    Q = np.zeros((N + K * T * L, N + K * T * L))
    for t in range(T):
        tnk = t * nk
        for li in range(L):
            ineq_qubo = gen_qubo_inequality(
                SPE[li, t].flatten(), M[li, t], K
            )
            Q[tnk : tnk + nk, tnk : tnk + nk] += ineq_qubo[0:nk, 0:nk]
            lid = N + get_idx((T, L, K), (t, li, 0))
            Q[tnk : tnk + nk, lid : lid + K] += ineq_qubo[
                0:nk, nk : nk + K
            ]
            Q[lid : lid + K, tnk : tnk + nk] += ineq_qubo[
                nk : nk + K, 0:nk
            ]
            Q[lid : lid + K, lid : lid + K] += ineq_qubo[
                nk : nk + K, nk : nk + K
            ]
    return Q


def add_inequality_to_qubo(
    Q: np.ndarray, a: np.ndarray, b: float, K: int
) -> np.ndarray:
    N = Q.shape[0]
    ineq_qubo = gen_qubo_inequality(a, b, K)

    full_qubo = np.zeros(shape=(N + K, N + K))
    full_qubo[0:N, 0:N] = Q
    full_qubo += ineq_qubo

    return full_qubo


@njit
def gen_qubo_inequality(a: np.ndarray, b: float, K: int) -> np.ndarray:
    """Add constraints of the form
    a^T x <= b
    for a, x in R^N, b in R.
    Add K additional slack variables.

    For x in {0,1}^N, z in {0, 1}^K
    we generate Q_{ineq} such that
    argmin_{x,z} [x, z]^T Q_{ineq} [x, z]
    is equivalent to minimizing
    P(x,z) = ( (2^K - 0.5) * (a^T x - c) / (b - c)
               - sum_{i=0}^{K-1} 2^i z_i  )^2
    which implies a^T x <= b
    """
    N = a.shape[0]

    # minimum value c = argmin_x a^T x
    if np.min(a) >= 0:
        c = 0
    else:
        c = np.sum(a[a < 0])

    M = (2**K - 0.5) / (b - c)

    constraint_qubo = np.zeros(shape=(N + K, N + K))

    for i in range(0, N):
        for j in range(0, N):
            if i == j:
                constraint_qubo[i, i] += -2 * c * M * M * a[i]
            else:
                constraint_qubo[i, j] += M * M * a[i] * a[j]
    for i in range(0, N):
        for k in range(0, K):
            constraint_qubo[i, N + k] += -2 * M * a[i] * (2**k)
    for k in range(0, K):
        for li in range(0, K):
            if k == li:
                constraint_qubo[N + k, N + k] += 2 * c * M * (2**k)
            else:
                constraint_qubo[N + k, N + li] += (2**k) * (2**li)
    return constraint_qubo
