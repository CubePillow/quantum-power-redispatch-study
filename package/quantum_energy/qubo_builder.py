import pickle

import numpy as np
import qe_native as qen
import qe_sparse as qeb
from numba import njit

from quantum_energy.problem.modeller import RedispatchModel
from quantum_energy.problem.qubo import (
    gen_qubo_cost,
    gen_qubo_switch_cost,
    gen_qubos_overload_unbalanced,
    gen_qubos_power_unb_pen,
    get_cost_bounds,
    get_overload_norm_bounds,
    get_power_norm_bounds,
    get_switch_bounds,
    norm_qubo,
)

# * Test


@njit
def get_idxes(bases, idx: int):
    base = 1
    for b in bases:
        base *= b
    idxes = np.zeros(len(bases), dtype="int")
    for i in range(len(bases)):
        base = int(base / bases[i])
        idxes[i] = int(idx // base)
        idx = idx % base
    return idxes


# Example QUBO constructor, "Minimum Viable Product"
def qubo_resos_modifier(T: int, n: int, k: int, SwC: np.ndarray):
    N = T * n * k
    Q = np.zeros((N, N))
    for idx in range(N):
        t1, j1, i1 = get_idxes((T, n, k), idx)
        for jdx in range(N):
            t2, j2, i2 = get_idxes((T, n, k), jdx)
            if j1 == j2 and t2 == t1 + 1:
                Q[idx, jdx] = SwC[j1, i1, i2]
    return Q


@njit
def qubo_resos_diagnose(T: int, n: int, k: int):
    N = T * n * k
    Q = np.zeros((N, N))
    for idx in range(N):
        t1, j1, i1 = get_idxes((T, n, k), idx)
        for jdx in range(N):
            t2, j2, i2 = get_idxes((T, n, k), jdx)
            Q[idx, jdx] = idx
    return Q


# * Util


def Q_dim(Rd: RedispatchModel):
    return Rd.T * Rd.n_resources * Rd.k


def qen_norm_qubo(Rd: RedispatchModel, qen_Q, lb, ub):
    qen.update_qubo_constant(
        qen_Q,
        -1 * lb,
        1,
    )


def get_norm_offset(lb, ub):
    return (ub - lb, -1 * lb)


# * qen Adapters
def qen_switch_cost(Rd: RedispatchModel, lam=1):
    Q = qen.create_symmetric_bf16(Q_dim(Rd))
    qen.update_qubo_switch_cost(
        Q, lam, Rd.T, Rd.n_resources, Rd.k, Rd.switch_cost
    )
    return Q


def qen_cost(Rd: RedispatchModel, lam=1):
    Q = qen.create_symmetric_bf16(Q_dim(Rd))
    qen.update_qubo_cost(Q, lam, Rd.cost_vals)
    return Q


def qen_overload(Rd: RedispatchModel, lam=1):
    Q = qen.create_symmetric_bf16(Q_dim(Rd))
    S = Rd.S_resos.astype("float64")
    M = (
        Rd.max_load
        - Rd.S_sgens @ Rd.sgen_prods
        - Rd.S_loads @ Rd.prod_static
    ).astype("float64")
    PE = Rd.prod_vals_eff
    qen.update_qubo_overload_unbalanced(Q, lam, M, S, PE)
    return Q


def qen_power(Rd: RedispatchModel, lam=1):
    Q = qen.create_symmetric_bf16(Q_dim(Rd))
    # Resos
    PE = Rd.prod_vals_eff
    tau = Rd.prod_target - Rd.sgen_prods.sum(axis=0)
    tau = tau + (tau * 0.01)
    qen.update_qubo_power_eq(Q, lam, PE, tau)
    return Q

# * qen Adapters
def qeb_switch_cost(Q, Rd: RedispatchModel, lam=1):
    qeb.update_qubo_switch_cost(Q, lam, Rd.T, Rd.switch_cost)
    return Q


def qeb_cost(Q, Rd: RedispatchModel, lam=1):
    qeb.update_qubo_cost(Q, lam, Rd.cost_vals)
    return Q


def qeb_overload(Q, Rd: RedispatchModel, lam=1):
    S = Rd.S_resos.astype("float64")
    M = (
        Rd.max_load
        - Rd.S_sgens @ Rd.sgen_prods
        - Rd.S_loads @ Rd.prod_static
    ).astype("float64")
    PE = Rd.prod_vals_eff
    qeb.update_qubo_overload_unbalanced(Q, lam, M, S, PE)
    return Q


def qeb_power(Q, Rd: RedispatchModel, lam=1):
    # Resos
    PE = Rd.prod_vals_eff
    Tau = Rd.prod_target - Rd.sgen_prods.sum(axis=0)
    Tau = Tau + (Tau * 0.01)
    qeb.update_qubo_power(Q, lam, PE, Tau)
    return Q

