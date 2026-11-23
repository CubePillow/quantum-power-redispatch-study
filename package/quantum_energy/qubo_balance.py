import matplotlib.pyplot as plt
import numpy as np
import qe_builder as qeb
import qubolite as ql
from numba import njit, prange
from tqdm import tqdm

from quantum_energy.problem.modeller import RedispatchModel
from quantum_energy.problem.qubo import RedispatchQUBO
from quantum_energy.qubo_builder import (
    Q_dim,
    qeb_cost,
    qeb_overload,
    qeb_power,
    qeb_switch_cost,
)
from quantum_energy.util import rand_x_free_adj


@njit(parallel=True)
def get_one_hot_xs(T, n, k, nsamp):
    xs = np.zeros((nsamp, T * n * k))
    for i in prange(nsamp):
        xs[i,:] = rand_x_free_adj(T, n, k)
    return xs

def get_energy_dist_ql(Q, nsamp):
    xs = np.random.randint(0, 2, size=(nsamp, Q.dim())).astype(
        np.float32
    )
    # es = ql.qubo(Q)(xs)
    es = Q.energies(xs)
    return es


@njit(parallel=True)
def get_energy_dist(T, n, k, Q, nsamp):
    es = np.zeros(nsamp, dtype=np.float64)
    for i in prange(nsamp):
        x = rand_x_free_adj(T, n, k).astype("float64")
        es[i] = x.T @ Q @ x
    return es


@njit
def get_Q_mean(Q):
    n = Q.shape[0]
    u = 0
    for i in range(n):
        for j in range(n):
            u += Q[i, j] / (2 ** (len(set([i, j]))))
    return u


@njit
def get_Q_var(Q):
    n = Q.shape[0]
    v = 0
    for i in range(n):
        for j in range(n):
            if Q[i, j] == 0:
                continue
            for k in range(n):
                for l in range(n):
                    v += (
                        Q[i, j]
                        * Q[k, l]
                        / (2 ** (len(set([i, j, k, l]))))
                    )
    return v


def plot_energy_dists(name_to_energies, axes=None):
    if axes is None:
        fig, axes = plt.subplots(
            1, len(name_to_energies), sharey=True, sharex=True
        )
    for i, (name, energies) in enumerate(name_to_energies.items()):
        axes[i].hist(energies, bins=100)
        axes[i].set_title(name)
        axes[i].set(xlabel="$x^T Q x$")
        axes[i].text(
            0.95,
            0.95,
            f"mean: {np.mean(energies):.4g}\nvar: {np.var(energies):.4g}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=axes[i].transAxes,
        )
        # axes[i].set_xlim(0,1)
        # axes[i].set_yscale("log")
    return axes


def make_Q(Rd: RedispatchModel):
    return qeb.create_sparse_symmetric(Q_dim(Rd))


def get_energy_dists(Rd: RedispatchModel, nsamp):
    T = Rd.T
    n = Rd.n_resources
    k = Rd.k
    name_to_Qs = {
        "power": qeb_power(make_Q(Rd), Rd),
        "overload": qeb_overload(make_Q(Rd), Rd),
        "cost": qeb_cost(make_Q(Rd), Rd),
        "switch": qeb_switch_cost(make_Q(Rd), Rd),
    }
    name_to_mean = {
        name: Q.comp_energy_mean() for name, Q in name_to_Qs.items()
    }
    name_to_variance = {
        name: Q.comp_energy_variance_2() for name, Q in name_to_Qs.items()
    }
    xs = get_one_hot_xs(Rd.T, Rd.n_resources, Rd.k, nsamp)
    name_to_energies = {
        name: Q.energies(xs)
        for name, Q in tqdm(name_to_Qs.items())
    }
    name_to_normed_energies = {
        n: (es - name_to_mean[n]) / np.sqrt(name_to_variance[n])
        for n, es in name_to_energies.items()
    }
    p, axes = plt.subplots(2, 4)
    plot_energy_dists(name_to_energies, axes=axes[0])
    plot_energy_dists(name_to_normed_energies, axes=axes[1])
    plt.tight_layout()
    plt.show()
