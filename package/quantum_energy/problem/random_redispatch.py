import numpy as np

from scipy.sparse.csgraph import minimum_spanning_tree as mst  # type: ignore [import-untyped]
from ..util import get_effective_prod_vals


# Goal: minimize cost while maintaining constraints
class RandomRedispatch:
    def __init__(
        self,
        n_resources=8,
        n_internal=7,
        n_output=2,
        n_handover=3,
        n_groups=7,
        T=192,
        k=4,
        m_sgens=8,
    ):
        self.n_resources = None
        self.n_handover = None
        self.n_groups = None

        self.T = None
        self.max_load = None  # L_max

        self.handover = []

        self.cost_vals = np.zeros(k * n_resources).reshape(
            n_resources, k
        )
        self.prod_vals = np.zeros(k * n_resources).reshape(
            n_resources, k
        )
        self.cap_vals = None

        self.q = None

        self.switch_cost = np.zeros(k * k * n_resources).reshape(
            n_resources, k, k
        )

        self.T = T
        self.n_resources = n_resources
        self.m_sgens = m_sgens
        self.n_handover = n_handover
        self.n_groups = n_groups
        self.n_internal = n_internal
        self.n_output = n_output
        self.k = k

        # randomly assign resources to handover points
        for j in range(self.n_resources):
            i = np.random.randint(self.n_handover)
            self.handover.append([i, j])

        self.sgen_prods = (
            np.random.random((self.m_sgens, self.T)) * 0.01
        )
        self.sgen_prices = self.sgen_prods * 95
        # randomly assign control steps and costs to each group
        for j in range(self.n_resources):
            self.switch_cost[j] = np.zeros(k * k).reshape((k, k))
            for y1 in range(k):
                for y2 in range(k):
                    self.switch_cost[j][y1, y2] = np.abs(y1 - y2)

            self.prod_vals[j] = np.random.random(size=k)
            self.prod_vals[j] = np.add.accumulate(
                self.prod_vals[j]
            )  # MW, monotonic incr.

        self.cap_vals = np.ones((self.n_resources, self.T))
        self.prod_vals_eff = get_effective_prod_vals(
            self.prod_vals, self.cap_vals
        )

        # somewhat realistic, I guess
        euro_per_kw = 0.05 + np.random.random() * 0.18

        # linear cost function
        self.cost_vals = self.prod_vals_eff * euro_per_kw * 1000

        self.q = np.zeros((self.n_resources, self.T))

        # derived
        self.n_internal_eff = self.n_internal - (
            n_output + n_handover
        )  # number of true internal nodes
        self.n_e_internal = (
            self.n_internal_eff + self.n_output - 1
        )  # number of internal edges (connections)

        self.max_load = (
            np.ones((self.n_e_internal + 2 * self.n_handover, self.T))
            * 0.75
        )
        self.prod_resos = (
            np.ones((self.n_e_internal + 2 * self.n_handover, self.T))
            * 0.5
        )
        self.prod_static = (
            np.random.random((self.n_resources, self.T)) * 0.01
        )
        self.prod_target = self.sgen_prods
        self.prod_target = self.prod_resos.sum(
            axis=0
        ) + self.sgen_prods.sum(axis=0)

        # self.max_load = np.random.rand(self.n_e_internal+2*self.n_handover,self.T) * 10

        # create random internal tree-structured connectivity
        A = np.random.random(size=self.n_internal_eff**2).reshape(
            (self.n_internal_eff, self.n_internal_eff)
        )
        self.internal = mst(A @ A.T)

        self.S_sgens = self.S_intern() @ self.S_extern()
        self.S_loads = self.S_intern() @ self.S_extern()
        self.S_resos = self.S_intern() @ self.S_extern()

        self.M = self.max_load - self.S_loads @ self.prod_static

    def S_extern(self):
        result = np.zeros((self.n_handover, self.n_resources))
        for h in self.handover:
            result[h[0], h[1]] = 1
        return result

    def S_intern(self):
        n_rows = self.n_internal - 1 + self.n_handover
        A = np.zeros((n_rows, self.n_handover))

        # no "loss" at handover point between internal and external networks
        A[n_rows - self.n_handover :, : self.n_handover] = np.eye(
            self.n_handover
        )
        A[
            n_rows - 2 * self.n_handover : n_rows - self.n_handover,
            : self.n_handover,
        ] = np.eye(self.n_handover)

        # fill each column (trnasmission/loss per internal connection w.r.t. connected group)
        for c in range(self.n_handover):
            v = np.random.random(self.n_e_internal)
            v /= v.sum()
            A[: self.n_e_internal, c] = v

        return A


# R = Redispatch()

# print('cost =',R.cost())

# Q,h,c = R.qubo()
# x = R.q_to_x()

# qubo_loss = x.T @ Q @ x + h @ x + c
# print('qubo_loss =',qubo_loss)
