import numpy as np

from quantum_energy.problem.modeller import RedispatchModel
from quantum_energy.util import (
    get_constant_sol,
    get_idx,
    get_n_switches,
    x_invalids,
    x_to_sol,
)


class RedispatchStats:
    def __init__(self, R: RedispatchModel):
        self.R = R
        self.N1 = R.T * R.n_resources * R.k
        self.L = self.R.S_loads @ self.R.prod_static
        self.K = R.M

    def rand_sol(self):
        return np.random.randint(
            0, self.R.k, size=self.R.n_resources * self.R.T
        )

    def get_all_same_sol(self, s):
        return s * np.ones(self.R.n_resources * self.R.T).astype(int)

    def get_idx(self, j, t):
        return int(j + t * self.R.n_resources)

    def sol_to_x(self, sol):
        return np.array(
            [
                np.eye(self.R.k)[int(s)] * (np.sign(s) if s != 0 else 1)
                for s in sol
            ]
        ).flatten()

    def x_to_sol(self, x):
        # return np.hstack(np.eye(self.R.k)[x.astype(int)])
        return (
            x_to_sol(x[0 : self.N1], self.R.k),
            (
                get_constant_sol(self.R.T, self.R.m_sgens, 1)
                if len(x) == self.N1
                else x_to_sol(x[self.N1 :], 2)
            ),
        )

    def Pz(self, sol):
        Pz = np.zeros((self.R.n_resources, self.R.T))
        for j in range(self.R.n_resources):
            for t in range(self.R.T):
                idx = get_idx((self.R.T, self.R.n_resources), (t, j))
                Pz[j, t] = (
                    self.R.prod_vals[j][sol[idx]]
                    * self.R.cap_vals[j, t]
                )
        return Pz

    def SP(self, sol):
        return self.R.S_resos @ self.Pz(sol)

    def Py(self, sol_s):
        return (
            self.R.sgen_prods
            * sol_s.reshape(self.R.T, self.R.m_sgens).T
        )

    def SPy(self, sol_s):
        return self.R.S_sgens @ self.Py(sol_s)

    def SP_all(self, sol_g, sol_s):
        return self.SP(sol_g) + self.SPy(sol_s)

    def ubp_h_sol(self, sol_g, sol_s):
        return self.K - self.SP(sol_g) - self.SPy(sol_s)

    def ubp_min_max_h(self):
        hs = self.ubp_h_sol(
            get_constant_sol(
                self.R.T, self.R.n_resources, self.R.k - 1
            ),
            get_constant_sol(self.R.T, self.R.m_sgens, 2),
        )
        return (np.min(hs), np.max(hs))

    def unbalanced_penalty(self, sol_g, sol_s, lam1=1, lam2=1):
        H = self.ubp_h_sol(sol_g, sol_s)
        return (
            lam2 * np.sum(np.power(H, 2))
            - np.sum(np.power(self.K, 2))
            - lam1 * np.sum(H)
            + np.sum(self.K)
        )

    def costs_reso(self, sol_g):
        Cs = np.zeros(self.R.T)
        for t in range(self.R.T):
            Cs[t] = 0
            for j in range(self.R.n_resources):
                idx = self.get_idx(j, t)
                Cs[t] += self.R.cost_vals[t, j, sol_g[idx]]
                if t < self.R.T - 1:
                    jdx = self.get_idx(j, t + 1)
                    Cs[t] += self.R.switch_cost[j][
                        sol_g[idx], sol_g[jdx]
                    ]
        return Cs

    def costs_sgen(self, sol_s):
        Cs = np.zeros(self.R.T)
        for t in range(self.R.T):
            Cs[t] = 0
            for g in range(self.R.m_sgens):
                idx = get_idx((self.R.T, self.R.m_sgens), (t, g))
                Cs[t] += self.R.sgen_prices[g, t] * (1, 0)[sol_s[idx]]
        return Cs

    def costs(self, sol_g, sol_s):
        Cs = np.zeros(self.R.T)
        for t in range(self.R.T):
            Cs[t] = 0
            for j in range(self.R.n_resources):
                idx = self.get_idx(j, t)
                Cs[t] += self.R.cost_vals[t, j, sol_g[idx]]
                if t < self.R.T - 1:
                    jdx = self.get_idx(j, t + 1)
                    Cs[t] += self.R.switch_cost[j][
                        sol_g[idx], sol_g[jdx]
                    ]
            for g in range(self.R.m_sgens):
                idx = get_idx((self.R.T, self.R.m_sgens), (t, g))
                Cs[t] += self.R.sgen_prices[g, t] * (1, 0)[sol_s[idx]]
        return Cs

    def cost(self, sol_g, sol_s):
        return np.sum(self.costs(sol_g, sol_s))

    def powers(self, sol_g, sol_s):
        return self.Pz(sol_g).sum(axis=0) + self.Py(sol_s).sum(axis=0)

    def power_eq(self, sol_g, sol_s):
        return np.linalg.norm(
            self.powers(sol_g, sol_s) - self.R.prod_target
        ) ** 2 - np.sum(np.power(self.R.prod_target, 2))

    def violate_max_load(self, sol_g, sol_s):
        # return (self.R.max_load < (self.SP(sol) - self.L))
        return self.K < self.SP_all(sol_g, sol_s)

    def violate_max_load_all(self, sol_g, sol_s):
        return np.logical_or.reduce(
            np.logical_or.reduce(self.violate_max_load(sol_g, sol_s))
        )

    def obj_ubp(self, sol_g, sol_s):
        C = self.cost(sol_g, sol_s)
        bal = self.power_eq(sol_g, sol_s)
        cap = self.unbalanced_penalty(sol_g, sol_s)
        return C + bal + cap

    def obj(self, sol_g, sol_s):
        C = self.cost(sol_g, sol_s)
        bal = self.power_eq(sol_g, sol_s)
        l = np.max(self.SP_all(sol_g, sol_s)) * 5
        cap = (
            np.linalg.norm(self.violate_max_load(sol_g, sol_s) * l) ** 2
        )
        return C + bal + cap

    def get_all_sols(self):
        n_res = self.R.n_resources * self.R.T
        return [
            np.array(
                list(np.base_repr(i, base=self.R.k).zfill(n_res))
            ).astype(np.int8)
            for i in range(self.R.k**n_res)
        ]

    def sort_sol_obj_pair(self, sols, obj):
        sorted_sols = [
            x for x, _ in sorted(zip(sols, obj), key=lambda p: p[1])
        ]
        return (sorted_sols, sorted(obj))

    def best_manual(self):
        all_sols = self.get_all_sols()
        all_obj = [self.obj(sol) for sol in all_sols]
        return self.sort_sol_obj_pair(all_sols, all_obj)

    def best_qubo(self, Q):
        all_sols = self.get_all_sols()
        all_obj = [Q(self.sol_to_x(sol)) for sol in all_sols]
        return self.sort_sol_obj_pair(all_sols, all_obj)

    def x_is_valid(self, x):
        xs = x.reshape(self.R.n_resources * self.R.T, self.R.k)
        return np.count_nonzero(np.sum(xs, axis=1) - 1) == 0

    def x_is_valid_count(self, x, n, k):
        xs = x.reshape(n * self.R.T, k)
        return np.count_nonzero(np.sum(xs, axis=1) - 1)

    def x_invalids(self, x):
        return x_invalids(
            x[0 : self.N1], self.R.T, self.R.n_resources, self.R.k
        ) + (
            [0] * self.R.T
            if len(x) == self.N1
            else x_invalids(x[self.N1 :], self.R.T, self.R.m_sgens, 2)
        )

    def x_invalids_count(self, x):
        return np.sum(self.x_invalids(x))

    def x_sol_to_stats(self, x):
        stats_over_time = self.x_sol_to_stats_over_time(x)
        stats = {}
        sol_g, sol_s = self.x_to_sol(x)
        powers_fufill = (self.powers(sol_g, sol_s) / self.R.prod_target) - 1
        for k, v in stats_over_time.items():
            if k == "t":
                continue
            stats[k] = np.mean(stats_over_time[k])
        stats["power_fufill"] = np.mean(powers_fufill)
        stats["n_power_fufill"] = (
            powers_fufill >= 1
        ).sum()

        return stats
        # sol_g, sol_s = self.x_to_sol(x)
        # n_violation = np.sum(self.violate_max_load(sol_g, sol_s)).item()
        # cost = self.cost(sol_g, sol_s)
        # power_fulfill = np.mean(
        #     (
        #         np.sum(self.Pz(sol_g), axis=0)
        #         + np.sum(self.Py(sol_s), axis=0)
        #     )
        #     / self.R.prod_target
        # )
        # n_non_valid = self.x_invalids_count(x)
        # return {
        #     "n_violations": n_violation,
        #     "n_non_valid": n_non_valid,
        #     "cost": cost,
        #     "power_fulfill": power_fulfill,
        # }

    def x_sol_to_stats_over_time(self, x):
        sol_g, sol_s = self.x_to_sol(x)
        n_violations = np.sum(
            self.violate_max_load(sol_g, sol_s), axis=0
        )
        costs = self.costs(sol_g, sol_s)
        powers = (self.powers(sol_g, sol_s)  / self.R.prod_target) - 1
        n_switches = get_n_switches(
            sol_g, self.R.T, self.R.n_resources
        ) + get_n_switches(sol_s, self.R.T, self.R.m_sgens)
        invalids = self.x_invalids(x)
        return {
            "t": list(range(self.R.T)),
            "overloads": n_violations,
            "invalids": invalids,
            "costs": costs,
            # "costs_reso": self.costs_reso(sol_g),
            # "costs_sgen": self.costs_sgen(sol_s),
            "powers": powers,
            "switches": n_switches,
        }
