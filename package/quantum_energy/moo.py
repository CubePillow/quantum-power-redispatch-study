import numpy as np
import qubolite as ql
from tqdm import tqdm


class WeightEnergy:
    def __init__(self, weights, energies, x):
        assert len(weights) == len(energies)
        self.weights = weights
        self.energies = energies
        self.x = x

    def __lt__(self, other):
        return list(self.weights) < list(other.weights)

    def __le__(self, other):
        return list(self.weights) <= list(other.weights)

    def __gt__(self, other):
        return list(self.weights) > list(other.weights)

    def __ge__(self, other):
        return list(self.weights) >= list(other.weights)

    def __eq__(self, other):
        return list(self.weights) == list(other.weights)

    def __ne__(self, other):
        return list(self.weights) != list(other.weights)

    def distance(self, other):
        return np.linalg.norm(self.energies - other.energies)

    def dominated_by(self, other):
        doms = np.all(other.energies <= self.energies)
        dom = np.any(other.energies < self.energies)
        return doms and dom

def pf_adaptive_avg(
    Qs: list[ql.qubo],
    constrs: list[ql.qubo],
    n_weights,
    solver,
    constr_scale=10,
):
    m = len(Qs)
    A: list[WeightEnergy] = []
    W: list[WeightEnergy] = []
    mixed: set[tuple] = set()
    for i in tqdm(range(n_weights)):
        if i < m:
            lambdas = np.zeros(m)
            lambdas[i] = 1.0
        else:
            max_d = np.float64(0.0)
            max_pair = (-1, -1)
            print("=" * 72)
            for j1 in range(len(W)):
                for j2 in range(j1 + 1, len(W)):
                    if (j1, j2) in mixed:
                        continue
                    d = W[j1].distance(W[j2])
                    print((W[j1].weights, W[j2].weights), ":", d)
                    if d > max_d:
                        max_d = d
                        max_pair = (j1, j2)
            print(W[max_pair[0]].weights, W[max_pair[1]].weights)
            print("=" * 72)
            mixed.add(max_pair)
            U = W[max_pair[0]].weights
            V = W[max_pair[1]].weights
            lambdas = (U + V) / 2
        cur_Q = np.zeros(Qs[0].m.shape)
        for j in range(len(Qs)):
            cur_Q += lambdas[j] * Qs[j].m
        for j in range(len(constrs)):
            cur_Q += constr_scale * constrs[j].m
        sols = solver(cur_Q)
        es = [Q(sols) for Q in Qs]
        for j in range(len(sols)):
            objs = np.array([es[Qi][j] for Qi in range(len(Qs))])
            we = WeightEnergy(lambdas, objs, sols[j])
            if j == 0:
                W.append(we)
            A.append(we)
    return A


def filter_non_dominated(wes: list[WeightEnergy], Rd_stats, p_thresh):
    new_wes: list[WeightEnergy] = []
    for we in wes:
        s = Rd_stats.x_sol_to_stats(we.x)
        if 0 in we.weights:
            continue
        if np.abs(s["power_fufill"]) > p_thresh:
            continue
        new_wes = [we2 for we2 in new_wes if not we2.dominated_by(we)]
        non_dominated = np.all(
            [not we.dominated_by(we2) for we2 in new_wes]
        )
        if non_dominated:
            new_wes.append(we)
    return new_wes


def filter_and_sort_pf(Rd_stats, wes, power_thresh=0.01):
    pf = filter_non_dominated(wes, Rd_stats, power_thresh)
    stats = {}
    for we in pf:
        s = Rd_stats.x_sol_to_stats(we.x)
        stats[tuple(we.weights)] = s
    return sorted(
        pf,
        key=lambda x: (
            stats[tuple(x.weights)]["overloads"],
            stats[tuple(x.weights)]["power_fufill"],
            stats[tuple(x.weights)]["costs"],
            stats[tuple(x.weights)]["switches"],
        ),
    )
