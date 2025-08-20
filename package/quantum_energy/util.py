import dimod
import numpy as np
from numba import njit  # type: ignore [import-untyped]


# * Index
@njit
def get_idx(bases, idxes):
    stride = 1
    idx = 0
    for i in np.arange(len(idxes))[::-1]:
        idx += stride * idxes[i]
        stride *= bases[i]
    return int(idx)


@njit
def get_idxes(bases, idx: int):
    base = 1
    for b in bases:
        base *= b
    idxes = np.zeros(len(bases))
    for i in range(len(bases)):
        base = int(base / bases[i])
        idxes[i] = idx // base
        idx = idx % base
    return idxes

@njit
def gen_tji_iter(T: int, n: int, k: int):
    tji = []
    for t in range(T):
        for j in range(n):
            for i in range(k):
                tji.append((t, j, i))
    return tji


@njit
def gen_ji_iter(n: int, k: int):
    ji = []
    for j in range(n):
        for i in range(k):
            ji.append((j, i))
    return ji


# * Helper fucntions for constructing QUBOs
@njit
def get_effective_prod_vals(P: np.ndarray, E: np.ndarray):
    """
    P = (n x k) matrix
    E = (n x T) matrix

    @returns (T x n x k) tensor
    """
    n, k = P.shape
    T = E.shape[1]
    PE = np.zeros((T, n, k))
    for t, j, i in gen_tji_iter(T, n, k):
        PE[t, j, i] = P[j, i] * E[j, t]
    return PE


@njit
def get_SPE(S: np.ndarray, PE: np.ndarray):
    """
    S  = (L x n) matrix
    PE = (T x n x k) matrix

    @returns (L x T x n x k) tensor
    """
    L = S.shape[0]
    T, n, k = PE.shape
    SPE = np.zeros((L, T, n, k))
    tij_iter = gen_tji_iter(T, n, k)
    for l in range(L):
        for t, j, i in tij_iter:
            SPE[l, t, j, i] = PE[t, j, i] * S[l, j]
    return SPE


# * Intergration with OCEAN


@njit
def QUBO_mat_to_params(Q: np.ndarray):
    N = len(Q)
    n_quad = np.sum(np.arange(N))
    quad = (
        np.zeros(n_quad, "int"),
        np.zeros(n_quad, "int"),
        np.zeros(n_quad),
    )
    linear = np.zeros(N)
    q_idx = 0
    for i in range(N):
        linear[i] = Q[i, i]
        for j in range(i + 1, N):
            quad[0][q_idx] = i
            quad[1][q_idx] = j
            quad[2][q_idx] = Q[i, j] + Q[j, i]
            q_idx += 1
    return (linear, quad)


def dict_to_array(d):
    if not isinstance(d, dict):
        return d
    a = np.zeros(len(d))
    for k, v in d.items():
        a[int(k)] = v
    return np.array(a)


def array_to_dict(a):
    if not isinstance(a, np.ndarray):
        return a
    d = {}
    for i in range(len(a)):
        d[i] = a[i]
    return d


def QUBO_mat_to_BQM(Q: np.ndarray):
    linear, quad = QUBO_mat_to_params(Q)
    return dimod.BQM.from_numpy_vectors(
        linear, quad, 0.0, dimod.BINARY, dtype="float64"
    )


def ocean_sampler_adapter(sampler):
    return lambda ql_Q: sampler.sample(
        QUBO_mat_to_BQM(ql_Q.m)
    ).first.sample


# * Solution handling


@njit
def sol_to_x(sol: np.ndarray, k: int):
    x_mat = np.zeros((len(sol), k), dtype="int8")
    for i in range(len(sol)):
        s = sol[i]
        x_mat[i] = np.eye(k)[int(s)] * (np.sign(s) if s != 0 else 1)
    return x_mat.flatten()


@njit
def x_invalids(x: np.ndarray, T: int, n: int, k: int):
    n_invalids = np.zeros(T)
    for j in range(n):
        prev_state = -1
        for t in range(T):
            idx1 = (j * k) + (t * n * k)
            one_ids = np.where(x[idx1 : idx1 + k] == 1)[0]
            if len(one_ids) != 1:
                n_invalids[t] += 1
                prev_state = -1
                continue
            elif prev_state == -1:
                prev_state = one_ids[0]
            elif prev_state != -1 and abs(prev_state - one_ids[0]) > 1:
                n_invalids[t] += 1
            prev_state = one_ids[0]
    return n_invalids


@njit
def x_to_sol(x: np.ndarray, k: int):
    ns = len(x) // k
    sol = np.zeros(ns, dtype="int")
    xm = x.reshape(ns, k)
    for i in range(ns):
        possible_sols = np.where(xm[i] == 1)[0]
        sol[i] = k - 1 if len(possible_sols) == 0 else possible_sols[-1]
    return sol


@njit
def get_adj_sols(k: int, cur_sol: int, d=1):
    return np.arange(max(0, cur_sol - d), min(k - 1, cur_sol + d) + 1)


@njit
def rand_sol_n(T: int, k: int, hs=None, es=None):
    sol = np.zeros(T, dtype='int')
    for t1 in range(0, int(T / 2)):
        t2 = T - 1 - t1
        dt = t2 - t1 + 1
        dk = dt - 2 + 1
        all_s1 = np.arange(k) if hs is None else get_adj_sols(k, hs)
        all_s2 = np.arange(k) if es is None else get_adj_sols(k, es)
        possible_combs = []
        for s1 in all_s1:
            for s2 in all_s2:
                if np.abs(s1 - s2) <= dk:
                    possible_combs.append((s1, s2))
        i = np.random.choice(np.arange(len(possible_combs)))
        s1, s2 = possible_combs[i]
        sol[t1], sol[t2] = possible_combs[i]
        hs = s1
        es = s2
    if T % 2 == 1:
        t = int(T / 2)
        if hs is None and es is None:
            sol[t] = np.random.randint(0, k)
        elif hs is None:
            sol[t] = np.random.choice(get_adj_sols(k, es))            
        elif es is None:
            sol[t] = np.random.choice(get_adj_sols(k, hs))            
        elif hs == es:
            sol[t] = np.random.choice(get_adj_sols(k, hs))            
        else:
            sol[t] = max(hs, es) - 1
    return sol.T


@njit
def rand_sol(T: int, n: int, k: int, hs=None, es=None):
    sols = np.zeros((T, n), dtype='int')
    for j in range(n):
        sols[:, j] = rand_sol_n(T, k)
    return sols.flatten()


@njit
def rand_sol_free_adj(T: int, n: int, k: int):
    return np.random.randint(0, k, T * n)


@njit
def rand_x(T: int, n: int, k: int):
    return sol_to_x(rand_sol(T, n, k), k)


@njit
def rand_x_free_adj(T: int, n: int, k: int):
    return sol_to_x(rand_sol_free_adj(T, n, k), k)


def get_constant_sol(t: int, n: int, s: int = 0):
    return s * np.ones(t * n, dtype=int)


def str_to_arr(s: str):
    return np.fromiter(s, dtype="int8")


def get_n_switches_mat(m: np.ndarray):
    next_m = np.r_["0,2", m[0, :], m[:-1, :]]
    return np.sum(next_m - m != 0, axis=1)


def get_n_switches(sol: np.ndarray, T: int, n: int):
    sol_mat = sol.reshape(T, n)
    return get_n_switches_mat(sol_mat)
