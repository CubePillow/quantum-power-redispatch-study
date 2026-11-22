"""
Line-aware reduction of a full SimBench redispatch instance.

Lines and generators are coupled through the sensitivity matrix S
(flow on line l = sum_a S[l, a] * p_a). A reduced instance therefore
first picks critical lines and then the generators that influence
those lines most. All non-selected generators (and, by default, all
sgens) are fixed at their simulated production; their flow is moved
into the line capacity M, and tau only contains the selected units.
"""

import copy

import numpy as np

from .modeller import RedispatchModel
from .simulator import GridResults


class ReducedRedispatchModel(RedispatchModel):
    """RedispatchModel for a reduced instance.

    The flow of fixed (non-selected) generators and sgens is subtracted
    from `max_load`, so `M` and the original, unchanged code that reads
    `max_load` (e.g. RedispatchQUBO) both see the remaining capacity.
    """

    def __init__(self, sim_res, k, S, fixed_flow, seed=0):
        super().__init__(sim_res, k, S=S, seed=seed)
        # (n_lines x T) flow caused by fixed generators / sgens
        self.fixed_flow = fixed_flow
        self.max_load = self.max_load - fixed_flow
        self.set_S(self.S)  # recompute M from the reduced max_load


def _column_blocks(net):
    n_gen, n_sgen = len(net.gen), len(net.sgen)
    gen_cols = np.arange(0, n_gen)
    sgen_cols = np.arange(n_gen, n_gen + n_sgen)
    return n_gen, n_sgen, gen_cols, sgen_cols


def line_utilization(sim_res: GridResults, S):
    """Flow / free capacity per line and time step at the simulated
    production (all generators and sgens at their profile values)."""
    net = sim_res.net
    n_gen, n_sgen, gen_cols, sgen_cols = _column_blocks(net)
    load_cols = np.arange(n_gen + n_sgen, S.shape[1])
    max_load = net.line["max_mw_per_line"].to_numpy()[:, None]
    M = max_load - S[:, load_cols] @ sim_res.load_prods
    flow = (
        S[:, gen_cols] @ sim_res.reso_prods
        + S[:, sgen_cols] @ sim_res.sgen_prods
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        util = np.where(M > 0, flow / M, np.inf)
    return util, flow, M


def select_lines(sim_res, S, n_lines=2):
    """Lines with the highest utilization (max over time)."""
    util, _, _ = line_utilization(sim_res, S)
    score = util.max(axis=1)
    score[~np.isfinite(score)] = -np.inf  # skip lines without capacity
    return np.argsort(-score)[:n_lines]


def select_generators(sim_res, S, line_ids, n_gen=3, min_range_mw=1.0):
    """Generators with the largest summed |S| on the chosen lines.
    Generators that cannot change their output are skipped."""
    net = sim_res.net
    _, _, gen_cols, _ = _column_blocks(net)
    p_range = (net.gen["max_p_mw"] - net.gen["min_p_mw"]).to_numpy()
    infl = np.abs(S[np.ix_(line_ids, gen_cols)]).sum(axis=0)
    infl[p_range < min_range_mw] = -np.inf
    return np.argsort(-infl)[:n_gen]


def reduced_capacity(sim_res, S, line_ids, gen_ids):
    """M of the chosen lines after fixing all other generators and all
    sgens at their simulated production, plus the flow range the
    selected generators can produce (all off / all at max)."""
    net = sim_res.net
    n_gen_all, _, _, sgen_cols = _column_blocks(net)
    _, _, M_full = line_utilization(sim_res, S)
    line_ids, gen_ids = np.asarray(line_ids), np.asarray(gen_ids)
    fixed_gen = np.setdiff1d(np.arange(n_gen_all), gen_ids)
    fixed_flow = (
        S[np.ix_(line_ids, fixed_gen)] @ sim_res.reso_prods[fixed_gen]
        + S[np.ix_(line_ids, sgen_cols)] @ sim_res.sgen_prods
    )
    M_new = M_full[line_ids] - fixed_flow
    max_p = net.gen["max_p_mw"].to_numpy()[gen_ids]
    S_sel = S[np.ix_(line_ids, gen_ids)]
    f_min = (S_sel * np.minimum(max_p, 0)).sum(axis=1)  # usually 0
    f_max = (S_sel * np.maximum(max_p, 0)).sum(axis=1)
    return M_new, fixed_flow, f_min, f_max


def is_controllable(sim_res, S, line_ids, gen_ids, margin=0.1):
    """True if, on every chosen line and time step, the selected
    generators can both cause and avoid an overload, i.e.
    f_min < M_new < f_max (with a relative safety margin)."""
    M_new, _, f_min, f_max = reduced_capacity(sim_res, S, line_ids, gen_ids)
    span = (f_max - f_min)[:, None]
    lo = f_min[:, None] + margin * span
    hi = f_max[:, None] - margin * span
    return bool(np.all((M_new > lo) & (M_new < hi)))


def select_lines_and_generators(
    sim_res, S, n_lines=2, n_gen=3, n_candidates=200, margin=0.1
):
    """Greedy joint choice: take the most utilized line whose top
    generators can switch it between overloaded / not overloaded, then
    add further lines that are controllable by the same generators."""
    util, _, _ = line_utilization(sim_res, S)
    score = util.max(axis=1)
    score[~np.isfinite(score)] = -np.inf
    candidates = np.argsort(-score)[:n_candidates]
    for first in candidates:
        gens = select_generators(sim_res, S, [first], n_gen)
        if not is_controllable(sim_res, S, [first], gens, margin):
            continue
        lines = [first]
        for ln in candidates:
            if len(lines) >= n_lines:
                break
            if ln in lines:
                continue
            if is_controllable(sim_res, S, lines + [ln], gens, margin):
                lines.append(ln)
        return np.array(lines), np.sort(gens)
    raise ValueError(
        "no controllable line found; increase n_gen / n_candidates "
        "or lower margin"
    )


def create_line_aware_subset(
    sim_res: GridResults,
    S,
    k=3,
    n_lines=2,
    n_gen=3,
    line_ids=None,
    gen_ids=None,
    seed=0,
):
    """
    Build a small RedispatchModel:
      1. pick critical lines that the selected generators can actually
         switch between overloaded / not overloaded (or use `line_ids`),
      2. pick the generators with most influence on them
         (or use `gen_ids`),
      3. fix all other generators and all sgens at their simulated
         production and subtract their flow from M.
    `sim_res` is not modified.
    """
    net = sim_res.net
    _, _, gen_cols, sgen_cols = _column_blocks(net)
    load_cols = np.arange(len(gen_cols) + len(sgen_cols), S.shape[1])

    if line_ids is None and gen_ids is None:
        line_ids, gen_ids = select_lines_and_generators(
            sim_res, S, n_lines, n_gen
        )
    elif line_ids is None:
        line_ids = select_lines(sim_res, S, n_lines)
    if gen_ids is None:
        gen_ids = select_generators(sim_res, S, line_ids, n_gen)
    line_ids = np.asarray(line_ids)
    gen_ids = np.sort(np.asarray(gen_ids))

    # flow of everything that is no longer a decision variable
    _, fixed_flow, _, _ = reduced_capacity(sim_res, S, line_ids, gen_ids)

    sub_net = copy.deepcopy(net)
    sub_net.gen = sub_net.gen.iloc[gen_ids]
    sub_net.sgen = sub_net.sgen.iloc[0:0]
    sub_net.line = sub_net.line.iloc[line_ids]

    T = sim_res.T
    sub_res = GridResults(
        sub_net,
        T,
        sim_res.T_gap,
        sim_res.line_loads[line_ids],
        sim_res.reso_group,
        sim_res.reso_prods[gen_ids],  # -> tau = selected units only
        sim_res.sgen_group,
        np.zeros((0, T)),
        sim_res.load_group,
        sim_res.load_prods,
    )
    S_sub = S[np.ix_(line_ids, np.hstack([gen_ids, load_cols]))]
    Rd = ReducedRedispatchModel(sub_res, k, S_sub, fixed_flow, seed=seed)
    Rd.orig_line_ids = line_ids
    Rd.orig_gen_ids = gen_ids
    return Rd


def describe_instance(Rd, n_samples=20000, seed=0):
    """Quick check whether a reduced instance is interesting.
    Enumerates all level combinations per time step (independent of
    ramp) if small enough, else samples randomly."""
    rng = np.random.default_rng(seed)
    n, k, T = Rd.n_resources, Rd.k, Rd.T
    n_comb = k**n
    if n_comb <= n_samples:
        z = np.array(np.unravel_index(np.arange(n_comb), (k,) * n)).T
    else:
        z = rng.integers(0, k, size=(n_samples, n))
    p = Rd.prod_vals[np.arange(n), z]  # (combos x n)
    cost_per_mwh = Rd.prod_cost_multi
    info = {
        "n_lines": Rd.S.shape[0],
        "n_gen": n,
        "k": k,
        "T": T,
        "qubits_one_hot": n * k * T,
    }
    for t in range(T):
        flow = p @ Rd.S_resos.T  # (combos x lines)
        over = (flow > Rd.M[:, t]).any(axis=1)
        D = (p.sum(axis=1) - Rd.prod_target[t]) ** 2
        cost = p @ cost_per_mwh
        # cheapest combination that roughly meets demand
        ok = D <= np.quantile(D, 0.05)
        cheapest = np.argmin(np.where(ok, cost, np.inf))
        info[f"t{t}"] = {
            "tau": float(Rd.prod_target[t]),
            "M": np.round(Rd.M[:, t], 1).tolist(),
            "share_overloaded": float(over.mean()),
            "cheapest_balanced_overloads": bool(over[cheapest]),
        }
    return info


def create_small_models(
    all_res, S, folder="data/models/small", ts=(1, 2, 3), k=3,
    n_lines=2, n_gen=4,
):
    """Build and pickle line-aware reduced instances for several T.
    `all_res` comes from simulator.run_and_record_profiles."""
    import os
    import pickle

    from .simulator import get_even_first_T_time_points

    os.makedirs(folder, exist_ok=True)
    for t in ts:
        sim_res = get_even_first_T_time_points(all_res, t, 2)
        Rd = create_line_aware_subset(
            sim_res, S, k=k, n_lines=n_lines, n_gen=n_gen
        )
        print(f"T={t}", describe_instance(Rd))
        with open(f"{folder}/T{t}_g{n_gen}_l{n_lines}_k{k}.pkl", "wb") as f:
            pickle.dump(Rd, f)
