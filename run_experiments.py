import json
import os
import pickle
import time

import dimod
import dwave.samplers as dws
import hybrid as dwh

# import iaisevo
import numpy as np
import pandas as pd
import polars as pl

# * Test C code
import qe_native as qen
import qubolite as ql
from numba import njit
from quantum_energy.decompose import ocean_subsolver_factory
from quantum_energy.dwave_solver import DWaveAnnealing, d_wave
from quantum_energy.problem.brute_force import RedispatchBruteForce
from quantum_energy.problem.modeller import RedispatchModel
from quantum_energy.problem.qubo import (
    RedispatchQUBO,
    RedispatchResosQUBO,
)
from quantum_energy.problem.simulator import (
    add_line_capacity_to_net,
    get_even_spaced_time_points,
    run_and_record_profiles,
)
from quantum_energy.util import (
    get_n_switches_mat,
    ocean_sampler_adapter,
    str_to_arr,
)

# import experiments.compare_decomposers as comp_decomp
# import experiments.compare_solvers as comp_solve
# import experiments.full_run as full_run
# import experiments.grid_search as grid_search
# import experiments.norm_penalty_compare as norm_pen_compare
# import experiments.penalty_type_compare as pen_type
# import experiments.unbalanced_penalty_compare as unb_pen_compare
import simbench as sb


# * Create data
def load_model():
    with open("experiments/data/models/S/0.pkl", "rb") as f:
        Rd = pickle.load(f)
    Rd_qubo = RedispatchQUBO(Rd)
    Rd_brute_force = RedispatchBruteForce(Rd)


def create_model_set(sim_res, k, S, folder, n_run=10):
    for i in range(n_run):
        if not os.path.exists(f"{folder}"):
            os.makedirs(f"{folder}")
        Rd = RedispatchModel(sim_res, k)
        Rd.set_S(S)
        with open(f"{folder}/{i}.pkl", "wb") as f:
            pickle.dump(Rd, f)


def get_subset_ids(net, S, n_gen=4, n_sgen=2):
    gen_ids = np.arange(0, n_gen)
    sgen_ids = np.arange(len(net.gen), len(net.gen) + n_sgen)
    other_ids = np.arange(len(net.gen) + len(net.sgen), S.shape[1])
    return (gen_ids, sgen_ids, other_ids)


def create_model_subset(sim_res, S, n_gen=4, n_sgen=2):
    gen_ids, sgen_ids, other_ids = get_subset_ids(
        sim_res.net, S, n_gen, n_sgen
    )
    subset_ids = np.hstack([gen_ids, sgen_ids, other_ids])
    # net subset
    sim_res.net.gen = sim_res.net.gen[0:n_gen]
    sim_res.net.sgen = sim_res.net.sgen[0:n_sgen]
    sim_res.reso_prods = sim_res.reso_prods[0:n_gen, :]
    sim_res.sgen_prods = sim_res.reso_prods[0:n_sgen, :]
    S = S[:, subset_ids]
    Rd = RedispatchModel(sim_res, 3)
    Rd.set_S(S)
    with open("package/tests/XS.pkl", "wb") as f:
        pickle.dump(Rd, f)


def create_models():
    folder = "experiments/data/models/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    grid_code = "1-EHV-mixed--0-sw"
    net = sb.get_simbench_net(grid_code)
    add_line_capacity_to_net(net)
    all_res = run_and_record_profiles(net, 0)
    # Create Redispatch problem from simulation results
    sim_res_t2 = get_even_spaced_time_points(all_res, 2)
    sim_res_t8 = get_even_spaced_time_points(all_res, 8)
    S = np.load(
        "experiments/data/S_mat_1-EHV-mixed--0-sw/S.npy"
    ).astype("float32")
    n_run = 10
    create_model_set(sim_res_t2, 3, S, f"{folder}/S", n_run)
    # create_model_set(sim_res_t2, 5, S, f"{folder}/SM", n_run)
    # create_model_set(sim_res_t2, 10, S, f"{folder}/MS", n_run)
    # create_model_set(sim_res_t8, 5, S, f"{folder}/M", n_run)

    # sim_res_t192 = get_even_spaced_time_points(all_res, 192)
    # create_model_set(sim_res_t192, 5, S, f"{folder}/XL", n_run)


# create_models()
# * Solvers
def run_on_ql_ss_single(Q):
    x, e = ql.solving.subspace_search(Q, 10000)
    return x


# * Result analyser
# ** Simulation Stats
def sim_res_stats(Rd: RedispatchModel):
    res_over_time = sim_res_stats_over_time(Rd)
    res = {}
    for k, v in res_over_time.items():
        if k == "t":
            continue
        res[k] = np.mean(res_over_time[k])
    return res


@njit
def get_n_invalid_switches(prod_vals, sim_prods):
    n, T = sim_prods.shape
    sol = np.zeros((n, T))
    n_invalids = 0
    for j in range(n):
        for t in range(T):
            s = (prod_vals[j] <= sim_prods[j, t]).nonzero()[0][-1]
            sol[j, t] = s
            if t > 0 and np.abs(s - sol[j, t - 1]) > 1:
                n_invalids += 1
    return n_invalids


def sim_res_stats_over_time(Rd: RedispatchModel):
    cost_mwh = Rd.prod_cost_multi
    switch_costs = np.abs(
        Rd.prod_resos
        - np.r_["1,2,0", Rd.prod_resos[:, 0], Rd.prod_resos[:, :-1]]
    ).sum(0)
    return {
        "t": list(range(Rd.T)),
        "sim_gen_powers": Rd.prod_target,
        "sim_costs": Rd.prod_resos.T @ cost_mwh + switch_costs,
        "sim_n_violations": np.sum(
            Rd.sim_res.line_loads > Rd.max_load, axis=0
        ),
        "sim_n_switches": get_n_switches_mat(Rd.prod_resos.T),
        "sim_n_invalid_switch": get_n_invalid_switches(
            Rd.prod_vals, Rd.sim_res.reso_prods
        ),
    }


def get_problem_stats(prob, n_runs=1):
    all_stats = {}
    for i in range(n_runs):
        with open(f"experiments/data/models/{prob}/{i}.pkl", "rb") as f:
            Rd = pickle.load(f)
        stats = sim_res_stats(Rd)
        for k, v in stats.items():
            all_stats[k] = all_stats.get(k, [])
            all_stats[k].append(v)
    return pd.DataFrame(all_stats)


# ** Average Results
def get_results(filepath, prob_size="S"):
    with open(filepath, "rb") as f:
        tmp_res = json.load(f)
    sim_n_overloads = get_problem_stats(prob_size)[
        "sim_n_violations"
    ].mean()
    df = (
        pl.DataFrame(data=tmp_res)
        .group_by(
            [
                "lam_power",
                "lam_overload",
                "lam_cost",
                "lam_switch",
            ]
        )
        .mean()
        .filter(
            (pl.col("n_power_fufill") == pl.col("n_power_fufill").max())
            & (pl.col("invalids") == 0)
            & (pl.col("overloads") <= sim_n_overloads)
        )
        .drop("run", "n_power_fufilled", "n_invalids")
        .sort(
            "costs",
            "overloads",
            # "costs_reso",
            # "costs_sgen",
            "switches",
            descending=[False, False, False],
        )
        # .filter.sort("n_violations", "")
    )
    return df


def get_avg_results(filepath="scripts/results/solvers.json"):
    with open(filepath, "rb") as f:
        tmp_res = json.load(f)
    tmp_res_avg = {}
    for name, res in tmp_res.items():
        tmp_res_avg["name"] = tmp_res_avg.get("name", [])
        tmp_res_avg["name"].append(name)
        for att, vals in tmp_res[name].items():
            if att in ["es", "lambdas"]:
                continue
            tmp_res_avg[att] = tmp_res_avg.get(att, [])
            tmp_res_avg[att].append(np.mean(vals))
            tmp_res_avg[att + "_std"] = tmp_res_avg.get(
                att + "_std", []
            )
            tmp_res_avg[att + "_std"].append(np.std(vals))
    # return tmp_res_avg
    return pd.DataFrame(data=tmp_res_avg)


def get_avg_results_single(filepath="scripts/results/solvers.json"):
    with open(filepath, "rb") as f:
        tmp_res = json.load(f)
    tmp_res_avg = {}
    for att, vals in tmp_res.items():
        if att != "xs" and att != "es":
            tmp_res_avg[att] = tmp_res_avg.get(att, [])
            tmp_res_avg[att].append(np.mean(vals))
            tmp_res_avg[att + "_std"] = tmp_res_avg.get(
                att + "_std", []
            )
            tmp_res_avg[att + "_std"].append(np.std(vals))
    # return tmp_res_avg
    return pd.DataFrame(data=tmp_res_avg)


# create_models()
# * Run
# ** Check noramlisation constant for unbalanced penalty
# unb_pen_compare.run(run_digital_annealer_single)
# unb_pen_compare.run(ocean_sampler_adapter(dws.TabuSampler()), "SM")
# ** Check noramlisation constant for all penalty
# norm_pen_compare.run(ocean_sampler_adapter(dws.TabuSampler()))
# norm_pen_compare.run(ocean_sampler_adapter(dws.TabuSampler()), "SM")
# ** Compare Slack variable and unbalanced penalty
# pen_type.run(run_on_ql_ss_single)
# ** Grid Search
# grid_search.run(ocean_sampler_adapter(dws.TabuSampler()), "S", True)
# ** Compare solvers
methods_to_test = {
    #    "tabu": ocean_sampler_adapter(dws.TabuSampler()),
    #    "simulated_annealing": ocean_sampler_adapter(
    #        dws.SimulatedAnnealingSampler()
    #    ),
    #    "steepest_descent": ocean_sampler_adapter(
    #        dws.SteepestDescentSampler()
    #    ),
    #    "digital_annealer": run_digital_annealer_single,
    "subspace_sampling": run_on_ql_ss_single,
}

# comp_solve.run(methods_to_test, model_size='XL')

# Rd_qubo = RedispatchQUBO(Rd)

# Rd = Rd_qubo.R
# Rd_brute_force = RedispatchBruteForce(Rd)
# Q = Rd_qubo.get_qubo_obj(1,1/Rd_qubo.C,1/(2*Rd_qubo.C**2),1)
# A = np.max(Q) * Rd_qubo.qubo_one_hot
# ql_Q = ql.qubo(Q + A)

# dwave = DWaveAnnealing(
#     token=os.environ["API_DWAVE"],region=None)

# full_run.run(
#     lambda Q: d_wave(Q, dwave, sample_set_flag=True, num_reads=10
#                      ).first.sample,
#     sub_size=128, prob_size='M', max_runs=50,
#     n_run=5, res_name="dwave_50iter")

# ** Compare decomposers
# subsize = 128
# name_to_subsolvers = {
#     'a-expansion': (subsize, run_digital_annealer_single),
#     'random': ocean_subsolver_factory(
#         dwh.RandomSubproblemDecomposer(size=subsize)
#         | DigitalAnnealerSubproblemSampler(os.environ["API_DIGITAL_ANNEALER"]) | dwh.SplatComposer()),
#     'energy': ocean_subsolver_factory(
#         dwh.EnergyImpactDecomposer(
#             size=subsize, rolling=True, rolling_history=1.0
#         ) | DigitalAnnealerSubproblemSampler(os.environ["API_DIGITAL_ANNEALER"]) | dwh.SplatComposer())
# }
# name_to_subsolvers = {
#    "a-expansion": (subsize, ocean_sampler_adapter(dws.TabuSampler())),
#    "random": ocean_subsolver_factory(
#        dwh.RandomSubproblemDecomposer(size=subsize)
#        | dwh.TabuSubproblemSampler()
#        | dwh.SplatComposer()
#    ),
#    "energy": ocean_subsolver_factory(
#        dwh.EnergyImpactDecomposer(
#            size=subsize, rolling=True, rolling_history=1.0
#        )
#        | dwh.TabuSubproblemSampler()
#        | dwh.SplatComposer()
#    ),
#

# comp_decomp.run(
#     name_to_subsolvers, 49, 2, 300, 50,
#     'decompose-128-max-time-120')
# comp_decomp.plot_convergence(
#     "scripts/results/decompose-128-max-time-120.json"
# )

# ** full run
# full_run.run(ocean_sampler_adapter(dws.TabuSampler()),
#              max_runs=np.inf,
#              n_run=2, prob_size='SM', sub_size=256)
# full_run.stats_over_time('full_run_MS', prob_size='SM')
# stats_over_time('full_run_MS', prob_size='SM')

# * Debug
# ** Compare solutions with and without adjacent constraints
import quantum_energy.solver.subsolvers as qe_subsolv
from quantum_energy.decompose import run_decomposer
from quantum_energy.problem.qubo import get_hard_constraint_penalty
from quantum_energy.util import dict_to_array, sol_to_x

with open("./experiments/data/models/S/0.pkl", "rb") as f:
    Rd = pickle.load(f)
Rd_qubo = RedispatchQUBO(Rd, False)
Rd_brute_force = RedispatchBruteForce(Rd)

# Q_ori = Rd_qubo.get_qubo_obj_normed(3, 2, 3, 1)
Q_ori = Rd_qubo.get_qubo_obj_normed(20, 100, 20, 0.0001)
Q_pen = get_hard_constraint_penalty(Q_ori)
Q_wt_adj = (
    Q_ori + Q_pen * Rd_qubo.qubo_one_hot + Q_pen * Rd_qubo.qubo_adjacent
)
Q_no_adj = Q_ori + Q_pen * Rd_qubo.qubo_one_hot

x_init = sol_to_x(np.zeros(Rd.T * Rd.n_resources), Rd.k)
# x_init = np.hstack([x_init, sol_to_x(np.zeros(Rd.T * Rd.m_sgens), 2)])
solver = ocean_sampler_adapter(dws.TabuSampler())
sol_x_wt_adj = solver(ql.qubo(Q_wt_adj))
res_wt_adj = Rd_brute_force.x_sol_to_stats(dict_to_array(sol_x_wt_adj))
res_wt_adj

x_init = sol_to_x(np.zeros(Rd.T * Rd.n_resources), Rd.k)
# x_init = np.hstack([x_init, sol_to_x(np.zeros(Rd.T * Rd.m_sgens), 2)])
solver = ocean_sampler_adapter(dws.TabuSampler())
sol_x_no_adj = solver(ql.qubo(Q_no_adj))
res_no_adj = Rd_brute_force.x_sol_to_stats(dict_to_array(sol_x_no_adj))
res_no_adj

res_wt_adj
res_no_adj


subsolver = BetaExpansion(
    Rd.T, Rd.n_resources, 0, Rd.k, 32, solver
).subsolver_ql
sol_x_alpha, es = run_decomposer(
    Q_ori,
    subsolver,
    x_init.copy(),
    bqm=None,
    max_runs=50,
    max_time=np.inf,
    max_stale=5,
)
res_alpha = Rd_brute_force.x_sol_to_stats(dict_to_array(sol_x_alpha))
res_alpha


res_init = Rd_brute_force.x_sol_to_stats(dict_to_array(x_init))
res_init
