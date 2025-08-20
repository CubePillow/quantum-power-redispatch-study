import os
import pickle

import dwave.samplers as dws
import numpy as np
import qe_builder as qeb
import qubolite as ql
from quantum_energy.decompose import dict_to_array, run_decomposer
from quantum_energy.dwave_solver import DWaveAnnealing, d_wave
from quantum_energy.moo import filter_and_sort_pf, pf_adaptive_avg
from quantum_energy.problem.qubo import RedispatchQUBO
from quantum_energy.problem.redispatch_stats import RedispatchStats
from quantum_energy.solver.subsolvers import BetaExpansion
from quantum_energy.util import sol_to_x
from tqdm import tqdm
from dimod import BinaryQuadraticModel
import http
from dwave.cloud.api.exceptions import RequestTimeout

qubos = {
    "power": (1, 0, 0, 0),
    "overload": (0, 1, 0, 0),
    "cost": (0, 0, 1, 0),
    "switch": (0, 0, 0, 1),
}


def tabu_solver(Q_ql):
    sampler = dws.TabuSampler()
    samp_set = sampler.sample(BinaryQuadraticModel(Q_ql.m, "BINARY"))
    return samp_set.first.sample


def dwave_solver(Q_ql, num_reads=100):
    dwave = DWaveAnnealing(
        token=os.environ["API_DWAVE"],
        region=None,
    )
    sol = None
    print("Sending QUBO...", end='')
    while sol is None:
        try:
            sol = d_wave(
                Q_ql, dwave, sample_set_flag=True, num_reads=num_reads
            ).first.sample
        except RequestTimeout:
            print("Request Timeout! Retrying...", end="")
            pass
        except http.client.RemoteDisconnected:
            print("Remote Disconnected! Retrying...", end="")
            pass
        finally:
            dwave.d_wave.close()
    return sol
    # sol = d_wave(
    #     Q_ql, dwave, sample_set_flag=True, num_reads=num_reads
    # ).first.sample
    # dwave.d_wave.close()
    # return sol


def a_exp_solver(
    Rd, Q, sub_size=150, max_runs=10000, solver=tabu_solver, n_sols=1
):
    x_init = sol_to_x(np.zeros(Rd.T * Rd.n_resources), Rd.k)
    x_init = {i: x_init[i] for i in range(len(x_init))}
    subsolver = BetaExpansion(
        Rd.T,
        Rd.n_resources,
        Rd.k,
        sub_size,
        solver,
    ).subsolver_ql
    sols = []
    for _ in tqdm(range(n_sols)):
        x, es = run_decomposer(
            Q,
            subsolver,
            x_init.copy(),
            max_runs=max_runs,
            max_stale=5,
        )
        sols.append(dict_to_array(x))
    return sols


def find_pfs(nrun=5, pf_size=25):
    with open("./data/models/T2.pkl", "rb") as f:
        Rd = pickle.load(f)
    Rd_qubo = RedispatchQUBO(Rd)
    qubos_names = list(qubos.keys())

    def solver(Q):
        return a_exp_solver(Rd, Q, n_sols=nrun)

    # Normalize
    ## Construct QUBOs
    norm_qubos = [
        ql.qubo(Rd_qubo.get_qubo_obj_normed_nt(*qubos[n]))
        for n in qubos_names
    ]
    c = 10000
    ## Get Pareto Front
    norm_pf = pf_adaptive_avg(
        norm_qubos[1:], [norm_qubos[0]], pf_size, solver, constr_scale=c
    )
    for we in norm_pf:
        we.weights = np.r_[c, we.weights]
    res_folder = "./new_results/"
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    with open("./new_results/norm_pf.pkl", "wb") as f:
        pickle.dump(norm_pf, f)
    # None
    none_qubos = [
        ql.qubo(Rd_qubo.get_qubo_obj(*qubos[n])) for n in qubos_names
    ]
    c = 1
    none_pf = pf_adaptive_avg(
        none_qubos[1:], [none_qubos[0]], pf_size, solver, constr_scale=c
    )
    for we in none_pf:
        we.weights = np.r_[c, we.weights]
    with open("./new_results/none_pf.pkl", "wb") as f:
        pickle.dump(none_pf, f)


def get_sds(Rd_qubo):
    sds = []
    Rd = Rd_qubo.R
    for n in list(qubos.keys()):
        Q = ql.qubo(Rd_qubo.get_qubo_obj(*qubos[n]))
        Qss = qeb.to_dense_symmetric(Q.m)
        var = Qss.variance_one_hot(Rd.T * Rd.n_resources, Rd.k)
        sds.append(np.sqrt(var))
    return np.array(sds)


def dry_run(prob):
    with open("./data/models/T2.pkl", "rb") as f:
        tmp_Rd = pickle.load(f)
    tmp_Rd_s = RedispatchStats(tmp_Rd)
    res_folder = "./new_results/"
    with open(f"{res_folder}/norm_pf.pkl", "rb") as f:
        pf = pickle.load(f)
    ws = filter_and_sort_pf(tmp_Rd_s, pf)[0].weights
    with open(f"./data/models/{prob}.pkl", "rb") as f:
        tmp_Rd = pickle.load(f)
    tmp_Rd_qubo = RedispatchQUBO(tmp_Rd)
    tmp_Rd_s = RedispatchStats(tmp_Rd)
    Q_norm = tmp_Rd_qubo.get_qubo_obj_normed_nt(*ws)
    x = a_exp_solver(tmp_Rd, Q_norm)[0]
    print(tmp_Rd_s.x_sol_to_stats(x))


def quantum(solver=tabu_solver, name="tabu", size="T8"):
    # Get Weights
    with open("./data/models/T2.pkl", "rb") as f:
        tmp_Rd = pickle.load(f)
    tmp_Rd_s = RedispatchStats(tmp_Rd)
    res_folder = "./new_results/"
    with open(f"{res_folder}/norm_pf.pkl", "rb") as f:
        pf = pickle.load(f)
    ws = filter_and_sort_pf(tmp_Rd_s, pf)[0].weights
    with open(f"./data/models/{size}.pkl", "rb") as f:
        tmp_Rd = pickle.load(f)
    tmp_Rd_qubo = RedispatchQUBO(tmp_Rd)
    Q = tmp_Rd_qubo.get_qubo_obj_normed_nt(*ws)
    for i in tqdm(range(3)):
        x = a_exp_solver(tmp_Rd, Q, solver=solver, max_runs=20)[0]
        with open(f"{res_folder}/compare_{name}_xs_{i}.pkl", "wb") as f:
            pickle.dump(x, f)


def apply_to_larger(
    runs, probs=["T2", "T4", "T6", "T8"]
):
    print("getting weights")
    res_folder = "./new_results/"
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    # Get best Weigths
    with open("./data/models/T2.pkl", "rb") as f:
        tmp_Rd = pickle.load(f)
    tmp_Rd_s = RedispatchStats(tmp_Rd)
    ## None
    with open(f"{res_folder}/none_pf.pkl", "rb") as f:
        pf = pickle.load(f)
    none_ws = filter_and_sort_pf(tmp_Rd_s, pf)[0].weights
    ## Normalize
    with open(f"{res_folder}/norm_pf.pkl", "rb") as f:
        pf = pickle.load(f)
    norm_ws = filter_and_sort_pf(tmp_Rd_s, pf)[0].weights
    norm_xs = {}
    none_xs = {}
    for prob in tqdm(probs):
        with open(f"./data/models/{prob}.pkl", "rb") as f:
            tmp_Rd = pickle.load(f)
        tmp_Rd_qubo = RedispatchQUBO(tmp_Rd)
        # Normalize
        Q_norm = tmp_Rd_qubo.get_qubo_obj_normed_nt(*norm_ws)
        norm_xs[prob] = a_exp_solver(tmp_Rd, Q_norm, n_sols=runs)
        # None
        Q_none = tmp_Rd_qubo.get_qubo_obj(*none_ws)
        none_xs[prob] = a_exp_solver(tmp_Rd, Q_none, n_sols=runs)
    with open(f"{res_folder}/norm_xs.pkl", "wb") as f:
        pickle.dump(norm_xs, f)
    with open(f"{res_folder}/none_xs.pkl", "wb") as f:
        pickle.dump(none_xs, f)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run scaling experiments."
    )
    parser.add_argument("-e", "--ex", type=str, help="Experiment Name")
    parser.add_argument(
        "-n", "--nruns", type=int, help="Number of runs"
    )
    parser.add_argument("-p", "--prob", type=int, help="Prob Size (T)")
    parser.add_argument("-s", "--solver", type=str, help="tabu or quan")
    args = parser.parse_args()
    if args.ex == "weights":
        find_pfs(args.nruns)
    elif args.ex == "apply":
        apply_to_larger(args.nruns)
    elif args.ex == "dry":
        for _ in range(10):
            dry_run(f"T{args.prob}")
    elif args.ex == "compare":
        if args.solver == "tabu":
            quantum(solver=tabu_solver, name="tabu")
        elif args.solver == "quan":
            quantum(solver=dwave_solver, name="quan")
