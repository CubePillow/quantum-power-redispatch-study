import pickle
import os

import simbench as sb
import numpy as np

from quantum_energy.problem.modeller import RedispatchModel
from quantum_energy.problem.simulator import (
    add_line_capacity_to_net,
    run_and_record_profiles,
    get_even_first_T_time_points
)

def create_model_set(sim_res, k, S, folder, n_run=10):
    for i in range(n_run):
        if not os.path.exists(f"{folder}"):
            os.makedirs(f"{folder}")
        Rd = RedispatchModel(sim_res, k, seed=i)
        Rd.set_S(S)
        with open(f"{folder}/{i}.pkl", "wb") as f:
            pickle.dump(Rd, f)

def create_model(sim_res, k, S, filename):
    Rd = RedispatchModel(sim_res, k, seed=0)
    Rd.set_S(S)
    with open(f"{filename}.pkl", "wb") as f:
        pickle.dump(Rd, f)
            

def create_model_subsets(sim_res, k, S, folder, n_run=10):
    for i in range(n_run):
        if not os.path.exists(f"{folder}"):
            os.makedirs(f"{folder}")
        Rd = create_model_subset(sim_res, S, k)
        with open(f"{folder}/{i}.pkl", "wb") as f:
            pickle.dump(Rd, f)


def get_subset_ids(net, S, n_gen=4, n_sgen=2):
    gen_ids =  np.arange(0,n_gen)
    sgen_ids = np.arange(len(net.gen), len(net.gen) + n_sgen)
    other_ids = np.arange(len(net.gen)+len(net.sgen), S.shape[1])
    return (gen_ids, sgen_ids, other_ids)


def create_model_subset(sim_res, S, k=3, n_gen=4, n_sgen=2):
    gen_ids, sgen_ids, other_ids = get_subset_ids(
        sim_res.net, S, n_gen, n_sgen)
    subset_ids = np.hstack([gen_ids, sgen_ids, other_ids])
    # net subset
    sim_res.net.gen = sim_res.net.gen[0:n_gen]
    sim_res.net.sgen = sim_res.net.sgen[0:n_sgen]
    sim_res.reso_prods = sim_res.reso_prods[0:n_gen, :]
    sim_res.sgen_prods = sim_res.reso_prods[0:n_sgen, :]
    S = S[:,subset_ids]
    Rd = RedispatchModel(sim_res, k)
    Rd.set_S(S)
    return Rd

            
def create_models():
    folder = "data/models/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    grid_code = "1-EHV-mixed--0-sw"
    net = sb.get_simbench_net(grid_code)
    add_line_capacity_to_net(net)
    all_res = run_and_record_profiles(net, 0)
    # Create Redispatch problem from simulation results
    S = np.load("data/sensitivity/S.npy").astype(
        "float32"
    )
    ts = [2, 4, 6, 8]
    cur_folder = "data/models/"
    if not os.path.exists(cur_folder):
        os.makedirs(cur_folder)
    for t in ts:
        sim_res = get_even_first_T_time_points(all_res, t, 2)
        create_model(sim_res, 5, S, f"{cur_folder}/T{t}")

if __name__ == "__main__":
    create_models()
