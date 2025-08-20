"""
Code from Patrick Völker
"""

import math
from dataclasses import dataclass

import numpy as np
import pandapower as pp  # type: ignore [import-untyped]
import simbench as sb
from tqdm import tqdm


@dataclass
class GridResults:
    net: pp.auxiliary.pandapowerNet
    T: int
    T_gap: float
    line_loads: np.ndarray
    reso_group: list[str]
    reso_prods: np.ndarray
    sgen_group: list[str]
    sgen_prods: np.ndarray
    load_group: list[str]
    load_prods: np.ndarray


def apply_absolute_values(net, abs_vals_dict, case_or_time_step):
    for elm_param in abs_vals_dict.keys():
        if abs_vals_dict[elm_param].shape[1]:
            # print(elm_param[0])
            elm = elm_param[0]
            param = elm_param[1]
            # print(param)
            net[elm].loc[:, param] = abs_vals_dict[elm_param].loc[
                case_or_time_step
            ]


def add_line_capacity_to_net(net):
    # Formular for Calculating max mw -> voltage is from bus
    difference_in_voltage = []
    max_mw_per_line = []
    # Check lines same voltage lvl
    for _, row in net.line.iterrows():
        from_vn_kv = net.bus.iloc[row.from_bus].vn_kv
        to_vn_kv = net.bus.iloc[row.to_bus].vn_kv
        if from_vn_kv != to_vn_kv:
            difference_in_voltage.append([row.from_bus, row.to_bus])
            print("Difference in voltage Level: Please check")
        # calulate Max possile MWs per line
        max_mw_per_line.append(
            net.bus.iloc[row.from_bus].vn_kv
            * row["max_i_ka"]
            * math.sqrt(3)
        )
    net.line["max_mw_per_line"] = max_mw_per_line
    return net


def get_time_gap(T=20, t_start=0, n_hours=48, T_unit=0.25):
    return int((n_hours // T) / T_unit)


def run_and_record_profiles(
    net,
    time_start,
    n_hours=48,
    t_gap=0.25,
    reso_group=["gen"],
    reso_signs=[1],
    sgen_group=["sgen"],
    sgen_signs=[1],
    load_group=["ext_grid", "load"],
    load_signs=[1, -1],
):
    n_time_steps = int(48 / t_gap)
    profiles = sb.get_absolute_values(
        net, profiles_instead_of_study_cases=True
    )
    net.trafo.tap_pos = 1
    reso_prods = []
    sgen_prods = []
    load_prods = []
    line_loads = np.zeros((len(net.line), n_time_steps))
    for t in tqdm(list(range(time_start, n_time_steps))):
        apply_absolute_values(net, profiles, t)
        pp.runpp(net, numba=True)
        if t < time_start:
            continue
        resos = [getattr(net, "res_" + name) for name in reso_group]
        sgens = [getattr(net, "res_" + name) for name in sgen_group]
        loads = [getattr(net, "res_" + name) for name in load_group]
        reso_prods.append(
            np.hstack(
                [
                    resos[i]["p_mw"] * reso_signs[i]
                    for i in range(len(resos))
                ]
            )
        )
        sgen_prods.append(
            np.hstack(
                [
                    sgens[i]["p_mw"] * sgen_signs[i]
                    for i in range(len(sgens))
                ]
            )
        )
        load_prods.append(
            np.hstack(
                [
                    loads[i]["p_mw"] * load_signs[i]
                    for i in range(len(loads))
                ]
            )
        )
        line_loads[:, t] = np.maximum.reduce(
            [net.res_line["p_from_mw"], net.res_line["p_to_mw"]]
        )
    return GridResults(
        net,
        n_time_steps,
        t_gap,
        line_loads,
        reso_group,
        np.vstack(reso_prods).T,
        sgen_group,
        np.vstack(sgen_prods).T,
        load_group,
        np.vstack(load_prods).T,
    )


def get_even_meaned(res, T):
    return np.column_stack(
        [s.mean(1) for s in np.array_split(res, T, axis=1)]
    )


def get_even_meaned_time_points(grid_res: GridResults, T):
    T1 = grid_res.T
    return GridResults(
        grid_res.net,
        T,
        T1 / T,
        get_even_meaned(grid_res.line_loads, T),
        grid_res.reso_group,
        get_even_meaned(grid_res.reso_prods, T),
        grid_res.sgen_group,
        get_even_meaned(grid_res.sgen_prods, T),
        grid_res.load_group,
        get_even_meaned(grid_res.load_prods, T),
    )


def get_even_spaced_time_points(grid_res: GridResults, T, T_max=-1):
    T_max = max(T, T_max)
    ts_f = np.linspace(0, grid_res.T - 1, T_max)
    gap = ts_f[1] - ts_f[0]
    ts = ts_f.astype(int)[0:T]
    return GridResults(
        grid_res.net,
        T,
        gap * grid_res.T_gap,
        grid_res.line_loads[:, ts],
        grid_res.reso_group,
        grid_res.reso_prods[:, ts],
        grid_res.sgen_group,
        grid_res.sgen_prods[:, ts],
        grid_res.load_group,
        grid_res.load_prods[:, ts],
    )


def get_even_first_T_time_points(
    grid_res: GridResults, T, n_hours=2
):
    gap = int(n_hours / grid_res.T_gap)
    ts = [i * gap for i in range(T)]
    return GridResults(
        grid_res.net,
        T,
        gap * grid_res.T_gap,
        grid_res.line_loads[:, ts],
        grid_res.reso_group,
        grid_res.reso_prods[:, ts],
        grid_res.sgen_group,
        grid_res.sgen_prods[:, ts],
        grid_res.load_group,
        grid_res.load_prods[:, ts],
    )


def run_profiles(
    net,
    time_start=0,
    time_steps=20,
    n_hours=48,
    reso_group=["gen"],
    reso_signs=[1],
    sgen_group=["sgen"],
    sgen_signs=[1],
    load_group=["ext_grid", "load"],
    load_signs=[1, -1],
):
    T_unit = 0.25
    time_stride = get_time_gap(time_steps, time_start, n_hours, T_unit)
    profiles = sb.get_absolute_values(
        net, profiles_instead_of_study_cases=True
    )
    net.trafo.tap_pos = 1
    # reso_prods = [np.zeros((len(n),len(time_steps))) for n in resos]
    # load_prods = [np.zeros((len(n),len(time_steps))) for n in loads]
    reso_prods = []
    load_prods = []
    line_loads = np.zeros((len(net.line), time_steps))
    all_reso_prods = []
    all_load_prods = []
    all_line_loads = np.zeros((len(net.line), time_steps * time_stride))
    for t in range(time_start, time_start + time_steps):
        for n in range(time_stride):
            apply_absolute_values(net, profiles, t * time_stride + n)
            pp.runpp(net)
            resos = [getattr(net, "res_" + name) for name in reso_group]
            loads = [getattr(net, "res_" + name) for name in load_group]
            all_reso_prods.append(
                np.hstack(
                    [
                        resos[i]["p_mw"] * reso_signs[i]
                        for i in range(len(resos))
                    ]
                )
            )
            all_load_prods.append(
                np.hstack(
                    [
                        loads[i]["p_mw"] * load_signs[i]
                        for i in range(len(loads))
                    ]
                )
            )
            all_line_loads[:, t * time_stride + n] = np.maximum.reduce(
                [net.res_line["p_from_mw"], net.res_line["p_to_mw"]]
            )
        reso_prods.append(
            np.hstack(
                [
                    resos[i]["p_mw"] * reso_signs[i]
                    for i in range(len(resos))
                ]
            )
        )
        load_prods.append(
            np.hstack(
                [
                    loads[i]["p_mw"] * load_signs[i]
                    for i in range(len(loads))
                ]
            )
        )
        # for ni in range(len(resos)):
        #     reso_prods[ni][:, t] = resos[ni]['p_mw'] * reso_signs[ni]
        # for ni in range(len(loads)):
        #     load_prods[ni][:, t] = loads[ni]['p_mw'] * load_signs[ni]
        line_loads[:, t] = np.maximum.reduce(
            [net.res_line["p_from_mw"], net.res_line["p_to_mw"]]
        )
    return GridResults(
        net,
        time_steps,
        0.25 * time_stride,
        reso_group,
        np.vstack(reso_prods).T,
        load_group,
        np.vstack(load_prods).T,
        line_loads,
        np.vstack(all_reso_prods).T,
        np.vstack(all_load_prods).T,
        all_line_loads,
    )


#### Simbecn Network###
# grid_code = "1-EHV-mixed--0-sw"
# net = sb.get_simbench_net(grid_code)
# add_line_capacity_to_net(net)
