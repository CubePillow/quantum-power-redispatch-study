import os
import time

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dimod
import hybrid as dwh
import iaisevo
import matplotlib.pyplot as plt
import numpy as np
import qubolite as ql
from quantum_energy.problem.qubo import (
    RedispatchQUBO,
    get_hard_constraint_penalty,
)
from quantum_energy.util import str_to_arr

import simbench as sb

# * PF finder

# * Plot Power

def plot_power_network(net):
    coords = net.bus[["name"]].join(net.bus_geodata[["x", "y"]])
    coords["type"] = "bus"
    for row in net.gen.iloc:
        coords["type"][row["bus"]] = "gen"
    for row in net.sgen.iloc:
        coords["type"][row["bus"]] = "sgen"
    for row in net.load.iloc:
        coords["type"][row["bus"]] = "load"
    for row in net.ext_grid.iloc:
        coords["type"][row["bus"]] = "ext_grid"

    line_coords = []
    for i in range(len(net.line)):
        line_coords.append(
            coords["x"][net.line[["from_bus", "to_bus"]].loc[i]]
        )
        line_coords.append(
            coords["y"][net.line[["from_bus", "to_bus"]].loc[i]]
        )
    # Plot borders and zoom into Germany
    ax = plt.axes(projection=ccrs.Mercator())
    # ax.set_extent([5, 16, 46.5, 55])
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.plot(
        *line_coords,
        transform=ccrs.PlateCarree(),
        color="#367FD0",
        linewidth=0.75,
    )
    ax.scatter(
        coords["x"],
        coords["y"],
        s=2,
        color="#CA3F3A",
        marker="o",
        transform=ccrs.PlateCarree(),
    )
    plt.savefig("./plots/power_network.pdf", bbox_inches='tight')

# * Main

if __name__ == "__main__":
    grid_code = "1-EHV-mixed--0-sw"
    net = sb.get_simbench_net(grid_code)
    plot_power_network(net)


def add_hard_cons(Rd_qubo: RedispatchQUBO, Q):
    P = get_hard_constraint_penalty(Q)
    return Q + P * Rd_qubo.qubo_one_hot + P * Rd_qubo.qubo_adjacent


def run_digital_annealer_single(Q: ql.qubo):
    n_shots = 5
    t = 5
    backend = iaisevo.Backend(
        os.environ["API_DIGITAL_ANNEALER"]
    )
    J = backend.run(Q.m, shots=n_shots, time=t)
    time.sleep(t)
    while not backend.all_completed(J):
        time.sleep(0.5)
    da_res = backend.get_counts(J)
    for j in J:
        j.delete()
    return da_res
    x = str_to_arr(max(da_res, key=da_res.get))
    return x


class DigitalAnnealerSubproblemSampler(dwh.Runnable):
    def __init__(self, n_shots=5, t=5, **runopts):
        super(DigitalAnnealerSubproblemSampler, self).__init__(
            **runopts
        )
        self.n_shots = n_shots
        self.t = t

    def __repr__(self):
        return (
            "{self}(n_shots={self.n_shots!r}, " "t={self.t!r})"
        ).format(self=self)

    def next(self, state, **runopts):
        ql_Q = ql.qubo.from_dict(
            state.subproblem.to_qubo()[0], relabel=True
        )[0]
        da_res = run_digital_annealer_single(ql_Q)
        xs_str = list(da_res.keys())
        xs = np.array([str_to_arr(x_str) for x_str in xs_str])
        xs_lab = [dict(zip(state.subproblem.variables, x)) for x in xs]
        energies = np.array([ql_Q(x) for x in xs])
        occurrences = np.array([da_res[x_str] for x_str in xs_str])
        sampleset = dimod.SampleSet.from_samples_bqm(
            xs_lab, state.subproblem, num_occurrences=occurrences
        )
        # sampleset = dimod.SampleSet.from_samples(
        #     xs_lab, "BINARY", energies, num_occurrences=occurrences
        # )
        ql_Q = None
        del ql_Q
        return state.updated(subsamples=sampleset)
