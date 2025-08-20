import numpy as np

from ..util import get_effective_prod_vals

# * Handover Structure Functions


def create_bus_to_node(*nodes):
    """
    Takes in a list of pd.DataFrame of nodes (gen, sgen, loads) and
    returns a dictionary of busses and the nodes attached to each bus
    """
    bus_to_node = {}
    for ns in nodes:
        for _, n in ns.iterrows():
            n_name = n["name"]
            n_bus = n["bus"]
            cur_bus_nodes = bus_to_node.get(n_bus, [])
            cur_bus_nodes.append(n_name)
            bus_to_node[n_bus] = cur_bus_nodes
    return bus_to_node


def create_node_att_to_id(att: str, *nodes):
    atts = []
    for ns in nodes:
        for v in ns[att]:
            if v not in atts:
                atts.append(v)
    node_att_to_id = {atts[i]: i for i in range(len(atts))}
    return node_att_to_id


class ResourceHandover:
    def __init__(self, bus_to_id, *resources):
        self.bus_to_id = bus_to_id
        self.n = None
        self.name_to_id = None
        self.S_extern = None
        self._create_S_extern(resources)

    def _create_S_extern(self, nodes):
        self.name_to_id = create_node_att_to_id("name", *nodes)
        self.n = len(self.name_to_id)
        self.S_extern = np.zeros((len(self.bus_to_id), self.n))
        for ns in nodes:
            for _, n in ns.iterrows():
                n_id = self.name_to_id[n["name"]]
                b_id = self.bus_to_id[n["bus"]]
                self.S_extern[b_id, n_id] = 1

    def get_S_extern(self):
        return self.S_extern

    def get_n_resources(self):
        return self.n


# * Resource Functions
def create_power_prod_levels(min_prod, max_prod, k_levels):
    ls = []
    d = 0
    # ensure 0 is ground state
    if min_prod != 0:
        ls.append(0)
        d = 1
    gap = (max_prod - min_prod) / (k_levels - (1 + d))
    return ls + [min_prod + gap * i for i in range(k_levels - d)]


# * Model Class
class RedispatchModel:
    def __init__(self, sim_res, k=3, S=None, seed=0):
        self.sim_res = sim_res
        self.k = k
        self.net = sim_res.net
        self.T = sim_res.T
        self.T_gap = sim_res.T_gap
        # build handover structure
        self.bus_to_id = create_node_att_to_id(
            "bus",
            self.net.gen,
            self.net.sgen,
            self.net.load,
            self.net.ext_grid,
        )
        self.line_to_id = create_node_att_to_id("name", self.net.line)
        self.reso_nodes = [
            getattr(self.net, name) for name in sim_res.reso_group
        ]
        self.sgen_nodes = [
            getattr(self.net, name) for name in sim_res.sgen_group
        ]
        self.load_nodes = [
            getattr(self.net, name) for name in sim_res.load_group
        ]
        self.resos = ResourceHandover(self.bus_to_id, *self.reso_nodes)
        self.sgens = ResourceHandover(self.bus_to_id, *self.sgen_nodes)
        self.loads = ResourceHandover(self.bus_to_id, *self.load_nodes)
        ## handover variables
        self.n_resources = self.resos.n  # number of resources
        self.m_sgens = self.sgens.n  # number of static resources
        self.loads.n  # number of load resources
        self.resos.S_extern  # handover gen
        self.loads.S_extern  # handover sgen, load, ext_grid
        # Sensitivity matrices
        if S is None:
            self.S = np.ones(
                (
                    sim_res.line_loads.shape[0],
                    int(self.resos.n + self.sgens.n + self.loads.n),
                )
            )
        else:
            self.S = S
        # Resource variables
        ## Production levels of resources
        self.prod_vals = np.vstack(
            [
                create_power_prod_levels(
                    n["min_p_mw"], n["max_p_mw"], self.k
                )
                for ns in self.reso_nodes
                for _, n in ns.iterrows()
            ]
        )
        maxmin_euro_mwh_per_type = {
            "hard coal": (50, 90),
            "gas": (40, 100),
            "solar": (30, 60),
            "nuclear": (80, 120),
            "offshore wind": (70, 120),
            "onshore wind": (40, 80),
            "waste": (80, 110),
            "lignite": (40, 70),
            "oil": (90, 160),
            "imp0": (30, 100),
            "imp1": (30, 100),
        }
        np.random.seed(seed)
        self.prod_cost_multi = np.array(
            [
                np.random.uniform(*maxmin_euro_mwh_per_type[n["type"]])
                # * self.T_gap
                for ns in self.reso_nodes
                for _, n in ns.iterrows()
            ]
        )
        ## Efficiency of resources (kappa in paper)
        self.cap_vals = np.ones((self.resos.n, self.T))
        self.prod_vals_eff = get_effective_prod_vals(
            self.prod_vals, self.cap_vals
        )
        ## Cost
        self.cost_vals = (
            get_effective_prod_vals(
                self.prod_vals, np.ones_like(self.cap_vals)
            )
            * self.prod_cost_multi[:, np.newaxis]
        )
        self.switch_cost = np.array(
            [
                np.reshape(
                    [
                        (
                            np.abs(
                                self.prod_vals[j][k1]
                                - self.prod_vals[j][k2]
                            )
                            if np.abs(k1 - k2) == 1
                            else 0
                        )
                        for k1 in range(self.k)
                        for k2 in range(self.k)
                    ],
                    (self.k, self.k),
                )
                for j in range(self.resos.n)
            ]
        )
        # Internal network variables
        ## vector for max capacity per line
        self.max_load = np.array(
            [
                [
                    line["max_mw_per_line"]
                    for _, line in self.net.line.iterrows()
                ]
            ]
            * self.T
        ).T
        ## sgen
        power_sell_price = 95
        self.sgen_prods = sim_res.sgen_prods
        self.sgen_prices = (
            sim_res.sgen_prods * power_sell_price * self.T_gap
        )
        ## target load over time
        self.prod_target = sim_res.reso_prods.sum(
            axis=0
        ) + sim_res.sgen_prods.sum(axis=0)
        self.prod_resos = sim_res.reso_prods
        self.prod_static = sim_res.load_prods
        self.set_S(self.S)

    def set_S(self, S):
        self.S = S
        self.S_resos = self.S[:, 0 : self.resos.n]
        self.S_sgens = self.S[
            :, self.resos.n : self.resos.n + self.sgens.n
        ]
        self.S_loads = self.S[
            :,
            self.resos.n + self.sgens.n : self.loads.n
            + self.sgens.n
            + self.resos.n,
        ]
        self.M = np.array(
            self.max_load - self.S_loads @ self.prod_static
        ).astype("float32")
