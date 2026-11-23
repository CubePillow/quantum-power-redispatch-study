"""Smoke test for qaoa_xmixer_standalone.py on the hand-built example.

Run:  python test_standalone.py   (or: pytest test_standalone.py)
No repository code needed. Expected values come from docs/build-instance.md.
"""
import numpy as np

import qaoa_xmixer_standalone as q


def _setup():
    tab = q.build_tables(q.DEFAULT_INSTANCE)
    return tab, tab["feasible"]


def test_counts():
    tab, f = _setup()
    assert tab["N"] == 12
    assert int(f.sum()) == 49          # 7 * 7 ramp-feasible schedules


def test_weighted_optima():
    tab, f = _setup()
    expected = {
        (1, 0, 0, 100): ([(2, 0), (2, 1)], 70.0),
        (1, 0, 100, 100): ([(1, 1), (1, 2)], 110.0),
        (1, 10, 100, 100): ([(1, 1), (1, 2)], 210.0),   # 110 + 10 * W (W = 10)
    }
    for w, (schedule, value) in expected.items():
        _, objective = q.build_cost(tab, w)
        s = int(np.argmin(np.where(f, objective, np.inf)))
        assert q.describe(s, tab) == schedule
        assert abs(objective[s] - value) < 1e-9


def test_uniform_start_feasibility():
    tab, f = _setup()
    cost, objective = q.build_cost(tab)
    psi = q.qaoa_state(np.array([]), cost, tab["N"])      # p = 0
    assert abs(q.metrics(psi, tab, objective)["feas_prob"] - 49 / 4096) < 1e-12


def test_mixer_keeps_norm():
    tab, _ = _setup()
    psi = np.zeros(2 ** tab["N"], dtype=complex)
    psi[0] = 1.0
    q.apply_rx_all(psi, 0.3, tab["N"])
    assert abs(np.sum(np.abs(psi) ** 2) - 1.0) < 1e-12
    assert abs(abs(psi[0]) ** 2 - np.cos(0.3) ** (2 * tab["N"])) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
