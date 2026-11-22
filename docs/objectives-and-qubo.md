# Redispatch Objectives, Constraints, and QUBO Functions

This note connects the Optimization problem in the one-page document to the small example and the existing QUBO code.

## Four objective terms

These are the C, W, O, and D terms in the one-page document.

| Term | Simple meaning | Calculation in this example |
| --- | --- | --- |
| C | Production cost | Add `A output + 3 * B output` over both times |
| W | Switching cost | Add the absolute output changes of A and B |
| O | Number of overloads | Count time steps with line flow greater than or equal to 19 MW |
| D | Demand mismatch | Add `(total output - demand) ** 2` over both times |

We minimize `mu_C * C + mu_W * W + mu_O * O + lambda_D * D`.

C, W, and O are competing objectives. D is a soft demand-tracking term: mismatch has a cost, but it is allowed. Line overload is also allowed at a cost; it is not one of the hard constraints.

## Two required constraints

1. **One-hot:** Each generator selects exactly one state at each time. In binary form, each block of three bits must contain exactly one 1.
2. **Ramp:** A generator can change by at most one state between times. For example, 0 to 1 is allowed; 0 to 2 is not.

There are `3 ** 4 = 81` possible schedules. For each generator, 7 of its 9 two-time schedules satisfy ramping. Therefore, `7 * 7 = 49` schedules satisfy both constraints.

## Existing QUBO functions for the next step

The repository already has these functions in `package/quantum_energy/problem/qubo.py`. They build QUBO matrices; they do not directly return the physical objective values.

| Part | Existing function |
| --- | --- |
| C | `gen_qubo_cost()` |
| W | `gen_qubo_switch_cost()` |
| O | `gen_qubos_overload_unbalanced()` |
| D | `gen_qubo_power_eq()` |
| One-hot | `gen_qubo_one_hot()` |
| Ramp | `gen_qubo_adjacent_lvls()` |

`RedispatchModel` in `problem/modeller.py` prepares data from a simulated grid. `RedispatchQUBO` in `problem/qubo.py` builds and combines the matrices. Our hand-built example does not need the grid-data preparation step.


