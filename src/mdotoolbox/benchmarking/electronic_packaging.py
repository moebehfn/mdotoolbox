# Copyright 2025 Mohamed Ali Belhafnaoui
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""src/mdotoolbox/benchmarking/electronic_packaging.py"""

import numpy as np

from ..core import Constraint, Function, Problem


def electronic_packaging_problem():
    """Electronic Packaging Problem"""
    V = 12.0
    T0 = 20.0

    var_sys = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "u2", "u3", "u11", "u12"]

    def system_objective(*args):
        x1, x2, x3, x4, x5, x6, x7, x8, u2, u3, u11, u12 = args
        u13 = x1 * x2 * 2 * (x3 + x4)
        u8 = V / (u2 + u3)
        u6 = u8**2 * u2
        u7 = u8**2 * u3
        u10 = u6 + u7
        u1 = u10 / u13
        return u1

    sys_func = Function(func=system_objective, x=var_sys, name="watt_density")

    def branch_current_equality(*args):
        x1, x2, x3, x4, x5, x6, x7, x8, u2, u3, u11, u12 = args
        u4 = V / u2
        u5 = V / u3
        return u4 - u5

    sys_constraints = [
        Constraint(
            func=Function(func=branch_current_equality, x=var_sys, name="current_eq"),
            ctype="eq",
            value=0.0,
        ),
        Constraint(
            func=Function(func=lambda *args: 85.0 - args[10], x=var_sys, name="temp1"),
            ctype="ge",
            value=0.0,
        ),
        Constraint(
            func=Function(func=lambda *args: 85.0 - args[11], x=var_sys, name="temp2"),
            ctype="ge",
            value=0.0,
        ),
    ]
    bounds_sys = np.array([
        [0.05, 0.15],
        [0.05, 0.15],
        [0.01, 0.1],
        [0.005, 0.05],
        [10.0, 1000.0],
        [0.004, 0.009],
        [10.0, 1000.0],
        [0.004, 0.009],
        [1.0, 200.0],
        [1.0, 200.0],
        [20.0, 100.0],
        [20.0, 100.0],
    ])
    system_problem = Problem(
        objective=sys_func,
        constraints=sys_constraints,
        ubounds=bounds_sys[:, 1],
        lbounds=bounds_sys[:, 0],
        name="electronic_packaging_system",
    )
    var_thermal = ["x1", "x2", "x3", "x4", "u2", "u3"]

    def thermal_analysis(x1, x2, x3, x4, u2, u3):
        u8 = V / (u2 + u3)
        u6 = u8**2 * u2
        u7 = u8**2 * u3
        A_sink = x1 * x2
        A_fins = 2 * (x3 + x4) * x1
        A_total = A_sink + A_fins
        h_conv = 10.0
        u11 = T0 + u6 / (h_conv * A_total)
        u12 = T0 + u7 / (h_conv * A_total)
        return np.array([u11, u12])

    thermal_func = Function(
        func=lambda x1, x2, x3, x4, u2, u3: thermal_analysis(x1, x2, x3, x4, u2, u3)[0],
        x=var_thermal,
        name="u11",
    )
    thermal_constraints = [
        Constraint(
            func=Function(
                func=lambda x1, x2, x3, x4, u2, u3: (
                    85.0 - thermal_analysis(x1, x2, x3, x4, u2, u3)[0]
                ),
                x=var_thermal,
                name="thermal_c1",
            ),
            ctype="ge",
            value=0.0,
        ),
        Constraint(
            func=Function(
                func=lambda x1, x2, x3, x4, u2, u3: (
                    85.0 - thermal_analysis(x1, x2, x3, x4, u2, u3)[1]
                ),
                x=var_thermal,
                name="thermal_c2",
            ),
            ctype="ge",
            value=0.0,
        ),
    ]
    bounds_thermal = bounds_sys[[0, 1, 2, 3, 8, 9], :]
    thermal_problem = Problem(
        objective=thermal_func,
        constraints=thermal_constraints,
        ubounds=bounds_thermal[:, 1],
        lbounds=bounds_thermal[:, 0],
        name="thermal_subsystem",
    )
    var_electrical = ["x5", "x6", "x7", "x8", "u11", "u12"]

    def electrical_analysis(x5, x6, x7, x8, u11, u12):
        u2 = x5 * (1 + x6 * (u11 - T0))
        u3 = x7 * (1 + x8 * (u12 - T0))
        return np.array([u2, u3])

    electrical_func = Function(
        func=lambda x5, x6, x7, x8, u11, u12: electrical_analysis(
            x5, x6, x7, x8, u11, u12
        )[0],
        x=var_electrical,
        name="u2",
    )
    electrical_constraints = [
        Constraint(
            func=Function(
                func=lambda x5, x6, x7, x8, u11, u12: 85.0 - u11,
                x=var_electrical,
                name="elec_c1",
            ),
            ctype="ge",
            value=0.0,
        ),
        Constraint(
            func=Function(
                func=lambda x5, x6, x7, x8, u11, u12: 85.0 - u12,
                x=var_electrical,
                name="elec_c2",
            ),
            ctype="ge",
            value=0.0,
        ),
    ]
    bounds_electrical = bounds_sys[[4, 5, 6, 7, 10, 11], :]
    electrical_problem = Problem(
        objective=electrical_func,
        constraints=electrical_constraints,
        ubounds=bounds_electrical[:, 1],
        lbounds=bounds_electrical[:, 0],
        name="electrical_subsystem",
    )
    subsystems = [
        {
            "problem": thermal_problem,
            "z_idxs": np.array([0, 1, 2, 3]),
            "x_idxs": np.array([]),
            "y_idxs": np.array([2, 3]),
            "y_coupled_idxs": [np.array([0, 1])],
        },
        {
            "problem": electrical_problem,
            "z_idxs": np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            "x_idxs": np.array([]),
            "y_idxs": np.array([2, 3]),
            "y_coupled_idxs": [np.array([2, 3])],
        },
    ]
    return {
        "system_problem": system_problem,
        "subsystems": subsystems,
        "n_shared": 8,
        "n_local": 0,
        "n_coupling": 4,
    }
