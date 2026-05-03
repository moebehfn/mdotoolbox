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

"""src/mdotoolbox/benchmarking/power_converter.py"""

import numpy as np

from ..core import Constraint, Function, Problem


def power_converter_problem():
    """Power Converter Problem"""
    var_sys = ["x1", "x2", "x3", "x4", "x5", "x6", "y3", "y5", "y6", "y7", "y8", "y2"]

    def system_objective(*args):
        x1, x2, x3, x4, x5, x6, y3, y5, y6, y7, y8, y2 = args
        core_weight = 0.0078 * (2 * x1 + x6) * x1**2
        winding_weight = 0.0089 * x2 * x3
        return core_weight + winding_weight

    sys_func = Function(func=system_objective, x=var_sys, name="weight")
    sys_constraints = [
        Constraint(
            func=Function(func=lambda *args: -args[6], x=var_sys, name="fill"),
            ctype="ge",
            value=0.0,
        ),
        Constraint(
            func=Function(func=lambda *args: -args[7], x=var_sys, name="ripple"),
            ctype="ge",
            value=0.0,
        ),
    ]
    bounds_sys = np.array([
        [0.001, 0.05],
        [1.0, 100.0],
        [1e-08, 1e-05],
        [1e-15, 0.001],
        [1e-05, 0.001],
        [0.001, 0.05],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [1.0, 100.0],
    ])
    system_problem = Problem(
        objective=sys_func,
        constraints=sys_constraints,
        ubounds=bounds_sys[:, 1],
        lbounds=bounds_sys[:, 0],
        name="power_converter_system",
    )
    var_elec = ["x1", "x2", "x3", "y2"]

    def electrical_analysis(x1, x2, x3, y2):
        y3 = x2 * x3
        y5 = x1 * y2
        return np.array([y3, y5])

    elec_func = Function(
        func=lambda x1, x2, x3, y2: electrical_analysis(x1, x2, x3, y2)[0],
        x=var_elec,
        name="y3",
    )
    elec_constraint = Constraint(
        func=Function(func=lambda *args: 1.0, x=var_elec, name="elec_dummy"),
        ctype="ge",
        value=0.0,
    )
    bounds_elec = bounds_sys[[0, 1, 2, 11], :]
    electrical_problem = Problem(
        objective=elec_func,
        constraints=[elec_constraint],
        ubounds=bounds_elec[:, 1],
        lbounds=bounds_elec[:, 0],
        name="electrical_subsystem",
    )
    var_loss = ["x4", "x5", "x6", "y3", "y5"]

    def loss_analysis(x4, x5, x6, y3, y5):
        y2 = y3 / y5 if abs(y5) > 1e-10 else 1.0
        y6 = x4 * x5
        y7 = x6 * y3
        y8 = y5 * x4
        return np.array([y2, y6, y7, y8])

    loss_func = Function(
        func=lambda x4, x5, x6, y3, y5: loss_analysis(x4, x5, x6, y3, y5)[0],
        x=var_loss,
        name="y2",
    )
    loss_constraint = Constraint(
        func=Function(func=lambda *args: 1.0, x=var_loss, name="loss_dummy"),
        ctype="ge",
        value=0.0,
    )
    bounds_loss = bounds_sys[[3, 4, 5, 6, 7], :]
    loss_problem = Problem(
        objective=loss_func,
        constraints=[loss_constraint],
        ubounds=bounds_loss[:, 1],
        lbounds=bounds_loss[:, 0],
        name="loss_subsystem",
    )
    subsystems = [
        {
            "problem": electrical_problem,
            "z_idxs": np.array([0, 1]),
            "x_idxs": np.array([0, 1]),
            "y_idxs": np.array([0, 1]),
            "y_coupled_idxs": [np.array([4])],
        },
        {
            "problem": loss_problem,
            "z_idxs": np.array([0, 1, 2, 3]),
            "x_idxs": np.array([2]),
            "y_idxs": np.array([2, 3, 4]),
            "y_coupled_idxs": [np.array([0, 1])],
        },
    ]
    return {
        "system_problem": system_problem,
        "subsystems": subsystems,
        "n_shared": 6,
        "n_local": 0,
        "n_coupling": 6,
    }
