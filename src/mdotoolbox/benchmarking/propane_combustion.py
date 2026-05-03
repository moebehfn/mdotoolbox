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

"""src/mdotoolbox/benchmarking/propane_combustion.py"""

import numpy as np
from numba import njit

from ..core import Constraint, Function, Problem


def propane_combustion_problem():
    """Combustion of Propane Problem (NASA MDO Test Suite)"""

    @njit(cache=True)
    def discipline1_compute(z1, z2, z3, z4, y2, y3):
        return (
            2.0 * z1 + z2 + np.abs(z3) + z4 + np.sqrt(np.abs(z2 * z3)) - 10.0 + y2 * y3
        )

    @njit(cache=True)
    def discipline2_compute(z1, z2, z3, y1, y3):
        return (
            z1
            + 2.0 * z2
            + z3**2
            + np.sqrt(np.abs(y1 * z1))
            - np.sqrt(np.abs(40.0 * z1 / (y3 + 1e-10)))
            - 5.0
        )

    @njit(cache=True)
    def discipline3_compute(z1, z2, z4, y1, y2):
        return (
            z1**2
            + z2
            + z4
            + np.sqrt(np.abs(z1 * z4))
            - np.sqrt(np.abs(40.0 * y1 / (y2 + 1e-10)))
            - 3.0
        )

    @njit(cache=True)
    def system_objective_compute(z1, z2, z3, z4, y1, y2, y3):
        return (
            2.0 * z1
            + y1
            + y2
            + z4
            + 2.0 * y3
            - 10.0
            + np.sqrt(np.abs(y1 * y2))
            - np.sqrt(np.abs(40.0 * z1 / (y3 + 1e-10))) * z3
            + np.sqrt(np.abs(y1 * z1))
            - np.sqrt(np.abs(40.0 * y2 / (y3 + 1e-10))) * z4
            + z1 * np.sqrt(np.abs(z3))
            - y2 * y2 * np.sqrt(np.abs(40.0 / (y3 + 1e-10)))
        )

    def discipline1(*args):
        z1, z2, z3, z4, y2, y3 = args
        return discipline1_compute(z1, z2, z3, z4, y2, y3)

    def discipline2(*args):
        z1, z2, z3, y1, y3 = args
        return discipline2_compute(z1, z2, z3, y1, y3)

    def discipline3(*args):
        z1, z2, z4, y1, y2 = args
        return discipline3_compute(z1, z2, z4, y1, y2)

    def system_objective(*args):
        z1, z2, z3, z4, y1, y2, y3 = args
        return system_objective_compute(z1, z2, z3, z4, y1, y2, y3)

    def sys_constraint_1(*args):
        z1, z2, z3, z4, y1, y2, y3 = args
        return z1 + 0.1

    def sys_constraint_2(*args):
        z1, z2, z3, z4, y1, y2, y3 = args
        return z2 + 0.1

    def sys_constraint_3(*args):
        z1, z2, z3, z4, y1, y2, y3 = args
        return z3 + 0.1

    def sys_constraint_4(*args):
        z1, z2, z3, z4, y1, y2, y3 = args
        return z4 + 0.1

    var_names_sys = ["z1", "z2", "z3", "z4", "y1", "y2", "y3"]
    var_names_d1 = ["z1", "z2", "z3", "z4", "y2", "y3"]
    var_names_d2 = ["z1", "z2", "z3", "y1", "y3"]
    var_names_d3 = ["z1", "z2", "z4", "y1", "y2"]
    sys_func = Function(func=system_objective, x=var_names_sys, name="propane_obj")
    sys_constraints = []
    for i, const_func in enumerate([
        sys_constraint_1,
        sys_constraint_2,
        sys_constraint_3,
        sys_constraint_4,
    ]):
        func = Function(func=const_func, x=var_names_sys, name=f"sys_c{i + 1}")
        sys_constraints.append(Constraint(func=func, ctype="ge", value=0.0))
    bounds_sys = np.array([
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
    ])
    system_problem = Problem(
        objective=sys_func,
        constraints=sys_constraints,
        ubounds=bounds_sys[:, 1],
        lbounds=bounds_sys[:, 0],
        name="propane_combustion_system",
    )
    d1_func = Function(func=discipline1, x=var_names_d1, name="d1_obj")

    def d1_constraint(*args):
        return 1.0

    d1_const_func = Function(func=d1_constraint, x=var_names_d1, name="d1_c1")
    d1_const = Constraint(func=d1_const_func, ctype="ge", value=0.0)
    bounds_d1 = np.array([
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
    ])
    subsystem1_problem = Problem(
        objective=d1_func,
        constraints=[d1_const],
        ubounds=bounds_d1[:, 1],
        lbounds=bounds_d1[:, 0],
        name="propane_discipline1",
    )
    d2_func = Function(func=discipline2, x=var_names_d2, name="d2_obj")

    def d2_constraint(*args):
        return 1.0

    d2_const_func = Function(func=d2_constraint, x=var_names_d2, name="d2_c1")
    d2_const = Constraint(func=d2_const_func, ctype="ge", value=0.0)
    bounds_d2 = np.array([
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
    ])
    subsystem2_problem = Problem(
        objective=d2_func,
        constraints=[d2_const],
        ubounds=bounds_d2[:, 1],
        lbounds=bounds_d2[:, 0],
        name="propane_discipline2",
    )
    d3_func = Function(func=discipline3, x=var_names_d3, name="d3_obj")

    def d3_constraint(*args):
        return 1.0

    d3_const_func = Function(func=d3_constraint, x=var_names_d3, name="d3_c1")
    d3_const = Constraint(func=d3_const_func, ctype="ge", value=0.0)
    bounds_d3 = np.array([
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
    ])
    subsystem3_problem = Problem(
        objective=d3_func,
        constraints=[d3_const],
        ubounds=bounds_d3[:, 1],
        lbounds=bounds_d3[:, 0],
        name="propane_discipline3",
    )
    z_idxs_1 = np.array([0, 1, 2, 3])
    x_idxs_1 = np.array([])
    y_idxs_1 = np.array([0])
    y_bar_coupled_1 = [np.array([1]), np.array([2])]
    z_idxs_2 = np.array([0, 1, 2])
    x_idxs_2 = np.array([])
    y_idxs_2 = np.array([1])
    y_bar_coupled_2 = [np.array([0]), np.array([2])]
    z_idxs_3 = np.array([0, 1, 3])
    x_idxs_3 = np.array([])
    y_idxs_3 = np.array([2])
    y_bar_coupled_3 = [np.array([0]), np.array([1])]
    subsystems = [
        {
            "problem": subsystem1_problem,
            "z_idxs": z_idxs_1,
            "x_idxs": x_idxs_1,
            "y_idxs": y_idxs_1,
            "y_coupled_idxs": y_bar_coupled_1,
        },
        {
            "problem": subsystem2_problem,
            "z_idxs": z_idxs_2,
            "x_idxs": x_idxs_2,
            "y_idxs": y_idxs_2,
            "y_coupled_idxs": y_bar_coupled_2,
        },
        {
            "problem": subsystem3_problem,
            "z_idxs": z_idxs_3,
            "x_idxs": x_idxs_3,
            "y_idxs": y_idxs_3,
            "y_coupled_idxs": y_bar_coupled_3,
        },
    ]
    return {
        "system_problem": system_problem,
        "subsystems": subsystems,
        "n_shared": 4,
        "n_local": 0,
        "n_coupling": 3,
        "n_disciplines": 3,
    }
