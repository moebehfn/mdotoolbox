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

"""src/mdotoolbox/benchmarking/sellar.py"""

import numpy as np
from numba import njit

from ..core import Constraint, Function, Problem


def sellar_problem():
    """Create the Sellar MDO benchmark problem with system and subsystem decomposition."""

    @njit(cache=True)
    def discipline1_compute(z1, z2, x1, y2):
        return z1**2 + z2 + x1 - 0.2 * y2

    @njit(cache=True)
    def discipline2_compute(z1, z2, y1):
        return np.sqrt(np.abs(y1)) + z1 + z2

    @njit(cache=True)
    def system_objective_compute(x1, z2, y1, y2):
        return x1**2 + z2 + y1 + np.exp(-y2)

    def discipline1(*args):
        """Compute y1 from z1, z2, x1, y2"""
        z1, z2, x1, y2 = args
        return discipline1_compute(z1, z2, x1, y2)

    def discipline2(*args):
        """Compute y2 from z1, z2, y1"""
        z1, z2, y1 = args
        return discipline2_compute(z1, z2, y1)

    def system_objective(*args):
        """System objective function"""
        z1, z2, x1, y1, y2 = args
        return system_objective_compute(x1, z2, y1, y2)

    def system_constraint1(*args):
        """System constraint 1"""
        z1, z2, x1, y1, y2 = args
        return y1 - 3.16

    def system_constraint2(*args):
        """System constraint 2"""
        z1, z2, x1, y1, y2 = args
        return 24.0 - y2

    var_names_sys = ["z1", "z2", "x1", "y1", "y2"]
    var_names_d1 = ["z1", "z2", "x1", "y2"]
    var_names_d2 = ["z1", "z2", "y1"]
    sys_func = Function(func=system_objective, x=var_names_sys, name="sellar_obj")
    sys_const1_func = Function(func=system_constraint1, x=var_names_sys, name="sys_c1")
    sys_const1 = Constraint(func=sys_const1_func, ctype="ge", value=0.0)
    sys_const2_func = Function(func=system_constraint2, x=var_names_sys, name="sys_c2")
    sys_const2 = Constraint(func=sys_const2_func, ctype="ge", value=0.0)
    bounds_sys = np.array([[-10, 10], [0, 10], [0, 10], [-100, 100], [-100, 100]])
    system_problem = Problem(
        objective=sys_func,
        constraints=[sys_const1, sys_const2],
        ubounds=bounds_sys[:, 1],
        lbounds=bounds_sys[:, 0],
        name="sellar_system",
    )
    d1_func = Function(func=discipline1, x=var_names_d1, name="d1_obj")

    def d1_constraint(*args):
        """Dummy constraint for discipline 1 (always satisfied)"""
        return 1.0

    d1_const_func = Function(func=d1_constraint, x=var_names_d1, name="d1_c1")
    d1_const = Constraint(func=d1_const_func, ctype="ge", value=0.0)
    bounds_d1 = np.array([[-10, 10], [0, 10], [0, 10], [-100, 100]])
    subsystem1_problem = Problem(
        objective=d1_func,
        constraints=[d1_const],
        ubounds=bounds_d1[:, 1],
        lbounds=bounds_d1[:, 0],
        name="sellar_discipline1",
    )
    d2_func = Function(func=discipline2, x=var_names_d2, name="d2_obj")

    def d2_constraint(*args):
        """Dummy constraint for discipline 2 (always satisfied)"""
        return 1.0

    d2_const_func = Function(func=d2_constraint, x=var_names_d2, name="d2_c1")
    d2_const = Constraint(func=d2_const_func, ctype="ge", value=0.0)
    bounds_d2 = np.array([[-10, 10], [0, 10], [-1000, 1000]])
    subsystem2_problem = Problem(
        objective=d2_func,
        constraints=[d2_const],
        ubounds=bounds_d2[:, 1],
        lbounds=bounds_d2[:, 0],
        name="sellar_discipline2",
    )
    z_idxs_1 = np.array([0, 1])
    x_idxs_1 = np.array([0])
    y_idxs_1 = np.array([0])
    y_bar_coupled_1 = [np.array([1])]
    z_idxs_2 = np.array([0, 1])
    x_idxs_2 = np.array([])
    y_idxs_2 = np.array([1])
    y_bar_coupled_2 = [np.array([0])]
    subsystems = []
    sub_dict = {
        "problem": subsystem1_problem,
        "z_idxs": z_idxs_1,
        "x_idxs": x_idxs_1,
        "y_idxs": y_idxs_1,
        "y_coupled_idxs": y_bar_coupled_1,
    }
    subsystems.append(sub_dict)
    sub_dict = {
        "problem": subsystem2_problem,
        "z_idxs": z_idxs_2,
        "x_idxs": x_idxs_2,
        "y_idxs": y_idxs_2,
        "y_coupled_idxs": y_bar_coupled_2,
    }
    subsystems.append(sub_dict)
    return {
        "system_problem": system_problem,
        "subsystems": subsystems,
        "n_shared": 2,
        "n_local": 1,
        "n_coupling": 2,
        "n_disciplines": 2,
    }
