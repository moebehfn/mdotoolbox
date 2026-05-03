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

"""src/mdotoolbox/benchmarking/heart_dipole.py"""

import numpy as np

from ..core import Constraint, Function, Problem


def heart_dipole_problem():
    """Heart Dipole Problem"""
    dAB = 0.485
    dXY = 0.369
    dA = -0.0739
    dB = 0.0739
    dC = 0.0739
    dD = 0.1239
    dE = 0.0739
    dF = 0.0739
    var_sys = ["x" + str(i) for i in range(1, 9)]

    def system_objective(*args):
        x1, x2, x3, x4, x5, x6, x7, x8 = args
        f5 = (
            x1 * (x5**2 - x7**2)
            - 2 * x3 * x5 * x7
            + x2 * (x6**2 - x8**2)
            - 2 * x4 * x6 * x8
            - dC
        )
        f6 = (
            x1 * x5 * x7
            + x3 * (x5**2 - x7**2)
            + x2 * x6 * x8
            + x4 * (x6**2 - x8**2)
            - dD
        )
        f7 = (
            x1 * x5 * (x5**2 - 3 * x7**2)
            + x3 * x7 * (x7**2 - 3 * x5**2)
            + x2 * x6 * (x6**2 - 3 * x8**2)
            + x4 * x8 * (x8**2 - 3 * x6**2)
            - dE
        )
        f8 = (
            x3 * x5 * (x5**2 - 3 * x7**2)
            - x1 * x7 * (x7**2 - 3 * x5**2)
            + x4 * x6 * (x6**2 - 3 * x8**2)
            - x2 * x8 * (x8**2 - 3 * x6**2)
            - dF
        )
        return f5**2 + f6**2 + f7**2 + f8**2

    sys_func = Function(func=system_objective, x=var_sys, name="residual")
    sys_constraints = [
        Constraint(
            func=Function(func=lambda *args: args[i], x=var_sys, name=f"pos_{i}"),
            ctype="ge",
            value=0.0,
        )
        for i in range(8)
    ]
    bounds_sys = np.array([[-1.0, 1.0]] * 8)
    system_problem = Problem(
        objective=sys_func,
        constraints=sys_constraints,
        ubounds=bounds_sys[:, 1],
        lbounds=bounds_sys[:, 0],
        name="heart_dipole_system",
    )
    var_d1 = ["x2", "x3", "x4", "x5", "x6", "x7"]

    def discipline1(x2, x3, x4, x5, x6, x7):
        x1 = dAB - x2
        x8 = (x5 * x1 - x6 * x2 - x7 * x3 - dA) / x4 if abs(x4) > 1e-10 else 0.0
        return np.array([x1, x8])

    d1_func = Function(
        func=lambda x2, x3, x4, x5, x6, x7: discipline1(x2, x3, x4, x5, x6, x7)[0],
        x=var_d1,
        name="x1",
    )
    d1_constraint = Constraint(
        func=Function(func=lambda *args: 1.0, x=var_d1, name="d1_dummy"),
        ctype="ge",
        value=0.0,
    )
    bounds_d1 = bounds_sys[[1, 2, 3, 4, 5, 6], :]
    subsystem1_problem = Problem(
        objective=d1_func,
        constraints=[d1_constraint],
        ubounds=bounds_d1[:, 1],
        lbounds=bounds_d1[:, 0],
        name="heart_dipole_d1",
    )
    var_d2 = ["x1", "x2", "x3", "x5", "x7", "x8"]

    def discipline2(x1, x2, x3, x5, x7, x8):
        x4 = dXY - x3
        x6 = (x7 * x1 + x5 * x3 - x8 * x2 - dB) / x4 if abs(x4) > 1e-10 else 0.0
        return np.array([x4, x6])

    d2_func = Function(
        func=lambda x1, x2, x3, x5, x7, x8: discipline2(x1, x2, x3, x5, x7, x8)[0],
        x=var_d2,
        name="x4",
    )
    d2_constraint = Constraint(
        func=Function(func=lambda *args: 1.0, x=var_d2, name="d2_dummy"),
        ctype="ge",
        value=0.0,
    )
    bounds_d2 = bounds_sys[[0, 1, 2, 4, 6, 7], :]
    subsystem2_problem = Problem(
        objective=d2_func,
        constraints=[d2_constraint],
        ubounds=bounds_d2[:, 1],
        lbounds=bounds_d2[:, 0],
        name="heart_dipole_d2",
    )
    subsystems = [
        {
            "problem": subsystem1_problem,
            "z_idxs": np.array([1, 2, 4, 6]),
            "x_idxs": np.array([]),
            "y_idxs": np.array([0, 3]),
            "y_coupled_idxs": [np.array([1, 2])],
        },
        {
            "problem": subsystem2_problem,
            "z_idxs": np.array([1, 2, 4, 6]),
            "x_idxs": np.array([]),
            "y_idxs": np.array([1, 2]),
            "y_coupled_idxs": [np.array([0, 3])],
        },
    ]
    return {
        "system_problem": system_problem,
        "subsystems": subsystems,
        "n_shared": 4,
        "n_local": 0,
        "n_coupling": 4,
    }
