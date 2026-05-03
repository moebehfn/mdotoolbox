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

"""src/mdotoolbox/benchmarking/speed_reducer.py"""

import numpy as np

from ..core import Constraint, Function, Problem


def speed_reducer_problem():
    """Golinski's Speed Reducer Problem (Tedford & Martins 2010 decomposition)"""
    C1 = 0.7854
    C2 = 3.3333
    C3 = 14.9334
    C4 = 43.0934
    C5 = 1.5079
    C6 = 7.477
    C7 = 40
    C8 = 12
    C9 = 3.6
    C10 = 27
    C11 = 397.5
    C12 = 5
    C13 = 2.6
    C14 = 3.9
    C15 = 1.5
    C16 = 1.9
    C17 = 1.93
    C18 = 1100
    C19 = 0.1
    C20 = 16930000.0
    C21 = 745
    C22 = 2.9
    C23 = 5.5
    C24 = 1.1
    C25 = 1.93
    C26 = 850
    C27 = 157500000.0
    C28 = 5
    g1 = lambda z1, z2: C10 / (z1**2 * z2)
    g2 = lambda z1, z2: C11 / (z1**2 * z2**2)
    g3 = lambda z1: C12 / z1
    g4 = C13
    g5 = lambda z1, z2, x2: (C17 * x2**3 / z1 / z2) ** (1 / 4)
    g6 = lambda z1, z2, x2: (
        (1 / C28 / C19 / np.sqrt(C20**2 * x2**2 / (z1**2 * z2**2) + C21)) ** (1 / 3)
    )
    g7 = C22
    g8 = lambda z1, z2, x3: (C25 * x3**3 / z1 / z2) ** (1 / 4)
    g9 = lambda z1, z2, x3: (
        (1 / C26 / C19 / np.sqrt(C20**2 * x3**2 / (z1**2 * z2**2) + C27)) ** (1 / 3)
    )
    g10 = C28
    var_names_sys = ["z1", "z2", "x2", "x3", "y1", "y2", "y3"]

    def system_objective(*args):
        z1, z2, x2, x3, y1, y2, y3 = args
        weight = (
            C1 * y1 * z1**2 * (C2 * z2**2 + C3 * z2 + C4)
            - C5 * y1 * (y2**2 + y3**2)
            + C6 * (y2**3 + y3**3)
            + C1 * (x2 * y2**2 + x3 * y3**2)
        )
        return weight

    sys_func = Function(
        func=system_objective, x=var_names_sys, name="speed_reducer_obj"
    )

    def sys_constraint(*args):
        z1, _, _, x3, _, _, _ = args
        return 1 - z1 * x3 / C7

    func = Function(func=sys_constraint, x=var_names_sys, name="c")
    sys_constraints = [Constraint(func=func, ctype="ge", value=0.0)]
    bounds_sys = np.array([
        [0.7, 0.8],
        [17.0, 28.0],
        [7.3, 8.3],
        [7.3, 8.3],
        [C13, 1000.0],
        [C22, 1000.0],
        [C28, 1000.0],
    ])
    system_problem = Problem(
        objective=sys_func,
        constraints=sys_constraints,
        ubounds=bounds_sys[:, 1],
        lbounds=bounds_sys[:, 0],
        name="speed_reducer_system",
    )
    var_names_d1 = ["z1", "z2", "y2", "y3"]

    def discipline1(*args):
        z1, z2, y2, y3 = args
        y1 = max(g1(z1, z2), g2(z1, z2), g3(z1), g4)
        return y1

    d1_func = Function(func=discipline1, x=var_names_d1, name="y1")
    g_1_1 = Constraint(
        func=Function(
            func=lambda z1, z2, y2, y3: 1 - discipline1(z1, z2, y2, y3) / C8 / z1,
            x=var_names_d1,
            name="g_1_1",
        ),
        ctype="ge",
        value=0.0,
    )
    g_1_2 = Constraint(
        func=Function(
            func=lambda z1, z2, y2, y3: 1 - discipline1(z1, z2, y2, y3) / C9,
            x=var_names_d1,
            name="g_1_2",
        ),
        ctype="ge",
        value=0.0,
    )
    bounds_d1 = np.array([[0.7, 0.8], [17.0, 28.0], [C22, 1000.0], [C28, 1000.0]])
    subsystem1_problem = Problem(
        objective=d1_func,
        constraints=[g_1_1, g_1_2],
        ubounds=bounds_d1[:, 1],
        lbounds=bounds_d1[:, 0],
        name="speed_reducer_discipline1",
    )
    z_idxs_1 = np.array([0, 1])
    x_idxs_1 = np.array([])
    y_idxs_1 = np.array([0])
    y_bar_coupled_1 = [np.array([1]), np.array([2])]
    var_names_d2 = ["z1", "z2", "x2", "y1", "y3"]

    def discipline2(*args):
        z1, z2, x2, y1, y3 = args
        y2 = max(g5(z1, z2, x2), g6(z1, z2, x2), g7)
        return y2

    d2_func = Function(func=discipline2, x=var_names_d2, name="y2")
    g_2_1 = Constraint(
        func=Function(
            func=lambda z1, z2, x2, y1, y3: 1 - discipline2(z1, z2, x2, y1, y3) / C14,
            x=var_names_d2,
            name="g_2_1",
        ),
        ctype="ge",
        value=0.0,
    )
    g_2_2 = Constraint(
        func=Function(
            func=lambda z1, z2, x2, y1, y3: (
                1 - discipline2(z1, z2, x2, y1, y3) * C15 * C16 / x2
            ),
            x=var_names_d2,
            name="g_2_2",
        ),
        ctype="ge",
        value=0.0,
    )
    bounds_d2 = np.array([
        [0.7, 0.8],
        [17.0, 28.0],
        [7.3, 8.3],
        [C13, 1000.0],
        [C28, 1000.0],
    ])
    subsystem2_problem = Problem(
        objective=d2_func,
        constraints=[g_2_1, g_2_2],
        ubounds=bounds_d2[:, 1],
        lbounds=bounds_d2[:, 0],
        name="speed_reducer_discipline2",
    )
    z_idxs_2 = np.array([0, 1])
    x_idxs_2 = np.array([0])
    y_idxs_2 = np.array([1])
    y_bar_coupled_2 = [np.array([0]), np.array([2])]
    var_names_d3 = ["z1", "z2", "x3", "y1", "y2"]

    def discipline3(*args):
        z1, z2, x3, y1, y2 = args
        y3 = max(g8(z1, z2, x3), g9(z1, z2, x3), g10)
        return y3

    d3_func = Function(func=discipline3, x=var_names_d3, name="y3")
    g_3_1 = Constraint(
        func=Function(
            func=lambda z1, z2, x3, y1, y2: 1 - discipline3(z1, z2, x3, y1, y2) / C23,
            x=var_names_d3,
            name="g_3_1",
        ),
        ctype="ge",
        value=0.0,
    )
    g_3_2 = Constraint(
        func=Function(
            func=lambda z1, z2, x3, y1, y2: (
                1 - discipline3(z1, z2, x3, y1, y2) * C24 * C16 / x3
            ),
            x=var_names_d3,
            name="g_3_2",
        ),
        ctype="ge",
        value=0.0,
    )
    bounds_d3 = np.array([
        [0.7, 0.8],
        [17.0, 28.0],
        [7.3, 8.3],
        [C22, 1000.0],
        [C28, 1000.0],
    ])
    subsystem3_problem = Problem(
        objective=d3_func,
        constraints=[g_3_1, g_3_2],
        ubounds=bounds_d3[:, 1],
        lbounds=bounds_d3[:, 0],
        name="speed_reducer_discipline3",
    )
    z_idxs_3 = np.array([0, 1])
    x_idxs_3 = np.array([1])
    y_idxs_3 = np.array([2])
    y_bar_coupled_3 = [np.array([0]), np.array([1])]
    subsystems = [
        dict(
            problem=subsystem1_problem,
            z_idxs=z_idxs_1,
            x_idxs=x_idxs_1,
            y_idxs=y_idxs_1,
            y_coupled_idxs=y_bar_coupled_1,
        ),
        dict(
            problem=subsystem2_problem,
            z_idxs=z_idxs_2,
            x_idxs=x_idxs_2,
            y_idxs=y_idxs_2,
            y_coupled_idxs=y_bar_coupled_2,
        ),
        dict(
            problem=subsystem3_problem,
            z_idxs=z_idxs_3,
            x_idxs=x_idxs_3,
            y_idxs=y_idxs_3,
            y_coupled_idxs=y_bar_coupled_3,
        ),
    ]
    return {
        "system_problem": system_problem,
        "subsystems": subsystems,
        "n_shared": 2,
        "n_local": 2,
        "n_coupling": 3,
        "n_disciplines": 3,
    }
