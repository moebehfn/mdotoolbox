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

"""src/mdotoolbox/benchmarking/scalable.py"""

import numpy as np
from numba import njit

from ..core import Constraint, Function, Problem


def scalable_mdo_problem(
    n_disciplines: int = 3,
    n_z_bar: int = 10,
    n_x_under_per_discipline: int = 5,
    n_y_per_discipline: int = 50,
    seed: int = 42,
):
    """Generate scalable MDO benchmark problem from Tedford & Martins (2010)."""
    n_local_total = n_disciplines * n_x_under_per_discipline
    n_coupling_total = n_disciplines * n_y_per_discipline
    rng = np.random.default_rng(seed)

    def create_random_matrix(rows, cols, flip_sign=False, rng_instance=rng):
        """Create random coefficient matrix using a specific generator."""
        mat = rng_instance.uniform(0, 10, (rows, cols))
        if flip_sign:
            mat = -mat
        return mat

    C_matrices = []
    Cx_matrices = []
    Cy_matrices = []
    for i in range(n_disciplines):
        C_matrices.append(create_random_matrix(n_y_per_discipline, n_z_bar))
        Cx_matrices.append(
            create_random_matrix(n_y_per_discipline, n_x_under_per_discipline)
        )
        cy_row = []
        for j in range(n_disciplines):
            if j != i:
                flip = j % 2 == 1
                cy_row.append(
                    create_random_matrix(n_y_per_discipline, n_y_per_discipline, flip)
                )
        Cy_matrices.append(cy_row)

    @njit(cache=True)
    def compute_coupling_jit(
        disc_idx,
        z_under,
        x_under_i,
        C_mat,
        Cx_mat,
        Cy_mats_list,
        y_bar_others_flat,
        n_coupling,
        n_other,
    ):
        """JIT-compiled coupling computation"""
        y_i = C_mat @ z_under + Cx_mat @ x_under_i
        for other_idx in range(n_other):
            start_idx = other_idx * n_coupling
            end_idx = start_idx + n_coupling
            y_bar_other = y_bar_others_flat[start_idx:end_idx]
            y_i += Cy_mats_list[other_idx] @ y_bar_other
        return y_i

    def compute_coupling(disc_idx, z_under, x_under_i, y_bar_others):
        """Compute coupling variables y_i for discipline i"""
        y_bar_others_flat = np.concatenate(y_bar_others)
        n_other = len(y_bar_others)
        Cy_array = np.array(Cy_matrices[disc_idx])
        return compute_coupling_jit(
            disc_idx,
            z_under,
            x_under_i,
            C_matrices[disc_idx],
            Cx_matrices[disc_idx],
            Cy_array,
            y_bar_others_flat,
            n_y_per_discipline,
            n_other,
        )

    @njit(cache=True)
    def system_objective_jit(variables):
        """JIT-compiled system objective"""
        return np.sum(variables**2)

    def system_objective(*variables):
        """System objective: sum of squares of all variables (z_bar, x_bar, y_bar)"""
        return system_objective_jit(np.array(variables))

    def system_constraint_factory(disc_idx):
        """Factory for system-level constraints"""

        def constraint(*variables):
            z_bar = np.array(variables[:n_z_bar])
            x_start = n_z_bar + disc_idx * n_x_under_per_discipline
            x_end = x_start + n_x_under_per_discipline
            x_under_i = np.array(variables[x_start:x_end])
            y_start = n_z_bar + n_local_total
            y_bar_vals = np.array(variables[y_start:])
            y_bar_others = []
            for j in range(n_disciplines):
                if j != disc_idx:
                    y_j_start = j * n_y_per_discipline
                    y_j_end = y_j_start + n_y_per_discipline
                    y_bar_others.append(y_bar_vals[y_j_start:y_j_end])
            y_i_computed = compute_coupling(disc_idx, z_bar, x_under_i, y_bar_others)
            y_i_start = disc_idx * n_y_per_discipline
            y_i_end = y_i_start + n_y_per_discipline
            y_bar_i = y_bar_vals[y_i_start:y_i_end]
            return 1.0 - np.sum((y_bar_i - y_i_computed) ** 2)

        return constraint

    var_names = (
        [f"z{i}" for i in range(n_z_bar)]
        + [
            f"x{i}_{j}"
            for i in range(n_disciplines)
            for j in range(n_x_under_per_discipline)
        ]
        + [f"y{i}_{j}" for i in range(n_disciplines) for j in range(n_y_per_discipline)]
    )
    system_func = Function(
        func=system_objective, x=var_names, name="scalable_mdo_system_obj"
    )
    system_constraints = []
    for i in range(n_disciplines):
        const_func = Function(
            func=system_constraint_factory(i), x=var_names, name=f"sys_const_{i}"
        )
        system_constraints.append(Constraint(func=const_func, ctype="ge", value=0.0))
    bounds_z_bar = np.array([[-10, 10]] * n_z_bar)
    bounds_x_under = np.array([[-10, 10]] * n_local_total)
    bounds_y_bar = np.array([[-100, 100]] * n_coupling_total)
    system_problem = Problem(
        objective=system_func,
        constraints=system_constraints,
        ubounds=np.vstack([bounds_z_bar, bounds_x_under, bounds_y_bar])[:, 1],
        lbounds=np.vstack([bounds_z_bar, bounds_x_under, bounds_y_bar])[:, 0],
        name="scalable_mdo_system",
    )
    subsystems = []
    for i in range(n_disciplines):

        def subsystem_objective_factory(disc_idx):

            def objective(*variables):
                z_under = np.array(variables[:n_z_bar])
                x_under_i = np.array(
                    variables[n_z_bar : n_z_bar + n_x_under_per_discipline]
                )
                y_bar_others_flat = np.array(
                    variables[n_z_bar + n_x_under_per_discipline :]
                )
                y_bar_others = []
                idx = 0
                for j in range(n_disciplines):
                    if j != disc_idx:
                        y_bar_j = y_bar_others_flat[idx : idx + n_y_per_discipline]
                        y_bar_others.append(y_bar_j)
                        idx += n_y_per_discipline
                y_i = compute_coupling(disc_idx, z_under, x_under_i, y_bar_others)
                return y_i

            return objective

        def subsystem_constraint_factory(disc_idx):

            def constraint(*variables):
                z_under = np.array(variables[:n_z_bar])
                x_under_i = np.array(
                    variables[n_z_bar : n_z_bar + n_x_under_per_discipline]
                )
                return 10.0 - np.sum(z_under**2) - np.sum(x_under_i**2)

            return constraint

        sub_var_names = (
            [f"z{j}" for j in range(n_z_bar)]
            + [f"x{i}_{j}" for j in range(n_x_under_per_discipline)]
            + [
                f"y{j}_{k}"
                for j in range(n_disciplines)
                if j != i
                for k in range(n_y_per_discipline)
            ]
        )
        sub_func = Function(
            func=subsystem_objective_factory(i), x=sub_var_names, name=f"sub_{i}_obj"
        )
        sub_const_func = Function(
            func=subsystem_constraint_factory(i), x=sub_var_names, name=f"sub_{i}_const"
        )
        sub_const = Constraint(func=sub_const_func, ctype="ge", value=0.0)
        bounds_sub_y_bar = np.array(
            [[-100, 100]] * (n_disciplines - 1) * n_y_per_discipline
        )
        subsystem_problem = Problem(
            objective=sub_func,
            constraints=[sub_const],
            ubounds=np.vstack([
                bounds_z_bar,
                bounds_x_under[:n_x_under_per_discipline],
                bounds_sub_y_bar,
            ])[:, 1],
            lbounds=np.vstack([
                bounds_z_bar,
                bounds_x_under[:n_x_under_per_discipline],
                bounds_sub_y_bar,
            ])[:, 0],
            name=f"scalable_mdo_subsystem_{i}",
        )
        z_idxs = np.arange(n_z_bar)
        x_idxs = np.arange(
            i * n_x_under_per_discipline, (i + 1) * n_x_under_per_discipline
        )
        y_idxs = np.arange(i * n_y_per_discipline, (i + 1) * n_y_per_discipline)
        y_coupled_idxs = []
        for j in range(n_disciplines):
            if j != i:
                y_j_idxs = np.arange(
                    j * n_y_per_discipline, (j + 1) * n_y_per_discipline
                )
                y_coupled_idxs.append(y_j_idxs)
        sub_dict = {
            "problem": subsystem_problem,
            "z_idxs": z_idxs,
            "x_idxs": x_idxs,
            "y_idxs": y_idxs,
            "y_coupled_idxs": y_coupled_idxs,
        }
        subsystems.append(sub_dict)
    return {
        "system_problem": system_problem,
        "subsystems": subsystems,
        "n_z_bar": n_z_bar,
        "n_x_under_per_discipline": n_x_under_per_discipline,
        "n_y_per_discipline": n_y_per_discipline,
        "n_disciplines": n_disciplines,
        "n_x_under": n_x_under_per_discipline * n_disciplines,
        "n_y": n_y_per_discipline * n_disciplines,
    }
