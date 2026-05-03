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

"""src/mdotoolbox/frameworks/framework_utils.py"""

import numpy as np

from ..core import compute_Ji


def extract_subsystem_targets(subsystem, z_bar, x_bar, y_bar):
    """Extract subsystem-specific targets from system variables."""
    x_bar_i = x_bar[subsystem.x_idxs] if len(subsystem.x_idxs) > 0 else np.array([])
    y_bar_i = y_bar[subsystem.y_idxs]
    z_bar_i = z_bar[subsystem.z_idxs]
    y_bar_coupled = []
    for coupled_idx_array in subsystem.y_coupled_idxs:
        for idx in coupled_idx_array:
            y_bar_coupled.append(y_bar[idx])
    y_bar_coupled = np.array(y_bar_coupled) if y_bar_coupled else np.array([])
    return (z_bar_i, x_bar_i, y_bar_i, y_bar_coupled)


def compute_all_subsystem_discrepancies(system, z_new, x_new, y_new):
    """Compute discrepancies J_i for all subsystems."""
    J_vals = []
    for subsystem in system.subsystems:
        x_bar_i = x_new[subsystem.x_idxs] if len(subsystem.x_idxs) > 0 else np.array([])
        y_bar_i = y_new[subsystem.y_idxs]
        y_bar_coupled = []
        for coupled_idx_array in subsystem.y_coupled_idxs:
            for idx in coupled_idx_array:
                y_bar_coupled.append(y_new[idx])
        y_bar_coupled = np.array(y_bar_coupled) if y_bar_coupled else np.array([])
        J_i_val, _ = compute_Ji(
            subsystem.problem.objective.func,
            z_new[subsystem.z_idxs],
            x_bar_i,
            y_bar_i,
            y_bar_coupled,
            subsystem.z_under_i,
            subsystem.x_under_i,
        )
        J_vals.append(J_i_val)
    return J_vals


def build_subsystem_objective_closure(
    subsystem,
    z_bar_i,
    x_bar_i,
    y_bar_i,
    y_bar_coupled,
    eval_history,
    best_result,
    system_iter,
    global_start_time,
):
    """Build objective function closure for subsystem optimization."""
    import time

    nz = len(subsystem.z_idxs)
    nx = len(subsystem.x_idxs)

    def objective(local_vars):
        """Objective: discrepancy J_i"""
        nonlocal eval_history, best_result
        z_under_i = local_vars[:nz]
        x_under_i = local_vars[nz:] if nx > 0 else np.array([])
        J_i_val, y_i = compute_Ji(
            subsystem.problem.objective.func,
            z_bar_i,
            x_bar_i,
            y_bar_i,
            y_bar_coupled,
            z_under_i,
            x_under_i,
        )
        input_vars = np.concatenate([z_under_i, x_under_i, y_bar_coupled])
        _, c_vals = subsystem.problem.evaluate(input_vars, f=False, c=True)
        c_vals_dict = {
            const.func.name: c_vals[i]
            for i, const in enumerate(subsystem.problem.constraints)
        }
        h_total = subsystem.problem.compute_constraint_violation(c_vals_dict)
        if J_i_val < best_result["J_i"] or (
            J_i_val <= best_result["J_i"] and h_total < best_result["h_total"]
        ):
            best_result.update({
                "x": local_vars.copy(),
                "J_i": J_i_val,
                "h_total": h_total,
                "z_under_i": z_under_i.copy(),
                "x_under_i": x_under_i.copy() if len(x_under_i) > 0 else np.array([]),
                "y_i": y_i.copy(),
            })
        eval_history.append({
            "system_iter": system_iter,
            "time_since_start": time.time() - global_start_time,
            "eval_type": "objective",
            "J_i": J_i_val,
            "h_total": h_total,
            "z_under_i": z_under_i.copy(),
            "x_under_i": x_under_i.copy() if len(x_under_i) > 0 else np.array([]),
            "y_i": y_i.copy(),
        })
        subsystem.y_i = y_i
        return J_i_val

    return objective


def build_subsystem_constraints_closure(subsystem, y_bar_coupled):
    """Build constraint function closure for subsystem optimization."""
    nz = len(subsystem.z_idxs)
    nx = len(subsystem.x_idxs)

    def constraints_func(local_vars):
        """Constraints: g_i(z_under_i, x_under_i, y_bar_coupled) >= 0"""
        z_under_i = local_vars[:nz]
        x_under_i = local_vars[nz:] if nx > 0 else np.array([])
        input_vars = np.concatenate([z_under_i, x_under_i, y_bar_coupled])
        c_vals_ineq = []
        c_vals_eq = []
        for const in subsystem.problem.constraints:
            if const.ctype == "ge":
                c_vals_ineq.append(const.func.func(*input_vars) - const.value)
            elif const.ctype == "le":
                c_vals_ineq.append(-const.func.func(*input_vars) + const.value)
            elif const.ctype == "eq":
                c_vals_eq.append(const.func.func(*input_vars) - const.value)
        return (np.array(c_vals_ineq), np.array(c_vals_eq))

    return {
        "ineq": lambda x: constraints_func(x)[0],
        "eq": lambda x: constraints_func(x)[1],
    }


def build_system_objective_closure(
    system, eval_history, best_result, system_iter, global_start_time, penalty_func=None
):
    """Build objective function closure for system optimization."""
    import time

    nz = len(system.z_bar)
    nx = len(system.x_bar)

    def objective(sys_vars):
        """System objective with optional penalty"""
        nonlocal eval_history, best_result
        f_sys_val, c_vals = system.problem.evaluate(sys_vars, f=True, c=True)
        if penalty_func is not None:
            penalty = penalty_func(sys_vars)
            obj_val = f_sys_val + penalty
        else:
            obj_val = f_sys_val
        c_vals_dict = {
            const.func.name: c_vals[i]
            for i, const in enumerate(system.problem.constraints)
        }
        h_total = system.problem.compute_constraint_violation(c_vals_dict)
        z_bar_eval = sys_vars[:nz]
        x_bar_eval = sys_vars[nz : nz + nx]
        y_bar_eval = sys_vars[nz + nx :]
        if f_sys_val < best_result["f_sys"] or (
            f_sys_val <= best_result["f_sys"] and h_total < best_result["h_total"]
        ):
            best_result.update({
                "f_sys": f_sys_val,
                "h_total": h_total,
                "z": z_bar_eval.copy(),
                "x": x_bar_eval.copy(),
                "y": y_bar_eval.copy(),
            })
        eval_history.append({
            "system_iter": system_iter,
            "time_since_start": time.time() - global_start_time,
            "eval_type": "objective",
            "f_sys": f_sys_val,
            "h_total": h_total,
            "z": z_bar_eval.copy(),
            "x": x_bar_eval.copy(),
            "y": y_bar_eval.copy(),
        })
        return obj_val

    return objective


def build_system_constraints_closure(system, constraint_type="equality"):
    """Build constraint function closure for system optimization."""
    nz = len(system.z_bar)
    nx = len(system.x_bar)

    def constraints_func(sys_vars, alpha=None):
        """System constraints including J_i consistency"""
        c_vals_ineq = []
        c_vals_eq = []
        z_new = sys_vars[:nz]
        x_new = sys_vars[nz : nz + nx]
        y_new = sys_vars[nz + nx :]
        input_vars = np.concatenate([z_new, x_new, y_new])
        for const in system.problem.constraints:
            if const.ctype == "ge":
                c_vals_ineq.append(const.func.func(*input_vars) - const.value)
            elif const.ctype == "le":
                c_vals_ineq.append(-const.func.func(*input_vars) + const.value)
            elif const.ctype == "eq":
                c_vals_eq.append(const.func.func(*input_vars) - const.value)
        if constraint_type != "none":
            J_vals = compute_all_subsystem_discrepancies(system, z_new, x_new, y_new)
            for J_i_val in J_vals:
                if constraint_type == "equality":
                    c_vals_eq.append(J_i_val)
                elif constraint_type == "inequality":
                    c_vals_ineq.append(alpha - J_i_val)
        return (np.array(c_vals_ineq), np.array(c_vals_eq))

    return {
        "ineq": lambda x: constraints_func(x)[0],
        "eq": lambda x: constraints_func(x)[1],
    }
