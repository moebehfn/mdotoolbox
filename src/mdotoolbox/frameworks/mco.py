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

"""src/mdotoolbox/frameworks/mco.py"""

import time
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from ..core import BudgetManager, compute_Ji
from ..optimizers import get_optimizer
from .base_classes import BaseSolver, BaseSubsystem, BaseSystem


@dataclass
class MCOSubsystem(BaseSubsystem):
    """Subsystem for Modified Collaborative Optimization."""

    pass


@dataclass
class MCOSystem(BaseSystem):
    """System-level coordinator for Modified Collaborative Optimization."""

    def _build_objective(
        self, _, _eval_history, _best_result, system_iter, global_start_time, **kwargs
    ):
        """Build system objective using averaged z_bar."""
        nx = len(self.x_bar)

        def objective(target_vars):
            """System objective with averaged z_bar"""
            nonlocal _eval_history, _best_result
            x_new = target_vars[:nx]
            y_new = target_vars[nx:]
            z_avg = np.zeros_like(self.z_bar)
            for subsystem in self.subsystems:
                z_avg[subsystem.z_idxs] += subsystem.z_under_i
            z_avg /= len(self.subsystems)
            sys_vars = np.concatenate([z_avg, x_new, y_new])
            f_sys_val, c_vals = self.problem.evaluate(sys_vars, f=True, c=True)
            c_vals_dict = {
                const.func.name: c_vals[i]
                for i, const in enumerate(self.problem.constraints)
            }
            h_total = self.problem.compute_constraint_violation(c_vals_dict)
            if f_sys_val < _best_result["f_sys"] or (
                f_sys_val <= _best_result["f_sys"] and h_total < _best_result["h_total"]
            ):
                _best_result = {
                    "f_sys": f_sys_val,
                    "h_total": h_total,
                    "z": z_avg.copy(),
                    "x": x_new.copy(),
                    "y": y_new.copy(),
                }
            _eval_history.append({
                "system_iter": system_iter,
                "time_since_start": time.time() - global_start_time,
                "eval_type": "objective",
                "f_sys": f_sys_val,
                "h_total": h_total,
                "z": z_avg.copy(),
                "x": x_new.copy(),
                "y": y_new.copy(),
            })
            return f_sys_val

        return objective

    def _build_constraints(self, **kwargs):
        """Build constraints using averaged z_bar."""
        nx = len(self.x_bar)

        def constraints_func(target_vars):
            """System constraints with averaged z_bar"""
            c_vals_ineq = []
            c_vals_eq = []
            x_new = target_vars[:nx]
            y_new = target_vars[nx:]
            z_avg = np.zeros_like(self.z_bar)
            for subsystem in self.subsystems:
                z_avg[subsystem.z_idxs] += subsystem.z_under_i
            z_avg /= len(self.subsystems)
            sys_vars = np.concatenate([z_avg, x_new, y_new])
            for const in self.problem.constraints:
                if const.ctype == "ge":
                    c_vals_ineq.append(const.func.func(*sys_vars) - const.value)
                elif const.ctype == "le":
                    c_vals_ineq.append(-const.func.func(*sys_vars) + const.value)
                elif const.ctype == "eq":
                    c_vals_eq.append(const.func.func(*sys_vars) - const.value)
            for subsystem in self.subsystems:
                x_bar_i = (
                    x_new[subsystem.x_idxs]
                    if len(subsystem.x_idxs) > 0
                    else np.array([])
                )
                y_bar_i = y_new[subsystem.y_idxs]
                y_bar_coupled = []
                for coupled_idx_array in subsystem.y_coupled_idxs:
                    for idx in coupled_idx_array:
                        y_bar_coupled.append(y_new[idx])
                y_bar_coupled = (
                    np.array(y_bar_coupled) if y_bar_coupled else np.array([])
                )
                J_i_val, _ = compute_Ji(
                    subsystem.problem.objective.func,
                    z_avg[subsystem.z_idxs],
                    x_bar_i,
                    y_bar_i,
                    y_bar_coupled,
                    subsystem.z_under_i,
                    subsystem.x_under_i,
                )
                c_vals_eq.append(J_i_val)
            return (np.array(c_vals_ineq), np.array(c_vals_eq))

        return {
            "ineq": lambda x: constraints_func(x)[0],
            "eq": lambda x: constraints_func(x)[1],
        }

    def solve(
        self,
        optimizer: Union[str, Callable],
        system_iter: int,
        global_start_time: float,
        maxiter: int = None,
        **kwargs,
    ):
        """Solve MCO system problem - optimizes only (x_bar, y_bar)."""
        _eval_history = []
        nx = len(self.x_bar)
        _best_result = {
            "f_sys": np.inf,
            "h_total": np.inf,
            "z": self.z_bar.copy(),
            "x": self.x_bar.copy(),
            "y": self.y_bar.copy(),
        }
        objective = self._build_objective(
            None, _eval_history, _best_result, system_iter, global_start_time, **kwargs
        )
        constraints = self._build_constraints(**kwargs)
        x0 = np.concatenate([self.x_bar, self.y_bar])
        bounds_mco = self.problem.bounds[len(self.z_bar) :]
        if isinstance(optimizer, str):
            opt_kwargs = (
                {"maxiter": maxiter} if optimizer != "cobyqa" else {"maxfev": maxiter}
            )
            opt = get_optimizer(optimizer, **opt_kwargs)
        else:
            opt = optimizer
        result_x, f_sys_val = opt(objective, x0, bounds_mco, constraints)
        from ..core import clip_to_bounds

        result_x = clip_to_bounds(result_x, bounds_mco)
        self.x_bar = result_x[:nx]
        self.y_bar = result_x[nx:]
        z_avg = np.zeros_like(self.z_bar)
        for subsystem in self.subsystems:
            z_avg[subsystem.z_idxs] += subsystem.z_under_i
        z_avg /= len(self.subsystems)
        self.z_bar = z_avg
        final_sys_vars = np.concatenate([self.z_bar, self.x_bar, self.y_bar])
        _, c_vals = self.problem.evaluate(final_sys_vars, f=False, c=True)
        c_vals_dict = {
            const.func.name: c_vals[i]
            for i, const in enumerate(self.problem.constraints)
        }
        h_total = self.problem.compute_constraint_violation(c_vals_dict)
        _eval_history.append({
            "system_iter": system_iter,
            "time_since_start": time.time() - global_start_time,
            "eval_type": "final",
            "f_sys": f_sys_val,
            "h_total": h_total,
            "z": self.z_bar.copy(),
            "x": self.x_bar.copy(),
            "y": self.y_bar.copy(),
        })
        self.iteration_history.extend(_eval_history)
        return (
            self.z_bar,
            self.x_bar,
            self.y_bar,
            f_sys_val,
            h_total,
            len(_eval_history),
        )


@dataclass
class ModifiedCollaborativeOptimization(BaseSolver):
    """Modified Collaborative Optimization solver with averaged z."""

    system: MCOSystem
    subsystem_optimizer: Union[str, Callable]
    system_optimizer: Union[str, Callable]
    epsilon_J: float = 1e-06
    epsilon_h: float = 1e-06
    budget: Union["BudgetManager", int] = 100
    max_iter: Union[int, None] = None
    max_eval: Union[int, None] = None
    solver: str = "MCO"
