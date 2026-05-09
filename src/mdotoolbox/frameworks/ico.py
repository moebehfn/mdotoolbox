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

"""src/mdotoolbox/frameworks/ico.py"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..core import BudgetManager, compute_Ji
from .base_classes import BaseSolver, BaseSubsystem, BaseSystem


@dataclass
class ICOSubsystem(BaseSubsystem):
    """Subsystem for Improved Collaborative Optimization."""

    pass


@dataclass
class ICOSystem(BaseSystem):
    """System-level coordinator for Improved Collaborative Optimization."""

    def _build_objective(
        self,
        _,
        _eval_history,
        _best_result,
        system_iter,
        global_start_time,
        gamma=1.0,
        relaxation_radius=0.0,
        **kwargs,
    ):
        """Build system objective with penalty term."""
        nz = len(self.z_bar)
        nx = len(self.x_bar)

        def objective(sys_vars_inner):
            """System objective with ICO penalty"""
            nonlocal _eval_history, _best_result
            f_sys_val, c_vals = self.problem.evaluate(sys_vars_inner, f=True, c=True)
            c_vals_dict = {
                const.func.name: c_vals[i]
                for i, const in enumerate(self.problem.constraints)
            }
            h_total = self.problem.compute_constraint_violation(c_vals_dict)
            z_bar_eval = sys_vars_inner[:nz]
            x_bar_eval = sys_vars_inner[nz : nz + nx]
            y_bar_eval = sys_vars_inner[nz + nx :]
            penalty = 0.0
            for subsystem in self.subsystems:
                x_bar_i = (
                    x_bar_eval[subsystem.x_idxs]
                    if len(subsystem.x_idxs) > 0
                    else np.array([])
                )
                y_bar_i = y_bar_eval[subsystem.y_idxs]
                y_bar_coupled = []
                for coupled_idx_array in subsystem.y_coupled_idxs:
                    for idx in coupled_idx_array:
                        y_bar_coupled.append(y_bar_eval[idx])
                y_bar_coupled = (
                    np.array(y_bar_coupled) if y_bar_coupled else np.array([])
                )
                J_i_val, _ = compute_Ji(
                    subsystem.problem.objective.func,
                    z_bar_eval[subsystem.z_idxs],
                    x_bar_i,
                    y_bar_i,
                    y_bar_coupled,
                    subsystem.z_under_i,
                    subsystem.x_under_i,
                )
                penalty += gamma * abs(J_i_val - relaxation_radius**2)
            obj_val = f_sys_val + penalty
            if f_sys_val < _best_result["f_sys"] or (
                f_sys_val <= _best_result["f_sys"] and h_total < _best_result["h_total"]
            ):
                _best_result = {
                    "f_sys": f_sys_val,
                    "h_total": h_total,
                    "z": z_bar_eval.copy(),
                    "x": x_bar_eval.copy(),
                    "y": y_bar_eval.copy(),
                }
            _eval_history.append({
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


@dataclass
class ImprovedCollaborativeOptimization(BaseSolver):
    """Improved Collaborative Optimization solver with dynamic penalty."""

    system: ICOSystem
    subsystem_optimizer: str | Callable
    system_optimizer: str | Callable
    epsilon_J: float = 1e-06
    epsilon_h: float = 1e-06
    budget: BudgetManager | int = 100
    max_iter: int | None = None
    max_eval: int | None = None
    gamma: float = 1.0
    relaxation_radius: float = 0.0
    delta: float = 1.1
    solver: str = "ICO"

    def _add_custom_history_columns(self):
        """Add gamma and radius columns to history tracking."""
        self._history["gamma"] = []
        self._history["radius"] = []

    def _custom_initialize(self):
        """Initialize penalty parameters."""
        self._gamma = self.gamma
        self._radius = self.relaxation_radius

    def _pre_system_solve_update(self):
        """Update penalty parameters before each system solve."""
        pass

    def _get_system_solve_kwargs(self):
        """Pass gamma and radius to system solve method."""
        return {"gamma": self._gamma, "relaxation_radius": self._radius}

    def _print_custom_budget_info(self):
        """Print penalty configuration."""
        print(f"Gamma: {self.gamma}")
        print(f"Relaxation radius: {self.relaxation_radius}")
        print(f"Delta: {self.delta}")

    def _record_subsystem_custom_history(self, eval_data):
        """Record gamma and radius for subsystem evaluations."""
        self._history["gamma"].append(self._gamma)
        self._history["radius"].append(self._radius)

    def _record_system_custom_history(self, eval_data):
        """Record gamma and radius for system evaluations."""
        self._history["gamma"].append(self._gamma)
        self._history["radius"].append(self._radius)
