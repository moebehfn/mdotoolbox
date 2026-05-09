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

"""src/mdotoolbox/frameworks/eco.py"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..core import BudgetManager, compute_Ji
from .base_classes import BaseSolver, BaseSubsystem, BaseSystem


@dataclass
class ECOSubsystem(BaseSubsystem):
    """Subsystem for Enhanced Collaborative Optimization framework."""

    pass


@dataclass
class ECOSystem(BaseSystem):
    """System-level coordinator for Enhanced Collaborative Optimization."""

    def _build_constraints(self, alpha=0.0, **kwargs):
        """Build constraints with relaxed consistency: J_i <= alpha."""
        nz = len(self.z_bar)
        nx = len(self.x_bar)

        def constraints_func(sys_vars):
            """System constraints including relaxed J_i <= alpha"""
            c_vals_ineq = []
            c_vals_eq = []
            z_new = sys_vars[:nz]
            x_new = sys_vars[nz : nz + nx]
            y_new = sys_vars[nz + nx :]
            input_vars = np.concatenate([z_new, x_new, y_new])
            for const in self.problem.constraints:
                if const.ctype == "ge":
                    c_vals_ineq.append(const.func.func(*input_vars) - const.value)
                elif const.ctype == "le":
                    c_vals_ineq.append(-const.func.func(*input_vars) + const.value)
                elif const.ctype == "eq":
                    c_vals_eq.append(const.func.func(*input_vars) - const.value)
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
                    z_new[subsystem.z_idxs],
                    x_bar_i,
                    y_bar_i,
                    y_bar_coupled,
                    subsystem.z_under_i,
                    subsystem.x_under_i,
                )
                c_vals_ineq.append(alpha - J_i_val)
            return (np.array(c_vals_ineq), np.array(c_vals_eq))

        return {
            "ineq": lambda x: constraints_func(x)[0],
            "eq": lambda x: constraints_func(x)[1],
        }


@dataclass
class EnhancedCollaborativeOptimization(BaseSolver):
    """Enhanced Collaborative Optimization solver with relaxation parameter alpha."""

    system: ECOSystem
    subsystem_optimizer: str | Callable
    system_optimizer: str | Callable
    epsilon_J: float = 1e-06
    epsilon_h: float = 1e-06
    budget: BudgetManager | int = 100
    max_iter: int | None = None
    max_eval: int | None = None
    alpha_initial: float = 10.0
    alpha_final: float = 1e-06
    alpha_schedule: str = "geometric"
    solver: str = "ECO"

    def _add_custom_history_columns(self):
        """Add alpha column to history tracking."""
        self._history["alpha"] = []

    def _custom_initialize(self):
        """Initialize alpha parameter."""
        self._alpha = self.alpha_initial

    def _pre_system_solve_update(self):
        """Update alpha parameter before each system solve."""
        self._update_alpha()

    def _get_system_solve_kwargs(self):
        """Pass alpha to system solve method."""
        return {"alpha": self._alpha}

    def _check_convergence(self, J_total: float, h_total: float = 0.0) -> bool:
        """Check convergence with alpha-aware criterion."""
        return (
            J_total <= self.epsilon_J
            and h_total <= self.epsilon_h
            and (J_total <= self._alpha)
        )

    def _update_alpha(self):
        """Update relaxation parameter according to schedule."""
        if self.max_iter is None:
            ratio = 0.0
        else:
            ratio = min(self._iter / self.max_iter, 1.0)
        if self.alpha_schedule == "linear":
            self._alpha = (
                self.alpha_initial + (self.alpha_final - self.alpha_initial) * ratio
            )
        elif self.alpha_schedule == "exponential":
            self._alpha = self.alpha_initial * np.exp(
                ratio * np.log(self.alpha_final / self.alpha_initial)
            )
        elif self.alpha_schedule == "geometric":
            if self._iter == 0:
                self._alpha = self.alpha_initial
            else:
                decay_rate = (self.alpha_final / self.alpha_initial) ** (
                    1.0 / (self.max_iter - 1)
                    if self.max_iter and self.max_iter > 1
                    else 0.1
                )
                self._alpha = max(self._alpha * decay_rate, self.alpha_final)

    def _print_custom_budget_info(self):
        """Print alpha schedule configuration."""
        print(f"Alpha schedule: {self.alpha_schedule}")
        print(f"Alpha initial: {self.alpha_initial}")
        print(f"Alpha final: {self.alpha_final}")

    def _record_subsystem_custom_history(self, eval_data):
        """Record alpha for subsystem evaluations."""
        self._history["alpha"].append(self._alpha)

    def _record_system_custom_history(self, eval_data):
        """Record alpha for system evaluations."""
        self._history["alpha"].append(self._alpha)
