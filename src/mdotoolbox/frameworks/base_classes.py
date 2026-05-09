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

"""src/mdotoolbox/frameworks/base_classes.py"""

import datetime
import time
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np

if TYPE_CHECKING:
    pass

from ..core import (
    BestSolution,
    BudgetManager,
    Problem,
    Results,
    TextColor,
    clip_to_bounds,
    compute_Ji,
    format_vars,
    pareto_archive_to_df,
    update_pareto,
)
from ..optimizers import get_optimizer


@dataclass
class BaseSubsystem(ABC):
    """Base class for all subsystem implementations."""

    problem: Problem
    z_idxs: np.ndarray
    x_idxs: np.ndarray
    y_idxs: np.ndarray
    y_coupled_idxs: List[np.ndarray]

    def __post_init__(self):
        """Initialize subsystem state."""
        self._initialize_state()

    def _initialize_state(self):
        """Common initialization for all subsystems."""
        self.z_idxs = np.array(self.z_idxs)
        self.x_idxs = np.array(self.x_idxs)
        self.y_idxs = np.array(self.y_idxs)
        self.z_under_i = None
        self.x_under_i = None
        self.y_i = None
        self.best_J_i = np.inf
        self.best_h = np.inf
        self.iteration_history = []

    def _extract_targets(self, z_bar, x_bar, y_bar):
        """Extract subsystem-specific targets from system variables."""
        x_bar_i = x_bar[self.x_idxs] if len(self.x_idxs) > 0 else np.array([])
        y_bar_i = y_bar[self.y_idxs]
        z_bar_i = z_bar[self.z_idxs]
        y_bar_coupled = []
        for coupled_idx_array in self.y_coupled_idxs:
            for idx in coupled_idx_array:
                y_bar_coupled.append(y_bar[idx])
        y_bar_coupled = np.array(y_bar_coupled) if y_bar_coupled else np.array([])
        return (z_bar_i, x_bar_i, y_bar_i, y_bar_coupled)

    def _build_objective(
        self,
        z_bar_i,
        x_bar_i,
        y_bar_i,
        y_bar_coupled,
        _eval_history,
        _best_result,
        system_iter,
        global_start_time,
    ):
        """Build objective function for subsystem optimization."""
        nz = len(self.z_idxs)
        nx = len(self.x_idxs)

        def objective(local_vars):
            """Objective: discrepancy J_i"""
            nonlocal _eval_history, _best_result
            z_under_i = local_vars[:nz]
            x_under_i = local_vars[nz:] if nx > 0 else np.array([])
            J_i_val, y_i = compute_Ji(
                self.problem.objective.func,
                z_bar_i,
                x_bar_i,
                y_bar_i,
                y_bar_coupled,
                z_under_i,
                x_under_i,
            )
            input_vars = np.concatenate([z_under_i, x_under_i, y_bar_coupled])
            _, c_vals = self.problem.evaluate(input_vars, f=False, c=True)
            c_vals_dict = {
                const.func.name: c_vals[i]
                for i, const in enumerate(self.problem.constraints)
            }
            h_total = self.problem.compute_constraint_violation(c_vals_dict)
            if J_i_val < _best_result["J_i"] or (
                J_i_val <= _best_result["J_i"] and h_total < _best_result["h_total"]
            ):
                _best_result = {
                    "x": local_vars.copy(),
                    "J_i": J_i_val,
                    "h_total": h_total,
                    "z_under_i": z_under_i.copy(),
                    "x_under_i": x_under_i.copy()
                    if len(x_under_i) > 0
                    else np.array([]),
                    "y_i": y_i.copy(),
                }
            _eval_history.append({
                "system_iter": system_iter,
                "time_since_start": time.time() - global_start_time,
                "eval_type": "objective",
                "J_i": J_i_val,
                "h_total": h_total,
                "z_under_i": z_under_i.copy(),
                "x_under_i": x_under_i.copy() if len(x_under_i) > 0 else np.array([]),
                "y_i": y_i.copy(),
            })
            self.y_i = y_i
            return J_i_val

        return objective

    def _build_constraints(self, y_bar_coupled):
        """Build constraint functions for subsystem optimization."""
        nz = len(self.z_idxs)
        nx = len(self.x_idxs)

        def constraints_func(local_vars):
            """Constraints: g_i(z_under_i, x_under_i, y_bar_coupled) >= 0"""
            z_under_i = local_vars[:nz]
            x_under_i = local_vars[nz:] if nx > 0 else np.array([])
            input_vars = np.concatenate([z_under_i, x_under_i, y_bar_coupled])
            c_vals_ineq = []
            c_vals_eq = []
            for const in self.problem.constraints:
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

    def solve(
        self,
        z_bar,
        x_bar,
        y_bar,
        optimizer: str | Callable,
        system_iter: int,
        global_start_time: float,
        maxiter: int = None,
    ):
        """Solve subsystem optimization problem."""
        start_time = time.time()
        _eval_history = []
        z_bar_i, x_bar_i, y_bar_i, y_bar_coupled = self._extract_targets(
            z_bar, x_bar, y_bar
        )
        nz = len(self.z_idxs)
        nx = len(self.x_idxs)
        _best_result = {
            "x": np.concatenate([z_bar_i, x_bar_i]),
            "J_i": np.inf,
            "h_total": np.inf,
            "z_under_i": z_bar_i.copy(),
            "x_under_i": x_bar_i.copy() if len(x_bar_i) > 0 else np.array([]),
            "y_i": None,
        }
        objective = self._build_objective(
            z_bar_i,
            x_bar_i,
            y_bar_i,
            y_bar_coupled,
            _eval_history,
            _best_result,
            system_iter,
            global_start_time,
        )
        constraints = self._build_constraints(y_bar_coupled)
        bounds = self.problem.bounds[: nz + nx]
        x0 = np.concatenate([z_bar_i, x_bar_i])
        n_vars = nz + nx
        min_evals = 2 * n_vars + 1
        if maxiter is not None and maxiter < min_evals:
            maxiter = min_evals
        from ..optimizers import get_optimizer

        if isinstance(optimizer, str):
            kwargs = (
                {"maxiter": maxiter} if optimizer != "cobyqa" else {"maxfev": maxiter}
            )
            opt = get_optimizer(optimizer, **kwargs)
            result_x, result_f = opt(
                objective, x0, bounds, constraints if self.problem.constraints else None
            )
        else:
            opt = optimizer
            result_x, result_f = opt(
                objective,
                x0,
                bounds,
                constraints if self.problem.constraints else None,
                maxiter=maxiter,
            )
        result_x = clip_to_bounds(result_x, bounds)
        self.z_under_i = result_x[:nz]
        self.x_under_i = result_x[nz:] if nx > 0 else np.array([])
        J_i_val = result_f
        input_vars = np.concatenate([self.z_under_i, self.x_under_i, y_bar_coupled])
        _, c_vals = self.problem.evaluate(input_vars, f=False, c=True)
        c_vals_dict = {
            const.func.name: c_vals[i]
            for i, const in enumerate(self.problem.constraints)
        }
        h_total = self.problem.compute_constraint_violation(c_vals_dict)
        _eval_history.append({
            "system_iter": system_iter,
            "time_since_start": time.time() - global_start_time,
            "eval_type": "final",
            "J_i": J_i_val,
            "h_total": h_total,
            "z_under_i": self.z_under_i.copy(),
            "x_under_i": self.x_under_i.copy()
            if len(self.x_under_i) > 0
            else np.array([]),
            "y_i": self.y_i.copy(),
        })
        if J_i_val < self.best_J_i or (
            J_i_val <= self.best_J_i and h_total < self.best_h
        ):
            self.best_h = h_total
            self.best_J_i = J_i_val
        self.iteration_history.extend(_eval_history)
        elapsed = time.time() - start_time
        return (
            self.z_under_i,
            self.x_under_i,
            self.y_i,
            J_i_val,
            h_total,
            elapsed,
            _eval_history,
        )


@dataclass
class BaseSystem(ABC):
    """Base system-level coordinator for CO frameworks."""

    problem: Problem
    subsystems: List[BaseSubsystem]

    def __post_init__(self):
        """Initialize system state."""
        self._initialize_state()

    def _initialize_state(self):
        """Common state initialization."""
        self.z_bar = None
        self.x_bar = None
        self.y_bar = None
        self.iteration_history = []
        self.best_J_total = np.inf
        self.best_h_total = np.inf
        self.best_f_sys = np.inf

    def _build_objective(
        self,
        sys_vars,
        _eval_history,
        _best_result,
        system_iter,
        global_start_time,
        **kwargs,
    ):
        """Build system-level objective function."""
        nz = len(self.z_bar)
        nx = len(self.x_bar)

        def objective(sys_vars_inner):
            """System objective"""
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
            return f_sys_val

        return objective

    def _build_constraints(self, **kwargs):
        """Build system-level constraints."""
        nz = len(self.z_bar)
        nx = len(self.x_bar)
        epsilon_J = kwargs.get("epsilon_J", 1.0)

        def constraints_func(sys_vars):
            """System constraints including J_i consistency"""
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
                c_vals_ineq.append(epsilon_J - J_i_val)
            return (np.array(c_vals_ineq), np.array(c_vals_eq))

        return {
            "ineq": lambda x: constraints_func(x)[0],
            "eq": lambda x: constraints_func(x)[1],
        }

    def solve(
        self,
        optimizer: str | Callable,
        system_iter: int,
        global_start_time: float,
        maxiter: int = None,
        **kwargs,
    ):
        """Solve system-level optimization."""
        _eval_history = []
        nz = len(self.z_bar)
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
        x0 = np.concatenate([self.z_bar, self.x_bar, self.y_bar])
        n_vars = len(x0)
        min_evals = 2 * n_vars + 1
        if maxiter is not None and maxiter < min_evals:
            maxiter = min_evals
        if isinstance(optimizer, str):
            opt_kwargs = (
                {"maxiter": maxiter} if optimizer != "cobyqa" else {"maxfev": maxiter}
            )
            opt = get_optimizer(optimizer, **opt_kwargs)
        else:
            opt = optimizer
        result_x, f_sys_val = opt(objective, x0, self.problem.bounds, constraints)
        result_x = clip_to_bounds(result_x, self.problem.bounds)
        self.z_bar = result_x[:nz]
        self.x_bar = result_x[nz : nz + nx]
        self.y_bar = result_x[nz + nx :]
        _, c_vals = self.problem.evaluate(result_x, f=False, c=True)
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
class BaseSolver(ABC):
    """Base solver for all CO framework variants."""

    system: BaseSystem
    subsystem_optimizer: str | Callable
    system_optimizer: str | Callable
    epsilon_J: float = 1e-06
    epsilon_h: float = 1e-06
    budget: BudgetManager | int = 100
    max_iter: int | None = None
    max_eval: int | None = None
    cache_dir: Path | None = None
    name: str = "Optimization Problem"
    solver: str = "CO"

    def __post_init__(self):
        """Initialize solver components."""
        if self.max_eval is not None:
            self.budget = self.max_eval
        self._initialize_budget()
        self._initialize_history()
        self._initialize_state()

    def _initialize_budget(self):
        """Convert budget to BudgetManager if needed."""
        if isinstance(self.budget, int):
            self.budget = BudgetManager(mode="shared", total_budget=self.budget)

    def _initialize_history(self):
        """Set up history dictionary structure."""
        self._history = {
            "iter": [],
            "eval": [],
            "tss": [],
            "improved": [],
            "level": [],
            "f_sys": [],
            "h_total": [],
            "J_total": [],
            "z": [],
            "x": [],
            "y": [],
        }
        for idx, _ in enumerate(self.system.subsystems):
            self._history.update({f"J{idx + 1:02d}": [], f"h{idx + 1:02d}": []})
        self._add_custom_history_columns()

    def _add_custom_history_columns(self):
        """Override to add framework-specific history columns."""
        pass

    def _initialize_state(self):
        """Common state initialization."""
        self._best_results = BestSolution()
        self._pareto_archive = []
        self._converged = False
        self._budget_exhausted = False
        self._iter = 0
        self._start_time = None
        self._eval = 0
        self.cache_dir = None

    def _save_iteration_cache(self):
        """Save the current iteration history to disk."""
        if self.cache_dir is None:
            return
        history_path = self.cache_dir / "state.pkl"
        temp_path = history_path.with_suffix(".tmp")
        import pandas as pd

        pd.DataFrame(self._history).to_pickle(temp_path)
        temp_path.replace(history_path)

    def initialize(self, z_bar0, x_bar0, y_bar0, z_under0, x_under0):
        """Initialize solver for optimization."""
        self._start_time = time.time()
        cache_identifier = f"{self.solver}-{self.name}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.cache_dir = Path.cwd() / ".mdo_cache" / cache_identifier
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        z_bar0 = np.asarray(z_bar0)
        x_bar0 = np.asarray(x_bar0)
        y_bar0 = np.asarray(y_bar0)
        z_under0 = np.asarray(z_under0)
        x_under0 = np.asarray(x_under0)
        self.system.z_bar = z_bar0.copy()
        self.system.x_bar = x_bar0.copy()
        self.system.y_bar = y_bar0.copy()
        for subsystem in self.system.subsystems:
            subsystem.z_under_i = z_under0[subsystem.z_idxs].copy()
            subsystem.x_under_i = (
                x_under0[subsystem.x_idxs].copy()
                if len(subsystem.x_idxs) > 0
                else np.array([])
            )
        self.budget.initialize(len(self.system.subsystems))
        self._custom_initialize()
        self._record_initial_state()

    def _record_initial_state(self):
        """Record the initial guess in the history as Iteration 0."""
        input_vars = np.concatenate([
            self.system.z_bar,
            self.system.x_bar,
            self.system.y_bar,
        ])
        f_sys, c_vals = self.system.problem.evaluate(input_vars, f=True, c=True)
        c_vals_dict = {
            const.func.name: c_vals[i]
            for i, const in enumerate(self.system.problem.constraints)
        }
        h_sys = self.system.problem.compute_constraint_violation(c_vals_dict)
        h_sub = 0.0
        J_total = 0.0
        subsystem_metrics = []
        for subsystem in self.system.subsystems:
            x_bar_i = (
                self.system.x_bar[subsystem.x_idxs]
                if len(subsystem.x_idxs) > 0
                else np.array([])
            )
            y_bar_i = self.system.y_bar[subsystem.y_idxs]
            y_bar_coupled = []
            for coupled_idx_array in subsystem.y_coupled_idxs:
                for idx in coupled_idx_array:
                    y_bar_coupled.append(self.system.y_bar[idx])
            y_bar_coupled = np.array(y_bar_coupled) if y_bar_coupled else np.array([])
            sub_input_vars = np.concatenate([
                subsystem.z_under_i,
                subsystem.x_under_i,
                y_bar_coupled,
            ])
            _, sub_c_vals = subsystem.problem.evaluate(sub_input_vars, f=False, c=True)
            sub_c_vals_dict = {
                const.func.name: sub_c_vals[i]
                for i, const in enumerate(subsystem.problem.constraints)
            }
            h_i = subsystem.problem.compute_constraint_violation(sub_c_vals_dict)
            h_sub += h_i
            J_i_val, _ = compute_Ji(
                subsystem.problem.objective.func,
                self.system.z_bar[subsystem.z_idxs],
                x_bar_i,
                y_bar_i,
                y_bar_coupled,
                subsystem.z_under_i,
                subsystem.x_under_i,
            )
            J_total += J_i_val
            subsystem_metrics.append((h_i, J_i_val))
        h_total = h_sys + h_sub
        self._best_results.update(
            z_new=self.system.z_bar.copy(),
            x_new=self.system.x_bar.copy(),
            y_new=self.system.y_bar.copy(),
            f_new=f_sys,
            h_new=h_total,
            J_new=J_total,
            epsilon_J=self.epsilon_J,
            epsilon_h=self.epsilon_h,
        )
        self._history["iter"].append(0)
        self._history["eval"].append(0)
        self._history["improved"].append(True)
        self._history["tss"].append(0.0)
        self._history["level"].append("Initial Guess")
        self._history["f_sys"].append(f_sys)
        self._history["h_total"].append(h_total)
        self._history["J_total"].append(J_total)
        self._history["z"].append(self.system.z_bar.copy())
        self._history["x"].append(self.system.x_bar.copy())
        self._history["y"].append(self.system.y_bar.copy())
        for idx, (h_i, J_i_val) in enumerate(subsystem_metrics):
            self._history[f"h{idx + 1:02d}"].append(h_i)
            self._history[f"J{idx + 1:02d}"].append(J_i_val)
        self._record_system_custom_history({})

    def _custom_initialize(self):
        """Override for framework-specific initialization."""
        pass

    def _solve_subsystems(self):
        """Solve all subsystems - common pattern."""
        J_stars = []
        h_total = 0.0
        for idx, subsystem in enumerate(self.system.subsystems):
            if self.budget.is_subsystem_exhausted(idx):
                J_stars.append(
                    subsystem.best_J_i if subsystem.best_J_i < np.inf else 0.0
                )
                continue
            maxiter = self.budget.get_subsystem_maxiter(idx)
            if maxiter <= 0:
                J_stars.append(
                    subsystem.best_J_i if subsystem.best_J_i < np.inf else 0.0
                )
                continue
            _, _, _, J_i_s, h_sub, _, eval_hist = subsystem.solve(
                self.system.z_bar,
                self.system.x_bar,
                self.system.y_bar,
                self.subsystem_optimizer,
                self._iter,
                self._start_time,
                maxiter=maxiter,
            )
            n_evals = len(eval_hist)
            J_stars.append(J_i_s)
            h_total += h_sub
            self.budget.record_subsystem_evals(idx, n_evals)
            self._record_subsystem_history(subsystem, idx, n_evals)
            if self.budget.is_total_exhausted():
                self._budget_exhausted = True
                break
        return (J_stars, h_total)

    def _record_subsystem_history(self, subsystem, idx, n_evals):
        """Record subsystem evaluation history."""
        for eval_data in subsystem.iteration_history[-n_evals:]:
            if eval_data.get("eval_type") == "final":
                continue
            self._eval += 1
            self._history["iter"].append(self._iter)
            self._history["eval"].append(self._eval)
            self._history["improved"].append(False)
            self._history["tss"].append(eval_data["time_since_start"])
            self._history["level"].append(f"Subsystem {idx + 1:02d}")
            self._history["z"].append(eval_data["z_under_i"].copy())
            self._history["x"].append(eval_data["x_under_i"].copy())
            self._history["y"].append(eval_data["y_i"].copy())
            self._history["f_sys"].append(np.nan)
            self._history["h_total"].append(np.nan)
            self._history["J_total"].append(np.nan)
            for jdx, _ in enumerate(self.system.subsystems):
                if jdx == idx:
                    self._history[f"h{jdx + 1:02d}"].append(eval_data["h_total"])
                    self._history[f"J{jdx + 1:02d}"].append(eval_data["J_i"])
                else:
                    self._history[f"h{jdx + 1:02d}"].append(np.nan)
                    self._history[f"J{jdx + 1:02d}"].append(np.nan)
            self._record_subsystem_custom_history(eval_data)

    def _record_subsystem_custom_history(self, eval_data):
        """Override to record framework-specific subsystem history."""
        pass

    def _pre_system_solve_update(self):
        """Override for parameter updates before system solve."""
        pass

    def _get_system_solve_kwargs(self):
        """Override to pass framework-specific parameters to system.solve()."""
        return {"epsilon_J": self.epsilon_J}

    def _compute_system_J_total(self):
        """Compute J_total at system level using current system targets."""
        J_total = 0.0
        for subsystem in self.system.subsystems:
            x_bar_i = (
                self.system.x_bar[subsystem.x_idxs]
                if len(subsystem.x_idxs) > 0
                else np.array([])
            )
            y_bar_i = self.system.y_bar[subsystem.y_idxs]
            y_bar_coupled = []
            for coupled_idx_array in subsystem.y_coupled_idxs:
                for idx in coupled_idx_array:
                    y_bar_coupled.append(self.system.y_bar[idx])
            y_bar_coupled = np.array(y_bar_coupled) if y_bar_coupled else np.array([])
            J_i_val, _ = compute_Ji(
                subsystem.problem.objective.func,
                self.system.z_bar[subsystem.z_idxs],
                x_bar_i,
                y_bar_i,
                y_bar_coupled,
                subsystem.z_under_i,
                subsystem.x_under_i,
            )
            J_total += J_i_val
        return J_total

    def _solve_system(self, J_total):
        """Solve system level - common pattern."""
        if self.budget.is_system_exhausted():
            self._budget_exhausted = True
            return (None, None, None)
        system_maxiter = self.budget.get_system_maxiter()
        if system_maxiter <= 0:
            self._budget_exhausted = True
            return (None, None, None)
        system_kwargs = self._get_system_solve_kwargs()
        _, _, _, f_sys, h_sys, n_sys_evals = self.system.solve(
            self.system_optimizer,
            self._iter,
            self._start_time,
            maxiter=system_maxiter,
            **system_kwargs,
        )
        self.budget.record_system_evals(n_sys_evals)
        return (f_sys, h_sys, n_sys_evals)

    def _record_system_history(self, n_sys_evals, J_total, h_sub=0.0):
        """Record system evaluation history."""
        n_subsystems = len(self.system.subsystems)
        for eval_data in self.system.iteration_history[-n_sys_evals:]:
            if eval_data.get("eval_type") == "final":
                continue
            self._eval += n_subsystems
            h_total = eval_data["h_total"] + h_sub
            improved = self._best_results.update(
                z_new=eval_data["z"].copy(),
                x_new=eval_data["x"].copy(),
                y_new=eval_data["y"].copy(),
                f_new=eval_data["f_sys"],
                h_new=h_total,
                J_new=J_total,
                epsilon_J=self.epsilon_J,
                epsilon_h=self.epsilon_h,
            )
            self._pareto_archive = update_pareto(
                self._pareto_archive,
                f=eval_data["f_sys"],
                h_total=h_total,
                J_total=J_total,
                z=eval_data["z"],
                x=eval_data["x"],
                y=eval_data["y"],
            )
            self._history["iter"].append(self._iter)
            self._history["eval"].append(self._eval)
            self._history["improved"].append(improved)
            self._history["tss"].append(eval_data["time_since_start"])
            self._history["level"].append("System")
            self._history["f_sys"].append(eval_data["f_sys"])
            self._history["h_total"].append(h_total)
            self._history["J_total"].append(J_total)
            self._history["z"].append(eval_data["z"].copy())
            self._history["x"].append(eval_data["x"].copy())
            self._history["y"].append(eval_data["y"].copy())
            for jdx, _ in enumerate(self.system.subsystems):
                self._history[f"h{jdx + 1:02d}"].append(np.nan)
                self._history[f"J{jdx + 1:02d}"].append(np.nan)
            self._record_system_custom_history(eval_data)

    def _record_system_custom_history(self, eval_data):
        """Override to record framework-specific system history."""
        pass

    def _check_convergence(self, J_total, h_total=0.0):
        """Check convergence criteria."""
        return J_total <= self.epsilon_J and h_total <= self.epsilon_h

    def iterate(self):
        """Execute one complete iteration (subsystems + system)."""
        f_sys, J_total, h_total, improved = (None, np.inf, np.inf, 0)
        J_stars, h_total = self._solve_subsystems()
        if self._budget_exhausted:
            J_total = np.sum(np.array(J_stars) if J_stars else np.array([np.inf]))
            return (f_sys, J_total, h_total, improved)
        J_total = np.sum(np.array(J_stars))
        self._pre_system_solve_update()
        f_sys, h_sys, n_sys_evals = self._solve_system(J_total)
        if f_sys is None:
            return (f_sys, J_total, h_total, improved)
        J_total = self._compute_system_J_total()
        h_sub = h_total
        h_total = h_sub + h_sys
        self._record_system_history(n_sys_evals, J_total, h_sub)
        improved = self._history["improved"][-1] if self._history["improved"] else 0
        if self.budget.is_total_exhausted():
            self._budget_exhausted = True
        if self._check_convergence(J_total, h_total):
            self._converged = True
        return (f_sys, J_total, h_total, improved)

    def _print_budget_info(self):
        """Print budget allocation information."""
        print(f"Budget mode: {self.budget.mode}")
        print(f"Total budget: {self.budget.total_budget}")
        if self.budget.mode in ["weighted", "fixed"]:
            print(f"Subsystem allocated: {self.budget._subsystem_allocated}")
            print(f"System allocated: {self.budget._system_allocated}")
            if self.budget.iteration_ratio:
                print(f"Iteration ratio: {self.budget.iteration_ratio}")
        self._print_custom_budget_info()
        print()

    def _print_custom_budget_info(self):
        """Override to print framework-specific configuration."""
        pass

    def _print_iteration(self, improved, f_sys, h_total, J_total, iterend, iterdur):
        """Print iteration progress."""
        evals = self._history["eval"][-1] if self._history["eval"] else self._eval
        if improved:
            improved_str = f"{TextColor.grn}Y{TextColor.clr}"
        else:
            improved_str = f"{TextColor.red}N{TextColor.clr}"
        if h_total < 1:
            h_str = f"{TextColor.grn}{format_vars(h_total, 3)}{TextColor.clr}"
        else:
            h_str = f"{TextColor.red}{format_vars(h_total, 3)}{TextColor.clr}"
        if J_total < 1:
            J_str = f"{TextColor.grn}{format_vars(J_total, 3)}{TextColor.clr}"
        else:
            J_str = f"{TextColor.red}{format_vars(J_total, 3)}{TextColor.clr}"
        print(
            f"iter={self._iter:>{len(str(self.budget.total_budget))}} |",
            f"eval={evals:>{len(str(self.budget.total_budget))}} /",
            f"{self.budget.total_budget:>{len(str(self.budget.total_budget))}} |",
            f"Improved={improved_str} |",
            f"h_total={h_str} |",
            f"J_total={J_str} |",
            f"f={format_vars(f_sys, 3)} |",
            f"TSince={iterend:>8.2f}s |",
            f"IterDur={iterdur:>7.3f}s",
        )

    def solve(self, z_bar0, x_bar0, y_bar0, z_under0, x_under0):
        """Solve the MDO problem."""
        self.initialize(z_bar0, x_bar0, y_bar0, z_under0, x_under0)
        self._converged = False
        self._budget_exhausted = False
        self._iter = 0
        self._eval = 0
        J_total = np.inf
        h_total = np.inf
        f_sys = np.inf
        iterend = 0.0
        self._print_budget_info()
        while not self.budget.is_total_exhausted():
            if self.max_iter is not None and self._iter >= self.max_iter:
                print(f"\nMaximum iterations ({self.max_iter}) reached.")
                break
            self._iter += 1
            iterstart = time.time() - self._start_time
            f_sys, J_total, h_total, improved = self.iterate()
            iterend = time.time() - self._start_time
            iterdur = iterend - iterstart
            if f_sys is None:
                if self._budget_exhausted:
                    print("\nEvaluation budget exhausted.")
                    status = self.budget.get_status()
                    print(
                        f"  Total used: {status['total_used']} / {status['total_budget']}"
                    )
                break
            self._print_iteration(improved, f_sys, h_total, J_total, iterend, iterdur)
            self._save_iteration_cache()
            if self._converged:
                print("\nConvergence criteria met.")
                break
            if self._budget_exhausted:
                print("\nEvaluation budget exhausted.")
                break
        import pandas as pd

        history_df = pd.DataFrame(self._history)
        history_df["identifier"] = self.name
        history_df["solver"] = self.solver
        history_df["epsilon_h"] = self.epsilon_h
        history_df["epsilon_J"] = self.epsilon_J
        pareto_df = pareto_archive_to_df(self._pareto_archive)
        pareto_df["identifier"] = self.name
        pareto_df["solver"] = self.solver
        pareto_df["epsilon_h"] = self.epsilon_h
        pareto_df["epsilon_J"] = self.epsilon_J
        return Results(
            best=self._best_results,
            converged=self._converged,
            iterations=self._iter,
            evaluations=self._eval,
            elapsed_time=iterend,
            history=history_df,
            pareto=pareto_df,
        )
