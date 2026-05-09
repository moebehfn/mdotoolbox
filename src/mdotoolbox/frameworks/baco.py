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

"""src/mdotoolbox/frameworks/baco.py"""

from __future__ import annotations

import copy
import inspect
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from smt.sampling_methods import LHS

if TYPE_CHECKING:
    from ..surrogates import TorchGPConfig
from ..core import (
    DoE,
    Problem,
    Results,
    TextColor,
    assess_progress,
    clip_to_bounds,
    compute_Ji,
    format_vars,
    pareto_archive_to_df,
    type_check,
    update_pareto,
)
from ..optimizers import get_optimizer
from ..surrogates import SMTGPConfig, build_surrogate
from .base_classes import BaseSolver


def _run_one_start_subsystem(
    x0,
    opt,
    acq_func,
    bounds,
    gp_J_i,
    gp_g_i,
    constraints,
    best_J_i,
    problem_tol,
    template_J_i,
    template_g_i,
    j_i_offsets,
    nz,
    nx,
):
    """One acquisition multi-start for BACOSubsystem. Self-contained - no closure captures."""
    _state = {
        "last_x": np.full(nz + nx, np.nan),
        "mu": {},
        "var": {},
        "buffer_J_i": template_J_i.copy(),
        "buffer_g_i": template_g_i.copy(),
    }
    off_z_bar, off_x_bar_i, off_y_bar_i, off_y_bar_coupled, off_z_under_i = j_i_offsets

    def _check_cache(local_vars):
        if not np.array_equal(local_vars, _state["last_x"]):
            np.copyto(_state["last_x"], local_vars)
            _state["mu"].clear()
            _state["var"].clear()

    def _cached_predict(gp, x_flat, cache_key):
        if cache_key in _state["mu"]:
            return (_state["mu"][cache_key], _state["var"][cache_key])
        x_2d = x_flat.reshape(1, -1)
        mu = gp.predict_values(x_2d).flatten()[0]
        var = gp.predict_variances(x_2d).flatten()[0]
        _state["mu"][cache_key] = mu
        _state["var"][cache_key] = var
        return (mu, var)

    def acquisition(local_vars):
        _check_cache(local_vars)
        buf = _state["buffer_J_i"]
        buf[off_y_bar_coupled:off_z_under_i] = local_vars[:nz]
        if nx > 0:
            buf[off_z_under_i:] = local_vars[nz:]
        mu, var = _cached_predict(gp_J_i, buf, "J_i")
        if not np.isfinite(mu) or not np.isfinite(var):
            warnings.warn(
                f"GP prediction returned non-finite value (mu={mu}, var={var}). Treating this start as failed.",
                UserWarning,
                stacklevel=3,
            )
            return np.inf
        std = np.sqrt(max(var, 0.0))
        alpha = acq_func(np.array([mu]), np.array([std]), best_J_i)[0]
        return -alpha

    def constraints_mean_func(local_vars):
        _check_cache(local_vars)
        buf = _state["buffer_g_i"]
        buf[:nz] = local_vars[:nz]
        if nx > 0:
            buf[nz : nz + nx] = local_vars[nz:]
        c_means_ineq = []
        c_means_eq = []
        for const in constraints:
            mu_c, _ = _cached_predict(gp_g_i[const.func.name], buf, const.func.name)
            if const.ctype == "ge":
                c_means_ineq.append(mu_c - const.value)
            elif const.ctype == "le":
                c_means_ineq.append(-mu_c + const.value)
            elif const.ctype == "eq":
                c_means_eq.append(mu_c - const.value)
        return (np.array(c_means_ineq), np.array(c_means_eq))

    if constraints:
        _constraints_arg = {"ineq": lambda x: constraints_mean_func(x)[0]}
        if any(c.ctype == "eq" for c in constraints):
            _constraints_arg["eq"] = lambda x: constraints_mean_func(x)[1]
    else:
        _constraints_arg = None
    try:
        return opt(acquisition, x0, bounds, _constraints_arg)
    except Exception:
        return (None, np.inf)


def _run_one_start_system(
    x0,
    opt,
    acq_func,
    bounds,
    gp_f,
    gp_c,
    gp_J_i_subsystems,
    subsystem_masks,
    constraints,
    epsilon_J,
    f_min,
    nsvars,
):
    """One acquisition multi-start for BACOSystem. Self-contained - no closure captures."""
    _state = {
        "last_x": np.full(nsvars, np.nan),
        "mu": {},
        "var": {},
        "buffers_J_i": [m["buffer_template"].copy() for m in subsystem_masks],
    }

    def _check_cache(sys_vars):
        if not np.array_equal(sys_vars, _state["last_x"]):
            np.copyto(_state["last_x"], sys_vars)
            _state["mu"].clear()
            _state["var"].clear()

    def _cached_predict(gp, x_flat, cache_key):
        if cache_key in _state["mu"]:
            return (_state["mu"][cache_key], _state["var"][cache_key])
        x_2d = x_flat.reshape(1, -1)
        mu = gp.predict_values(x_2d).flatten()[0]
        var = gp.predict_variances(x_2d).flatten()[0]
        _state["mu"][cache_key] = mu
        _state["var"][cache_key] = var
        return (mu, var)

    def acquisition(sys_vars, _f_min=f_min):
        _check_cache(sys_vars)
        mu, var = _cached_predict(gp_f, sys_vars, "f")
        if not np.isfinite(mu) or not np.isfinite(var):
            warnings.warn(
                f"GP prediction returned non-finite value (mu={mu}, var={var}). Treating this start as failed.",
                UserWarning,
                stacklevel=3,
            )
            return np.inf
        std = np.sqrt(max(var, 0.0))
        alpha = acq_func(np.array([mu]), np.array([std]), _f_min)[0]
        return -alpha

    def constraints_mean_func(sys_vars):
        _check_cache(sys_vars)
        c_means_ineq = []
        c_means_eq = []
        for const in constraints:
            mu_c, _ = _cached_predict(gp_c[const.func.name], sys_vars, const.func.name)
            if const.ctype == "ge":
                c_means_ineq.append(mu_c - const.value)
            elif const.ctype == "le":
                c_means_ineq.append(-mu_c + const.value)
            elif const.ctype == "eq":
                c_means_eq.append(mu_c - const.value)
        for i, mask in enumerate(subsystem_masks):
            buf = _state["buffers_J_i"][i]
            buf[: mask["n_dynamic"]] = sys_vars[mask["src_indices"]]
            mu_J_i, _ = _cached_predict(gp_J_i_subsystems[i], buf, f"J_i_{i}")
            c_means_ineq.append(epsilon_J - mu_J_i)
        return (np.array(c_means_ineq), np.array(c_means_eq))

    _constraints_arg = {"ineq": lambda x: constraints_mean_func(x)[0]}
    if any(c.ctype == "eq" for c in constraints):
        _constraints_arg["eq"] = lambda x: constraints_mean_func(x)[1]
    try:
        return opt(acquisition, x0, bounds, _constraints_arg)
    except Exception:
        return (None, np.inf)


@dataclass
class BACOSubsystem:
    """Subsystem for Bayesian Collaborative Optimization framework."""

    problem: Problem
    z_idxs: np.ndarray
    x_idxs: np.ndarray
    y_idxs: np.ndarray
    y_coupled_idxs: list[np.ndarray]
    surrogate_config: SMTGPConfig | TorchGPConfig | None = None

    def __post_init__(self):
        self.z_idxs = np.array(self.z_idxs)
        self.x_idxs = np.array(self.x_idxs)
        self.y_idxs = np.array(self.y_idxs)
        if self.surrogate_config is None:
            self.surrogate_config = SMTGPConfig()
        self.doe_J_i = None
        self.doe_g_i = None
        self.gp_J_i = None
        self.gp_g_i = {}
        self.z_under_i = None
        self.x_under_i = None
        self.y_i = None
        self.best_J_i = np.inf
        self.best_h = np.inf

    def initialize_doe(self, n_samples, z_bar, x_bar, y_bar):
        """Initialize Design of Experiments using Latin Hypercube Sampling."""
        nz = len(self.z_idxs)
        nx = len(self.x_idxs)
        if type_check(n_samples, Callable):
            n_samples = n_samples(nz + nx)
        elif type_check(n_samples, (int, float)) and n_samples < 0:
            warnings.warn(
                "n should be a positive integer, converting to absolute value.",
                stacklevel=2,
            )
            n_samples = abs(n_samples)
        elif type_check(n_samples, float):
            n_samples = int(n_samples)
        elif not type_check(n_samples, int):
            raise ValueError(
                "n must be an integer or a callable function of the number of variables."
            )
        if n_samples < 2:
            raise ValueError(
                f"n_samples must be >= 2 for KPLSK training (got {n_samples}). KPLSK requires at least 2 training points."
            )
        from smt.sampling_methods import LHS

        bounds_local = self.problem.bounds[: nz + nx]
        if self.z_under_i is not None and self.x_under_i is not None:
            initial_guess_local = np.concatenate([self.z_under_i, self.x_under_i])
            lhs_samples = LHS(xlimits=bounds_local, seed=42)(n_samples)
            samples = np.vstack([initial_guess_local, lhs_samples])
        else:
            samples = LHS(xlimits=bounds_local, seed=42)(n_samples)
        x_bar_i = x_bar[self.x_idxs] if len(self.x_idxs) > 0 else np.array([])
        y_bar_i = y_bar[self.y_idxs]
        y_bar_coupled = []
        for coupled_idx_array in self.y_coupled_idxs:
            for idx in coupled_idx_array:
                y_bar_coupled.append(y_bar[idx])
        y_bar_coupled = np.array(y_bar_coupled) if y_bar_coupled else np.array([])
        J_i_vals = []
        yi_vals = []
        h_i_vals = []
        x_J_i = []
        x_g_i = []
        g_i_vals = {const.func.name: [] for const in self.problem.constraints}
        for sample in samples:
            z_under_i = sample[:nz]
            x_under_i = sample[nz:] if nx > 0 else np.array([])
            J_i_val, y_i = compute_Ji(
                self.problem.objective.func,
                z_bar[self.z_idxs],
                x_bar_i,
                y_bar_i,
                y_bar_coupled,
                z_under_i,
                x_under_i,
            )
            if not np.isfinite(J_i_val):
                warnings.warn(
                    "True function evaluation returned NaN/Inf. Skipping DoE update for this sample.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            J_i_vals.append(J_i_val)
            yi_vals.append(y_i.item() if y_i.size == 1 else y_i)
            x_entry = np.concatenate([
                z_bar,
                x_bar_i,
                y_bar_i,
                y_bar_coupled,
                z_under_i,
                x_under_i,
            ])
            x_J_i.append(x_entry)
            input_vars = np.concatenate([z_under_i, x_under_i, y_bar_coupled])
            _, c_vals = self.problem.evaluate(input_vars, f=False, c=True)
            c_vals_dict = {
                const.func.name: c_vals[i]
                for i, const in enumerate(self.problem.constraints)
            }
            h_i_val = self.problem.compute_constraint_violation(c_vals_dict)
            h_i_vals.append(h_i_val)
            for i, const in enumerate(self.problem.constraints):
                g_i_vals[const.func.name].append(c_vals[i])
            if self.problem.constraints:
                x_g_i.append(np.concatenate([z_under_i, x_under_i, y_bar_coupled]))
        self.doe_J_i = DoE(
            x=np.array(x_J_i),
            y={"J_i": np.array(J_i_vals), "y_i": np.array(yi_vals)},
            constraint_violation=np.array(h_i_vals),
        )
        if self.problem.constraints:
            g_i_y = {name: np.array(vals) for name, vals in g_i_vals.items()}
            self.doe_g_i = DoE(x=np.array(x_g_i), y=g_i_y)

    def build_surrogate_J_i(self):
        """Train Gaussian Process surrogate for discrepancy J_i."""
        self.gp_J_i = build_surrogate(
            self.doe_J_i, y_key="J_i", config=self.surrogate_config
        )

    def build_surrogate_g_i(self):
        """Train Gaussian Process surrogates for all subsystem constraints."""
        if self.problem.constraints:
            for const in self.problem.constraints:
                self.gp_g_i[const.func.name] = build_surrogate(
                    self.doe_g_i, y_key=const.func.name, config=self.surrogate_config
                )

    def solve_acquisition(
        self,
        z_bar,
        x_bar,
        y_bar,
        optimizer: str | Callable,
        acq_func: Callable,
        n_multistart: int = 10,
    ):
        """Optimize acquisition function to find next evaluation point."""
        _start_time = time.time()
        x_bar_i = x_bar[self.x_idxs] if len(self.x_idxs) > 0 else np.array([])
        y_bar_i = y_bar[self.y_idxs]
        y_bar_coupled = []
        for coupled_idx_array in self.y_coupled_idxs:
            for idx in coupled_idx_array:
                y_bar_coupled.append(y_bar[idx])
        y_bar_coupled = np.array(y_bar_coupled) if y_bar_coupled else np.array([])
        nz = len(self.z_idxs)
        nx = len(self.x_idxs)
        _j_i_off_z_bar = len(z_bar)
        _j_i_off_x_bar_i = _j_i_off_z_bar + len(x_bar_i)
        _j_i_off_y_bar_i = _j_i_off_x_bar_i + len(y_bar_i)
        _j_i_off_y_bar_coupled = _j_i_off_y_bar_i + len(y_bar_coupled)
        _j_i_off_z_under_i = _j_i_off_y_bar_coupled + nz
        _j_i_size = _j_i_off_z_under_i + nx
        _template_J_i = np.empty(_j_i_size)
        _template_J_i[:_j_i_off_z_bar] = z_bar
        if len(x_bar_i) > 0:
            _template_J_i[_j_i_off_z_bar:_j_i_off_x_bar_i] = x_bar_i
        _template_J_i[_j_i_off_x_bar_i:_j_i_off_y_bar_i] = y_bar_i
        if len(y_bar_coupled) > 0:
            _template_J_i[_j_i_off_y_bar_i:_j_i_off_y_bar_coupled] = y_bar_coupled
        _g_i_size = nz + nx + len(y_bar_coupled)
        _template_g_i = np.empty(_g_i_size)
        if len(y_bar_coupled) > 0:
            _template_g_i[nz + nx :] = y_bar_coupled
        if self.best_J_i == np.inf:
            self.best_J_i = self.doe_J_i.get_f_min(
                objective_key="J_i", tol=self.problem.tol
            )
        bounds = self.problem.bounds[: nz + nx]
        from smt.sampling_methods import LHS

        lhs_starts = LHS(xlimits=bounds, seed=42)(n_multistart - 1)
        x0_default = np.concatenate([z_bar, x_bar_i])
        start_points = [x0_default] + [lhs_starts[i] for i in range(n_multistart - 1)]
        from ..optimizers import get_optimizer

        if isinstance(optimizer, str):
            opt = get_optimizer(optimizer)
        else:
            opt = optimizer
        from joblib import Parallel, delayed

        if n_multistart >= 4:
            start_results = Parallel(n_jobs=-1, backend="loky")(
                delayed(_run_one_start_subsystem)(
                    x0,
                    opt,
                    acq_func,
                    bounds,
                    self.gp_J_i,
                    self.gp_g_i,
                    self.problem.constraints,
                    self.best_J_i,
                    self.problem.tol,
                    _template_J_i.copy(),
                    _template_g_i.copy(),
                    (
                        _j_i_off_z_bar,
                        _j_i_off_x_bar_i,
                        _j_i_off_y_bar_i,
                        _j_i_off_y_bar_coupled,
                        _j_i_off_z_under_i,
                    ),
                    nz,
                    nx,
                )
                for x0 in start_points
            )
        else:
            start_results = [
                _run_one_start_subsystem(
                    x0,
                    opt,
                    acq_func,
                    bounds,
                    self.gp_J_i,
                    self.gp_g_i,
                    self.problem.constraints,
                    self.best_J_i,
                    self.problem.tol,
                    _template_J_i.copy(),
                    _template_g_i.copy(),
                    (
                        _j_i_off_z_bar,
                        _j_i_off_x_bar_i,
                        _j_i_off_y_bar_i,
                        _j_i_off_y_bar_coupled,
                        _j_i_off_z_under_i,
                    ),
                    nz,
                    nx,
                )
                for x0 in start_points
            ]
        best_result = None
        best_acq_val = np.inf
        for result_x, acq_val in start_results:
            if result_x is not None and acq_val < best_acq_val:
                best_acq_val = acq_val
                best_result = result_x
        if best_result is None:
            warnings.warn(
                f"All {len(start_points)} multi-start acquisition attempts failed. Falling back to x0_default. Check optimizer and constraint configuration.",
                UserWarning,
                stacklevel=2,
            )
            best_result = x0_default
        best_result = clip_to_bounds(best_result, bounds)
        self.z_under_i = best_result[:nz]
        self.x_under_i = best_result[nz:] if nx > 0 else np.array([])
        J_i_val, self.y_i = compute_Ji(
            self.problem.objective.func,
            z_bar[self.z_idxs],
            x_bar_i,
            y_bar_i,
            y_bar_coupled,
            self.z_under_i,
            self.x_under_i,
        )
        input_vars = np.concatenate([self.z_under_i, self.x_under_i, y_bar_coupled])
        _, c_vals = self.problem.evaluate(input_vars, f=False, c=True)
        c_vals_dict = {
            const.func.name: c_vals[i]
            for i, const in enumerate(self.problem.constraints)
        }
        h_i = self.problem.compute_constraint_violation(c_vals_dict)
        if assess_progress(
            h_total=h_i,
            h_best=self.best_h,
            J_total=0.0,
            J_best=0.0,
            f=J_i_val,
            f_best=self.best_J_i,
        ):
            self.best_h = h_i
            self.best_J_i = J_i_val
        x_J_i_new = np.concatenate([
            z_bar,
            x_bar_i,
            y_bar_i,
            y_bar_coupled,
            self.z_under_i,
            self.x_under_i,
        ])
        yi_val = self.y_i.item() if self.y_i.size == 1 else self.y_i
        self.doe_J_i.update_DoE(x_J_i_new, {"J_i": J_i_val, "y_i": yi_val})
        if self.problem.constraints:
            x_g_i_new = np.concatenate([self.z_under_i, self.x_under_i, y_bar_coupled])
            self.doe_g_i.update_DoE(x_g_i_new, c_vals_dict)
        elapsed = time.time() - _start_time
        return (self.z_under_i, self.x_under_i, self.y_i, J_i_val, h_i, elapsed)

    def _describe_setup(self):
        return "\n".join([
            f"problem: {self.problem}",
            f"z_idxs: {self.z_idxs}",
            f"x_idxs: {self.x_idxs}",
            f"y_idxs: {self.y_idxs}",
            f"y_coupled_idxs: {self.y_coupled_idxs}",
            f"surrogate_config: {self.surrogate_config}",
        ])


@dataclass
class BACOSystem:
    """System-level problem for BACO"""

    problem: Problem
    subsystems: list[BACOSubsystem]
    surrogate_config: SMTGPConfig | TorchGPConfig | None = None

    def __post_init__(self):
        if self.surrogate_config is None:
            from ..surrogates import SMTGPConfig

            self.surrogate_config = SMTGPConfig()
        self.z_bar = None
        self.x_bar = None
        self.y_bar = None
        self.doe_sys = None
        self.gp_f = None
        self.gp_c = {}
        self.best_J = np.inf
        self.best_h = np.inf
        self.best_f = np.inf

    def initialize_doe(self, n_samples):
        """Initialize system DoE"""
        if (
            self.z_bar is not None
            and self.x_bar is not None
            and (self.y_bar is not None)
        ):
            x0 = np.concatenate([self.z_bar, self.x_bar, self.y_bar])
            f0, c0 = self.problem.evaluate(x0, f=True, c=True)
            c0_dict = {
                const.func.name: c0[i]
                for i, const in enumerate(self.problem.constraints)
            }
            h0 = self.problem.compute_constraint_violation(c0_dict)
            doe_rest = self.problem.initial_DoE(n_samples)
            x_all = np.vstack([x0, doe_rest.x])
            y_all = {"obj": np.concatenate([[f0], doe_rest.y["obj"]])}
            for key in c0_dict:
                y_all[key] = np.concatenate([[c0_dict[key]], doe_rest.y[key]])
            h_all = np.concatenate([[h0], doe_rest.constraint_violation])
            self.doe_sys = DoE(x_all, y_all, h_all)
        else:
            self.doe_sys = self.problem.initial_DoE(n_samples)
        sample = self.doe_sys.x[0]
        nz = len(np.unique(np.concatenate([sub.z_idxs for sub in self.subsystems])))
        nx = sum(len(sub.x_idxs) for sub in self.subsystems)
        self.z_bar = sample[:nz]
        self.x_bar = sample[nz : nz + nx]
        self.y_bar = sample[nz + nx :]

    def build_surrogate_system(self):
        """Build GPs for system objective and constraints"""
        self.gp_f = build_surrogate(
            self.doe_sys, y_key="obj", config=self.surrogate_config
        )
        for const in self.problem.constraints:
            self.gp_c[const.func.name] = build_surrogate(
                self.doe_sys, y_key=const.func.name, config=self.surrogate_config
            )

    def solve_acquisition(
        self,
        optimizer: str | Callable,
        acq_func: Callable,
        f_min: float,
        n_multistart: int = 10,
        epsilon_J: float = 1.0,
    ):
        """Solve system: maximize alpha_f subject to mu_c >= 0, mu_J_i <= epsilon_J"""
        _start_time = time.time()
        fit_tasks = []
        fit_tasks.append(("gp_f", self.doe_sys, "obj", self.surrogate_config, None))
        for const in self.problem.constraints:
            fit_tasks.append((
                "gp_c",
                self.doe_sys,
                const.func.name,
                self.surrogate_config,
                const.func.name,
            ))
        for i, subsystem in enumerate(self.subsystems):
            fit_tasks.append((
                "gp_J_i",
                subsystem.doe_J_i,
                "J_i",
                subsystem.surrogate_config,
                i,
            ))
        min_doe = min(t[1].x.shape[0] for t in fit_tasks)
        use_loky = len(fit_tasks) >= 3 and min_doe >= 10

        def _fit_one(task_idx, kind, doe, y_key, config, tag):
            task_config = copy.copy(config)
            task_config.seed = config.seed + task_idx
            return (kind, tag, build_surrogate(doe, y_key=y_key, config=task_config))

        from joblib import Parallel, delayed

        if use_loky:
            fit_results = Parallel(n_jobs=-1, backend="loky")(
                (delayed(_fit_one)(i, *task) for i, task in enumerate(fit_tasks))
            )
        else:
            fit_results = [_fit_one(i, *task) for i, task in enumerate(fit_tasks)]
        for kind, tag, gp in fit_results:
            if kind == "gp_f":
                self.gp_f = gp
            elif kind == "gp_c":
                self.gp_c[tag] = gp
            elif kind == "gp_J_i":
                self.subsystems[tag].gp_J_i = gp
        nz = len(self.z_bar)
        nx = len(self.x_bar)
        _nsvars = nz + nx + len(self.y_bar)
        _subsystem_masks = []
        for subsystem in self.subsystems:
            coupled_flat = []
            for coupled_idx_array in subsystem.y_coupled_idxs:
                for idx in coupled_idx_array:
                    coupled_flat.append(idx)
            coupled_flat = (
                np.array(coupled_flat, dtype=int)
                if coupled_flat
                else np.array([], dtype=int)
            )
            parts = [np.arange(nz)]
            if len(subsystem.x_idxs) > 0:
                parts.append(nz + subsystem.x_idxs)
            parts.append(nz + nx + subsystem.y_idxs)
            if len(coupled_flat) > 0:
                parts.append(nz + nx + coupled_flat)
            src_indices = np.concatenate(parts)
            n_dynamic = len(src_indices)
            static_part = np.concatenate([subsystem.z_under_i, subsystem.x_under_i])
            buffer_template = np.empty(n_dynamic + len(static_part))
            buffer_template[n_dynamic:] = static_part
            _subsystem_masks.append({
                "src_indices": src_indices,
                "n_dynamic": n_dynamic,
                "buffer_template": buffer_template,
            })
        if np.isinf(f_min):
            f_min = self.doe_sys.get_f_min("obj", self.problem.tol)
        bounds = self.problem.bounds
        lhs_starts = LHS(xlimits=bounds, seed=42)(n_multistart - 1)
        x0_default = np.concatenate([self.z_bar, self.x_bar, self.y_bar])
        start_points = [x0_default] + [lhs_starts[i] for i in range(n_multistart - 1)]
        if isinstance(optimizer, str):
            opt = get_optimizer(optimizer)
        else:
            opt = optimizer
        gp_J_i_subsystems = [sub.gp_J_i for sub in self.subsystems]
        if n_multistart >= 4:
            start_results = Parallel(n_jobs=-1, backend="loky")(
                delayed(_run_one_start_system)(
                    x0,
                    opt,
                    acq_func,
                    bounds,
                    self.gp_f,
                    self.gp_c,
                    gp_J_i_subsystems,
                    _subsystem_masks,
                    self.problem.constraints,
                    epsilon_J,
                    f_min,
                    _nsvars,
                )
                for x0 in start_points
            )
        else:
            start_results = [
                _run_one_start_system(
                    x0,
                    opt,
                    acq_func,
                    bounds,
                    self.gp_f,
                    self.gp_c,
                    gp_J_i_subsystems,
                    _subsystem_masks,
                    self.problem.constraints,
                    epsilon_J,
                    f_min,
                    _nsvars,
                )
                for x0 in start_points
            ]
        best_result = None
        best_acq_val = np.inf
        for result_x, acq_val in start_results:
            if result_x is not None and acq_val < best_acq_val:
                best_acq_val = acq_val
                best_result = result_x
        if best_result is None:
            warnings.warn(
                f"All {len(start_points)} multi-start acquisition attempts failed. Falling back to x0_default. Check optimizer and constraint configuration.",
                UserWarning,
                stacklevel=2,
            )
            best_result = x0_default
        best_result = clip_to_bounds(best_result, bounds)
        nz = len(self.z_bar)
        nx = len(self.x_bar)
        self.z_bar = best_result[:nz]
        self.x_bar = best_result[nz : nz + nx]
        self.y_bar = best_result[nz + nx :]
        f_val, c_vals = self.problem.evaluate(best_result, f=True, c=True)
        c_vals_dict = {
            const.func.name: c_vals[i]
            for i, const in enumerate(self.problem.constraints)
        }
        h_total = self.problem.compute_constraint_violation(c_vals_dict)
        y_dict = {"obj": f_val}
        for i, const in enumerate(self.problem.constraints):
            y_dict[const.func.name] = c_vals[i]
        self.doe_sys.update_DoE(best_result, y_dict, h_total)
        z_snap = self.z_bar
        x_bar_snap = self.x_bar
        y_bar_snap = self.y_bar

        def _compute_one_J_i(subsystem):
            x_bar_i = (
                x_bar_snap[subsystem.x_idxs]
                if len(subsystem.x_idxs) > 0
                else np.array([])
            )
            y_bar_i = y_bar_snap[subsystem.y_idxs]
            y_bar_coupled = []
            for coupled_idx_array in subsystem.y_coupled_idxs:
                for idx in coupled_idx_array:
                    y_bar_coupled.append(y_bar_snap[idx])
            y_bar_coupled = np.array(y_bar_coupled) if y_bar_coupled else np.array([])
            J_i_val, y_i = compute_Ji(
                subsystem.problem.objective.func,
                z_snap[subsystem.z_idxs],
                x_bar_i,
                y_bar_i,
                y_bar_coupled,
                subsystem.z_under_i,
                subsystem.x_under_i,
            )
            x_J_i_new = np.concatenate([
                z_snap,
                x_bar_i,
                y_bar_i,
                y_bar_coupled,
                subsystem.z_under_i,
                subsystem.x_under_i,
            ])
            return (J_i_val, x_J_i_new, y_i)

        J_total = 0.0
        for subsystem in self.subsystems:
            J_i_val, x_J_i_new, y_i = _compute_one_J_i(subsystem)
            J_total += J_i_val
            yi_val = y_i.item() if y_i.size == 1 else y_i
            subsystem.doe_J_i.update_DoE(x_J_i_new, {"J_i": J_i_val, "y_i": yi_val})
        elapsed = time.time() - _start_time
        return (self.z_bar, self.x_bar, self.y_bar, elapsed, J_total)

    def _describe_setup(self):
        return "\n".join([
            f"problem: {self.problem}",
            f"subsystems: {len(self.subsystems)}",
            f"surrogate_config: {self.surrogate_config}",
        ])


@dataclass
class BayesianCollaborativeOptimization(BaseSolver):
    """BACO solver"""

    system: BACOSystem
    subsystem_optimizer: str | Callable
    system_optimizer: str | Callable
    n_initial: int | Callable | None = None
    acq_func: Callable | None = None
    n_multistart: int = 10
    solver: str = "BACO"

    def __post_init__(self):
        super().__post_init__()
        if self.n_initial is None:
            self.n_initial = 2 * len(
                np.array([self.system.problem.objective.x]).ravel()
            )
        if self.acq_func is None:
            from ..surrogates import log_ei

            self.acq_func = log_ei
        self.setup_description = [
            80 * "=",
            "BACO Optimization Setup:",
            80 * "=",
            "n_initial:"
            + (
                f"{self.n_initial}"
                if isinstance(self.n_initial, int)
                else inspect.getsource(self.n_initial).strip()
            ),
            f"epsilon_J: {self.epsilon_J}",
            f"epsilon_h: {self.epsilon_h}",
            f"max_eval: {self.budget.total_budget}",
            f"acq_func: {self.acq_func}",
            f"n_multistart: {self.n_multistart}",
            f"name: {self.name}",
            f"solver: {self.solver}",
            "",
            80 * "=",
            "SYSTEM SETUP",
            80 * "=",
            f"system_optimizer: {self.system_optimizer}",
            self.system._describe_setup(),
        ]
        for i, subsystem in enumerate(self.system.subsystems, 1):
            self.setup_description += [
                "",
                80 * "=",
                f"SUBSYSTEM {i:02d} SETUP",
                80 * "=",
                f"subsystem_optimizer: {self.subsystem_optimizer}",
                subsystem._describe_setup(),
            ]

    def _custom_initialize(self):
        """Perform BACO-specific initialization (DoE generation)."""
        if isinstance(self.n_initial, Callable):
            n_init = self.n_initial(
                len(self.system.z_bar) + len(self.system.x_bar) + len(self.system.y_bar)
            )
        else:
            n_init = self.n_initial
        self.system.initialize_doe(n_init)
        for subsystem in self.system.subsystems:
            subsystem.initialize_doe(
                self.n_initial, self.system.z_bar, self.system.x_bar, self.system.y_bar
            )

    def iterate(self):
        """Execute one BACO iteration"""
        J_stars = []
        h_total = 0.0
        z_bar = self.system.z_bar
        x_bar = self.system.x_bar
        y_bar = self.system.y_bar
        gp_fit_tasks = []
        task_map = []
        for idx, subsystem in enumerate(self.system.subsystems):
            gp_fit_tasks.append((subsystem.doe_J_i, "J_i", subsystem.surrogate_config))
            task_map.append((idx, "J_i", None))
            for const in subsystem.problem.constraints:
                gp_fit_tasks.append((
                    subsystem.doe_g_i,
                    const.func.name,
                    subsystem.surrogate_config,
                ))
                task_map.append((idx, "g_i", const.func.name))
        min_doe = min(t[0].x.shape[0] for t in gp_fit_tasks)
        use_loky = len(gp_fit_tasks) >= 3 and min_doe >= 10
        from joblib import Parallel, delayed

        if use_loky:
            gp_results = Parallel(n_jobs=-1, backend="loky")(
                (
                    delayed(build_surrogate)(doe, key, copy.copy(cfg))
                    for doe, key, cfg in gp_fit_tasks
                )
            )
        else:
            gp_results = [
                build_surrogate(doe, key, cfg) for doe, key, cfg in gp_fit_tasks
            ]
        for (sub_idx, kind, cname), gp in zip(task_map, gp_results):
            subsystem = self.system.subsystems[sub_idx]
            if kind == "J_i":
                subsystem.gp_J_i = gp
            else:
                subsystem.gp_g_i[cname] = gp
        results = []
        for idx, subsystem in enumerate(self.system.subsystems):
            z_ss, x_ss, y_ss, J_i, h_i, elap = subsystem.solve_acquisition(
                z_bar,
                x_bar,
                y_bar,
                self.subsystem_optimizer,
                self.acq_func,
                n_multistart=self.n_multistart,
            )
            results.append((idx, z_ss, x_ss, y_ss, J_i, h_i, elap))
        for idx, z_ss, x_ss, y_ss, J_i, h_i, elap in results:
            J_stars.append(J_i)
            h_total += h_i
            self._eval += 1
            self._history["iter"].append(self._iter)
            self._history["eval"].append(self._eval)
            self._history["tss"].append(
                elap + self._history["tss"][-1] if self._history["tss"] else elap
            )
            self._history["improved"].append(False)
            self._history["level"].append(f"Subsystem {idx + 1:02d}")
            self._history["z"].append(z_ss.copy())
            self._history["x"].append(x_ss.copy())
            self._history["y"].append(y_ss.copy())
            for key in self._history:
                if key in ["iter", "eval", "tss", "improved", "level", "z", "x", "y"]:
                    continue
                elif key == f"J{idx + 1:02d}":
                    self._history[key].append(J_i)
                elif key == f"h{idx + 1:02d}":
                    self._history[key].append(h_i)
                else:
                    self._history[key].append(np.nan)
        _, _, _, elap, J_total = self.system.solve_acquisition(
            self.system_optimizer,
            self.acq_func,
            self._best_results.f,
            n_multistart=self.n_multistart,
            epsilon_J=self.epsilon_J,
        )
        h_sys = self.system.doe_sys.constraint_violation[-1]
        h_total += h_sys
        f_sys = self.system.doe_sys.y["obj"][-1]
        converge = J_total <= self.epsilon_J and h_total <= self.epsilon_h
        self._eval += len(self.system.subsystems)
        self._history["iter"].append(self._iter)
        self._history["eval"].append(self._eval)
        self._history["tss"].append(
            self._history["tss"][-1] + elap if self._history["tss"] else elap
        )
        self._history["level"].append("System")
        self._history["f_sys"].append(f_sys)
        self._history["h_total"].append(h_total)
        self._history["J_total"].append(J_total)
        self._history["z"].append(self.system.z_bar.copy())
        self._history["x"].append(self.system.x_bar.copy())
        self._history["y"].append(self.system.y_bar.copy())
        for idx, _ in enumerate(self.system.subsystems):
            self._history[f"h{idx + 1:02d}"].append(np.nan)
            self._history[f"J{idx + 1:02d}"].append(np.nan)
        if converge:
            self._converged = True

    def solve(self, z0, x_bar0, y_bar0, z_hat0, x0):
        """Execute BACO"""
        setup_log = (
            list(self.setup_description)
            if isinstance(self.setup_description, list)
            else [self.setup_description]
        )
        setup_log += [
            "",
            80 * "=",
            "INITIAL SETUP",
            80 * "=",
            "z0: " + np.array2string(z0),
            "x_bar0: " + np.array2string(x_bar0),
            "y_bar0: " + np.array2string(y_bar0),
            "z_hat0: " + np.array2string(z_hat0),
            "x0: " + np.array2string(x0),
        ]
        self.initialize(z0, x_bar0, y_bar0, z_hat0, x0)
        import pandas as pd

        setup_text = "\n".join(setup_log)
        with open(self.cache_dir / "setup.txt", "w") as f:
            f.write(setup_text)
        self._converged = False
        self._iter = 0
        self._eval = len(self.system.doe_sys.x) * len(self.system.subsystems) + sum(
            len(sub.doe_J_i.x) for sub in self.system.subsystems
        )
        nz = len(self.system.z_bar)
        nx = len(self.system.x_bar)
        doe = self.system.doe_sys
        cv = doe.constraint_violation
        eval_counter = 0
        for i in range(len(doe.x)):
            sample = doe.x[i]
            f_i = float(doe.y["obj"][i])
            h_i = float(cv[i]) if cv is not None else 0.0
            z_i = sample[:nz].copy()
            x_under_i_all = sample[nz : nz + nx].copy()
            y_i_all = sample[nz + nx :].copy()
            J_total = 0.0
            for idx, subsystem in enumerate(self.system.subsystems):
                x_bar_i_sample = (
                    x_under_i_all[subsystem.x_idxs]
                    if len(subsystem.x_idxs) > 0
                    else np.array([])
                )
                y_bar_i_sample = y_i_all[subsystem.y_idxs]
                y_bar_coupled_sample = []
                for coupled_idx_array in subsystem.y_coupled_idxs:
                    for idx_c in coupled_idx_array:
                        y_bar_coupled_sample.append(y_i_all[idx_c])
                y_bar_coupled_sample = (
                    np.array(y_bar_coupled_sample)
                    if y_bar_coupled_sample
                    else np.array([])
                )
                J_i_val, _ = compute_Ji(
                    subsystem.problem.objective.func,
                    z_i[subsystem.z_idxs],
                    x_bar_i_sample,
                    y_bar_i_sample,
                    y_bar_coupled_sample,
                    subsystem.z_under_i,
                    subsystem.x_under_i,
                )
                J_total += J_i_val
                eval_counter += 1
            self._pareto_archive = update_pareto(
                self._pareto_archive,
                f=f_i,
                h_total=h_i,
                J_total=J_total,
                z=z_i,
                x=x_under_i_all,
                y=y_i_all,
            )
            self._history["iter"].append(0)
            self._history["eval"].append(eval_counter)
            self._history["tss"].append(0.0)
            self._history["improved"].append(False)
            self._history["level"].append("System-DoE")
            self._history["f_sys"].append(f_i)
            self._history["h_total"].append(h_i)
            self._history["J_total"].append(J_total)
            self._history["z"].append(z_i)
            self._history["x"].append(x_under_i_all)
            self._history["y"].append(y_i_all)
            for jdx, _ in enumerate(self.system.subsystems):
                self._history[f"h{jdx + 1:02d}"].append(np.nan)
                self._history[f"J{jdx + 1:02d}"].append(np.nan)
        for idx, subsystem in enumerate(self.system.subsystems):
            J_i_vals = subsystem.doe_J_i.y["J_i"]
            yi_vals = subsystem.doe_J_i.y["y_i"]
            hi_vals = subsystem.doe_J_i.constraint_violation
            nz_bar = len(self.system.z_bar)
            nx_bar_i = len(subsystem.x_idxs)
            ny_bar_i = len(subsystem.y_idxs)
            ny_coupled = sum(len(c) for c in subsystem.y_coupled_idxs)
            offset = nz_bar + nx_bar_i + ny_bar_i + ny_coupled
            nz_ss = len(subsystem.z_idxs)
            nx_ss = len(subsystem.x_idxs)
            for j in range(len(subsystem.doe_J_i.x)):
                sample_ss = subsystem.doe_J_i.x[j]
                z_under_i_sample = sample_ss[offset : offset + nz_ss]
                x_under_i_sample = sample_ss[offset + nz_ss : offset + nz_ss + nx_ss]
                eval_counter += 1
                J_i_j = float(J_i_vals[j])
                yi_j = np.array(yi_vals[j])
                hi_j = float(hi_vals[j]) if hi_vals is not None else np.nan
                self._history["iter"].append(0)
                self._history["eval"].append(eval_counter)
                self._history["tss"].append(0.0)
                self._history["improved"].append(False)
                self._history["level"].append(f"Subsystem{idx + 1:02d}-DoE")
                self._history["f_sys"].append(np.nan)
                self._history["h_total"].append(np.nan)
                self._history["J_total"].append(np.nan)
                self._history["z"].append(z_under_i_sample)
                self._history["x"].append(x_under_i_sample)
                self._history["y"].append(yi_j)
                for jdx, _ in enumerate(self.system.subsystems):
                    if jdx == idx:
                        self._history[f"h{jdx + 1:02d}"].append(hi_j)
                        self._history[f"J{jdx + 1:02d}"].append(J_i_j)
                    else:
                        self._history[f"h{jdx + 1:02d}"].append(np.nan)
                        self._history[f"J{jdx + 1:02d}"].append(np.nan)
        self._eval = eval_counter
        if self._eval >= self.budget.total_budget:
            iterend = time.time() - self._start_time
            print(
                f"DoE evaluations ({self._eval}) >= budget ({self.budget.total_budget}). Returning best DoE solution."
            )
            return Results(
                best=self._best_results,
                converged=False,
                iterations=0,
                evaluations=self._eval,
                elapsed_time=iterend,
                history=pd.DataFrame(self._history),
                pareto=pareto_archive_to_df(self._pareto_archive),
            )
        J_total = np.inf
        h_total = np.inf
        iterend = 0.0
        pd.DataFrame(self._history).to_csv(self.cache_dir / "DoE.csv", index=False)
        while self._eval < self.budget.total_budget:
            self._iter += 1
            iterstart = time.time() - self._start_time
            self.iterate()
            iterend = time.time() - self._start_time
            iterdur = iterend - iterstart
            f_sys = self._history["f_sys"][-1]
            J_total = self._history["J_total"][-1]
            h_total = self._history["h_total"][-1]
            improved = self._best_results.update(
                z_new=self.system.z_bar,
                x_new=self.system.x_bar,
                y_new=self.system.y_bar,
                f_new=f_sys,
                h_new=h_total,
                J_new=J_total,
                epsilon_J=self.epsilon_J,
                epsilon_h=self.epsilon_h,
            )
            self._history["improved"].append(improved)
            pd.DataFrame(self._history).to_pickle(self.cache_dir / "state.pkl")
            self._pareto_archive = update_pareto(
                self._pareto_archive,
                f=f_sys,
                h_total=h_total,
                J_total=J_total,
                z=self.system.z_bar.copy(),
                x=self.system.x_bar.copy(),
                y=self.system.y_bar.copy(),
            )
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
                f"eval={self._eval:>{len(str(self.budget.total_budget))}} /",
                f"{self.budget.total_budget:>{len(str(self.budget.total_budget))}} |",
                f"Improved={improved_str} |",
                f"h_total={h_str} |",
                f"J_total={J_str} |",
                f"f={format_vars(f_sys, 3)} |",
                f"TSince={iterend:>8.2f}s |",
                f"IterDur={iterdur:>7.3f}s",
            )
            if self._converged:
                print("\nConvergence criteria met.")
                self._history["epsilon_J"] = [self.epsilon_J] * len(
                    self._history["iter"]
                )
                self._history["epsilon_h"] = [self.epsilon_h] * len(
                    self._history["iter"]
                )
                break
        history_df = pd.DataFrame(self._history)
        history_df["identifier"] = self.name
        history_df["solver"] = self.solver
        history_df["epsilon_J"] = self.epsilon_J
        history_df["epsilon_h"] = self.epsilon_h
        pareto_df = pareto_archive_to_df(self._pareto_archive)
        pareto_df["identifier"] = self.name
        pareto_df["solver"] = self.solver
        pareto_df["epsilon_J"] = self.epsilon_J
        pareto_df["epsilon_h"] = self.epsilon_h
        return Results(
            best=self._best_results,
            converged=self._converged,
            iterations=self._iter,
            evaluations=self._eval,
            elapsed_time=iterend,
            history=history_df,
            pareto=pareto_df,
        )
