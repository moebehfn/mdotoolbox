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

"""src/mdotoolbox/optimizers/wrappers.py"""

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass
from ..core.utils import assess_progress

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
OPTIMIZERS = {}


def register(name):
    """Decorator to register optimizer function in global OPTIMIZERS dict."""

    def decorator(func):
        OPTIMIZERS[name] = func
        return func

    return decorator


def get_optimizer(name, **kwargs):
    """Get optimizer function by string name with optional kwargs."""
    if name not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer: {name}. Available: {list(OPTIMIZERS.keys())}"
        )
    opt_func = OPTIMIZERS[name]
    if not kwargs:
        return opt_func
    return lambda objective, x0, bounds, constraints=None: opt_func(
        objective, x0, bounds, constraints, **kwargs
    )


def _scipy_bounds(bounds):
    """Convert bounds array to SciPy format."""
    bounds = np.atleast_2d(bounds)
    return [(bounds[i, 0], bounds[i, 1]) for i in range(bounds.shape[0])]


def _scipy_constraints(constraints, x0):
    """Convert constraints to scipy format"""
    if constraints is None:
        return None
    scipy_constraints = []
    if isinstance(constraints, dict):
        for ctype in ["ineq", "eq"]:
            if ctype in constraints:
                c_result = (
                    constraints[ctype](x0)
                    if callable(constraints[ctype])
                    else constraints[ctype]
                )
                c_array = np.atleast_1d(c_result)
                n = len(c_array)
                if callable(constraints[ctype]):
                    for i in range(n):
                        scipy_constraints.append({
                            "type": ctype,
                            "fun": lambda x, i=i, f=constraints[ctype]: np.atleast_1d(
                                f(x)
                            )[i],
                        })
                else:
                    for i in range(n):
                        scipy_constraints.append({
                            "type": ctype,
                            "fun": lambda x, i=i, val=c_array[i]: val,
                        })
    else:
        n = len(np.atleast_1d(constraints(x0)))
        for i in range(n):
            scipy_constraints.append({
                "type": "ineq",
                "fun": lambda x, i=i, f=constraints: np.atleast_1d(f(x))[i],
            })
    return scipy_constraints if scipy_constraints else None


def _bounds_as_constraints(bounds):
    """Convert variable bounds to inequality constraints for COBYLA."""
    bounds = np.atleast_2d(bounds)
    constraints = []
    for i in range(bounds.shape[0]):
        constraints.append({"type": "ineq", "fun": lambda x, i=i: x[i] - bounds[i, 0]})
        constraints.append({"type": "ineq", "fun": lambda x, i=i: bounds[i, 1] - x[i]})
    return constraints


@register("cobyla")
def cobyla(objective, x0, bounds, constraints=None, maxiter=1000):
    """COBYLA - Constrained Optimization BY Linear Approximations."""
    from scipy.optimize import OptimizeWarning, minimize

    warnings.filterwarnings("ignore", category=OptimizeWarning)
    x0 = np.atleast_1d(x0)
    cobyla_constraints = _scipy_constraints(constraints, x0) or []
    if bounds is not None:
        cobyla_constraints.extend(_bounds_as_constraints(bounds))
    result = minimize(
        objective,
        x0,
        method="COBYLA",
        constraints=cobyla_constraints,
        options={"maxiter": maxiter},
    )
    return (result.x, result.fun)


@register("cobyqa")
def cobyqa(objective, x0, bounds, constraints=None, maxfev=1000, maxiter=1000):
    """COBYQA - Constrained Optimization BY Quadratic Approximations."""
    from scipy.optimize import OptimizeWarning, minimize

    warnings.filterwarnings("ignore", category=OptimizeWarning)
    x0 = np.atleast_1d(x0)
    result = minimize(
        objective,
        x0,
        method="COBYQA",
        bounds=_scipy_bounds(bounds),
        constraints=_scipy_constraints(constraints, x0),
        options={"maxfev": maxfev, "maxiter": maxiter},
    )
    return (result.x, result.fun)


@register("slsqp")
def slsqp(objective, x0, bounds, constraints=None, maxfev=1000, maxiter=1000):
    """SLSQP - Sequential Least SQuares Programming."""
    from scipy.optimize import OptimizeWarning, minimize

    warnings.filterwarnings("ignore", category=OptimizeWarning)
    x0 = np.atleast_1d(x0)
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=_scipy_bounds(bounds),
        constraints=_scipy_constraints(constraints, x0),
        options={"maxfev": maxfev, "maxiter": maxiter},
    )
    return (result.x, result.fun)


@register("mads")
def mads(
    objective,
    x0,
    bounds,
    constraints=None,
    maxiter=1000,
    tol=1e-06,
    initial_mesh_size=None,
    min_mesh_size=1e-10,
    mesh_refine_factor=0.5,
    mesh_coarsen_factor=1.0,
    use_improvement_tracking=True,
    improvement_patience=10,
):
    """Mesh Adaptive Direct Search (MADS) optimizer"""
    x0 = np.atleast_1d(x0).astype(float)
    bounds = np.atleast_2d(bounds).astype(float)
    n = len(x0)
    if initial_mesh_size is None:
        bound_ranges = bounds[:, 1] - bounds[:, 0]
        initial_mesh_size = 0.1 * np.mean(bound_ranges)
    x_current = np.clip(x0, bounds[:, 0], bounds[:, 1])
    mesh_size = initial_mesh_size
    n_eval = 1
    no_improvement_count = 0

    def _evaluate_constraints(x):
        """Returns constraint violation (0 if feasible, >0 if violated)"""
        if constraints is None:
            return 0.0
        violation = 0.0
        if isinstance(constraints, dict):
            if "ineq" in constraints:
                ineq_vals = np.atleast_1d(constraints["ineq"](x))
                violation += np.sum(np.maximum(0, -ineq_vals))
            if "eq" in constraints:
                eq_vals = np.atleast_1d(constraints["eq"](x))
                violation += np.sum(np.abs(eq_vals))
        else:
            c_vals = np.atleast_1d(constraints(x))
            violation += np.sum(np.maximum(0, -c_vals))
        return violation

    def _penalized_objective(x, current_mesh_size):
        """Objective with penalty for constraint violations"""
        f = objective(x)
        cv = _evaluate_constraints(x)
        if cv > 0:
            penalty_factor = 1000000.0 * (
                1.0 + 1.0 / max(current_mesh_size, min_mesh_size)
            )
            return f + penalty_factor * cv
        return f

    f_current = _penalized_objective(x_current, mesh_size)
    best_x = x_current.copy()
    best_f = f_current
    best_h = _evaluate_constraints(x_current)
    h_current = best_h
    if use_improvement_tracking:
        f_current_raw = objective(x_current)
        n_eval += 1
        best_f_raw = f_current_raw
    no_improvement_iters = 0

    def _generate_directions(n_dims):
        """Generate comprehensive poll directions"""
        directions = []
        for i in range(n_dims):
            d = np.zeros(n_dims)
            d[i] = 1.0
            directions.append(d)
            directions.append(-d)
        if n_dims >= 2:
            directions.append(np.ones(n_dims))
            directions.append(-np.ones(n_dims))
            if n_dims <= 4:
                for i in range(n_dims):
                    for j in range(i + 1, n_dims):
                        d = np.zeros(n_dims)
                        d[i] = 1.0
                        d[j] = 1.0
                        directions.append(d)
                        directions.append(-d)
        return np.array(directions)

    directions = _generate_directions(n)
    for iteration in range(maxiter):
        if mesh_size < min_mesh_size or mesh_size < tol:
            break
        improved = False
        if iteration % 10 == 0 and mesh_size > 10 * min_mesh_size:
            n_search = min(5, n)
            for _ in range(n_search):
                rand_dir = np.random.randn(n)
                rand_dir = rand_dir / (np.linalg.norm(rand_dir) + 1e-16)
                x_search = x_current + mesh_size * 2.0 * rand_dir
                x_search = np.clip(x_search, bounds[:, 0], bounds[:, 1])
                f_search = _penalized_objective(x_search, mesh_size)
                n_eval += 1
                if f_search < f_current:
                    x_current = x_search
                    f_current = f_search
                    if use_improvement_tracking:
                        f_current_raw = objective(x_search)
                        n_eval += 1
                    h_current = _evaluate_constraints(x_search)
                    improved = True
                    if f_search < best_f:
                        best_x = x_search.copy()
                        best_f = f_search
                        if use_improvement_tracking:
                            best_f_raw = f_current_raw
                        best_h = h_current
                    break
        if not improved:
            for direction in directions:
                if n_eval >= maxiter:
                    break
                x_trial = x_current + mesh_size * direction
                x_trial = np.clip(x_trial, bounds[:, 0], bounds[:, 1])
                if np.linalg.norm(x_trial - x_current) < 1e-12:
                    continue
                f_trial = _penalized_objective(x_trial, mesh_size)
                n_eval += 1
                if f_trial < f_current - 1e-12:
                    x_current = x_trial
                    f_current = f_trial
                    if use_improvement_tracking:
                        f_current_raw = objective(x_trial)
                        n_eval += 1
                    h_current = _evaluate_constraints(x_trial)
                    improved = True
                    if f_trial < best_f:
                        best_x = x_trial.copy()
                        best_f = f_trial
                        if use_improvement_tracking:
                            best_f_raw = f_current_raw
                        best_h = h_current
                    mesh_size = min(mesh_size * mesh_coarsen_factor, initial_mesh_size)
                    no_improvement_count = 0
                    break
        if not improved:
            mesh_size *= mesh_refine_factor
            no_improvement_count += 1
            if no_improvement_count > 20:
                mesh_size = initial_mesh_size * 0.1
                no_improvement_count = 0
        if use_improvement_tracking and iteration > 0:
            improved_flag = assess_progress(
                h_total=h_current,
                h_best=best_h,
                J_total=0.0,
                J_best=0.0,
                f=f_current_raw,
                f_best=best_f_raw,
            )
            if not improved_flag:
                no_improvement_iters += 1
                if no_improvement_iters >= improvement_patience:
                    break
            else:
                no_improvement_iters = 0
                mesh_size = min(mesh_size * mesh_coarsen_factor, initial_mesh_size)
        if n_eval >= maxiter:
            break
    return (best_x, best_f)


@register("nomad")
def nomad(
    objective,
    x0,
    bounds,
    constraints=None,
    maxiter=1000,
    tol=1e-06,
    initial_mesh_size=None,
    min_mesh_size=1e-10,
    mesh_refine_factor=0.5,
    mesh_coarsen_factor=2.0,
    use_improvement_tracking=True,
    improvement_patience=15,
):
    """NOMAD-style optimizer (Ortho-MADS variant)"""
    x0 = np.atleast_1d(x0).astype(float)
    bounds = np.atleast_2d(bounds).astype(float)
    n = len(x0)
    if initial_mesh_size is None:
        bound_ranges = bounds[:, 1] - bounds[:, 0]
        initial_mesh_size = 0.1 * np.mean(bound_ranges)
    x_current = np.clip(x0, bounds[:, 0], bounds[:, 1])
    mesh_size = initial_mesh_size
    n_eval = 1
    no_improvement_count = 0

    def _evaluate_constraints(x):
        """Returns constraint violation (0 if feasible, >0 if violated)"""
        if constraints is None:
            return 0.0
        violation = 0.0
        if isinstance(constraints, dict):
            if "ineq" in constraints:
                ineq_vals = np.atleast_1d(constraints["ineq"](x))
                violation += np.sum(np.maximum(0, -ineq_vals))
            if "eq" in constraints:
                eq_vals = np.atleast_1d(constraints["eq"](x))
                violation += np.sum(np.abs(eq_vals))
        else:
            c_vals = np.atleast_1d(constraints(x))
            violation += np.sum(np.maximum(0, -c_vals))
        return violation

    def _barrier_objective(x, h_max):
        """Objective with progressive barrier for constraints"""
        f = objective(x)
        h = _evaluate_constraints(x)
        if h <= h_max:
            return (f, h)
        else:
            return (np.inf, h)

    f_current = objective(x_current)
    h_current = _evaluate_constraints(x_current)
    h_max = max(h_current, 0.001)
    best_x = x_current.copy()
    best_f = f_current
    best_h = h_current
    if use_improvement_tracking:
        f_current_raw = f_current
        n_eval += 1
        best_f_raw = f_current_raw
    no_improvement_iters = 0

    def _generate_orthogonal_directions(n_dims, iteration):
        """Generate orthogonal poll directions using Householder reflections"""
        directions = []
        basis = np.eye(n_dims)
        if iteration > 0:
            np.random.seed(iteration)
            v = np.random.randn(n_dims)
            v = v / (np.linalg.norm(v) + 1e-16)
            H = np.eye(n_dims) - 2 * np.outer(v, v)
            basis = H @ basis
        for i in range(n_dims):
            directions.append(basis[:, i])
            directions.append(-basis[:, i])
        return np.array(directions)

    for iteration in range(maxiter):
        if mesh_size < min_mesh_size or mesh_size < tol:
            break
        improved = False
        if iteration % 5 == 0 and mesh_size > 10 * min_mesh_size:
            n_search = min(10, 2 * n)
            for k in range(n_search):
                alpha = np.random.uniform(-2.0, 2.0, size=n)
                x_search = x_current + mesh_size * alpha
                x_search = np.clip(x_search, bounds[:, 0], bounds[:, 1])
                f_search, h_search = _barrier_objective(x_search, h_max)
                n_eval += 1
                if h_search <= h_current and f_search < f_current:
                    x_current = x_search
                    f_current = f_search
                    h_current = h_search
                    improved = True
                    if use_improvement_tracking:
                        f_current_raw = objective(x_search)
                        n_eval += 1
                    if h_search <= best_h and f_search < best_f:
                        best_x = x_search.copy()
                        best_f = f_search
                        best_h = h_search
                        if use_improvement_tracking:
                            best_f_raw = f_current_raw
                    break
                if n_eval >= maxiter:
                    break
        if not improved and n_eval < maxiter:
            directions = _generate_orthogonal_directions(n, iteration)
            for direction in directions:
                if n_eval >= maxiter:
                    break
                x_trial = x_current + mesh_size * direction
                x_trial = np.clip(x_trial, bounds[:, 0], bounds[:, 1])
                if np.linalg.norm(x_trial - x_current) < 1e-12:
                    continue
                f_trial, h_trial = _barrier_objective(x_trial, h_max)
                n_eval += 1
                accept = False
                if h_trial <= h_current:
                    if f_trial < f_current - 1e-12:
                        accept = True
                elif h_trial < h_current:
                    accept = True
                if accept:
                    x_current = x_trial
                    f_current = f_trial
                    h_current = h_trial
                    improved = True
                    if use_improvement_tracking:
                        f_current_raw = objective(x_trial)
                        n_eval += 1
                    if h_trial <= best_h and f_trial <= best_f:
                        best_x = x_trial.copy()
                        best_f = f_trial
                        best_h = h_trial
                        if use_improvement_tracking:
                            best_f_raw = f_current_raw
                    mesh_size = min(mesh_size * mesh_coarsen_factor, initial_mesh_size)
                    no_improvement_count = 0
                    break
        if not improved:
            mesh_size *= mesh_refine_factor
            no_improvement_count += 1
            h_max = max(h_max * 0.9, h_current, 1e-06)
            if no_improvement_count > 25:
                mesh_size = initial_mesh_size * 0.1
                h_max = max(h_current * 2.0, 0.001)
                no_improvement_count = 0
        if use_improvement_tracking and iteration > 0:
            improved_flag = assess_progress(
                h_total=h_current,
                h_best=best_h,
                J_total=0.0,
                J_best=0.0,
                f=f_current_raw if use_improvement_tracking else f_current,
                f_best=best_f_raw if use_improvement_tracking else best_f,
            )
            if not improved_flag:
                no_improvement_iters += 1
                if no_improvement_iters >= improvement_patience:
                    break
            else:
                no_improvement_iters = 0
                mesh_size = min(mesh_size * mesh_coarsen_factor, initial_mesh_size)
        if n_eval >= maxiter:
            break
    return (best_x, best_f)
