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

from __future__ import annotations

"\nsrc/mdotoolbox/core/base.py\n\nDefines baseline objects for the library.\n"
import warnings  # noqa: E402
from collections.abc import Callable, Iterable  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import TYPE_CHECKING, Literal

import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
from jaxtyping import Bool, Float, Int  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd
from .utils import (  # noqa: E402
    assess_progress,
    format_time_multi,
    format_vars,
    is_valid_matrix,
    type_check,
)

PyScalar = int | float | bool
NumPyScalar = np.bool_ | np.integer | np.floating
ScalarLike = (
    PyScalar
    | Float[np.ndarray, ""]  # noqa: F722
    | Int[np.ndarray, ""]  # noqa: F722
    | Bool[np.ndarray, ""]  # noqa: F722
)
ArrayLike = (
    list[PyScalar]
    | Float[np.ndarray, "N"]  # noqa: F821
    | Int[np.ndarray, "N"]  # noqa: F821
    | Bool[np.ndarray, "N"]  # noqa: F821
)
MatrixLike = (
    list[list[PyScalar]]
    | Float[np.ndarray, "H W"]  # noqa: F722
    | Int[np.ndarray, "H W"]  # noqa: F722
    | Bool[np.ndarray, "H W"]  # noqa: F722
)


@dataclass
class DoE:
    """Design of Experiments (DoE) container for storing sampling data."""

    x: list | np.ndarray
    y: dict | list | np.ndarray
    constraint_violation: np.ndarray | None = None

    def __post_init__(self):
        """Validate and normalize data after initialization."""
        if type_check(self.y, dict):
            first_key = list(self.y.keys())[0]
            n_outputs = len(self.y[first_key])
            for key, val in self.y.items():
                if not type_check(val, ArrayLike | MatrixLike):
                    raise TypeError(
                        f"y['{key}'] must be an ArrayLike or a MatrixLike object."
                    )
                if not len(val) == n_outputs:
                    raise ValueError(
                        f"All outputs must have same length. y['{key}'] has length {len(val)}, expected {n_outputs}"
                    )
            self.y = {key: np.array(val) for key, val in self.y.items()}
        elif type_check(self.y, ArrayLike):
            n_outputs = len(self.y)
            self.y = {"y": np.array(self.y)}
        if type_check(self.x, MatrixLike):
            if not is_valid_matrix(self.x):
                raise ValueError(
                    "x must be a valid MatrixLike object. Verify the number of items in each entry."
                )
            n_rows = self._get_n_rows(self.x)
            if n_rows != n_outputs:
                raise ValueError(
                    f"Number of rows in x ({n_rows}) must match length of outputs ({n_outputs})"
                )
        elif type_check(self.x, ArrayLike):
            if len(self.x) != n_outputs:
                raise ValueError(
                    f"Length of x ({len(self.x)}) must match length of outputs ({n_outputs})"
                )
        else:
            raise ValueError("x must either be a vector or a matrix.")
        self.x = np.array(self.x)
        if self.constraint_violation is not None:
            self.constraint_violation = np.array(self.constraint_violation).ravel()
            if len(self.constraint_violation) != n_outputs:
                raise ValueError(
                    f"constraint_violation length ({len(self.constraint_violation)}) must match number of samples ({n_outputs})"
                )

    @staticmethod
    def _get_n_rows(matrix: np.array) -> int:
        """Get number of rows in a matrix."""
        if type_check(matrix, np.ndarray):
            return matrix.shape[0]
        return len(matrix)

    def update_DoE(
        self,
        x_n: ArrayLike | ScalarLike,
        y_n: ArrayLike | ScalarLike,
        constraint_violation_n: float | None = None,
        tol: float = 1e-10,
        drop_duplicates: bool = True,
    ):
        """Add a new sample point to the Design of Experiments."""
        if not type_check(tol, float):
            raise TypeError("tol must be a float.")
        if not type_check(drop_duplicates, bool):
            raise TypeError("drop_duplicates must be a bool.")
        if isinstance(y_n, dict):
            if set(y_n.keys()) != set(self.y.keys()):
                raise ValueError(
                    f"y_n keys {set(y_n.keys())} must match existing keys {set(self.y.keys())}"
                )
            for key, val in y_n.items():
                if not type_check(val, ArrayLike | ScalarLike):
                    raise TypeError(
                        f"y_n['{key}'] must be a ScalarLike or an ArrayLike object."
                    )
        elif len(self.y) == 1:
            key = list(self.y.keys())[0]
            y_n = {key: y_n}
        else:
            raise ValueError(
                f"DoE has multiple outputs {list(self.y.keys())}, y_n must be a dictionary"
            )
        if not type_check(x_n, ArrayLike | ScalarLike):
            raise TypeError("x_n must be a ScalarLike or an ArrayLike object.")
        x_n = np.array([x_n]).ravel()
        x_0 = np.array([self.x[0]]).ravel()
        if len(x_n) != len(x_0):
            raise ValueError(f"New DoE x input must contain {len(x_0)} values.")
        n_samples = self.x.shape[0]
        duplicates = []
        for i in range(n_samples):
            if np.allclose(self.x[i], x_n, rtol=tol, atol=tol):
                duplicates.append(i)
        if duplicates and drop_duplicates:
            return
        if self.x.ndim == 1:
            self.x = np.append(self.x, x_n)
        else:
            self.x = np.vstack([self.x, x_n.reshape(1, -1)])
        for key in self.y.keys():
            val = np.array(y_n[key]).ravel()
            if self.y[key].ndim == 1:
                self.y[key] = np.append(self.y[key], val)
            else:
                self.y[key] = np.vstack([self.y[key], val.reshape(1, -1)])
        if self.constraint_violation is not None or constraint_violation_n is not None:
            if self.constraint_violation is None:
                self.constraint_violation = np.zeros(n_samples)
            if constraint_violation_n is None:
                constraint_violation_n = 0.0
            self.constraint_violation = np.append(
                self.constraint_violation, constraint_violation_n
            )

    def to_dict(self) -> dict:
        """Export DoE data as a dictionary."""
        result = {"x": self.x}
        result.update(self.y)
        if self.constraint_violation is not None:
            result["constraint_violation"] = self.constraint_violation
        return result

    def get_feasible_indices(self, tol: float = 1e-06) -> np.ndarray:
        """Get indices of feasible points in the DoE."""
        if self.constraint_violation is None:
            return np.arange(len(self.x))
        return np.where(self.constraint_violation <= tol)[0]

    def get_best_feasible(self, objective_key: str = "obj", tol: float = 1e-06):
        """Find the best (minimum objective) feasible point in the DoE."""
        feasible_idx = self.get_feasible_indices(tol)
        if len(feasible_idx) == 0:
            min_violation_idx = np.argmin(self.constraint_violation)
            return (
                self.x[min_violation_idx],
                self.y[objective_key][min_violation_idx],
                self.constraint_violation[min_violation_idx],
            )
        feasible_objectives = self.y[objective_key][feasible_idx]
        best_feasible_idx = feasible_idx[np.argmin(feasible_objectives)]
        return (
            self.x[best_feasible_idx],
            self.y[objective_key][best_feasible_idx],
            self.constraint_violation[best_feasible_idx]
            if self.constraint_violation is not None
            else 0.0,
        )

    def get_f_min(self, objective_key: str = "obj", tol: float = 1e-06) -> float:
        """Get the minimum objective value from feasible points."""
        _, f_min, _ = self.get_best_feasible(objective_key, tol)
        return f_min

    def __str__(self) -> str:
        """String representation of DoE"""
        n_samples = len(self.x)
        n_vars = len(self.x[0]) if len(self.x.shape) > 1 else 1
        output_keys = list(self.y.keys())
        has_cv = self.constraint_violation is not None
        lines = []
        lines.append(f"DoE: {n_samples} samples, {n_vars} variables")
        lines.append(f"Outputs: {', '.join(output_keys)}")
        if has_cv:
            n_feasible = len(self.get_feasible_indices())
            lines.append(f"Feasible points: {n_feasible}/{n_samples}")
        return " | ".join(lines)

    def __repr__(self) -> str:
        """Detailed representation of DoE"""
        return f"DoE(x={self.x.shape}, outputs={list(self.y.keys())}, n_samples={len(self.x)})"


@dataclass
class Function:
    """Callable function wrapper for optimization problems."""

    func: Callable
    x: str | list[str] | np.typing.NDArray[str]
    name: str = ""

    def __post_init__(self):
        if not type_check(self.func, Callable):
            raise TypeError("func must be a callable function.")
        if not type_check(self.x, str | list[str] | np.typing.NDArray[str]):
            raise TypeError(
                "x is a vector therefore x_str must be a list-like of strings"
            )
        self.x = np.array([self.x], dtype=str).ravel()
        if not type_check(self.name, str):
            raise TypeError("name must be a string.")

    def __str__(self) -> str:
        """String representation of Function"""
        name_str = f"{self.name}: " if self.name else ""
        vars_str = ", ".join(self.x)
        return f"{name_str}f({vars_str})"

    def __repr__(self) -> str:
        """Detailed representation of Function"""
        return f"Function(name='{self.name}', variables={list(self.x)})"


@dataclass
class Constraint:
    """Constraint definition for optimization problems."""

    func: Function
    ctype: str = "ge"
    value: int | float = 0.0

    def __post_init__(self):
        if not type_check(self.func, Function):
            raise TypeError("func must be a Function object.")
        if not type_check(self.ctype, str):
            raise TypeError("ctype must be a string.")
        if not type_check(self.value, int | float):
            raise TypeError("value must be int or float.")
        if self.ctype not in ["ge", "le", "eq"]:
            raise ValueError(
                "Only:\n\t- ge: greater than;\n\t- le: smaller than;\n\t- eq: equal to;\nare acceptable."
            )

    def __str__(self) -> str:
        """String representation of Constraint"""
        func_name = self.func.name if self.func.name else "f"
        vars_str = ", ".join(self.func.x)
        if self.ctype == "ge":
            op = ">="
        elif self.ctype == "le":
            op = "<="
        else:
            op = "="
        return f"{func_name}({vars_str}) {op} {self.value}"

    def __repr__(self) -> str:
        """Detailed representation of Constraint"""
        return f"Constraint(type='{self.ctype}', value={self.value}, func={self.func.name})"


@dataclass
class Problem:
    """Complete optimization problem definition."""

    objective: Function
    constraints: Iterable
    ubounds: ArrayLike
    lbounds: ArrayLike
    maximize: bool = False
    tol: float = 1e-06
    name: str = ""

    def __post_init__(self):
        """Verify input lengths"""
        if not type_check(self.objective, Function):
            raise TypeError("objective must be a Function object.")
        if not type_check(self.constraints, (list, set, tuple, np.ndarray)):
            raise TypeError(
                "constraints must be a list or array of Constraint objects."
            )
        for idx, const in enumerate(self.constraints):
            if not type_check(const, Constraint):
                raise TypeError(f"Constraint {idx} must be a Constraint object.")
            if type_check(const.func.x, npt.NDArray):
                if const.func.x.shape != self.objective.x.shape:
                    raise ValueError(
                        "The constraints must share the same variables as the objective function."
                    )
            const.func.name = f"c{idx}-{const.ctype}-{const.value}"
        if not type_check(self.ubounds, ArrayLike):
            raise TypeError("ubounds must be an ArrayLike object.")
        if not type_check(self.lbounds, ArrayLike):
            raise TypeError("lbounds must be an ArrayLike object.")
        if len(self.lbounds) != len(self.objective.x):
            raise ValueError(
                f"There must be {len(self.objective.x)} lower bounds, only {len(self.lbounds)} were given"
            )
        self.lbounds = np.array(self.lbounds)
        if len(self.ubounds) != len(self.objective.x):
            raise ValueError(
                f"There must be {len(self.objective.x)} upper bounds, only {len(self.ubounds)} were given"
            )
        self.ubounds = np.array(self.ubounds)
        self.bounds = np.array([self.lbounds, self.ubounds]).T
        if not type_check(self.maximize, bool):
            raise TypeError("maximize must be a boolean.")
        if self.maximize:
            self.objective.func = lambda *args: -1.0 * self.objective.func(*args)
        if not type_check(self.tol, float):
            raise TypeError("tol must be a float.")
        if not type_check(self.name, str):
            raise TypeError("name must be a string.")

    def evaluate(self, x, f: bool, c: bool):
        """Evaluate objective and/or constraints at a given point."""
        if f and c:
            f_val = self.objective.func(*x)
            f_arr = np.asarray(f_val)
            if f_arr.size != 1:
                raise ValueError(
                    f"Objective function '{self.objective.name}' returned shape {f_arr.shape}, expected scalar. Callable must return a single numeric value."
                )
            c_list = []
            for const in self.constraints:
                c_val = const.func.func(*x)
                c_arr = np.asarray(c_val)
                if c_arr.size != 1:
                    raise ValueError(
                        f"Constraint function '{const.func.name}' returned shape {c_arr.shape}, expected scalar. Callable must return a single numeric value."
                    )
                c_list.append(float(c_arr.flat[0]))
            return (f_arr.flat[0], np.array(c_list))
        elif f and (not c):
            f_val = self.objective.func(*x)
            f_arr = np.asarray(f_val)
            if f_arr.size != 1:
                raise ValueError(
                    f"Objective function '{self.objective.name}' returned shape {f_arr.shape}, expected scalar. Callable must return a single numeric value."
                )
            return (f_arr.flat[0], None)
        elif not f and c:
            c_list = []
            for const in self.constraints:
                c_val = const.func.func(*x)
                c_arr = np.asarray(c_val)
                if c_arr.size != 1:
                    raise ValueError(
                        f"Constraint function '{const.func.name}' returned shape {c_arr.shape}, expected scalar. Callable must return a single numeric value."
                    )
                c_list.append(float(c_arr.flat[0]))
            return (None, np.array(c_list))
        else:
            raise ValueError(
                "You must evaluate either f, or c, or both, but not neither."
            )

    def initial_DoE(self, n) -> DoE:
        """Generate initial Design of Experiments using Latin Hypercube Sampling."""
        if type_check(n, Callable):
            n = n(len(self.bounds))
        elif type_check(n, (int, float)) and n < 0:
            warnings.warn(
                "n should be a positive integer, converting to absolute value.",
                stacklevel=2,
            )
            n = abs(n)
        elif type_check(n, float):
            n = int(n)
        elif not type_check(n, int):
            raise ValueError(
                "n must be an integer or a callable function of the number of variables."
            )
        lims = []
        for bound in self.bounds:
            if np.isfinite(bound[0]) and np.isfinite(bound[1]):
                lims.append(bound)
            elif not np.isfinite(bound[0]) and np.isfinite(bound[1]):
                lims.append([min(-bound[1], bound[1] - 50), bound[1]])
            elif np.isfinite(bound[0]) and (not np.isfinite(bound[1])):
                lims.append([bound[0], max(-bound[0], bound[0] + 50)])
            else:
                lims.append([-100.0, +100.0])
        lims = np.array(lims)
        from smt.sampling_methods import LHS

        x = LHS(xlimits=lims)(n)
        y = {"obj": np.array([self.objective.func(*row) for row in x])}
        yc = {}
        for const in self.constraints:
            yc[const.func.name] = np.array([const.func.func(*row) for row in x])
        y.update(yc)
        if self.constraints:
            constraint_violation = self.compute_constraint_violation(yc)
        else:
            constraint_violation = np.zeros(n)
        doe = DoE(x, y, constraint_violation)
        return doe

    def compute_constraint_violation(self, constraint_values: dict) -> np.ndarray:
        """Compute total constraint violation for each sample point."""
        violation = 0
        if self.constraints:
            for const in self.constraints:
                c_vals = constraint_values[const.func.name]
                if const.ctype == "ge":
                    violation += np.maximum(0, const.value - c_vals) ** 2
                elif const.ctype == "le":
                    violation += np.maximum(0, c_vals - const.value) ** 2
                elif const.ctype == "eq":
                    violation += np.abs(c_vals - const.value) ** 2
        return violation

    def __str__(self) -> str:
        """String representation of Problem"""
        lines = []
        obj_type = "max" if self.maximize else "min"
        obj_name = self.objective.name if self.objective.name else "f"
        vars_str = ", ".join(self.objective.x)
        lines.append(f"{obj_type} {obj_name}({vars_str})")
        if len(self.constraints) > 0:
            lines.append("subject to:")
            for i, const in enumerate(self.constraints):
                func_name = const.func.name if const.func.name else f"c{i}"
                const_vars = ", ".join(const.func.x)
                if const.ctype == "ge":
                    op = ">="
                elif const.ctype == "le":
                    op = "<="
                else:
                    op = "="
                lines.append(f"    {func_name}({const_vars}) {op} {const.value}")
        if len(self.objective.x) > 0:
            lines.append("bounds:")
            for i, var in enumerate(self.objective.x):
                lb = self.lbounds[i]
                ub = self.ubounds[i]
                lb_str = f"{lb:.2g}" if np.isfinite(lb) else "-inf"
                ub_str = f"{ub:.2g}" if np.isfinite(ub) else "+inf"
                lines.append(f"    {lb_str} <= {var} <= {ub_str}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Detailed representation of Problem"""
        n_vars = len(self.objective.x)
        n_constraints = len(self.constraints)
        return f"Problem(name='{self.name}', n_vars={n_vars}, n_constraints={n_constraints}, maximize={self.maximize})"


def _dominates(u: tuple[float, float], v: tuple[float, float]) -> bool:
    """Return True if u dominates v in a minimisation bi-objective space."""
    return (u[0] <= v[0] and u[1] <= v[1]) and (u[0] < v[0] or u[1] < v[1])


_SPACES: list[tuple[int, str, str]] = [
    (1, "f", "h_total"),
    (2, "f", "J_total"),
    (4, "h_total", "J_total"),
]


@dataclass
class ParetoEntry:
    """A single entry in the Pareto set."""

    f: float
    h_total: float
    J_total: float
    z: np.ndarray
    x: np.ndarray
    y: np.ndarray
    code: int

    def metric(self, key: str) -> float:
        return {"f": self.f, "h_total": self.h_total, "J_total": self.J_total}[key]

    @property
    def dominated_in_all(self) -> bool:
        return self.code == 0

    @property
    def nondominated_all_spaces(self) -> bool:
        """True if non-dominated in all three bi-objective spaces simultaneously."""
        return self.code == 7


def update_pareto(
    pareto_set: list[ParetoEntry],
    f: float,
    h_total: float,
    J_total: float,
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> list[ParetoEntry]:
    """Insert a new iterate into the Pareto set if it is non-dominated
    in at least one bi-objective space, and update membership codes of existing
    entries."""
    candidate = {"f": f, "h_total": h_total, "J_total": J_total}
    code = 0
    for s, key_a, key_b in _SPACES:
        a = (candidate[key_a], candidate[key_b])
        dominated = any(
            _dominates((e.metric(key_a), e.metric(key_b)), a)
            for e in pareto_set
            if e.code // s % 2 == 1
        )
        if not dominated:
            code += s
    if code == 0:
        return pareto_set
    for e in pareto_set:
        for s, key_a, key_b in _SPACES:
            if e.code // s % 2 == 1:
                b = (e.metric(key_a), e.metric(key_b))
                a = (candidate[key_a], candidate[key_b])
                if _dominates(a, b):
                    e.code -= s
    pareto_set = [e for e in pareto_set if not e.dominated_in_all]
    pareto_set.append(
        ParetoEntry(
            f=f,
            h_total=h_total,
            J_total=J_total,
            z=z.copy(),
            x=x.copy(),
            y=y.copy(),
            code=code,
        )
    )
    return pareto_set


PARETO_COLUMNS = ["f_sys", "J_total", "h_total", "z", "x", "y", "code"]


def pareto_archive_to_df(pareto_set: list[ParetoEntry]) -> pd.DataFrame:
    """Convert an internal Pareto set list to a DataFrame."""
    import pandas as pd

    if not pareto_set:
        return pd.DataFrame(columns=PARETO_COLUMNS)
    df = pd.DataFrame([
        {
            "f_sys": e.f,
            "J_total": e.J_total,
            "h_total": e.h_total,
            "z": e.z,
            "x": e.x,
            "y": e.y,
            "code": e.code,
        }
        for e in pareto_set
    ])
    fh = []
    fj = []
    hj = []
    for entry in df.code:
        if entry == 1:
            fh.append(1)
            fj.append(0)
            hj.append(0)
        elif entry == 2:
            fh.append(0)
            fj.append(1)
            hj.append(0)
        elif entry == 3:
            fh.append(1)
            fj.append(1)
            hj.append(0)
        elif entry == 4:
            fh.append(0)
            fj.append(0)
            hj.append(1)
        elif entry == 5:
            fh.append(1)
            fj.append(0)
            hj.append(1)
        elif entry == 6:
            fh.append(0)
            fj.append(1)
            hj.append(1)
        elif entry == 7:
            fh.append(1)
            fj.append(1)
            hj.append(1)
    df["fh"] = fh
    df["fJ"] = fj
    df["hJ"] = hj
    return df


@dataclass
class BestSolution:
    """Tracks the best iterate found during optimization under the lexicographic
    progress criterion."""

    z: np.ndarray = field(default_factory=lambda: np.array([np.nan]))
    x: np.ndarray = field(default_factory=lambda: np.array([np.nan]))
    y: np.ndarray = field(default_factory=lambda: np.array([np.nan]))
    f: float = np.inf
    h_total: float = np.inf
    J_total: float = np.inf

    def update(
        self,
        z_new: np.ndarray,
        x_new: np.ndarray,
        y_new: np.ndarray,
        f_new: float,
        h_new: float,
        J_new: float,
        eta: float = 1e-08,
        epsilon_h: float = 1e-06,
        epsilon_J: float = 1e-06,
    ) -> bool:
        """Update the stored best iterate if the new point represents progress."""
        improved = assess_progress(
            h_total=h_new,
            h_best=self.h_total,
            J_total=J_new,
            J_best=self.J_total,
            f=f_new,
            f_best=self.f,
            eta=eta,
            epsilon_h=epsilon_h,
            epsilon_J=epsilon_J,
        )
        if improved:
            self.z = z_new.copy()
            self.x = x_new.copy()
            self.y = y_new.copy()
            self.f = f_new
            self.h_total = h_new
            self.J_total = J_new
        return improved

    def to_dict(self) -> dict:
        """Export best results as a dictionary."""
        return {
            "z": self.z,
            "x": self.x,
            "y": self.y,
            "f": self.f,
            "h_total": self.h_total,
            "J_total": self.J_total,
        }

    def __str__(self):
        results = [
            "Best Attained Results:",
            f"\t- (z) Best Shared Variables                 : {format_vars(self.z, 6)}",
            f"\t- (x) Best Local Variables                  : {format_vars(self.x, 6)}",
            f"\t- (y) Best Coupling Variables               : {format_vars(self.y, 6)}",
            f"\t- (f) Best Objective Value                  : {format_vars(self.f, 6)}",
            f"\t- (J_total) Best Total Discrepancy          : {format_vars(self.J_total, 6)}",
            f"\t- (h_total) Best Total Constraint Violation : {format_vars(self.h_total, 6)}",
        ]
        return "\n".join(results)

    def __repr__(self) -> str:
        return f"BestSolution(f={self.f:.6e}, h_total={self.h_total:.6e}, J_total={self.J_total:.6e})"


@dataclass
class Results:
    """Container for the complete output of an optimization run."""

    best: BestSolution = None
    converged: bool = None
    iterations: int = None
    evaluations: int = None
    elapsed_time: float = None
    history: pd.DataFrame | None = None
    pareto: pd.DataFrame = field(
        default_factory=lambda: __import__("pandas").DataFrame(columns=PARETO_COLUMNS)
    )

    def __post_init__(self):
        if self.best is None:
            self.best = BestSolution()
        self._pareto_archive: list[ParetoEntry] = []
        if not self.pareto.empty:
            self._pareto_archive = [
                ParetoEntry(
                    f=row["f_sys"],
                    h_total=row["h_total"],
                    J_total=row["J_total"],
                    z=row["z"],
                    x=row["x"],
                    y=row["y"],
                    code=int(row["code"]),
                )
                for _, row in self.pareto.iterrows()
            ]

    def update_pareto(
        self,
        f: float,
        h_total: float,
        J_total: float,
        z: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Pass the current iterate to the Pareto set update procedure."""
        self._pareto_archive = update_pareto(
            self._pareto_archive,
            f=f,
            h_total=h_total,
            J_total=J_total,
            z=z,
            x=x,
            y=y,
        )
        self.pareto = pareto_archive_to_df(self._pareto_archive)

    @property
    def pareto_all(self) -> pd.DataFrame:
        """Archive entries non-dominated in all three bi-objective spaces."""
        if self.pareto.empty:
            return self.pareto
        return self.pareto[self.pareto["code"] == 7].reset_index(drop=True)

    def __str__(self):
        conv_txt = (
            TextColor.grn + "Converged" + TextColor.clr
            if self.converged
            else TextColor.red + "Did not converge" + TextColor.clr
        )
        ts = format_time_multi(
            self.elapsed_time,
            self.elapsed_time / self.evaluations if self.evaluations else 0,
            self.elapsed_time / self.iterations if self.iterations else 0,
        )
        results = [
            80 * "=",
            f"{conv_txt}",
            80 * "-",
            "Results:",
            f"\t- Iterations                          : {self.iterations:>{max(len(str(self.iterations)), len(str(self.evaluations)))}}",
            f"\t- Evaluations                         : {self.evaluations:>{max(len(str(self.iterations)), len(str(self.evaluations)))}}",
            f"\t- Elapsed Time                        : {ts[0]}",
            f"\t- Average Time per Evaluation         : {ts[1]}",
            f"\t- Average Time per Iteration          : {ts[2]}",
            80 * "-",
            str(self.best),
            80 * "=",
        ]
        return "\n".join(results)

    def __repr__(self) -> str:
        status = "converged" if self.converged else "not_converged"
        return f"Results({status}, iters={self.iterations}, evals={self.evaluations}, {self.best!r})"


@dataclass
class BudgetManager:
    """Manages evaluation budget allocation for Collaborative Optimization frameworks."""

    mode: Literal["shared", "weighted", "fixed"] = "weighted"
    total_budget: int = None
    system_ratio: float = 0.5
    subsystem_weights: list[float] = None
    iteration_ratio: float = 0.05
    subsystem_budgets: list[int] = None
    system_budget: int = None

    def __post_init__(self):
        self._subsystem_used = []
        self._system_used = 0
        self._total_used = 0
        self._subsystem_allocated = None
        self._system_allocated = None
        self._validate()

    def _validate(self):
        """Validate budget configuration"""
        if self.mode == "shared":
            if self.total_budget is None:
                raise ValueError("total_budget required for 'shared' mode")
        elif self.mode == "weighted":
            if self.total_budget is None:
                raise ValueError("total_budget required for 'weighted' mode")
            if self.subsystem_weights is None:
                raise ValueError("subsystem_weights required for 'weighted' mode")
            if not np.isclose(sum(self.subsystem_weights), 1.0):
                raise ValueError("subsystem_weights must sum to 1.0")
            if not 0 < self.system_ratio < 1:
                raise ValueError("system_ratio must be between 0 and 1")
        elif self.mode == "fixed":
            if self.subsystem_budgets is None:
                raise ValueError("subsystem_budgets required for 'fixed' mode")
            if self.system_budget is None:
                raise ValueError("system_budget required for 'fixed' mode")
        else:
            raise ValueError("mode must be one of 'shared', 'weighted', or 'fixed'")

    def initialize(self, n_subsystems: int):
        """Initialize budget tracking for given number of subsystems"""
        self._subsystem_used = [0] * n_subsystems
        self._system_used = 0
        self._total_used = 0
        if self.mode == "weighted":
            if len(self.subsystem_weights) != n_subsystems:
                raise ValueError(
                    f"subsystem_weights length ({len(self.subsystem_weights)}) must match n_subsystems ({n_subsystems})"
                )
            subsystem_pool = self.total_budget * (1 - self.system_ratio)
            self._subsystem_allocated = [
                int(subsystem_pool * w) for w in self.subsystem_weights
            ]
            self._system_allocated = int(self.total_budget * self.system_ratio)
        elif self.mode == "fixed":
            if len(self.subsystem_budgets) != n_subsystems:
                raise ValueError(
                    f"subsystem_budgets length ({len(self.subsystem_budgets)}) must match n_subsystems ({n_subsystems})"
                )
            self._subsystem_allocated = list(self.subsystem_budgets)
            self._system_allocated = self.system_budget
            self.total_budget = sum(self.subsystem_budgets) + self.system_budget
        elif self.mode == "shared":
            pass

    def get_subsystem_maxiter(self, subsystem_idx: int) -> int:
        """Get maxiter for a subsystem."""
        if self.mode == "shared":
            remaining = self.total_budget - self._total_used
            return max(1, remaining)
        allocated = self._subsystem_allocated[subsystem_idx]
        used = self._subsystem_used[subsystem_idx]
        remaining = allocated - used
        if remaining <= 0:
            return 0
        if self.iteration_ratio is not None:
            per_iter = max(1, int(np.floor(allocated * self.iteration_ratio)))
            return min(per_iter, remaining)
        return remaining

    def get_system_maxiter(self) -> int:
        """Get maxiter for system optimization."""
        if self.mode == "shared":
            remaining = self.total_budget - self._total_used
            return max(1, remaining)
        remaining = self._system_allocated - self._system_used
        if remaining <= 0:
            return 0
        if self.iteration_ratio is not None:
            per_iter = max(
                1, int(np.floor(self._system_allocated * self.iteration_ratio))
            )
            return min(per_iter, remaining)
        return remaining

    def record_subsystem_evals(self, subsystem_idx: int, n_evals: int):
        """Record evaluations used by a subsystem"""
        self._subsystem_used[subsystem_idx] += n_evals
        self._total_used += n_evals

    def record_system_evals(self, n_evals: int):
        """Record evaluations used by system"""
        self._system_used += n_evals
        self._total_used += n_evals

    def is_subsystem_exhausted(self, subsystem_idx: int) -> bool:
        """Check if subsystem budget is exhausted"""
        if self.mode == "shared":
            return self._total_used >= self.total_budget
        return (
            self._subsystem_used[subsystem_idx]
            >= self._subsystem_allocated[subsystem_idx]
        )

    def is_system_exhausted(self) -> bool:
        """Check if system budget is exhausted"""
        if self.mode == "shared":
            return self._total_used >= self.total_budget
        return self._system_used >= self._system_allocated

    def is_total_exhausted(self) -> bool:
        """Check if total budget is exhausted"""
        return self._total_used >= self.total_budget

    def get_remaining_total(self) -> int:
        """Get remaining total budget"""
        return max(0, self.total_budget - self._total_used)

    def get_status(self) -> dict:
        """Get current budget status"""
        status = {
            "mode": self.mode,
            "total_budget": self.total_budget,
            "total_used": self._total_used,
            "total_remaining": self.get_remaining_total(),
            "subsystem_used": self._subsystem_used.copy(),
            "system_used": self._system_used,
        }
        if self.mode in ["weighted", "fixed"]:
            status["subsystem_allocated"] = self._subsystem_allocated.copy()
            status["system_allocated"] = self._system_allocated
            status["subsystem_remaining"] = [
                a - u for a, u in zip(self._subsystem_allocated, self._subsystem_used)
            ]
            status["system_remaining"] = self._system_allocated - self._system_used
        return status

    def __str__(self) -> str:
        """Print budget allocation summary"""
        lines = []
        lines.append(f"Mode: {self.mode}")
        lines.append(f"Total Budget: {self.total_budget}")
        if self._subsystem_allocated is None:
            lines.append("(Not initialized - call initialize() first)")
            return "\n".join(lines)
        if self.mode == "shared":
            lines.append(f"Max per iteration: {self.total_budget} (shared pool)")
        elif self.mode in ["weighted", "fixed"]:
            lines.append("\nSubsystem Allocations:")
            for i, alloc in enumerate(self._subsystem_allocated):
                max_per_iter = (
                    max(1, int(np.floor(alloc * self.iteration_ratio)))
                    if self.iteration_ratio is not None
                    else alloc
                )
                lines.append(
                    f"  Subsystem {i + 1}: {alloc} total, {max_per_iter} max/iter"
                )
            max_sys_per_iter = (
                max(1, int(np.floor(self._system_allocated * self.iteration_ratio)))
                if self.iteration_ratio is not None
                else self._system_allocated
            )
            lines.append(
                f"\nSystem Allocation: {self._system_allocated} total, {max_sys_per_iter} max/iter"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Detailed representation of BudgetManager"""
        return f"BudgetManager(mode='{self.mode}', total_budget={self.total_budget}, used={self._total_used})"


class TextColor:
    """ANSI escape sequences for text formatting in terminal"""

    clr = "\x1b[0m"
    blk = "\x1b[30m"
    red = "\x1b[31m"
    grn = "\x1b[32m"
    ylw = "\x1b[33m"
    blu = "\x1b[34m"
    mgy = "\x1b[35m"
    cyn = "\x1b[36m"
    wht = "\x1b[37m"
    brw = "\x1b[38;5;94m"
    org = "\x1b[38;5;208m"
