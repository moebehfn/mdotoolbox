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

"""Multi-Disciplinary Optimization (MDO) problem definitions."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .base import Constraint, Function, type_check


@dataclass
class Discipline:
    """Single discipline (subsystem) in a Multi-Disciplinary Optimization problem."""

    name: str
    outputs: Function
    constraints: List[Constraint] = None
    shared_var_indices: np.ndarray = None
    local_var_indices: np.ndarray = None
    coupling_output_indices: np.ndarray = None
    coupling_input_indices: List[np.ndarray] = None

    def __post_init__(self):
        if not type_check(self.name, str):
            raise TypeError("name must be a string")
        if not type_check(self.outputs, Function):
            raise TypeError("outputs must be a Function")
        if self.constraints is None:
            self.constraints = []
        else:
            if not type_check(self.constraints, (list, tuple, np.ndarray)):
                raise TypeError("constraints must be a list of Constraint objects")
            for idx, const in enumerate(self.constraints):
                if not type_check(const, Constraint):
                    raise TypeError(f"Constraint {idx} must be a Constraint object")
        if self.shared_var_indices is not None:
            self.shared_var_indices = np.array(self.shared_var_indices)
        else:
            self.shared_var_indices = np.array([])
        if self.local_var_indices is not None:
            self.local_var_indices = np.array(self.local_var_indices)
        else:
            self.local_var_indices = np.array([])
        if self.coupling_output_indices is not None:
            self.coupling_output_indices = np.array(self.coupling_output_indices)
        else:
            self.coupling_output_indices = np.array([])
        if self.coupling_input_indices is None:
            self.coupling_input_indices = []
        else:
            self.coupling_input_indices = [
                np.array(idx_arr) for idx_arr in self.coupling_input_indices
            ]

    def evaluate(
        self, z_under: np.ndarray, x_under: np.ndarray, y_bar_coupled: np.ndarray
    ) -> np.ndarray:
        """Evaluate discipline outputs given current variable values."""
        z_under_local = (
            z_under[self.shared_var_indices]
            if len(self.shared_var_indices) > 0
            else np.array([])
        )
        x_under_local = (
            x_under[self.local_var_indices]
            if len(self.local_var_indices) > 0
            else np.array([])
        )
        inputs = np.concatenate([z_under_local, x_under_local, y_bar_coupled])
        return np.atleast_1d(self.outputs.func(*inputs))

    def evaluate_constraints(
        self, z_under: np.ndarray, x_under: np.ndarray, y_bar_coupled: np.ndarray
    ) -> np.ndarray:
        """Evaluate discipline-level constraints."""
        if not self.constraints:
            return np.array([])
        z_under_local = (
            z_under[self.shared_var_indices]
            if len(self.shared_var_indices) > 0
            else np.array([])
        )
        x_under_local = (
            x_under[self.local_var_indices]
            if len(self.local_var_indices) > 0
            else np.array([])
        )
        inputs = np.concatenate([z_under_local, x_under_local, y_bar_coupled])
        c_vals = []
        for const in self.constraints:
            c_vals.append(const.func.func(*inputs))
        return np.array(c_vals)


@dataclass
class MDOProblem:
    """Multi-Disciplinary Optimization (MDO) problem definition."""

    system_objective: Function
    disciplines: List[Discipline]
    n_shared: int
    n_local: List[int]
    n_coupling: List[int]
    shared_bounds: np.ndarray
    local_bounds: List[np.ndarray]
    system_constraints: List[Constraint] = None
    maximize: bool = False
    tol: float = 1e-06
    name: str = ""

    def __post_init__(self):
        """Validate MDO problem structure"""
        if not type_check(self.system_objective, Function):
            raise TypeError("system_objective must be a Function")
        if not type_check(self.disciplines, (list, tuple)):
            raise TypeError("disciplines must be a list of Discipline objects")
        for idx, disc in enumerate(self.disciplines):
            if not type_check(disc, Discipline):
                raise TypeError(f"Discipline {idx} must be a Discipline object")
        if not type_check(self.n_shared, int):
            raise TypeError("n_shared must be an integer")
        if self.n_shared < 0:
            raise ValueError("n_shared must be non-negative")
        if not type_check(self.n_local, (list, tuple, np.ndarray)):
            raise TypeError("n_local must be a list of integers")
        if len(self.n_local) != len(self.disciplines):
            raise ValueError("n_local must have one entry per discipline")
        if not type_check(self.n_coupling, (list, tuple, np.ndarray)):
            raise TypeError("n_coupling must be a list of integers")
        if len(self.n_coupling) != len(self.disciplines):
            raise ValueError("n_coupling must have one entry per discipline")
        if not type_check(self.shared_bounds, (list, tuple, np.ndarray)):
            raise TypeError("shared_bounds must be array-like")
        self.shared_bounds = np.array(self.shared_bounds)
        if self.shared_bounds.shape != (self.n_shared, 2):
            raise ValueError(f"shared_bounds must be ({self.n_shared}, 2)")
        if not type_check(self.local_bounds, (list, tuple)):
            raise TypeError("local_bounds must be a list of array-like bounds")
        if len(self.local_bounds) != len(self.disciplines):
            raise ValueError("local_bounds must have one entry per discipline")
        self.local_bounds = [np.array(bounds) for bounds in self.local_bounds]
        for idx, bounds in enumerate(self.local_bounds):
            if bounds.shape != (self.n_local[idx], 2):
                raise ValueError(
                    f"local_bounds[{idx}] must be ({self.n_local[idx]}, 2)"
                )
        if self.system_constraints is None:
            self.system_constraints = []
        else:
            for idx, const in enumerate(self.system_constraints):
                if not type_check(const, Constraint):
                    raise TypeError(
                        f"System constraint {idx} must be a Constraint object"
                    )
        if not type_check(self.maximize, bool):
            raise TypeError("maximize must be a boolean")
        if self.maximize:
            self.system_objective.func = lambda *args: (
                -1.0 * self.system_objective.func(*args)
            )
        if not type_check(self.tol, float):
            raise TypeError("tol must be a float")
        if not type_check(self.name, str):
            raise TypeError("name must be a string")
        self._validate_coupling_structure()

    def _validate_coupling_structure(self):
        """Validate that coupling indices are consistent"""
        total_coupling = sum(self.n_coupling)
        for disc in self.disciplines:
            n_outputs = len(disc.coupling_output_indices)
            if not n_outputs <= total_coupling:
                raise ValueError(
                    f"Discipline {disc.name} has more outputs than total coupling variables"
                )
            for coupled_idx_arr in disc.coupling_input_indices:
                if not all((idx < total_coupling for idx in coupled_idx_arr)):
                    raise ValueError(
                        f"Discipline {disc.name} references invalid coupling indices"
                    )

    def evaluate_system(
        self, z_bar: np.ndarray, x_bar: np.ndarray, y_bar: np.ndarray
    ) -> float:
        """Evaluate system objective."""
        inputs = np.concatenate([z_bar, x_bar, y_bar])
        return self.system_objective.func(*inputs)

    def evaluate_disciplines(
        self, z_bar: np.ndarray, x_bar: np.ndarray, y_bar: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Evaluate all discipline outputs."""
        outputs = {}
        x_offset = 0
        for disc in self.disciplines:
            n_local_disc = len(disc.local_var_indices)
            y_bar_coupled_disc = []
            for coupled_idx_arr in disc.coupling_input_indices:
                for idx in coupled_idx_arr:
                    y_bar_coupled_disc.append(y_bar[idx])
            y_bar_coupled_disc = (
                np.array(y_bar_coupled_disc) if y_bar_coupled_disc else np.array([])
            )
            x_bar_disc = (
                x_bar[x_offset : x_offset + n_local_disc]
                if n_local_disc > 0
                else np.array([])
            )
            outputs[disc.name] = disc.evaluate(z_bar, x_bar_disc, y_bar_coupled_disc)
            x_offset += n_local_disc
        return outputs

    def compute_coupling_residuals(
        self, z_bar: np.ndarray, x_bar: np.ndarray, y_bar: np.ndarray
    ) -> np.ndarray:
        """Compute coupling residuals: ||y_computed - y_bar||^2"""
        outputs = self.evaluate_disciplines(z_bar, x_bar, y_bar)
        residuals = []
        for disc in self.disciplines:
            y_computed = outputs[disc.name]
            y_bar_target = y_bar[disc.coupling_output_indices]
            if len(y_bar_target) > 0:
                residual = np.linalg.norm(y_computed - y_bar_target) ** 2
            else:
                residual = 0.0
            residuals.append(residual)
        return np.array(residuals)
