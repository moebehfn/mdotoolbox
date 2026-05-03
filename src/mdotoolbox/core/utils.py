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

"""src/mdotoolbox/core/utils.py"""

import numpy as np
import typeguard


def type_check(obg, typ) -> bool:
    """Check if object matches given type or Union of types."""
    try:
        typeguard.check_type(obg, typ)
        return True
    except typeguard.TypeCheckError:
        return False


def is_valid_matrix(obj) -> bool:
    """Check if object is a valid 2D matrix with numeric or boolean values."""
    if type_check(obj, np.ndarray):
        return obj.ndim == 2 and (
            np.issubdtype(obj.dtype, np.number) or np.issubdtype(obj.dtype, np.bool_)
        )
    if not type_check(obj, (list, np.ndarray)):
        return False
    if not obj:
        return False
    if not all((type_check(row, (list, np.ndarray)) for row in obj)):
        return False
    first_row_length = len(obj[0])
    for row in obj:
        if not row:
            return False
        if len(row) != first_row_length:
            return False
    return True


def assess_progress(
    h_total: float,
    h_best: float,
    J_total: float,
    J_best: float,
    f: float,
    f_best: float,
    eta: float = 1e-08,
    epsilon_h: float = 1e-06,
    epsilon_J: float = 1e-06,
    mode: str = "hJf",
) -> bool:
    """Assess whether the current iterate represents lexicographic progress."""
    _VALID_MODES = {"hJf", "Jf", "hf", "f"}
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
    if eta < 0:
        raise ValueError("eta must be non-negative")
    if epsilon_h <= 0 or epsilon_J <= 0:
        raise ValueError("epsilon_h and epsilon_J must be strictly positive")
    eta = min(eta, epsilon_h / 100, epsilon_J / 100)
    if mode == "f":
        return f < f_best
    if mode == "hf":
        h_newly_converged = h_total <= epsilon_h and h_best > epsilon_h
        if h_newly_converged or h_total < h_best - eta:
            return True
        h_maintained = h_total <= h_best + eta or h_total <= epsilon_h
        if h_maintained and f < f_best:
            return True
        return False
    if mode == "Jf":
        J_i_newly_converged = J_total <= epsilon_J and J_best > epsilon_J
        if J_i_newly_converged or J_total < J_best - eta:
            return True
        J_i_maintained = J_total <= J_best + eta or J_total <= epsilon_J
        if J_i_maintained and f < f_best:
            return True
        return False
    h_newly_converged = h_total <= epsilon_h and h_best > epsilon_h
    if h_newly_converged or h_total < h_best - eta:
        return True
    h_maintained = h_total <= h_best + eta or h_total <= epsilon_h
    J_i_newly_converged = J_total <= epsilon_J and J_best > epsilon_J
    J_i_improved = J_i_newly_converged or J_total < J_best - eta
    if h_maintained and J_i_improved:
        return True
    J_i_maintained = J_total <= J_best + eta or J_total <= epsilon_J
    if h_maintained and J_i_maintained and (f < f_best):
        return True
    return False


def format_vars(var, precision: int = 6) -> str:
    """Format scalar or array variable to string with scientific notation."""
    if type_check(var, np.ndarray):
        return np.array2string(
            var,
            precision=precision,
            separator=", ",
            formatter={"float_kind": lambda x: f"{x:+.{precision}e}"},
        )
    else:
        return f"{var:+.{precision}e}"


def decompose_seconds(td: float):
    """Decompose time duration in seconds into components."""
    d, r = divmod(td, 3600 * 24)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    d = int(d)
    h = int(h)
    m = int(m)
    s_int = int(s)
    ms = (s - s_int) * 1000
    return (d, h, m, s_int, ms)


def tokenize_time(d, h, m, s, ms):
    """Convert time components to formatted string tokens."""
    fields = [f"{d} D", f"{h} H", f"{m} M", f"{s} S", f"{ms:.3f} MS"]
    while fields and (fields[0].startswith("0 ") or fields[0].startswith("0.000")):
        fields.pop(0)
    if not fields:
        fields = ["0 S"]
    return fields


def format_time_multi(*durations):
    """Format multiple time durations with aligned columns."""
    token_rows = []
    for td in durations:
        parts = tokenize_time(*decompose_seconds(td))
        token_rows.append(parts)
    max_cols = max((len(row) for row in token_rows))
    for row in token_rows:
        row[:0] = [""] * (max_cols - len(row))
    col_widths = []
    for col in range(max_cols):
        col_widths.append(max((len(row[col]) for row in token_rows)))
    result = []
    for row in token_rows:
        aligned = " ".join((row[col].rjust(col_widths[col]) for col in range(max_cols)))
        result.append(aligned)
    return result


def compute_Ji(y_i_func, z_bar, x_bar_i, y_bar_i, y_bar_coupled, z_under_i, x_under_i):
    """Compute subsystem discrepancy J_i for collaborative optimization."""
    input_vars = np.concatenate([z_under_i, x_under_i, y_bar_coupled])
    y_i = np.atleast_1d(y_i_func(*input_vars))
    J_i = (
        np.linalg.norm(z_bar - z_under_i) ** 2
        + (np.linalg.norm(x_bar_i - x_under_i) ** 2 if len(x_bar_i) > 0 else 0.0)
        + np.linalg.norm(y_bar_i - y_i) ** 2
    )
    return (J_i, y_i)


def clip_to_bounds(x, bounds):
    """Enforce variable bounds by clipping values to feasible range."""
    x_clipped = x.copy()
    for i, (lower, upper) in enumerate(bounds):
        x_clipped[i] = np.clip(x_clipped[i], lower, upper)
    return x_clipped
