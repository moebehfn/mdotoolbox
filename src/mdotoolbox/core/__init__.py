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

from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "base": [
            "DoE",
            "Function",
            "Constraint",
            "Problem",
            "ParetoEntry",
            "update_pareto",
            "pareto_archive_to_df",
            "BestSolution",
            "Results",
            "BudgetManager",
            "TextColor",
        ],
        "exceptions": ["BudgetExhausted", "ConstraintError"],
        "mdo": ["Discipline", "MDOProblem"],
        "utils": [
            "type_check",
            "is_valid_matrix",
            "assess_progress",
            "format_vars",
            "decompose_seconds",
            "tokenize_time",
            "format_time_multi",
            "compute_Ji",
            "clip_to_bounds",
        ],
    },
)
if TYPE_CHECKING:
    from .base import (  # noqa: F401
        BestSolution,
        BudgetManager,
        Constraint,
        DoE,
        Function,
        ParetoEntry,
        Problem,
        Results,
        TextColor,
        pareto_archive_to_df,
        update_pareto,
    )
    from .exceptions import BudgetExhausted, ConstraintError  # noqa: F401
    from .mdo import Discipline, MDOProblem  # noqa: F401
    from .utils import (  # noqa: F401
        assess_progress,
        clip_to_bounds,
        compute_Ji,
        decompose_seconds,
        format_time_multi,
        format_vars,
        is_valid_matrix,
        tokenize_time,
        type_check,
    )
