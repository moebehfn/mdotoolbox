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
        "baco": ["BACOSubsystem", "BACOSystem", "BayesianCollaborativeOptimization"],
        "co": ["COSubsystem", "COSystem", "CollaborativeOptimization"],
        "eco": ["ECOSubsystem", "ECOSystem", "EnhancedCollaborativeOptimization"],
        "ico": ["ICOSubsystem", "ICOSystem", "ImprovedCollaborativeOptimization"],
        "mco": ["MCOSubsystem", "MCOSystem", "ModifiedCollaborativeOptimization"],
        "base_classes": ["BaseSubsystem", "BaseSystem", "BaseSolver"],
        "framework_utils": [
            "extract_subsystem_targets",
            "compute_all_subsystem_discrepancies",
            "build_subsystem_objective_closure",
            "build_subsystem_constraints_closure",
            "build_system_objective_closure",
            "build_system_constraints_closure",
        ],
    },
)
if TYPE_CHECKING:
    from .baco import (  # noqa: F401
        BACOSubsystem,
        BACOSystem,
        BayesianCollaborativeOptimization,
    )
    from .base_classes import BaseSolver, BaseSubsystem, BaseSystem  # noqa: F401
    from .co import CollaborativeOptimization, COSubsystem, COSystem  # noqa: F401
    from .eco import (  # noqa: F401
        ECOSubsystem,
        ECOSystem,
        EnhancedCollaborativeOptimization,
    )
    from .framework_utils import (  # noqa: F401
        build_subsystem_constraints_closure,
        build_subsystem_objective_closure,
        build_system_constraints_closure,
        build_system_objective_closure,
        compute_all_subsystem_discrepancies,
        extract_subsystem_targets,
    )
    from .ico import (  # noqa: F401
        ICOSubsystem,
        ICOSystem,
        ImprovedCollaborativeOptimization,
    )
    from .mco import (  # noqa: F401
        MCOSubsystem,
        MCOSystem,
        ModifiedCollaborativeOptimization,
    )
