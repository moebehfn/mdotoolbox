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
        "electronic_packaging": ["electronic_packaging_problem"],
        "heart_dipole": ["heart_dipole_problem"],
        "power_converter": ["power_converter_problem"],
        "propane_combustion": ["propane_combustion_problem"],
        "scalable": ["scalable_mdo_problem"],
        "sellar": ["sellar_problem"],
        "speed_reducer": ["speed_reducer_problem"],
    },
)
if TYPE_CHECKING:
    from .electronic_packaging import electronic_packaging_problem  # noqa: F401
    from .heart_dipole import heart_dipole_problem  # noqa: F401
    from .power_converter import power_converter_problem  # noqa: F401
    from .propane_combustion import propane_combustion_problem  # noqa: F401
    from .scalable import scalable_mdo_problem  # noqa: F401
    from .sellar import sellar_problem  # noqa: F401
    from .speed_reducer import speed_reducer_problem  # noqa: F401
