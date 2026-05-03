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
        "acquisition": ["ei", "pi", "ucb", "log_ei", "log_pi", "ei_cf", "log_ei_cf"],
        "builders": ["build_surrogate", "build_surrogate_dict"],
        "configurations": [
            "SMTGPConfig",
            "TorchGPConfig",
            "FastGPConfig",
            "StudentTConfig",
        ],
        "surrogates": ["TorchGP", "FastGP", "StudentTProcess"],
    },
)
if TYPE_CHECKING:
    from .acquisition import ei, ei_cf, log_ei, log_ei_cf, log_pi, pi, ucb  # noqa: F401
    from .builders import build_surrogate, build_surrogate_dict  # noqa: F401
    from .configurations import (  # noqa: F401
        FastGPConfig,
        SMTGPConfig,
        StudentTConfig,
        TorchGPConfig,
    )
    from .surrogates import FastGP, StudentTProcess, TorchGP  # noqa: F401
