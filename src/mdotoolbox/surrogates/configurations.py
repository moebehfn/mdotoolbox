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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class SMTGPConfig:
    """Configuration for Gaussian Process surrogate models."""

    model: Any = "KPLSK"
    n_start: int = 10
    hyper_opt: str = "Cobyla"
    seed: int = 42
    print_global: bool = False


@dataclass
class TorchGPConfig:
    """Configuration for GPyTorch-based Gaussian Process surrogate models."""

    n_iter: int = 100
    lr: float = 0.1
    seed: int = 42
    device: str = "cpu"
    print_global: bool = False


@dataclass
class FastGPConfig:
    """Configuration for Fast GP (scikit-learn + PCA) surrogate."""

    n_components: int = 2
    n_restarts_optimizer: int = 5
    alpha: float = 1e-10
    seed: int = 42
    normalize_y: bool = True


class StudentTConfig:
    """Configuration for Student t-Process surrogate models."""

    n_samples: int = 1000
    n_tune: int = 500
    nu: float = 3.0
    length_scale_prior: tuple = (0.1, 2.0)
    variance_prior: tuple = (2.0, 1.0)
    noise_prior: tuple = (1.0, 1.0)
    seed: int = 42
    print_progress: bool = False
