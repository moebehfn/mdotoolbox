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

"""src/mdotoolbox/frameworks/co.py"""

from dataclasses import dataclass

from .base_classes import BaseSolver, BaseSubsystem, BaseSystem


@dataclass
class COSubsystem(BaseSubsystem):
    """Subsystem for Collaborative Optimization framework."""

    pass


@dataclass
class COSystem(BaseSystem):
    """System-level coordinator for Collaborative Optimization."""

    pass


@dataclass
class CollaborativeOptimization(BaseSolver):
    """Main solver for Collaborative Optimization framework."""

    pass
