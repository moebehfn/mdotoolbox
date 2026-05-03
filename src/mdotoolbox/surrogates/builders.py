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

from ..core import DoE
from .configurations import FastGPConfig, SMTGPConfig, StudentTConfig, TorchGPConfig
from .surrogates import FastGP, StudentTProcess, TorchGP


def build_surrogate(doe: DoE, y_key: str = "obj", config=None):
    """Build and train a Gaussian Process surrogate from DoE data."""
    if config is None:
        config = SMTGPConfig()
    if y_key not in doe.y:
        raise ValueError(
            f"Key '{y_key}' not found in DoE.y. Available: {list(doe.y.keys())}"
        )
    if isinstance(config, TorchGPConfig):
        gp = TorchGP(config)
        gp.set_training_values(doe.x, doe.y[y_key])
        gp.train()
        return gp
    if isinstance(config, SMTGPConfig):
        if config.model == "KPLSK":
            from smt.surrogate_models import KPLSK

            model_class = KPLSK
        else:
            model_class = config.model
        gp = model_class(
            print_global=config.print_global,
            n_start=config.n_start,
            hyper_opt=config.hyper_opt,
            seed=config.seed,
        )
        n_comp = gp.options["n_comp"]
        n_required = n_comp + 1
        n_actual = doe.x.shape[0]
        if n_actual < n_required:
            raise ValueError(
                f"build_surrogate() requires at least {n_required} training points for KPLSK with n_comp={n_comp} (got {n_actual}). Increase the DoE size or reduce n_comp."
            )
        gp.set_training_values(doe.x, doe.y[y_key])
        gp.train()
        return gp
    if isinstance(config, FastGPConfig):
        gp = FastGP(config=config)
        gp.set_training_values(doe.x, doe.y[y_key])
        gp.train()
        return gp
    if isinstance(config, StudentTConfig):
        tp = StudentTProcess(config=config)
        tp.set_training_values(doe.x, doe.y[y_key])
        tp.train()
        return tp


def build_surrogate_dict(doe: DoE, y_keys: list[str], config=None) -> dict:
    """Build multiple GPs for different outputs."""
    return {key: build_surrogate(doe, key, config) for key in y_keys}
