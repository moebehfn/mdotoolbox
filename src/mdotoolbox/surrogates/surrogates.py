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

"""src/mdotoolbox/surrogates/surrogates.py"""

import warnings
from typing import Optional

import numpy as np

from .configurations import FastGPConfig, StudentTConfig, TorchGPConfig


class TorchGP:
    """GPyTorch-backed GP surrogate with SMT-compatible API."""

    def __init__(self, config: Optional[TorchGPConfig] = None):
        import torch

        self.config = config if config is not None else TorchGPConfig()
        self._device = torch.device(self.config.device)
        self._model = None
        self._likelihood = None
        self._train_x_under_i = None
        self._train_y_i = None
        self._cache_key = None
        self._cache_mean = None
        self._cache_var = None
        self._pred_ctx = None

    def set_training_values(self, x_under_i, y_i):
        """Store training data (numpy arrays)."""
        import torch

        x_under_i = np.atleast_2d(x_under_i)
        y_i = np.atleast_1d(y_i).ravel()
        self._train_x_under_i = torch.tensor(
            x_under_i, dtype=torch.float32, device=self._device
        )
        self._train_y_i = torch.tensor(y_i, dtype=torch.float32, device=self._device)

    def train(self):
        """Fit GP hyper-parameters via Adam + exact MLL."""
        import gpytorch
        import torch

        if self._train_x_under_i is None:
            raise ValueError("Must call set_training_values before train")

        class _ExactGPModel(gpytorch.models.ExactGP):
            """Exact GP with ScaleKernel(RBFKernel(ARD))."""

            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        torch.manual_seed(self.config.seed)
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self._device)
        self._model = _ExactGPModel(
            self._train_x_under_i, self._train_y_i, self._likelihood
        ).to(self._device)
        self._model.train()
        self._likelihood.train()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.config.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)
        use_mps = self._device.type == "mps"
        settings = [gpytorch.settings.cholesky_jitter(0.0001)]
        if use_mps:
            settings.append(gpytorch.settings.cholesky_max_tries(6))
        with gpytorch.settings.fast_computations(
            covar_root_decomposition=False, log_prob=False, solves=False
        ):
            for i in range(self.config.n_iter):
                optimizer.zero_grad()
                output = self._model(self._train_x_under_i)
                loss = -mll(output, self._train_y_i)
                loss.backward()
                optimizer.step()
                if self.config.print_global and (
                    i % 25 == 0 or i == self.config.n_iter - 1
                ):
                    print(
                        f"  TorchGP iter {i:>4d}/{self.config.n_iter}  loss={loss.item():.4f}"
                    )
        self._model.eval()
        self._likelihood.eval()
        self._pred_ctx = self._build_prediction_context()
        self._pred_ctx.__enter__()
        self._cache_key = None

    def predict_values(self, x_under_i):
        """Predict mean. Returns ndarray of shape (n, 1)."""
        self._ensure_cache(x_under_i)
        return self._cache_mean

    def predict_variances(self, x_under_i):
        """Predict variance. Returns ndarray of shape (n, 1)."""
        self._ensure_cache(x_under_i)
        return self._cache_var

    def _ensure_cache(self, x_under_i):
        """Run a single forward pass and cache both mean and variance."""
        import torch

        x_under_i = np.atleast_2d(x_under_i)
        key = x_under_i.tobytes()
        if key == self._cache_key:
            return
        test_x_under_i = torch.tensor(
            x_under_i, dtype=torch.float32, device=self._device
        )
        with torch.no_grad():
            pred = self._likelihood(self._model(test_x_under_i))
            self._cache_mean = pred.mean.cpu().numpy().reshape(-1, 1)
            self._cache_var = pred.variance.cpu().numpy().reshape(-1, 1)
        self._cache_key = key

    def _build_prediction_context(self):
        """Create a reusable context manager for prediction settings."""
        from contextlib import ExitStack

        import gpytorch

        stack = ExitStack()
        stack.enter_context(
            gpytorch.settings.fast_computations(
                covar_root_decomposition=False, log_prob=False, solves=False
            )
        )
        stack.enter_context(gpytorch.settings.fast_pred_var())
        if self._device.type == "mps":
            stack.enter_context(gpytorch.settings.max_cholesky_size(0))
            stack.enter_context(gpytorch.settings.max_root_decomposition_size(0))
        return stack


class FastGP:
    """Fast Gaussian Process using scikit-learn + PCA dimension reduction."""

    def __init__(self, config: Optional[FastGPConfig] = None):
        from sklearn.decomposition import PCA
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

        self.config = config if config is not None else FastGPConfig()
        self.pca = PCA(
            n_components=self.config.n_components, random_state=self.config.seed
        )
        kernel = ConstantKernel(1.0, constant_value_bounds=(0.001, 1000.0)) * RBF(
            length_scale=1.0, length_scale_bounds=(0.01, 100.0)
        ) + WhiteKernel(noise_level=1e-05, noise_level_bounds=(1e-10, 1.0))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.config.n_restarts_optimizer,
            alpha=self.config.alpha,
            normalize_y=self.config.normalize_y,
            random_state=self.config.seed,
        )
        self.x_under_i_train = None
        self.y_i_train = None
        self.is_trained = False

    def set_training_values(self, x_under_i, y_i):
        """Set training data."""
        self.x_under_i_train = np.atleast_2d(x_under_i)
        self.y_i_train = np.atleast_1d(y_i).flatten()

    def train(self):
        """Train the GP with PCA dimension reduction."""
        if self.x_under_i_train is None or self.y_i_train is None:
            raise ValueError("Must call set_training_values before train")
        n_samples, n_features = self.x_under_i_train.shape
        n_comp = min(self.config.n_components, n_features, n_samples - 1)
        if n_comp != self.config.n_components:
            warnings.warn(
                f"Adjusted n_components from {self.config.n_components} to {n_comp} based on data dimensions",
                stacklevel=2,
            )
        self.pca.n_components = n_comp
        x_under_i_reduced = self.pca.fit_transform(self.x_under_i_train)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.gp.fit(x_under_i_reduced, self.y_i_train)
        self.is_trained = True

    def predict_values(self, x_under_i):
        """Predict mean values at test points."""
        if not self.is_trained:
            raise ValueError("Must train model before making predictions")
        x_under_i = np.atleast_2d(x_under_i)
        x_under_i_reduced = self.pca.transform(x_under_i)
        y_i_pred = self.gp.predict(x_under_i_reduced, return_std=False)
        return y_i_pred.reshape(-1, 1)

    def predict_variances(self, x_under_i):
        """Predict variance at test points."""
        if not self.is_trained:
            raise ValueError("Must train model before making predictions")
        x_under_i = np.atleast_2d(x_under_i)
        x_under_i_reduced = self.pca.transform(x_under_i)
        _, std = self.gp.predict(x_under_i_reduced, return_std=True)
        variance = std**2
        return variance.reshape(-1, 1)


class StudentTProcess:
    """Student t-Process regression model using PyMC for Bayesian inference."""

    def __init__(self, config: Optional[StudentTConfig] = None):
        self.config = config if config is not None else StudentTConfig()
        self.model = None
        self.trace = None
        self.x_under_i_train = None
        self.y_i_train = None
        self.x_under_i_mean = None
        self.x_under_i_std = None
        self.y_i_mean = None
        self.y_i_std = None

    def _normalize_data(self, x_under_i, y_i=None):
        """Normalize inputs and outputs to zero mean and unit variance."""
        if self.x_under_i_mean is None:
            self.x_under_i_mean = np.mean(x_under_i, axis=0)
            self.x_under_i_std = np.std(x_under_i, axis=0) + 1e-08
        x_under_i_norm = (x_under_i - self.x_under_i_mean) / self.x_under_i_std
        if y_i is not None:
            if self.y_i_mean is None:
                self.y_i_mean = np.mean(y_i)
                self.y_i_std = np.std(y_i) + 1e-08
            y_i_norm = (y_i - self.y_i_mean) / self.y_i_std
            return (x_under_i_norm, y_i_norm)
        return x_under_i_norm

    def _denormalize_predictions(self, y_i_norm, var_norm=None):
        """Transform predictions back to original data scale."""
        y_i = y_i_norm * self.y_i_std + self.y_i_mean
        if var_norm is not None:
            var = var_norm * self.y_i_std**2
            return (y_i, var)
        return y_i

    def set_training_values(self, x_under_i, y_i):
        """Set training data for the Student t-Process."""
        self.x_under_i_train = np.atleast_2d(x_under_i)
        self.y_i_train = np.atleast_1d(y_i).flatten()

    def train(self):
        """Train the Student t-Process using MCMC sampling (NUTS algorithm)."""
        import pymc as pm

        if self.x_under_i_train is None or self.y_i_train is None:
            raise ValueError("Must call set_training_values before train")
        x_under_i_norm, y_i_norm = self._normalize_data(
            self.x_under_i_train, self.y_i_train
        )
        with pm.Model() as self.model:
            length_scale = pm.InverseGamma(
                "length_scale",
                alpha=self.config.length_scale_prior[0],
                beta=self.config.length_scale_prior[1],
                shape=x_under_i_norm.shape[1],
            )
            variance = pm.InverseGamma(
                "variance",
                alpha=self.config.variance_prior[0],
                beta=self.config.variance_prior[1],
            )
            noise_var = pm.InverseGamma(
                "noise_var",
                alpha=self.config.noise_prior[0],
                beta=self.config.noise_prior[1],
            )
            cov_func = variance * pm.gp.cov.ExpQuad(
                input_dim=x_under_i_norm.shape[1], ls=length_scale
            )
            gp = pm.gp.Latent(cov_func=cov_func)
            f = gp.prior("f", X=x_under_i_norm)
            pm.StudentT(
                "y",
                nu=self.config.nu,
                mu=f,
                sigma=pm.math.sqrt(noise_var),
                observed=y_i_norm,
            )
            if self.config.print_progress:
                self.trace = pm.sample(
                    draws=self.config.n_samples,
                    tune=self.config.n_tune,
                    random_seed=self.config.seed,
                    return_inferencedata=True,
                )
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.trace = pm.sample(
                        draws=self.config.n_samples,
                        tune=self.config.n_tune,
                        random_seed=self.config.seed,
                        return_inferencedata=True,
                        progressbar=False,
                    )

    def predict_values(self, x_under_i):
        """Predict mean values at test points."""
        if self.trace is None:
            raise ValueError("Must train model before making predictions")
        x_under_i = np.atleast_2d(x_under_i)
        x_under_i_norm = self._normalize_data(x_under_i)
        x_under_i_train_norm = self._normalize_data(self.x_under_i_train)
        length_scale_samples = self.trace.posterior["length_scale"].values
        variance_samples = self.trace.posterior["variance"].values
        f_samples = self.trace.posterior["f"].values
        n_chains, n_draws = length_scale_samples.shape[:2]
        n_samples_total = n_chains * n_draws
        ls_flat = length_scale_samples.reshape(n_samples_total, -1)
        var_flat = variance_samples.reshape(n_samples_total)
        f_flat = f_samples.reshape(n_samples_total, -1)
        ls_mean = ls_flat.mean(axis=0)
        var_mean = var_flat.mean()
        f_train_mean = f_flat.mean(axis=0)
        dist_train = np.sum(
            (x_under_i_train_norm[:, None, :] - x_under_i_train_norm[None, :, :]) ** 2
            / ls_mean**2,
            axis=2,
        )
        K_train = var_mean * np.exp(-0.5 * dist_train)
        dist_test_train = np.sum(
            (x_under_i_norm[:, None, :] - x_under_i_train_norm[None, :, :]) ** 2
            / ls_mean**2,
            axis=2,
        )
        K_test_train = var_mean * np.exp(-0.5 * dist_test_train)
        K_train_inv = np.linalg.inv(K_train + 1e-06 * np.eye(len(self.y_i_train)))
        y_i_pred_norm = K_test_train @ K_train_inv @ f_train_mean
        y_i_pred = self._denormalize_predictions(y_i_pred_norm)
        return y_i_pred.reshape(-1, 1)

    def predict_variances(self, x_under_i):
        """Predict variance at test points."""
        if self.trace is None:
            raise ValueError("Must train model before making predictions")
        x_under_i = np.atleast_2d(x_under_i)
        x_under_i_norm = self._normalize_data(x_under_i)
        x_under_i_train_norm = self._normalize_data(self.x_under_i_train)
        length_scale_samples = self.trace.posterior["length_scale"].values
        variance_samples = self.trace.posterior["variance"].values
        noise_var_samples = self.trace.posterior["noise_var"].values
        n_chains, n_draws = length_scale_samples.shape[:2]
        n_samples_total = n_chains * n_draws
        ls_flat = length_scale_samples.reshape(n_samples_total, -1)
        var_flat = variance_samples.reshape(n_samples_total)
        noise_flat = noise_var_samples.reshape(n_samples_total)
        ls_mean = ls_flat.mean(axis=0)
        var_mean = var_flat.mean()
        noise_mean = noise_flat.mean()
        dist_train = np.sum(
            (x_under_i_train_norm[:, None, :] - x_under_i_train_norm[None, :, :]) ** 2
            / ls_mean**2,
            axis=2,
        )
        K_train = var_mean * np.exp(-0.5 * dist_train)
        dist_test_train = np.sum(
            (x_under_i_norm[:, None, :] - x_under_i_train_norm[None, :, :]) ** 2
            / ls_mean**2,
            axis=2,
        )
        K_test_train = var_mean * np.exp(-0.5 * dist_test_train)
        dist_test = np.sum(
            (x_under_i_norm[:, None, :] - x_under_i_norm[None, :, :]) ** 2 / ls_mean**2,
            axis=2,
        )
        K_test = var_mean * np.exp(-0.5 * dist_test)
        K_train_inv = np.linalg.inv(K_train + 1e-06 * np.eye(len(self.y_i_train)))
        y_i_var_norm = (
            np.diag(K_test - K_test_train @ K_train_inv @ K_test_train.T) + noise_mean
        )
        _, y_i_var = self._denormalize_predictions(
            np.zeros_like(y_i_var_norm), y_i_var_norm
        )
        return y_i_var.reshape(-1, 1)
