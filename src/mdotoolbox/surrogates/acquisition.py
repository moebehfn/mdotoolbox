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

"""/src/mdotoolbox/surrogates/acquisition.py"""

import numpy as np

__all__ = ["ei", "log_ei", "pi", "log_pi", "ucb", "ei_cf", "log_ei_cf"]


def _log1mexp(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(1 - exp(x)) for x < 0."""
    result = np.empty_like(x, dtype=float)
    m1 = x > -2.0 * np.log(2.0)
    result[m1] = np.log(-np.expm1(x[m1]))
    result[~m1] = np.log1p(-np.exp(x[~m1]))
    return result


def ei(mu: np.ndarray, std: np.ndarray, f_min: float) -> np.ndarray:
    """Expected Improvement."""
    from scipy.stats import norm

    mu = np.asarray(mu, dtype=float)
    std = np.asarray(std, dtype=float)
    out = np.zeros_like(mu)
    mask = std > 0
    z = (f_min - mu[mask]) / std[mask]
    out[mask] = (f_min - mu[mask]) * norm.cdf(z) + std[mask] * norm.pdf(z)
    return out


def pi(mu: np.ndarray, std: np.ndarray, f_min: float) -> np.ndarray:
    """Probability of Improvement."""
    from scipy.stats import norm

    mu = np.asarray(mu, dtype=float)
    std = np.asarray(std, dtype=float)
    out = np.zeros_like(mu)
    mask = std > 0
    out[mask] = norm.cdf((f_min - mu[mask]) / std[mask])
    return out


def ucb(mu: np.ndarray, std: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    """Upper Confidence Bound (negated for minimisation)."""
    mu = np.asarray(mu, dtype=float)
    std = np.asarray(std, dtype=float)
    return -(mu - kappa * std)


def log_ei(mu: np.ndarray, std: np.ndarray, f_min: float) -> np.ndarray:
    """Log Expected Improvement (ament2023). Numerically stable over three
    regimes of z = (f_min - mu) / std:"""
    from scipy.special import erfcx
    from scipy.stats import norm

    mu = np.asarray(mu, dtype=float)
    std = np.asarray(std, dtype=float)
    eps = np.finfo(np.float64).eps
    c1 = 0.5 * np.log(2.0 * np.pi)
    c2 = 0.5 * np.log(0.5 * np.pi)
    z = np.full_like(mu, np.nan)
    mask_s = (std > 0).ravel()
    z[mask_s] = (f_min - mu[mask_s]) / std[mask_s]
    m1 = mask_s & (z > -1.0).ravel()
    m2 = mask_s & ((z > -1.0 / np.sqrt(eps)) & (z <= -1.0)).ravel()
    m3 = mask_s & (z <= -1.0 / np.sqrt(eps)).ravel()
    log_h = np.empty_like(z)
    log_h[m1] = np.log(norm.pdf(z[m1]) + z[m1] * norm.cdf(z[m1]))
    log_h[m2] = (
        -(z[m2] ** 2) / 2.0
        - c1
        + _log1mexp(np.log(erfcx(-z[m2] / np.sqrt(2.0)) * np.abs(z[m2])) + c2)
    )
    log_h[m3] = -(z[m3] ** 2) / 2.0 - c1 - 2.0 * np.log(np.abs(z[m3]))
    out = np.full_like(mu, -np.inf)
    out[mask_s] = log_h[mask_s] + np.log(std[mask_s])
    return out


def log_pi(mu: np.ndarray, std: np.ndarray, f_min: float) -> np.ndarray:
    """Log Probability of Improvement (ament2023). Uses scipy.special.log_ndtr
    for numerical stability in the tails."""
    from scipy.special import log_ndtr

    mu = np.asarray(mu, dtype=float)
    std = np.asarray(std, dtype=float)
    out = np.full_like(mu, -np.inf)
    mask = std > 0
    out[mask] = log_ndtr((f_min - mu[mask]) / std[mask])
    return out


def ei_cf(
    mu_xi: np.ndarray,
    C_xi: np.ndarray,
    outer_func,
    f_min: float,
    n_samples: int = 512,
    rng=None,
) -> float:
    """Expected Improvement for Composite Functions (astudillo2019). MC estimator:"""
    if rng is None:
        rng = np.random.default_rng()
    mu_xi = np.asarray(mu_xi, dtype=float).ravel()
    C_xi = np.asarray(C_xi, dtype=float)
    d_xi = mu_xi.shape[0]
    z = rng.standard_normal((n_samples, d_xi))
    xi_samples = mu_xi + z @ C_xi.T
    f_samples = np.asarray(outer_func(xi_samples)).ravel()
    improvement = f_min - f_samples
    return float(np.mean(np.maximum(0.0, improvement)))


def log_ei_cf(
    mu_xi: np.ndarray,
    C_xi: np.ndarray,
    outer_func,
    f_min: float,
    n_samples: int = 512,
    tau: float = 0.001,
    rng=None,
) -> float:
    """Log Expected Improvement for Composite Functions (Novel). Replaces
    max(0, u) with softplus_tau(u) for differentiability everywhere, then
    takes the log of the sample mean via logsumexp:"""
    from scipy.special import logsumexp

    if rng is None:
        rng = np.random.default_rng()
    mu_xi = np.asarray(mu_xi, dtype=float).ravel()
    C_xi = np.asarray(C_xi, dtype=float)
    d_xi = mu_xi.shape[0]
    z = rng.standard_normal((n_samples, d_xi))
    xi_samples = mu_xi + z @ C_xi.T
    f_samples = np.asarray(outer_func(xi_samples)).ravel()
    improvement = f_min - f_samples
    log_u_tilde = np.log(tau) + np.logaddexp(0.0, improvement / tau)
    return float(logsumexp(log_u_tilde) - np.log(n_samples))
