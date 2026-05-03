# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-03-18

### Added

- **Initial Public Release** of MDOToolbox.
- Five MDO frameworks: BACO, CO, ICO, MCO, and ECO.
- Bayesian optimization integration with Gaussian Process surrogates (SMT/GPyTorch/Torch).
- Parallel multi-start acquisition optimization for robust global search.
- Suite of 7 built-in benchmark problems (Sellar, Scalable, Speed Reducer, etc.).
- Comprehensive documentation using MkDocs.
- Public test suite with regression tests for all frameworks.

### Fixed

- Fixed `StudentTConfig` lifecycle issues.
- Improved error handling for NaN/Inf values during GP training and prediction.
- Resolved consistency issues in BACO initial guess handling.
- Fixed constraints dictionary structure when no equality constraints are present.

### Performance

- Parallelized multi-start acquisition optimization using `joblib` with `loky` backend.
- Optimized DoE validation and KPLSK training workflow.
