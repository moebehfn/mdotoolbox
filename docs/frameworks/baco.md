# BACO Framework

Extracted documentation from `src/mdotoolbox/frameworks/baco.py`.

## Overview

### Key Innovations

- GP surrogates for subsystem objectives (J_i) and constraints (g_i)
    - Acquisition function optimization (Expected Improvement)
    - Latin Hypercube Sampling for efficient DoE initialization
    - Reduced subsystem evaluations through surrogate modeling

### Mathematical Formulation

System Level: (same as CO)
        min_{z,x_bar,y_bar} f(z, x_bar, y_bar)
        s.t.  g_sys(z, x_bar, y_bar) >= 0
              J_i <= epsilon  for all i

    Subsystem Level i: (Bayesian optimization)
        - Build GP surrogates: mu_{J_i}(z_under_i, x_under_i), mu_{g_i}(z_under_i, x_under_i)
        - Optimize acquisition: max alpha_{EI}[mu_{J_i}] s.t. mu_{g_i} >= 0
        - Update DoE with new evaluation
        - Retrain surrogates

### Advantages

- Reduced subsystem calls (5-10x fewer than CO)
    - Better for expensive analysis codes
    - Uncertainty quantification via GP variance
    - Adaptive sampling focuses on promising reg_ions

### Configuration

- SMTGPConfig: GP hyperparameters, kernel settings
    - DoE size: Initial sampling (typically 5-20 points per subsystem)
    - Acquisition function: Expected Improvement (log_ei)

## BACOSubsystem

### Attributes

problem: Problem
        Local subsystem problem definition
    z_idxs: ndarray
        Indices of shared design variables
    x_idxs: ndarray
        Indices of local design variables
    y_idxs: ndarray
        Indices of subsystem outputs
    y_coupled_idxs: List[ndarray]
        Coupled variable indices from other subsystems
    surrogate_config: SMTGPConfig
        GP configuration (kernel, hyperparameters)
    doe_J_i: DoE
        Design of Experiments for discrepancy J_i
    doe_g_i: DoE
        Design of Experiments for constraints g_i
    gp_J_i: GP model
        Trained surrogate for J_i objective
    gp_g_i: dict
        Trained surrogates for each constraint {name: GP}
    z_under_i, x_under_i, y_i: ndarray
        Current subsystem variable values
    best_J_i, best_h: float
        Best achieved discrepancy and constraint violation

Workflow:
    1. initialize_doe(): LHS sampling to create initial DoE
    2. build_surrogate_J_i(), build_surrogate_g_i(): Train GP surrogates
    3. solve_acquisition(): Optimize acquisition function
    4. Update DoE and retrain (repeat)

## initialize_doe

### Notes

- Uses Latin Hypercube for space-filling design
    - Samples from subsystem variable bounds [z_under_i, x_under_i]
    - Evaluates actual discipline for each sample
    - Forms baseline for GP training

Example:
    >>> subsystem.initialize_doe(
    ...     n_samples=20,
    ...     z_bar=np.array([1.0, 2.0]),
    ...     x_bar=np.array([0.5]),
    ...     y_bar=np.array([1.0, 1.5])
    ... )

## build_surrogate_J_i

### Notes

- Must call initialize_doe() first
    - GP trained on all DoE samples
    - Hyperparameters optimized during training
    - Model ready for predict_values() calls

Example:
    >>> subsystem.initialize_doe(n_samples=20, z_bar, x_bar, y_bar)
    >>> subsystem.build_surrogate_J_i()
    >>> # Now can predict: subsystem.gp_J_i.predict_values(X_new)

## build_surrogate_g_i

### Notes

- Must call initialize_doe() first
    - One GP per constraint in self.problem.constraints
    - Each GP predicts g_i(z_under_i, x_under_i, y_coupled)
    - Used to enforce constraints in acquisition optimization

Example:
    >>> subsystem.build_surrogate_g_i()
    >>> # Predict constraint 'stress' at new point
    >>> X_test = np.array([[1.0, 2.0, 3.0]])
    >>> g_pred = subsystem.gp_g_i['stress'].predict_values(X_test)

## solve_acquisition

### Notes

- Acquisition maximized = minimize negative acquisition
    - GP mean predictions used for constraint handling
    - True function evaluated at optimal acquisition point
    - DoE automatically updated for next GP training

Example:
    >>> from mdotoolbox.surrogates import log_ei
    >>> subsystem.solve_acquisition(
    ...     z_bar=np.array([1.0, 2.0]),
    ...     x_bar=np.array([0.5]),
    ...     y_bar=np.array([1.0, 1.5]),
    ...     optimizer='cobyqa',
    ...     acq_func=log_ei,
    ...     n_multistart=20
    ... )

