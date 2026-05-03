# CO Framework

Extracted documentation from `src/mdotoolbox/frameworks/co.py`.

## Overview

### Mathematical Formulation

System Level:
        min_{z_bar, x_bar, y_bar} f(z_bar, x_bar, y_bar)
        s.t.  g_sys(z_bar, x_bar, y_bar) >= 0
              J_i = 0  for all i  (strict consistency)

    Subsystem Level i:
        min_{z_under_i, x_under_i}  J_i(z_under_i, x_under_i; z_bar, x_bar_i, y_bar_i)
        s.t.  g_i(z_under_i, x_under_i, y_bar_coupled) >= 0

        where J_i = ||z_bar - z_under_i||**2 + ||x_bar_i - x_under_i||**2 + ||y_bar_i - y_i(z_under_i, x_under_i)||**2

Key Components:
    - COSubsystem: Individual discipline (inherits from BaseSubsystem)
    - COSystem: System-level coordinator (inherits from BaseSystem)
    - CollaborativeOptimization: Main solver (inherits from BaseSolver)

For implementation details of the base bilevel structure, see:
    - BaseSubsystem: Common subsystem optimization logic
    - BaseSystem: Common system-level coordination
    - BaseSolver: Common iteration and budget management

Features:
    - Parallel subsystem optimization capability
    - Flexible budget management (system vs subsystem allocation)
    - Comprehensive iteration history tracking
    - Convergence based on coupling discrepancy epsilon

References:
    Braun, R. D., & Kroo, I. M. (1997). "Development and Application of the
    Collaborative Optimization Architecture in a Multidisciplinary Design
    Environment." In Multidisciplinary design optimization: State of the art.

## COSubsystem

### Attributes

problem: Problem
        Local subsystem optimization problem with objective and constraints
    z_idxs: ndarray
        Indices of shared design variables in the global variable vector
    x_idxs: ndarray
        Indices of local design variables in the global variable vector
    y_idxs: ndarray
        Indices of subsystem outputs in the global output vector
    y_bar_coupled_idxs: List[ndarray]
        List of index arrays for coupled variables (targets) from other subsystems
    z_under_i: ndarray
        Subsystem's copy of shared variables (optimized to match z_bar)
    x_under_i: ndarray
        Subsystem's local variables
    y_i: ndarray
        Actual subsystem outputs from discipline analysis
    best_J_i: float
        Best (lowest) discrepancy achieved
    best_h: float
        Best (lowest) constraint violation achieved
    iteration_history: List[dict]
        History of subsystem optimization iterations

Example:
    >>> from mdotoolbox.core import Problem, Function
    >>> # Define discipline problem
    >>> y_func = Function(func=lambda z, x: z**2 + x, x=['z', 'x'], name='disc1')
    >>> problem = Problem(objective=y_func, lbounds=[0, 0], ubounds=[10, 10])
    >>>
    >>> # Create subsystem
    >>> subsystem = COSubsystem(
    ...     problem=problem,
    ...     z_idxs=[0],  # First variable is shared
    ...     x_idxs=[1],  # Second variable is local
    ...     y_idxs=[0],  # First output
    ...     y_bar_coupled_idxs=[]  # No coupling
    ... )

## COSystem

### Attributes

problem: Problem
        System-level optimization problem with objective and constraints
    subsystems: List[COSubsystem]
        All subsystems in the MDO problem
    z_bar: ndarray
        Current shared design variables
    x_bar: ndarray
        System-level targets for local variables (concatenated from all subsystems)
    y_bar: ndarray
        System-level targets for coupling variables (concatenated from all subsystems)
    iteration_history: List[dict]
        System-level optimization iteration metrics
    best_J_total: float
        Best total discrepancy (sum of all J_i)
    best_h_total: float
        Best system constraint violation
    best_f_bar: float
        Best system objective value

Workflow:
    1. Receive discrepancies J_i from subsystems
    2. Optimize system objective subject to J_i = 0 and system constraints
    3. Return updated targets (z_bar, x_bar, y_bar) to subsystems
    4. Repeat until convergence

Example:
    >>> system = COSystem(
    ...     problem=system_problem,
    ...     subsystems=[subsystem1, subsystem2]
    ... )
    >>> # System solves for optimal targets
    >>> z_bar, x_bar, y_bar, f, h, n_evals = system.solve(
    ...     optimizer='cobyqa',
    ...     system_iter=1,
    ...     global_start_time=time.time()
    ... )

## CollaborativeOptimization

### Attributes

system: COSystem
        System-level coordinator with subsystems
    subsystem_optimizer: str or Callable
        Optimizer for subsystem minimization (e.g., 'cobyqa', 'cobyla')
    system_optimizer: str or Callable
        Optimizer for system-level coordination
    epsilon_J: float, optional
        Convergence tolerance for coupling discrepancy J_total (default: 1e-6)
    epsilon_h: float, optional
        Convergence tolerance for constraint violation h_total (default: 1e-6)
    budget: BudgetManager or int, optional
        Budget allocation strategy. If int, creates weighted budget manager.
        (default: 100)
    max_iter: int or None, optional
        Maximum number of CO iterations. None for unlimited. (default: None)

Budget Management:
    Three modes available via BudgetManager:

    1. Weighted (recommended):
        - Distributes total_budget proportionally
        - system_ratio: fraction for system level
        - subsystem_weights: relative allocation per subsystem
        - iteration_ratio: min budget per iteration

    2. Fixed:
        - Explicit budgets for each subsystem and system
        - system_budget: evaluations for system
        - subsystem_budgets: list of budgets per subsystem

    3. Shared:
        - Single pool shared by all levels
        - First-come, first-served allocation

### Examples

>>> # Basic usage with integer budget
    >>> solver = CollaborativeOptimization(
    ...     system=co_system,
    ...     subsystem_optimizer='cobyqa',
    ...     system_optimizer='cobyqa',
    ...     epsilon_J=1e-6,
    ...     epsilon_h=1e-6,
    ...     budget=400
    ... )
    >>>
    >>> # Weighted budget allocation
    >>> budget = BudgetManager(
    ...     mode='weighted',
    ...     total_budget=500,
    ...     system_ratio=0.4,
    ...     subsystem_weights=[0.6, 0.4],  # 60% to subsys 1, 40% to subsys 2
    ...     iteration_ratio=0.05
    ... )
    >>> solver = CollaborativeOptimization(
    ...     system=co_system,
    ...     subsystem_optimizer='cobyla',
    ...     system_optimizer='slsqp',
    ...     budget=budget
    ... )
    >>>
    >>> # Solve the problem
    >>> z_bar0 = np.array([1.0, 2.0])
    >>> x_bar0 = np.array([0.5])
    >>> y_bar0 = np.array([1.0, 1.0])
    >>> z_under0 = np.array([1.0, 2.0])
    >>> x_under0 = np.array([0.5])
    >>> result = solver.solve(z_bar0, x_bar0, y_bar0, z_under0, x_under0)
    >>>
    >>> print(f"Optimal objective: {result.best.f_bar}")
    >>> print(f"Final discrepancy: {result.best.J_total}")

### Notes

- Convergence requires all J_i <= epsilon_J AND feasibility (h_total <= epsilon_h)
    - System targets (z_bar, x_bar, y_bar) updated each iteration
    - Subsystems optimize in parallel conceptually
    - Budget exhaustion triggers early termination

