# Core Base API

Extracted documentation from `src/mdotoolbox/core/base.py`.

## DoE

### Attributes

x (Union[list, np.ndarray]): Input samples as a matrix (n_samples x n_vars)
        or vector (for 1D problems). Each row represents one sample point.
    y (Union[dict, list, np.ndarray]): Output evaluations. Can be:
        - dict: Multiple outputs with named keys {name: array of values}
        - list/array: Single output (converted to dict with key 'y')
    constraint_violation (Union[np.ndarray, None]): Optional array of constraint
        violation values for each sample. None indicates no constraints or
        all points are feasible. Default is None.

### Examples

>>> # Single output
    >>> doe = DoE(x=[[1, 2], [3, 4]], y=[0.5, 1.2])
    >>>
    >>> # Multiple outputs with constraints
    >>> doe = DoE(
    ...     x=[[1, 2], [3, 4]],
    ...     y={'obj': [0.5, 1.2], 'c1': [0.1, -0.2]},
    ...     constraint_violation=[0.0, 0.2]
    ... )

Raises:
    TypeError: If x or y are not of the correct type.
    ValueError: If dimensions of x, y, and constraint_violation don't match.

## update_DoE

### Examples

>>> doe = DoE(x=[[1, 2]], y=[0.5])
    >>> doe.update_DoE(x_n=[3, 4], y_n=1.2)
    >>> print(len(doe.x))  # Now has 2 samples
    2

## to_dict

### Examples

>>> doe = DoE(x=[[1, 2]], y={'obj': [0.5]}, constraint_violation=[0.1])
    >>> data = doe.to_dict()
    >>> print(data.keys())
    dict_keys(['x', 'obj', 'constraint_violation'])

## get_feasible_indices

### Examples

>>> doe = DoE(x=[[1], [2], [3]], y=[1, 2, 3],
    ...           constraint_violation=[0, 0.1, 0.0001])
    >>> feasible_idx = doe.get_feasible_indices(tol=0.01)
    >>> print(feasible_idx)  # Points 0 and 2
    [0 2]

## get_best_feasible

### Examples

>>> doe = DoE(
    ...     x=[[1], [2], [3]],
    ...     y={'obj': [10, 5, 8]},
    ...     constraint_violation=[0.5, 0.0, 0.1]
    ... )
    >>> x_best, f_best, cv_best = doe.get_best_feasible(tol=0.2)
    >>> print(f"Best: x={x_best}, f={f_best}, cv={cv_best}")
    Best: x=[3], f=8, cv=0.1

## get_f_min

### Examples

>>> doe = DoE(x=[[1], [2]], y={'obj': [10, 5]})
    >>> f_min = doe.get_f_min()
    >>> print(f_min)
    5

## Function

### Attributes

func (Callable): The actual function to evaluate. Should accept
        individual arguments matching the variable names in x.
    x (Union[str, List[str], np.ndarray[str]]): Variable names that the
        function depends on. Can be a single variable name or a list of names.
    name (str, optional): Descriptive name for the function (e.g., "drag",
        "lift", "constraint_1"). Defaults to empty string.

### Examples

>>> # Simple quadratic function
    >>> def quad(x, y):
    ...     return x**2 + y**2
    >>> f = Function(func=quad, x=['x', 'y'], name='quadratic')
    >>>
    >>> # Constraint function
    >>> def constraint(x):
    ...     return x - 5
    >>> c = Function(func=constraint, x='x', name='lower_bound')

Raises:
    TypeError: If func is not callable, x is not string-like, or name is not a string.

### Notes

The function is called with positional arguments in the order specified by x.
    For example, if x=['a', 'b', 'c'], func will be called as func(a_val, b_val, c_val).

## Constraint

### Attributes

func (Function): The constraint function wrapped in a Function object.
    ctype (str, optional): Constraint type. Must be one of:
        - 'ge': Greater than or equal (func(x) >= value)
        - 'le': Less than or equal (func(x) <= value)
        - 'eq': Equal to (func(x) = value)
        Defaults to 'ge'.
    value (Union[int, float], optional): Right-hand side value of the
        constraint. Defaults to 0.0.

### Examples

>>> # Box constraint: x >= 0
    >>> def x_func(x):
    ...     return x
    >>> f = Function(func=x_func, x='x', name='x_value')
    >>> c = Constraint(func=f, ctype='ge', value=0.0)
    >>>
    >>> # Nonlinear constraint: x^2 + y^2 <= 1 (unit circle)
    >>> def circle(x, y):
    ...     return x**2 + y**2
    >>> f = Function(func=circle, x=['x', 'y'], name='circle')
    >>> c = Constraint(func=f, ctype='le', value=1.0)

Raises:
    TypeError: If func is not a Function object, ctype is not a string,
        or value is not numeric.
    ValueError: If ctype is not one of 'ge', 'le', or 'eq'.

### Notes

- Internally, constraints are converted to the form func(x) OP value
    - For inequality constraints, violation is measured as the amount by
      which the constraint is not satisfied

## Problem

### Attributes

objective (Function): Objective function to minimize (or maximize if
        maximize=True).
    constraints (Iterable): Collection of Constraint objects defining the
        feasible region. Can be empty for unconstrained problems.
    ubounds (ArrayLike): Upper bounds for each variable. Must have same
        length as objective.x. Use np.inf for unbounded variables.
    lbounds (ArrayLike): Lower bounds for each variable. Must have same
        length as objective.x. Use -np.inf for unbounded variables.
    maximize (bool, optional): If True, maximize the objective instead of
        minimizing. Internally converted to minimization. Defaults to False.
    tol (float, optional): Tolerance for numerical comparisons and
        convergence checks. Defaults to 1e-6.
    name (str, optional): Descriptive name for the problem. Defaults to "".

### Examples

>>> # Unconstrained problem: minimize (x-2)^2
    >>> def obj(x):
    ...     return (x - 2)**2
    >>> f = Function(func=obj, x='x', name='quadratic')
    >>> problem = Problem(
    ...     objective=f,
    ...     constraints=[],
    ...     lbounds=[-10],
    ...     ubounds=[10],
    ...     name='simple_quadratic'
    ... )
    >>>
    >>> # Constrained problem: minimize f(x,y) subject to x+y >= 1
    >>> def obj(x, y):
    ...     return x**2 + y**2
    >>> def con(x, y):
    ...     return x + y
    >>> f = Function(func=obj, x=['x', 'y'])
    >>> c = Constraint(func=Function(func=con, x=['x', 'y']), ctype='ge', value=1.0)
    >>> problem = Problem(
    ...     objective=f,
    ...     constraints=[c],
    ...     lbounds=[0, 0],
    ...     ubounds=[10, 10]
    ... )

Raises:
    TypeError: If inputs are not of the correct type.
    ValueError: If bounds length doesn't match number of variables or
        constraint variables don't match objective variables.

### Notes

- If maximize=True, the objective is internally negated for minimization
    - Bounds are stored as self.bounds (n_vars x 2) array
    - Constraint names are auto-generated if not provided

## evaluate

### Examples

>>> # Evaluate objective only
    >>> f_val, _ = problem.evaluate(x=[1.0, 2.0], f=True, c=False)
    >>>
    >>> # Evaluate constraints only
    >>> _, c_vals = problem.evaluate(x=[1.0, 2.0], f=False, c=True)
    >>>
    >>> # Evaluate both
    >>> f_val, c_vals = problem.evaluate(x=[1.0, 2.0], f=True, c=True)

### Notes

The objective is called with unpacked x values as positional arguments.

## initial_DoE

### Examples

>>> # Fixed number of samples
    >>> doe = problem.initial_DoE(n=20)
    >>>
    >>> # Adaptive number based on problem dimension
    >>> doe = problem.initial_DoE(n=lambda n: 3*n + 1)

### Notes

- For infinite bounds, uses +/-100 or +/-(bound + 50) as limits
    - Evaluates objective as 'obj' and constraints as 'c{i}-{type}-{value}'
    - Automatically computes constraint violations

## compute_constraint_violation

### Examples

>>> # For constraints c1(x) >= 0 and c2(x) <= 5
    >>> constraint_vals = {
    ...     'c0-ge-0.0': np.array([1.0, -0.5, 0.1]),  # c1 values
    ...     'c1-le-5.0': np.array([3.0, 6.0, 4.9])    # c2 values
    ... }
    >>> violations = problem.compute_constraint_violation(constraint_vals)
    >>> # violations = [0, 0.25+1.0, 0] = [0, 1.25, 0]

### Notes

- For 'ge' constraints: violation = max(0, value - c_val)^2
    - For 'le' constraints: violation = max(0, c_val - value)^2
    - For 'eq' constraints: violation = |c_val - value|^2
    - Total violation is the sum across all constraints

## ParetoEntry

### Attributes

f:    Objective value.
    h:    Total constraint violation h_total.
    J_i:  Total coupling discrepancy J_i.
    z_bar: Shared design variables at this iterate.
    x_bar: Local design variables at this iterate.
    y_bar: Coupling variables at this iterate.
    code: Integer in {1,...,7}. Bit 1 set if non-dominated in (f,h);
          bit 2 if non-dominated in (f,J); bit 4 if non-dominated in (h,J).

## BestSolution

### Attributes

z_bar:     Shared design variables at the best iterate.
    x_bar:     Local design variables at the best iterate.
    y_bar:     Coupling variables at the best iterate.
    f:         Objective value at the best iterate.
    h:         Running minimum of h_total (used in strict improvement check).
    J_i:       Coupling discrepancy J_i at the best iterate.
    h_history: All observed h_total values in order of observation.

## Results

### Attributes

best:         Best iterate found under the lexicographic criterion.
    converged:    True if termination criteria were met.
    iterations:   Number of system-level iterations performed.
    evaluations:  Total number of discipline evaluations performed.
    elapsed_time: Wall-clock time in seconds.
    history:      DataFrame with one row per iteration.
    pareto:       Pareto set at termination.  DataFrame with columns
                  [f_sys, J_i, h_total, z_bar, x_bar, y_bar, code].  *code* is an
                  integer bitmask (1-7) encoding membership in bi-objective
                  spaces (f,h), (f,J), (h,J).

## BudgetManager

### Attributes

mode (Literal["shared", "weighted", "fixed"]): Budget allocation strategy:
        - "shared": Single shared pool, first-come-first-served
        - "weighted": Proportional split based on ratios
        - "fixed": User-specified exact allocations
        Defaults to "weighted".
    total_budget (int, optional): Total evaluation budget. Required for
        "shared" and "weighted" modes.
    system_ratio (float, optional): Fraction of budget for system-level
        (weighted mode only). Must be in (0, 1). Defaults to 0.5.
    subsystem_weights (List[float], optional): Relative weights for each
        subsystem (weighted mode). Must sum to ~1.0.
    subsystem_budgets (List[int], optional): Exact budgets for each
        subsystem (fixed mode). Required if mode="fixed".
    system_budget (int, optional): Exact budget for system-level (fixed mode).
        Required if mode="fixed".
    iteration_ratio (float, optional): Fraction of allocated budget to use
        per iteration (0, 1). If None, use full budget per iteration.
        Defaults to None.

### Examples

>>> # Mode 1: Shared pool (400 total, use until exhausted)
    >>> budget = BudgetManager(mode="shared", total_budget=400)
    >>>
    >>> # Mode 2: Weighted split (50% system, 50% subsystems)
    >>> budget = BudgetManager(
    ...     mode="weighted",
    ...     total_budget=400,
    ...     system_ratio=0.5,
    ...     subsystem_weights=[0.3, 0.5, 0.2]
    ... )
    >>>
    >>> # Mode 3: Weighted with iteration ratio (5% per iteration)
    >>> budget = BudgetManager(
    ...     mode="weighted",
    ...     total_budget=400,
    ...     system_ratio=0.5,
    ...     subsystem_weights=[0.3, 0.5, 0.2],
    ...     iteration_ratio=0.05
    ... )
    >>>
    >>> # Mode 4: Fixed budgets with iteration ratio
    >>> budget = BudgetManager(
    ...     mode="fixed",
    ...     subsystem_budgets=[100, 70],
    ...     system_budget=120,
    ...     iteration_ratio=0.05
    ... )

### Notes

- Call initialize(n_subsystems) before using the budget manager
    - Use get_subsystem_max_iter() and get_system_max_iter() to get
      available evaluations for each call
    - Record actual usage with record_subsystem_evals() and record_system_evals()
    - Check exhaustion with is_subsystem_exhausted(), is_system_exhausted(),
      is_total_exhausted()

Raises:
    ValueError: If mode is invalid, weights don't sum to 1, or required
        parameters are missing for the selected mode.

