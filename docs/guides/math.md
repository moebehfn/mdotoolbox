# Mathematical Foundation

Extracted documentation from `src/mdotoolbox/core/utils.py`.

## assess_progress

### Examples

>>> # Full hierarchy - feasibility improved
    >>> assess_progress(h=0.01, h_best=0.1,
    ...                 J_i=2.0, J_i_best=1.0, f=20.0, f_best=10.0)
    True

    >>> # Jf mode - coupling improved, no h check
    >>> assess_progress(h=0.0, h_best=0.0,
    ...                 J_i=0.5, J_i_best=1.0, f=20.0, f_best=10.0,
    ...                 mode="Jf")
    True

    >>> # f mode - strictly objective
    >>> assess_progress(h=0.0, h_best=0.0,
    ...                 J_i=0.0, J_i_best=0.0, f=9.0, f_best=10.0,
    ...                 mode="f")
    True

## format_time_multi

### Notes

- All rows padded to same number of columns
    - Columns right-aligned for readability
    - Useful for timing comparisons in output tables

## compute_Ji

### Mathematical Formulation

J_i = ||z_bar - z_under_i||**2 + ||x_bar_i - x_under_i||**2 + ||y_bar_i - y_i||**2

Args:
    y_i_func: Callable
        Subsystem output function: y_i = f(z_under_i, x_under_i, y_bar_coupled)
    z_bar: ndarray
        System-level shared design variables
    x_bar_i: ndarray
        System-level target for subsystem local variables
    y_bar_i: ndarray
        System-level target for subsystem outputs
    y_bar_coupled: ndarray
        Target values for coupled variables from other subsystems
    z_under_i: ndarray
        Subsystem copy of shared variables (optimization variable)
    x_under_i: ndarray
        Subsystem local variables (optimization variable)

Returns:
    J_i: float
        Total discrepancy (sum of squared deviations)
    y_i: ndarray
        Actual subsystem outputs from y_i_func evaluation

Example:
    >>> def subsystem_output(z, x, y_coupled):
    ...     return np.array([z[0]**2 + x[0]])
    >>>
    >>> z_bar = np.array([2.0, 3.0])
    >>> x_bar = np.array([1.0])
    >>> y_bar = np.array([5.0])
    >>> y_bar_coupled = np.array([])
    >>> z_under = np.array([2.1, 3.0])
    >>> x_under = np.array([0.9])
    >>>
    >>> J_i, y_i = compute_Ji(
    ...     subsystem_output, z_bar, x_bar, y_bar, y_bar_coupled, z_under, x_under
    ... )

### Notes

- Lower J_i indicates better consistency with system targets
    - J_i = 0 means perfect coupling consistency
    - Used in CO subsystem optimization objective

## clip_to_bounds

### Notes

- Creates a copy, does not modify input
    - Useful for enforcing bounds after optimizer steps
    - Common in CO/BACO after subsystem optimization

