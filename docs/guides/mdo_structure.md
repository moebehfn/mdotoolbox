# MDO Structure

Extracted documentation from `src/mdotoolbox/core/mdo.py`.

## Discipline

### Attributes

name (str): Unique identifier for this discipline (e.g., "aerodynamics",
        "structures", "propulsion").
    outputs (Function): Function that computes discipline outputs. Takes
        concatenated inputs [z_under_i, x_under_i, y_bar_coupled] and returns output array.
    constraints (List[Constraint], optional): Discipline-level constraints
        that must be satisfied. Defaults to empty list.
    shared_var_indices (np.ndarray, optional): Indices into the global shared
        variable vector (z_bar) that this discipline uses. Defaults to empty array.
    local_var_indices (np.ndarray, optional): Indices into the global local
        variable vector (x_bar) that this discipline owns. Defaults to empty array.
    coupling_output_indices (np.ndarray, optional): Indices into the global
        coupling variable vector (y_bar) that this discipline produces.
        Defaults to empty array.
    coupling_input_indices (List[np.ndarray], optional): List of index arrays,
        one per upstream discipline, specifying which coupling outputs from
        those disciplines are inputs to this discipline. Defaults to empty list.

### Examples

>>> # Aerodynamics discipline
    >>> def aero_analysis(mach, altitude, angle_of_attack, structural_deflection):
    ...     # Compute lift and drag
    ...     lift = ...
    ...     drag = ...
    ...     return np.array([lift, drag])
    >>>
    >>> aero = Discipline(
    ...     name="aerodynamics",
    ...     outputs=Function(
    ...         func=aero_analysis,
    ...         x=['mach', 'altitude', 'aoa', 'deflection']
    ...     ),
    ...     shared_var_indices=[0, 1],      # mach, altitude
    ...     local_var_indices=[2],          # angle_of_attack
    ...     coupling_output_indices=[0, 1], # lift, drag (outputs)
    ...     coupling_input_indices=[[2]]    # structural_deflection (input from structures)
    ... )

### Notes

- Variable indices refer to positions in global vectors (z_bar, x_bar, y_bar)
    - The outputs function receives: [z_under_i, x_under_i, y_bar_coupled]
    - Coupling creates interdependencies between disciplines

## evaluate

### Examples

>>> z_under = np.array([0.8, 10000.0])  # mach, altitude
    >>> x_under = np.array([5.0])            # angle of attack
    >>> y_bar_coupled = np.array([0.1])    # structural deflection
    >>> outputs = aero.evaluate(z_under, x_under, y_bar_coupled)
    >>> print(outputs)  # [lift, drag]
    [15000.0, 1200.0]

### Notes

- Internally builds input as [z_under_local, x_under_local, y_bar_coupled]
    - The output function is called with unpacked inputs

## evaluate_constraints

### Examples

>>> # Discipline with stress constraint
    >>> c_vals = structure.evaluate_constraints(z_under, x_under, y_bar_coupled)
    >>> print(c_vals)  # [stress_margin]
    [0.15]  # Positive means constraint is satisfied

### Notes

- Returns empty array if self.constraints is empty
    - Constraint functions receive same input format as outputs function

## MDOProblem

### Attributes

system_objective (Function): System-level objective function to minimize
        (or maximize if maximize=True). Takes all variables [z_bar, x_bar, y_bar].
    disciplines (List[Discipline]): List of coupled discipline objects.
        Each discipline analyzes part of the system.
    n_shared (int): Number of shared design variables (z_bar) that affect
        multiple disciplines.
    n_local (List[int]): Number of local design variables (x_bar_i) for each discipline.
        Length must match len(disciplines).
    n_coupling (List[int]): Number of coupling variables (y_bar_i) output by each
        discipline. Length must match len(disciplines).
    shared_bounds (np.ndarray): Bounds for shared variables, shape (n_shared, 2)
        where each row is [lower_bound, upper_bound].
    local_bounds (List[np.ndarray]): Bounds for local variables of each
        discipline. Each element has shape (n_local[i], 2).
    system_constraints (List[Constraint], optional): System-level constraints
        that involve variables from multiple disciplines. Defaults to empty list.
    maximize (bool, optional): If True, maximize the objective instead of
        minimizing. Defaults to False.
    tol (float, optional): Numerical tolerance for comparisons. Defaults to 1e-6.
    name (str, optional): Descriptive name for the problem. Defaults to "".

### Examples

>>> # Two-discipline problem (e.g., aerodynamics + structures)
    >>> problem = MDOProblem(
    ...     system_objective=Function(func=total_weight, x=['z', 'x1', 'x2', 'y1', 'y2']),
    ...     disciplines=[aero_discipline, structure_discipline],
    ...     n_shared=2,           # e.g., mach number, altitude
    ...     n_local=[1, 2],       # aero has 1 local, structure has 2
    ...     n_coupling=[2, 1],    # aero outputs 2, structure outputs 1
    ...     shared_bounds=np.array([[0.5, 0.9], [0, 15000]]),
    ...     local_bounds=[
    ...         np.array([[0, 10]]),           # aero local bounds
    ...         np.array([[0, 1], [0, 100]])   # structure local bounds
    ...     ],
    ...     system_constraints=[weight_constraint],
    ...     name="aircraft_design"
    ... )

### Notes

- Variable ordering: z_bar (shared), x_bar (concatenated local), y_bar (coupling)
    - Disciplines are evaluated in sequence to resolve coupling
    - System-level optimization coordinates discipline-level optimizations

Raises:
    TypeError: If inputs are not of the correct type.
    ValueError: If dimensions are inconsistent (e.g., len(n_local) != len(disciplines)).

