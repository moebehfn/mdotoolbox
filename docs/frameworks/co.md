# Collaborative Optimization Framework

## Mathematical Formulation

System Level:

$$
\begin{align*}
    &\min_{
        \overline{\mathbf{z}},
        \overline{\mathbf{x}},
        \overline{\mathbf{y}}
    }\ &&f(
        \overline{\mathbf{z}},
        \overline{\mathbf{x}},
        \overline{\mathbf{y}}
    ) \tag{\(P_\text{sys}\)} \\
    &\text{subject to:}\ &&\mathbf{c}(
        \overline{\mathbf{z}},
        \overline{\mathbf{x}},
        \overline{\mathbf{y}},
    ) &&\geq \mathbf{0}\nonumber\\
    %
    &  && J(
        \overline{\mathbf{z}},
        \overline{\mathbf{x}}_i,
        \overline{\mathbf{y}},
        \underline{\mathbf{z}}_i^*,
        \underline{\mathbf{x}}_i^*
    ) &&= 0\ \text{for all}\ i\in\{1, \ldots, N\}
\end{align*}
$$

Subsystem $i$ level:

$$
\begin{equation}
    \min_{
        \underline{\mathbf{z}}_i,
        \underline{\mathbf{x}}_i
    }\ J(
        \overline{\mathbf{z}},
        \overline{\mathbf{x}}_i,
        \overline{\mathbf{y}},
        \underline{\mathbf{z}}_i,
        \underline{\mathbf{x}}_i
    ) \tag{\(P_i\)}\
    \text{subject to:}\ \mathbf{g}_i(
        \overline{\mathbf{y}}_{j\neq i},
        \underline{\mathbf{z}}_i,
        \underline{\mathbf{x}}_i
    )\geq\mathbf{0}
\end{equation}
$$

The discrepancy function:

$$
\begin{equation*}
    J(
        \overline{\mathbf{z}},
        \overline{\mathbf{x}}_i,
        \overline{\mathbf{y}},
        \underline{\mathbf{z}}_i,
        \underline{\mathbf{x}}_i
    ) = J_i =
        \bigl\|\overline{\mathbf{z}}-\underline{\mathbf{z}}_i\bigr\|^2 +
        \bigl\|\overline{\mathbf{x}}_i-\underline{\mathbf{x}}_i\bigr\|^2 +
        \bigl\|\overline{\mathbf{y}}_i-\mathbf{y}_i\bigl(
        \overline{\mathbf{y}}_{j\neq i},
        \underline{\mathbf{z}}_i,
        \underline{\mathbf{x}}_i
    \bigr)\bigr\|^2
\end{equation*}
$$

References:

- Braun, R. D., & Kroo, I. M. (1997). "Development and Application of the
- Collaborative Optimization Architecture in a Multidisciplinary Design
- Environment." In Multidisciplinary design optimization: State of the art.

## Implementation

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

### COSubsystem

| Attribute        | Type         | Description                                                                 |
|:-----------------|:-------------|:----------------------------------------------------------------------------|
| `problem`        | `Problem`    | Local subsystem optimization problem with objective and constraints.        |
| `z_idxs`         | `np.ndarray` | Indices of shared design variables in the global variable vector.           |
| `x_idxs`         | `np.ndarray` | Indices of local design variables in the global variable vector.            |
| `y_idxs`         | `np.ndarray` | Indices of subsystem outputs in the global output vector.                   |
| `y_coupled_idxs` | `np.ndarray` | list of index arrays for coupled variables (targets) from other subsystems. |

Example:

```python
from mdotoolbox.core import Problem, Function

# Define discipline problem
y_func = Function(func=lambda z, x: z**2 + x, x=["z", "x"], name="disc1")
problem = Problem(objective=y_func, lbounds=[0, 0], ubounds=[10, 10])

# Create subsystem
subsystem = COSubsystem(
    problem=problem,
    z_idxs=[0],  # First variable is shared
    x_idxs=[1],  # Second variable is local
    y_idxs=[0],  # First output
    y_bar_coupled_idxs=[],  # No coupling
)
```

### COSystem

| Attribute    | Type                | Description                                                       |
|:-------------|:--------------------|:------------------------------------------------------------------|
| `problem`    | `Problem`           | System-level optimization problem with objective and constraints. |
| `subsystems` | `list[COSubsystem]` | All subsystems in the MDO problem.                                |

Workflow:

1. Receive discrepancies J_i from subsystems
2. Optimize system objective subject to J_i = 0 and system constraints
3. Return updated targets (z_bar, x_bar, y_bar) to subsystems
4. Repeat until convergence

Example:

```python
f_func = Function(func=lambda z, x: z**2 + x, x=["z", "x"], name="disc1")
system_problem = Problem(objective=y_func, lbounds=[0, 0], ubounds=[10, 10])

system = COSystem(problem=system_problem, subsystems=[subsystem1, subsystem2])
```

### CollaborativeOptimization

| Attribute             | Type                     | Description                                                          |
|:----------------------|:-------------------------|:---------------------------------------------------------------------|
| `system`              | `COSystem`               | System-level coordinator with subsystems.                            |
| `subsystem_optimizer` | `str` or `Callable`      | Optimizer for subsystem minimization (e.g., 'cobyqa', 'cobyla').     |
| `system_optimizer`    | `str` or `Callable`      | Optimizer for system-level coordination.                             |
| `epsilon_J`           | `float`                  | Convergence tolerance for coupling discrepancy $J_\text{total}$.     |
| `epsilon_h`           | `float`                  | Convergence tolerance for constraint violation $h_\text{total}$.     |
| `budget`              | `BudgetManager` or `int` | Budget allocation strategy. If int, creates weighted budget manager. |
| `max_iter`            | `int`                    | Maximum number of CO iterations. None for unlimited.                 |
| `max_eval`            | `int`                    | Equivalent to `budget` as integer.                                   |
| `cache_dir`           | `Path`                   | Directory to save intermediate progress in case code crashes.        |
| `name`                | `str`                    | Field to define problem name.                                        |
| `solver`              | `str`                    | Field to define solver name.                                         |

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

### Example - Basic usage with integer budget

```python
solver = CollaborativeOptimization(
    system=co_system,
    subsystem_optimizer="cobyqa",
    system_optimizer="cobyqa",
    epsilon_J=1e-6,
    epsilon_h=1e-6,
    budget=400,
)

# Weighted budget allocation
budget = BudgetManager(
    mode="weighted",
    total_budget=500,
    system_ratio=0.4,
    subsystem_weights=[0.6, 0.4],  # 60% to subsys 1, 40% to subsys 2
    iteration_ratio=0.05,
)
solver = CollaborativeOptimization(
    system=co_system,
    subsystem_optimizer="cobyla",
    system_optimizer="slsqp",
    budget=budget,
)

# Solve the problem
z_bar0 = np.array([1.0, 2.0])
x_bar0 = np.array([0.5])
y_bar0 = np.array([1.0, 1.0])
z_under0 = np.array([1.0, 2.0])
x_under0 = np.array([0.5])
result = solver.solve(z_bar0, x_bar0, y_bar0, z_under0, x_under0)

print(f"Optimal objective: {result.best.f_bar}")
print(f"Final discrepancy: {result.best.J_total}")
```

### Notes

- Convergence requires $\sum_i^N J_i = J_\text{total} \leq \epsilon_J$ AND feasibility $h_\text{total} <= \epsilon_h$
- System targets (z_bar, x_bar, y_bar) updated each iteration
- Subsystems optimize in parallel conceptually
- Budget exhaustion triggers early termination
