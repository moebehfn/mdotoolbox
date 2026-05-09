# Bayesian Algorithm for Collaborative Optimization Framework

## Mathematical Formulation

System Level:

$$
\begin{align*}
    &\max_{
        \overline{\mathbf{z}},
        \overline{\mathbf{x}},
        \overline{\mathbf{y}}
    }\ && \alpha_f(
        \overline{\mathbf{z}},
        \overline{\mathbf{x}},
        \overline{\mathbf{y}}
    ) \tag{\(\widetilde{P}_\text{sys}\)} \\
    &\text{subject to:}\
    &&\mathbf{\mu_c}(
        \overline{\mathbf{z}},
        \overline{\mathbf{x}},
        \overline{\mathbf{y}},
    ) &&\geq \mathbf{0}\\
    %
    &  && \mu_{J_i}(
        \overline{\mathbf{z}},
        \overline{\mathbf{x}}_i,
        \overline{\mathbf{y}},
        \underline{\mathbf{z}}_i^*,
        \underline{\mathbf{x}}_i^*
    ) &&= 0\ \text{for all}\ i\in\{1\cdots N\}
\end{align*}
$$

Subsystem $i$ level:

$$
\begin{equation*}
    \max_{
        \underline{\mathbf{z}}_i,
        \underline{\mathbf{x}}_i
    }\ \alpha_{J_i}(
        \overline{\mathbf{z}},
        \overline{\mathbf{x}}_i,
        \overline{\mathbf{y}},
        \underline{\mathbf{z}}_i,
        \underline{\mathbf{x}}_i
    ) \tag{\(\widetilde{P}_i\)}\
    \text{subject to:}\
    \mathbf{\mu}_{\mathbf{g}_i}(
        \overline{\mathbf{y}}_{j\neq i},
        \underline{\mathbf{z}}_i,
        \underline{\mathbf{x}}_i
    )\geq\mathbf{0}
\end{equation*}
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

### Key Innovations

- GP surrogates for subsystem objectives (J_i) and constraints (g_i)
- Acquisition function optimization (Expected Improvement)
- Latin Hypercube Sampling for efficient DoE initialization
- Reduced subsystem evaluations through surrogate modeling

### Advantages

- Reduced subsystem calls
- Better for expensive analysis codes
- Uncertainty quantification via GP variance
- Adaptive sampling focuses on promising reg_ions

## Implementation

### BACOSubsystem

| Attribute          | Type                             | Description                                                                 |
|:-------------------|:---------------------------------|:----------------------------------------------------------------------------|
| `problem`          | `Problem`                        | Local subsystem optimization problem with objective and constraints.        |
| `z_idxs`           | `np.ndarray`                     | Indices of shared design variables in the global variable vector.           |
| `x_idxs`           | `np.ndarray`                     | Indices of local design variables in the global variable vector.            |
| `y_idxs`           | `np.ndarray`                     | Indices of subsystem outputs in the global output vector.                   |
| `y_coupled_idxs`   | `np.ndarray`                     | list of index arrays for coupled variables (targets) from other subsystems. |
| `surrogate_config` | `SMTGPConfig` or `TorchGPConfig` | GP configuration.                                                           |

Workflow:

1. initialize_doe(): LHS sampling to create initial DoE
2. build_surrogate_J_i(), build_surrogate_g_i(): Train GP surrogates
3. solve_acquisition(): Optimize acquisition function
4. Update DoE and retrain (repeat)

### BACOSystem

| Attribute          | Type                             | Description                                                       |
|:-------------------|:---------------------------------|:------------------------------------------------------------------|
| `problem`          | `Problem`                        | System-level optimization problem with objective and constraints. |
| `subsystems`       | `list[COSubsystem]`              | All subsystems in the MDO problem.                                |
| `surrogate_config` | `SMTGPConfig` or `TorchGPConfig` | GP configuration.                                                 |

### BayesianCollaborativeOptimization

| Attribute             | Type                | Description                                                                           |
|:----------------------|:--------------------|:--------------------------------------------------------------------------------------|
| `system`              | `COSystem`          | System-level coordinator with subsystems.                                             |
| `subsystem_optimizer` | `str` or `Callable` | Optimizer for subsystem minimization (e.g., 'cobyqa', 'cobyla')                       |
| `system_optimizer`    | `str` or `Callable` | Optimizer for system-level coordination.                                              |
| `n_initial`           | `int` or `Callable` | Size of the initial Design of Experiments.                                            |
| `acq_func`            | `Callable`          | Choice of acquisition function.                                                       |
| `n_multistart`        | `int`               | Number of multi-starts when solving $\widetilde{P}_\text{sys}$ and $\widetilde{P}_i$. |
| `epsilon_J`           | `float`             | Convergence tolerance for coupling discrepancy $J_\text{total}$.                      |
| `epsilon_h`           | `float`             | Convergence tolerance for constraint violation $h_\text{total}$.                      |
| `max_eval`            | `int`               | Evaluation budget.                                                                    |
| `cache_dir`           | `Path`              | Directory to save intermediate progress in case code crashes.                         |
| `name`                | `str`               | Field to define problem name.                                                         |
| `solver`              | `str`               | Field to define solver name.                                                          |
