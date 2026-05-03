# Building MDO Problems from Scratch in MDOToolbox

This guide provides a detailed, step-by-step walkthrough for defining a Multi-Disciplinary Optimization (MDO) problem from scratch and solving it using Collaborative Optimization (**CO**) or Bayesian Collaborative Optimization (**BACO**).

---

## 1. Conceptual Decompositon

Before writing code, identify the three types of variables in your MDO problem:

1.  **Shared Variables ($z$):** Design variables that are inputs to multiple disciplines (e.g., altitude, Mach number).
2.  **Local Variables ($x_i$):** Design variables that are inputs to only one specific discipline $i$ (e.g., wing thickness for aerodynamics).
3.  **Coupling Variables ($y$):** Outputs from one discipline that serve as inputs to another (e.g., aerodynamic loads acting on the structure).

---

## 2. Defining Core Components

Every problem is built using `Function`, `Constraint`, and `Problem` objects from `mdotoolbox.core`.

### A. Define the Mathematical Functions
Functions must be Python callables (or JIT-compiled with `@njit`) that accept individual scalar arguments.

```python
import numpy as np
from mdotoolbox.core import Function, Constraint, Problem


# Discipline 1: y1 = z1^2 + z2 + x1 - 0.2*y2
def d1_output_func(z1, z2, x1, y2):
    return z1**2 + z2 + x1 - 0.2 * y2


# Discipline 2: y2 = sqrt(y1) + z1 + z2
def d2_output_func(z1, z2, y1):
    return np.sqrt(np.abs(y1)) + z1 + z2


# System Objective: f = x1^2 + z2 + y1 + exp(-y2)
def system_obj_func(z1, z2, x1, y1, y2):
    return x1**2 + z2 + y1 + np.exp(-y2)
```

### B. Wrap in `Function` Objects
The `Function` object maps your Python function to specific variable names.

```python
# Variables for Discipline 1: [shared, local, coupled_input]
f_d1 = Function(func=d1_output_func, x=["z1", "z2", "x1", "y2"], name="y1_calc")

# Variables for Discipline 2: [shared, coupled_input]
f_d2 = Function(func=d2_output_func, x=["z1", "z2", "y1"], name="y2_calc")

# System objective variables: [all variables]
f_sys = Function(
    func=system_obj_func, x=["z1", "z2", "x1", "y1", "y2"], name="total_obj"
)
```

### C. Define Constraints
Constraints wrap a `Function` with a comparison type (`ge` for $\ge$, `le` for $\le$, `eq` for $=$) and a reference value.

```python
# System constraint: y1 >= 3.16
def c1_func(z1, z2, x1, y1, y2):
    return y1


c_sys1 = Constraint(
    func=Function(func=c1_func, x=["z1", "z2", "x1", "y1", "y2"]),
    ctype="ge",
    value=3.16,
)
```

---

## 3. Assembling the Hierarchical Structure

### Step 1: Create Subsystem Problems
Each discipline needs its own `Problem` object. For CO/BACO, the **objective** of a subsystem problem is the function that computes its **coupling outputs**.

```python
# Subsystem 1 Problem
sub1_prob = Problem(
    objective=f_d1,
    constraints=[],  # Local constraints go here
    lbounds=[-10, 0, 0, -100],  # [z1, z2, x1, y2]
    ubounds=[10, 10, 10, 100],
    name="sub1",
)

# Subsystem 2 Problem
sub2_prob = Problem(
    objective=f_d2,
    constraints=[],
    lbounds=[-10, 0, -100],  # [z1, z2, y1]
    ubounds=[10, 10, 100],
    name="sub2",
)
```

### Step 2: Define Index Mappings
You must tell the framework how variables in the global vectors ($z, x, y$) map to each subsystem.

*   `z_idxs`: Indices in the shared variable vector $z$.
*   `x_idxs`: Indices in the **local** variable vector $x$.
*   `y_idxs`: Indices in the **coupling** variable vector $y$ that this subsystem **produces**.
*   `y_coupled_idxs`: A list of arrays. Each array contains indices in $y$ that this subsystem **receives** as input.

```python
# Subsystem 1 Mappings
# z = [z1, z2], x = [x1], y = [y1, y2]
sub1_config = {
    "problem": sub1_prob,
    "z_idxs": [0, 1],  # Uses z1, z2
    "x_idxs": [0],  # Uses x1
    "y_idxs": [0],  # Produces y1 (index 0 in y)
    "y_coupled_idxs": [[1]],  # Receives y2 (index 1 in y)
}

# Subsystem 2 Mappings
sub2_config = {
    "problem": sub2_prob,
    "z_idxs": [0, 1],  # Uses z1, z2
    "x_idxs": [],  # Uses no local variables
    "y_idxs": [1],  # Produces y2 (index 1 in y)
    "y_coupled_idxs": [[0]],  # Receives y1 (index 0 in y)
}
```

---

## 4. Solving with Collaborative Optimization (CO)

CO is a deterministic bilevel approach. It is suitable for problems where disciplines are cheap to evaluate.

```python
from mdotoolbox.frameworks import CollaborativeOptimization, COSubsystem, COSystem

# 1. Instantiate Subsystems
subsystems = [COSubsystem(**sub1_config), COSubsystem(**sub2_config)]

# 2. Define System Problem (The coordinator)
sys_prob = Problem(
    objective=f_sys,
    constraints=[c_sys1],
    lbounds=[-10, 0, 0, -100, -100],  # [z1, z2, x1, y1, y2]
    ubounds=[10, 10, 10, 100, 100],
    name="system",
)

# 3. Instantiate System
system = COSystem(problem=sys_prob, subsystems=subsystems)

# 4. Setup Solver
solver = CollaborativeOptimization(
    system=system,
    subsystem_optimizer="cobyqa",  # Optimizer for J_i minimization
    system_optimizer="cobyqa",  # Optimizer for coordination
    budget=500,  # Max total evaluations
)

# 5. Solve from initial guess
z0 = np.array([5.0, 2.0])
x0 = np.array([1.0])
y0 = np.array([1.0, 1.0])
results = solver.solve(z0, x0, y0)

print(f"Optimal f: {results.best.f}")
```

---

## 5. Solving with Bayesian Collaborative Optimization (BACO)

BACO uses Gaussian Process surrogates and is highly efficient for expensive black-box simulations.

```python
from mdotoolbox.frameworks import (
    BayesianCollaborativeOptimization,
    BACOSubsystem,
    BACOSystem,
)

# 1. Instantiate Subsystems
subsystems = [BACOSubsystem(**sub1_config), BACOSubsystem(**sub2_config)]

# 2. Instantiate System
system = BACOSystem(problem=sys_prob, subsystems=subsystems)

# 3. Setup Solver
solver = BayesianCollaborativeOptimization(
    system=system,
    subsystem_optimizer="cobyqa",
    system_optimizer="cobyqa",
    n_initial=10,  # Number of initial LHS samples for surrogates
    max_eval=100,  # Total budget of real discipline evaluations
    epsilon_J=1e-4,  # Consistency tolerance
)

# 4. Solve
results = solver.solve(z0, x0, y0)
```

## Key Differences Summary

| Feature | Collaborative Optimization (CO) | Bayesian CO (BACO) |
| :--- | :--- | :--- |
| **Philosophy** | Direct optimization of disciplines | Surrogate-based (Bayesian) optimization |
| **Efficiency** | High evaluation cost | Very low evaluation cost |
| **Robustness** | Sensitive to initial guess | Robust to noise and local minima |
| **Parallelism** | Subsystems can run in parallel | Subsystems and GP training are parallelized |
| **Best For** | Cheap/Analytical models | Expensive simulations (CFD, FEA) |
