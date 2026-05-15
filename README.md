# MDOToolbox: Multidisciplinary Design Optimization with Bayesian Collaborative Frameworks

MDOToolbox is a Python library for solving multi-disciplinary optimization (MDO) problems using hierarchical decomposition and Bayesian optimization. It provides five optimization frameworks, Gaussian Process surrogates, and a suite of built-in benchmark problems for rapid prototyping and research.

## Cite us (preprint)

```bibtex
@misc{baco2026,
      title         = {Bayesian Algorithm for Collaborative Optimization with Application to Aircraft Design}, 
      author        = {Mohamed Ali Belhafnaoui and Youssef Diouane},
      year          = {2026},
      eprint        = {2605.05474},
      archivePrefix = {arXiv},
      primaryClass  = {math.OC},
      url           = {https://arxiv.org/abs/2605.05474}, 
}
```

[Paper PDF](https://arxiv.org/pdf/2605.05474)

## Features

- **Multiple MDO frameworks** CO, BACO, ICO, MCO, and ECO for different problem classes
- **Bayesian optimization** with Gaussian Process surrogates for expensive black-box functions
- **7 built-in benchmark problems** covering analytical, mechanical, thermal, biomedical, and chemical domains
- **Parallel multi-start acquisition optimization** for robust global search
- **Support for constrained and unconstrained optimization**

## Installation

**Basic install:**

```bash
pip install mdotoolbox
```

**Development install (with test dependencies):**

```bash
pip install -e ".[test]"
```

**Dependencies:** numpy, scipy, smt, joblib, numba, matplotlib, scikit-learn, gpytorch, torch, jaxtyping, typeguard, pandas, openmdao, openaerostruct, pymc.

## Quick Start

This example solves the classic Sellar benchmark problem using the Bayesian Collaborative Optimization (BACO) framework:

```python
import numpy as np
from mdotoolbox.benchmarking import sellar_problem
from mdotoolbox.frameworks import (
    BayesianCollaborativeOptimization,
    BACOSubsystem,
    BACOSystem,
)

# Load benchmark problem
problem_dict = sellar_problem()

# Build system
subsystems = [BACOSubsystem(**sub) for sub in problem_dict["subsystems"]]
system = BACOSystem(
    problem=problem_dict["system_problem"],
    subsystems=subsystems,
)

# Create and run solver
solver = BayesianCollaborativeOptimization(
    system=system,
    subsystem_optimizer="cobyqa",
    system_optimizer="cobyqa",
    epsilon_J=1e-6,
    epsilon_h=1e-6,
    max_eval=50,
    n_multistart=5,
    n_initial=8,
)

z0 = np.array([5.0, 2.0])
x0 = np.array([1.0])
y0 = np.array([1.0, 1.0])
results = solver.solve(z0, x0, y0)
print(results)
```

## Framework Selection Guide

| Framework | Description | Best For |
|-----------|-------------|----------|
| **CO** (Collaborative Optimization) | Classical bilevel formulation | Well-behaved problems; fastest when it works |
| **BACO** (Bayesian Collaborative Optimization) | CO with GP surrogates | Expensive simulations; efficient with function evaluations |
| **ICO** (Improved Collaborative Optimization) | Enhanced convergence | When CO has convergence issues |
| **MCO** (Modified Collaborative Optimization) | Alternative formulation | Experimental problems |
| **ECO** (Enhanced Collaborative Optimization) | Advanced convergence criteria | Best convergence guarantees; slowest but most reliable |

## Built-in Benchmark Problems

| Function | Description |
|----------|-------------|
| `sellar_problem()` | Classic 2-discipline analytical benchmark |
| `scalable_mdo_problem()` | Configurable scalable problem with adjustable size |
| `speed_reducer_problem()` | Mechanical engineering design problem |
| `electronic_packaging_problem()` | Thermal management problem |
| `heart_dipole_problem()` | Biomedical application |
| `power_converter_problem()` | Power electronics design |
| `propane_combustion_problem()` | Chemical process optimization |

All benchmark functions return a dictionary with `system_problem` and `subsystems` keys, ready to pass into any framework.

## Project Structure

```
.
├── src/mdotoolbox/   # Core library source
├── docs/             # Documentation source
├── pyproject.toml    # Build and dependency configuration
└── README.md         # Project overview
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing instructions, and contribution guidelines.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Links

- **Documentation:** [https://moebehfn.github.io/mdotoolbox/](https://moebehfn.github.io/mdotoolbox/)
- **Repository:** [https://github.com/moebehfn/mdotoolbox](https://github.com/moebehfn/mdotoolbox)
- **Author:** Mohamed Ali Belhafnaoui (<mohamed-ali.belhafnaoui@etud.polymtl.ca>)

## Please consider supporting our other project: fluxwing

Python library for simulating the aerodynamics of a wing at a given condition using VLM, as well as the structural loads using an Euler-Bernoulli solver:

- [github](https://github.com/moebehfn/fluxwing)
- [documentation](https://moebehfn.github.io/fluxwing)
