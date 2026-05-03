# Contributing to MDOToolbox

Thank you for your interest in contributing to MDOToolbox! This guide covers the development setup, testing workflow, and contribution process.

## Development Setup

1. **Clone the repository:**

```bash
git clone https://github.com/moebehfn/mdotoolbox.git
cd mdotoolbox
```

1. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. **Install in editable mode with test dependencies:**

```bash
pip install -e ".[test]"
```

## Running Tests

We use `pytest` for testing. The configuration is in `pyproject.toml`.

**Fast suite** (skips slow GP training and full BACO iterations):

```bash
pytest -m "not slow"
```

**Full suite:**

```bash
pytest
```

**Specific test file:**

```bash
pytest tests/test_regression.py -v
```

The `slow` marker is used for tests that train real Gaussian Process models or run full BACO optimization iterations.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use type hints for function signatures.
- Maintain consistency with existing code's naming conventions (e.g., `z` for shared variables, `y` for coupling variables).

## Project Structure

| Directory | Contents |
|-----------|----------|
| `src/mdotoolbox/` | Core library source code |
| `docs/` | Documentation source (MkDocs) |

<!-- ## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.
2. **Implement** your changes with clear, focused commits.
3. **Ensure all tests pass** before submitting.
4. **Include tests** for any new functionality.
5. **Update** `CHANGELOG.md` with your changes.
6. **Submit** a pull request with a clear description of the changes. -->

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
