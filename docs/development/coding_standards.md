# Coding Standards

To maintain a clean and consistent codebase, we enforce strict coding standards using automated tools.

## Python Version

We target **Python 3.11+**. Please avoid using features that are deprecated in these versions. We use `from __future__ import annotations` to support modern type hinting across all modules.

## Style and Formatting

We use **Ruff** for both linting and formatting. Ruff is configured in `pyproject.toml` with a line length limit of **140 characters**.

### Running Checks

To check your code for style issues:
```bash
ruff check .
```

To automatically fix most issues:
```bash
ruff check --fix .
```

To format your code:
```bash
ruff format .
```

## Type Checking

We use **Mypy** for static type checking. All core library code (`src/bice/core/`) should be fully typed and pass Mypy checks without errors.

To run type checks:
```bash
mypy src/bice
```

## Docstrings

We follow the **NumPy-style docstring convention**. Every public class and method must have a descriptive docstring.

Example:
```python
def solve(self, A: Matrix, M: Matrix | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the eigenproblem A*x = v*x.

    Parameters
    ----------
    A : Matrix
        The system matrix.
    M : Matrix, optional
        The mass matrix, by default None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing (eigenvalues, eigenvectors).
    """
```
