# 3. System Scope and Context

## 3.1 Business Context

The system context shows how _bice_ interacts with its environment.

```{mermaid}
graph LR
    User[User / Researcher] -->|Configures Problem & Equations| Bice[bice]
    Bice -->|Uses for Linear Algebra| NumPy[NumPy / SciPy]
    Bice -->|Uses for Derivatives| Findiff[findiff / numdifftools]
    Bice -->|Generates Plots| Matplotlib[Matplotlib]
```

- **User**: Scientists or researchers who define mathematical equations (ODEs/PDEs) and parameters.
- **NumPy / SciPy**: Low-level linear algebra solvers and array operations.
- **findiff / numdifftools**: Tools for computing finite difference derivatives.
- **Matplotlib**: Used for visualizing results and bifurcation diagrams.

## 3.2 Technical Context

_bice_ is used as a Python library. Users typically write scripts or Jupyter notebooks to define their problems and call _bice_'s continuation routines.
