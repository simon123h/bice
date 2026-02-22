# 5. Building Block View

## 5.1 Level 1: Core System Overview

The core system is decomposed into several packages:

```{mermaid}
classDiagram
    class Problem {
        +Equation eq
        +ContinuationStepper continuation_stepper
        +NewtonSolver newton_solver
        +EigenSolver eigen_solver
        +continuation_step()
        +newton_solve()
    }
    class Equation {
        +u: Array
        +rhs(u)
        +jacobian(u)
    }
    class ContinuationStepper {
        <<interface>>
        +step(problem)
    }
    class NewtonSolver {
        <<interface>>
        +solve(f, u0, jac)
    }
    class FiniteDifferencesEquation {
        +build_FD_matrices()
    }
    class PseudoArclengthContinuation {
        +step(problem)
    }

    Problem o-- Equation
    Problem o-- ContinuationStepper
    Problem o-- NewtonSolver
    Equation <|-- FiniteDifferencesEquation
    ContinuationStepper <|-- PseudoArclengthContinuation
```

### 5.1.1 Core Components

- **Problem**: Acts as the central hub, aggregating equations, solvers, and settings.
- **Equation**: Base class for defining the mathematical model.
- **ContinuationStepper**: Strategy for parameter continuation (e.g., Natural, Pseudo-Arclength).
- **NewtonSolver**: Algorithms for solving the resulting nonlinear systems.
