# 4. Solution Strategy

The following architectural decisions have been made for _bice_:

- **Problem-Equation-Solver Decomposition**: Separating the mathematical problem definition (`Problem`) from its governing equations (`Equation`) and the numerical algorithms (`ContinuationStepper`, `NewtonSolver`).
- **Flexible Discretization**: Providing base classes (`Equation`, `FiniteDifferencesEquation`) to allow users to implement different numerical schemes.
- **Pseudo-Arclength Continuation**: Defaulting to pseudo-arclength continuation for its robustness near folds and turning points.
