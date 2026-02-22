# 6. Runtime View

## 6.1 Parameter Continuation Step

This scenario shows what happens when `Problem.continuation_step()` is called.

```{mermaid}
sequenceDiagram
    User->>Problem: continuation_step()
    Problem->>ProblemHistory: update()
    Problem->>ContinuationStepper: step(problem)
    activate ContinuationStepper
    ContinuationStepper->>Problem: jacobian(u)
    Problem->>Equation: jacobian(u)
    Equation-->>Problem: Matrix
    Problem-->>ContinuationStepper: Matrix
    ContinuationStepper->>ContinuationStepper: Predictor-Step (Tangent)
    ContinuationStepper->>ContinuationStepper: Corrector-Step (Newton)
    deactivate ContinuationStepper
    Problem->>BifurcationDiagram: add_solution_point(sol)
```
