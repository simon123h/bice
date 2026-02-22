# 8. Concepts

## 8.1 Numerical Path Continuation

Numerical path continuation (also called homotopy continuation) is used to track the solutions $u$ of a nonlinear system $f(u, p) = 0$ as a parameter $p$ is varied.

### 8.1.1 Pseudo-Arclength Continuation

In pseudo-arclength continuation, an additional arclength constraint is added to the system, allowing for the tracking of solution branches around limit points (folds) where natural continuation fails.

## 8.2 Discretization Schemes

- **Finite Differences**: $d^n u / dx^n$ is approximated using grid points and stencils.
- **Pseudospectral Methods**: (Reserved for future implementation).
