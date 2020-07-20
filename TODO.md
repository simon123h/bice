
# TODO list

## General

- find backronym
- follow the TODO's in the code

## Architecture

- A problem should consist of one or multiple "Equations" and is then assembled. --> generalize this assembly process.
- Add a general NewtonSolver class and use it

## Mathematical features

- Implement a default norm (L2-norm?) in Problem class
- Add default interfaces for FEM, FD and pseudospectral discretizations
- Mesh adaption
- implement implicit BDF(n) time-stepping schemes
- implement a generic Runge-Kutta(n) scheme for neatness?

## Bifurcations, Continuation & Eigenproblems

- Implement automatic recording of branches in the Problem class
- Implement Eigensolver
- stability detection
- bifurcation detection
- branch switching
- bifurcation tracking (augmented systems)
- continuation of time-periodic solutions
- Implement constraints as 'Equations'

## Optimization

- check performance / increase performance of solvers
- use a better (external) Newton solver in PseudoArclengthContinuation

## Documentation

- fill the main README with info
