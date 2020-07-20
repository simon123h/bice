
# TODO list

## General

- find backronym
- follow the TODO's in the code

## Architecture

- A problem should consist of one or multiple "Equations" and is then assembled. --> generalize this assembly process.
- Add a general NewtonSolver class and use it

## Bugs & issues

- Unsure whether Runge-Kutta-Fehlberg-45 works properly or if there is some bug. Does adaptive step-size work properly?

## Mathematical features

- Implement a default norm (L2-norm?) in Problem class
- Add default interfaces for FEM, FD and pseudospectral discretizations
- Mesh adaption
- implement implicit BDF(n) time-stepping schemes
- implement a generic Runge-Kutta(n) scheme for neatness?

## Bifurcations, Continuation & Eigenproblems

- Implement automatic recording of branches in the Problem class
- Implement Eigensolver
- Stability detection
- Bifurcation detection
- Branch switching
- Bifurcation tracking (augmented systems)
- Continuation of time-periodic solutions
- Implement constraints as 'Equations'

## Optimization

- check performance / increase performance of solvers
- use a better (external) Newton solver in PseudoArclengthContinuation

## Documentation

- fill the main README with info
