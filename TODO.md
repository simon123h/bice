
# TODO list

## General

- find backronym
- follow the TODO's in the code

## Architecture

- A problem should consist of one or multiple "Equations" and is then assembled. --> generalize this assembly process.
- Save/load solution points & everything that is needed
- Store parameters in a np.array and pass index for continuation

## Bugs & issues

- Unsure whether Runge-Kutta-Fehlberg-45 works properly or if there is some bug. Does adaptive step-size work properly?

## Mathematical features

- Add default interfaces for FEM, FD and pseudospectral discretizations
- Mesh adaption
- implement implicit BDF(n) time-stepping schemes
- implement a generic Runge-Kutta(n) scheme for neatness?

## Bifurcations, Continuation & Eigenproblems

- Branch switching
- Bifurcation tracking (augmented systems)
- Continuation of time-periodic solutions
- Implement constraints as 'Equations'

## Optimization

- check performance / increase performance of solvers
- use a better (external) Newton solver in PseudoArclengthContinuation

## Documentation

- fill the main README with info
