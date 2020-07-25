
# TODO list
**NOTE:**

**This list is deprecated. The issues are now found in the GitLab repo!**


## General

- Find backronym
- Follow the TODO's in the code

## Architecture

- Ability to save/load solution points & everything that is needed to/from disk
- Store parameters in a np.array and pass index for continuation, make use of @property decorator
- Standardize the vector of spatial coordinates and their spatial dimension for all Equations
- Make sure that equations & constraints work for arbitrary spatial dimension

## Bugs & issues

- Unsure whether Runge-Kutta-Fehlberg-45 works properly or if there is some bug. Does adaptive step-size work properly?

## Mathematical features

- Add default interfaces for FD and pseudospectral discretizations
- Implement some FEM interface
- Mesh adaption
- Implement a generic Runge-Kutta(n) and BDF(n) scheme for neatness?

## Bifurcations, Continuation & Eigenproblems

- Bifurcation tracking (augmented systems)
- Converging exactly onto bifurcation points (either by bisection or with an augmented system)
- Branch switching
- Continuation of time-periodic solutions

## Optimization

- Check performance / increase performance of solvers
- Make sure that we use a good, sensible third-party Newton solver in the NewtonSolver class
- Try using the NewtonSolver class also in PseudoArclengthContinuation, instead of its own for-loop

## Documentation

- Fill the main README with info
