import numpy as np
from bice.core.problem import Problem
from bice.core.profiling import profile


class LyapunovExponentCalculator():
    """
    Calculates the spectrum of Lyapunov exponents for a given problem.
    Uses the algorithm reported in:
    Wolf, A., Swift, J. B., Swinney, H. L., & Vastano, J. A. (1985).
    "Determining Lyapunov exponents from a time series"
    Physica D: Nonlinear Phenomena, 16(3), 285-317.
    """

    def __init__(self,
                 problem: Problem,
                 nexponents: int = 1,
                 epsilon: float = 1e-6,
                 nintegration_steps: int = 1) -> None:
        #: reference to the problem
        self.problem = problem
        #: the number of exponents to be calculated
        self.nexponents = nexponents
        #: the norm of the perturbation
        self.epsilon = epsilon
        #: the number of time-integration steps for each trajectory
        self.nintegration_steps = nintegration_steps
        #: cumulative variable for the total integration time
        self.T = 0
        # storage for the perturbation vectors and the reference trajectory
        self.perturbations = None
        self.generate_perturbation_vectors()
        # cumulative sum of the exponents, the actual exponents are calculated from sum / T
        self.__sum = np.zeros(nexponents)

    # return the Lyapunov exponents
    @property
    def exponents(self) -> np.ndarray:
        # calculate average from sum
        return self.__sum / self.T

    # generate a new set of orthonormal perturbation vectors
    def generate_perturbation_vectors(self) -> None:
        self.perturbations = [np.random.rand(self.problem.ndofs)
                              for i in range(self.nexponents)]
        self.orthonormalize()

    # orthonormalize the set of perturbation vectors using Gram-Schmidt-Orthonormalization
    @profile
    def orthonormalize(self) -> np.ndarray:
        # construct orthogonal vectors using Gram-Schmidt-method
        for i in range(self.nexponents):
            for j in range(i):
                self.perturbations[i] -= self.perturbations[j] * \
                    np.sum(self.perturbations[i]*self.perturbations[j]) / \
                    np.sum(self.perturbations[j]*self.perturbations[j])
        # normalize vectors and return each norm
        norms = np.array([np.linalg.norm(t) for t in self.perturbations])
        for i in range(self.nexponents):
            self.perturbations[i] /= norms[i]
        return norms

    # integrate dt, reorthonormalize and update Lyapunov exponents
    @profile
    def step(self) -> None:
        # if the number of points changed, regenerate the perturbation vectors
        if self.perturbations[0].size != self.problem.u.size:
            self.generate_perturbation_vectors()
        # load reference trajectory from problem, in case it has changed
        reference = self.problem.u.copy()
        # generate perturbed trajectories from reference and perturbations
        trajectories = [reference + ptb *
                        self.epsilon for ptb in self.perturbations]
        trajectories.append(reference)
        # integrate every trajectory, including reference
        time = self.problem.time
        dt = self.problem.time_stepper.dt
        for i in range(self.nexponents+1):
            self.problem.u = trajectories[i]
            self.problem.time = time
            self.problem.time_stepper.dt = dt
            self.problem.history.clear()
            for _ in range(self.nintegration_steps):
                self.problem.time_step()
            trajectories[i] = self.problem.u.copy()
        # calculate new perturbation vectors from difference to reference
        reference = trajectories[-1]
        for i in range(self.nexponents):
            self.perturbations[i] = trajectories[i] - reference
        # re-orthonormalize
        norms = self.orthonormalize()
        # add to total exponents sum and increment time
        self.__sum += np.log(norms / self.epsilon)
        self.T += self.problem.time - time
