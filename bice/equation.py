import numpy as np

# NOTE: this class is not yet used at all, future work!


class Equation:

    def __init__(self):
        # The vector of unknowns
        self.u = None
        # Is the left-hand side (lhs) = du/dt (time evolution / Cauchy equation) or is it zero?
        self.equals_dudt = True

    # The (vector) dimension of the equation
    @property
    def dim(self):
        return self.u.size

    # Calculate the right-hand side of the equation 0 = rhs(u)
    def rhs(self, u):
        raise NotImplementedError(
            "No right-hand side (rhs) implemented for this equation!")

    # The mass matrix determines the linear relation of the rhs to the temporal derivatives dudt
    def mass_matrix(self):
        # default case: assume the identity matrix I
        return np.eye(self.dim)


class VolumeConstraint(Equation):

    def __init__(self, reference_equation):
        super().__init__()
        self.ref_eq = reference_equation

    def rhs(self, u):
        # calculate the difference in volumes
        # between current and previous unknowns of the
        # reference equation
        return np.trapz(u-self.ref_eq.u, self.ref_eq.x)

    # TODO: add mass_matrix

class TranslationConstraint(Equation):

    def __init__(self, reference_equation):
        super().__init__()
        self.ref_eq = reference_equation

    def rhs(self, u):
        # calculate the difference in center of masses
        # between current and previous unknowns of the
        # reference equation
        return np.dot(self.ref_eq.x, u-self.ref_eq.u)

    # TODO: add mass_matrix