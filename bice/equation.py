
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
