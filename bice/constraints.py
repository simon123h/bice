from .equation import Equation
import numpy as np


class VolumeConstraint(Equation):
    """
    A volume constraint (or mass constraint) assures the conservation of
    the integral of the unknowns of some given equation when solving the system.
    We may even prescribe the target volume (or mass) with a parameter,
    but we don't have to.
    The constraint equation comes with an additional (unknown) Lagrange
    multiplier that can be interpreted as an influx into the system.
    """
    # TODO: is this constraint implemented correctly?
    #  @simon: should this simply keep the volume constant? then it's correct i guess.
    #  Should it keep the volume at a value specified by a parameter, i.e. enabling
    #  continuation in the volume, then rhs should return something like:
    #  np.trapz(self.ref_eq.u - volume_parameter, self.ref_eq.x)

    def __init__(self, reference_equation):
        super().__init__()
        # on which equation/unknowns should the constraint be imposed?
        self.ref_eq = reference_equation
        # the constraint equation couples to some other equation of the problem
        self.is_coupled = True
        # this equation brings a single extra degree of freedom (influx Lagrange multiplier)
        self.u = np.array([0])
        # This parameter allows for prescribing a fixed volume (unless it is None)
        self.fixed_volume = None

    def rhs(self, u):
        # generate empty vector of residual contributions
        res = np.zeros((u.size))
        # employ the constraint equation
        if self.fixed_volume is None:
            # calculate the difference in volumes between current
            # and previous unknowns of the reference equation
            if len(self.ref_eq.x) == 2:
                res[self.idx] = np.mean(u[self.ref_eq.idx] - self.ref_eq.u)
            else:
                res[self.idx] = np.trapz(u[self.ref_eq.idx] -
                                         self.ref_eq.u, self.ref_eq.x[0])
        else:
            # parametric constraint: calculate the difference between current
            # volume and the prescribed fixed_volume parameter
            res[self.idx] = np.trapz(u[self.ref_eq.idx],
                                     self.ref_eq.x[0]) - self.fixed_volume
        # Add the constraint to the reference equation: unknown influx is the Langrange multiplier
        res[self.ref_eq.idx] = u[self.idx]
        return res

    def mass_matrix(self):
        # couples to no time-derivatives
        return np.zeros(self.problem.dim)

    def plot(self, ax):
        # nothing to plot
        pass


class TranslationConstraint(Equation):
    """
    A translation constraint assures that the center of mass of some
    reference equation's unknowns does not move when solving the system.
    The additional constraint equations (one per spatial dimension) come
    with Lagrange multipliers, that correspond to the velocities of a comoving
    frame (advection term).
    """

    def __init__(self, reference_equation):
        # call parent constructor
        super().__init__()
        # on which equation/unknowns should the constraint be imposed?
        self.ref_eq = reference_equation
        # the dimension of this equation is equal to the spatial dimension of the reference eq
        # TODO: fix for higher than 1 dimensions, i.e. make it possible to chose, which direction to fix.
        dim = 1
        # initialize unknowns (velocity vector) to zero
        self.u = np.zeros(dim)
        # the constraint equation couples to some other equation of the problem
        self.is_coupled = True

    def rhs(self, u):
        # TODO: fix for higher than 1 dimensions
        #  @simon: how about storing x, y, z, ... arrays as list in eq.x and then iterate lines 64-68 over list items
        #  don't know if that works for irregularly spaced grids
        # set up the vector of the residual contributions
        res = np.zeros(self.problem.dim)
        # define some variables
        eq = self.ref_eq
        eq_u = u[eq.idx]
        eq_u_old = eq.u
        velocity = u[self.idx]
        # add constraint to residuals of reference equation (velocity is the langrange multiplier)
        try: # if method first_spatial_derivative is implemented, use this
            eq_dudx = eq.first_spatial_derivative(eq_u)
        except AttributeError:  # if not, get it from the gradient
            eq_dudx = np.gradient(eq_u, eq.x[0])
        res[eq.idx] = velocity * eq_dudx
        # calculate the difference in center of masses between current
        # and previous unknowns of the reference equation
        #res[self.idx] = np.dot(eq.x, eq_u-eq_u_old)
        res[self.idx] = np.dot(eq_dudx, eq_u - eq_u_old)
        return res

    def mass_matrix(self):
        # couples to no time-derivatives
        return np.zeros(self.problem.dim)

    def plot(self, ax):
        # nothing to plot
        pass
