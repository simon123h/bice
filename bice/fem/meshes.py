import numpy as np
from bice.core.profiling import profile
from .elements import Node, Element1d, TriangleElement2d


class Mesh:
    """
    Base class for meshes
    """

    def __init__(self):
        # spatial dimension of the mesh
        self.dim = None
        # storage for the nodes
        self.nodes = []
        # storage for the elements
        self.elements = []
        # error thresholds for refinement
        self.min_refinement_error = 1e-5
        self.max_refinement_error = 1e-3
        # minimal edge length of an element, for mesh adaption
        self.min_element_dx = 1e-9
        # maximal edge length of an element, for mesh adaption
        self.max_element_dx = 1e9

    # adapt mesh to the values given by the unknowns u
    def adapt(self, error_estimate):
        pass


class OneDimMesh(Mesh):
    """
    One-dimensional mesh from elements with linear shape functions
    """

    def __init__(self, N, L, L0=0):
        # call parent constructor
        super().__init__()
        # spatial dimension
        self.dim = 1
        # generate x
        x = np.linspace(L0, L0+L, N, endpoint=True)
        # add equidistant nodes
        for i in range(N):
            node = Node(x=np.array([x[i]]))
            self.nodes.append(node)
            # mark boundary nodes
            if i in [0, N-1]:
                node.is_boundary_node = True
        # generate the elements
        for i in range(N-1):
            nodes = [self.nodes[i], self.nodes[i+1]]
            self.elements.append(Element1d(nodes))

    @profile
    def adapt(self, error_estimate):
        # check the errors for each node and store whether they should be (un)refined
        for node, error in zip(self.nodes, error_estimate):
            node.can_be_unrefined = abs(error) < self.min_refinement_error
            node.should_be_refined = abs(error) > self.max_refinement_error

        # unrefinement loop
        i = 0
        while i < len(self.elements)-1:
            # store reference to nodes
            node_l = self.elements[i].nodes[0]
            node_m = self.elements[i].nodes[1]
            node_r = self.elements[i+1].nodes[1]
            # unrefine if all three nodes call for unrefinement
            if node_l.can_be_unrefined and node_m.can_be_unrefined and node_r.can_be_unrefined:
                # check if element has maximum size already
                if node_r.x[0] - node_l.x[0] >= 0.5 * self.max_element_dx:
                    break
                # delete the old elements
                self.elements.pop(i).purge()
                self.elements.pop(i).purge()
                # delete middle node
                self.nodes.remove(node_m)
                # create new element
                self.elements.insert(i, Element1d([node_l, node_r]))
                # this element should not be unrefined any further (for now)
                node_l.should_be_refined = False
            i += 1

        # refinement loop
        i = 0
        while i < len(self.elements):
            # store reference to the nodes
            node_l = self.elements[i].nodes[0]
            node_r = self.elements[i].nodes[1]
            # refine if any of the nodes was marked for refinement
            if node_l.should_be_refined or node_r.should_be_refined:
                # check if element has minimal size already
                if node_r.x[0] - node_l.x[0] <= 2 * self.min_element_dx:
                    break
                # generate new node in the middle and insert after node_l
                node_m = Node((node_l.x+node_r.x)/2)
                n = self.nodes.index(node_l)
                self.nodes.insert(n+1, node_m)
                # interpolate the unknowns
                node_m.u = (node_l.u + node_r.u)/2
                # delete old element
                self.elements.pop(i).purge()
                # generate two new elements and insert at the position of the old element
                self.elements.insert(i, Element1d([node_l, node_m]))
                self.elements.insert(i+1, Element1d([node_m, node_r]))
                # skip refinement of the newly created element
                i += 1
            i += 1


class TriangleMesh(Mesh):
    """
    Two-dimensional rectangular mesh with
    triangular elements and linear shape functions
    """

    def __init__(self, Nx, Ny, Lx, Ly, Lx0=0, Ly0=0):
        # call parent constructor
        super().__init__()
        # spatial dimension
        self.dim = 2
        # generate x,y-space
        x = np.linspace(Lx0, Lx0+Lx, Nx, endpoint=True)
        y = np.linspace(Ly0, Ly0+Ly, Ny, endpoint=True)
        # add equidistant nodes
        for i in range(Nx):
            for j in range(Ny):
                node = Node(x=np.array([x[i], y[j]]))
                self.nodes.append(node)
                # mark boundary nodes
                if i in [0, Nx-1] or j in [0, Ny-1]:
                    node.is_boundary_node = True

        # generate the elements
        for i in range(Nx-1):
            for j in range(Ny-1):
                # just like setting up a rectangular mesh, but we
                # divide each rectangle into two triangular elements
                # using counter-clockwise order of the nodes
                nodes = [
                    self.nodes[i*Nx+j],
                    self.nodes[(i+1)*Nx+j],
                    self.nodes[i*Nx+(j+1)],
                    self.nodes[(i+1)*Nx+(j+1)],
                    self.nodes[i*Nx+j]
                ]
                offs = (i+j) % 2
                nodes_a = [nodes[i] for i in (0, 1, 2+offs)]
                nodes_b = [nodes[i] for i in (1-offs, 3, 2)]
                self.elements.append(TriangleElement2d(nodes_a))
                self.elements.append(TriangleElement2d(nodes_b))

    @profile
    def adapt(self, error_estimate):
        # check the errors for each node and store whether they should be (un)refined
        for node, error in zip(self.nodes, error_estimate):
            node.can_be_unrefined = abs(error) < self.min_refinement_error
            node.should_be_refined = abs(error) > self.max_refinement_error

        # unrefinement loop, using node fusion
        i = 0
        while i < len(self.elements):
            # store reference to the current element
            elem1 = self.elements[i]
            # store reference to the nodes
            node_a = elem1.nodes[0]
            node_b = elem1.nodes[1]
            node_c = elem1.nodes[2]
            # unrefine if all of the nodes were marked for unrefinement
            if node_a.can_be_unrefined and node_b.can_be_unrefined and node_c.can_be_unrefined:
                # find the nodes with smallest distance
                min_index = elem1.edge_lengths.index(min(elem1.edge_lengths))
                node_b = elem1.nodes[min_index]
                node_c = elem1.nodes[(min_index+1) % 3]
                # check if they're both boundary nodes
                if node_b.is_boundary_node or node_c.is_boundary_node:
                    # TODO: we could deal with this case, though...
                    i += 1
                    continue
                # find the element that shares the edge: node_b--node_c
                elem2 = [
                    e for e in node_b.elements if node_c in e.nodes and e is not elem1][0]
                # generate a new node in the middle
                x_m = (node_b.x + node_c.x) / 2
                node_m = Node(x_m)
                node_m.can_be_unrefined = False
                node_m.should_be_refined = False
                # interpolate the unknowns
                node_m.u = (node_b.u + node_c.u) / 2
                # we will now join the nodes node_b and node_c
                # all the elements with node_b and node_c will be recreated using node_m
                # in order to abort in case of bad mesh quality, we'll save the replacements
                # to a list first and will (maybe) apply them later
                replacement_elements = []
                aborted = False  # flag whether the unrefinement is aborted
                # for each element that surrounds the principal nodes
                for node in [node_b, node_c]:
                    for element in node.elements:
                        if element in [elem1, elem2]:
                            continue
                        # calculate orientation of the old element
                        orientation_old = element.orientation()
                        # create the new nodes and element
                        new_nodes = [
                            n if n != node else node_m for n in element.nodes]
                        new_element = TriangleElement2d(new_nodes)
                        # store the replacement in the replacements-list
                        replacement_elements.append((element, new_element))
                        # calculate orientation of the old element
                        orientation_new = new_element.orientation()
                        # check whether the orientation of the triangle flipped
                        if orientation_old * orientation_new < 0:
                            # if yes, abort the current unrefinement
                            aborted = True
                        # check whether the angles would become very small
                        if min(new_element.angles()) < 0.2:
                            aborted = True
                    # break if unrefinement is aborted
                    if aborted:
                        break
                # apply the unrefinement, unless aborted
                if not aborted:
                    # destroy the collapsed elements
                    self.elements.pop(i).purge()  # elem1
                    i2 = self.elements.index(elem2)
                    self.elements.pop(i2).purge()  # elem2
                    # overwrite the surrounding elements
                    for old_e, new_e in replacement_elements:
                        index = self.elements.index(old_e)
                        old_e.purge()
                        self.elements[index] = new_e
                    # insert the new middle node
                    n = self.nodes.index(node_b)
                    self.nodes.insert(n, node_m)
                    # delete the nodes node_b and node_c
                    self.nodes.remove(node_b)
                    self.nodes.remove(node_c)
                else:
                    # purge the new elements
                    for old_e, new_e in replacement_elements:
                        new_e.purge()
            i += 1

        # refinement loop
        i = 0
        while i < len(self.elements):
            # store reference to the nodes
            node_a = self.elements[i].nodes[0]
            node_b = self.elements[i].nodes[1]
            node_c = self.elements[i].nodes[2]
            # refine if all of the nodes were marked for refinement
            if node_a.should_be_refined and node_b.should_be_refined and node_c.should_be_refined:
                # if both node_a and node_b were at a border, node_m is also at a border
                if node_a.is_boundary_node and node_b.is_boundary_node:
                    # if we don't unrefine boundary nodes, we should not refine them either
                    i += 1
                    continue
                # check if element has minimal size already
                if np.linalg.norm(node_a.x - node_b.x) <= 2 * self.min_element_dx:
                    i += 1
                    continue
                # generate new node in the middle of the first two nodes (longest edge)
                x_m = (node_a.x + node_b.x) / 2
                node_m = Node(x_m)
                node_m.should_be_refined = False
                # and insert after node_a
                n = self.nodes.index(node_a)
                self.nodes.insert(n+1, node_m)
                # interpolate the unknowns
                node_m.u = (node_a.u + node_b.u) / 2
                # delete old element
                self.elements.pop(i).purge()
                # generate two new elements and insert at the position of the old element
                self.elements.insert(
                    i, TriangleElement2d([node_c, node_a, node_m]))
                self.elements.insert(
                    i+1, TriangleElement2d([node_b, node_c, node_m]))
                # else, find the element, that shares the edge: node_a--node_b
                neighbor_element = [
                    e for e in node_a.elements if e in node_b.elements]
                # if no neighboring element was found, we must be at a border
                if neighbor_element:
                    neighbor_element = neighbor_element[0]
                    neighbor_node = [
                        n for n in neighbor_element.nodes if n not in [node_a, node_b]][0]
                    index = self.elements.index(neighbor_element)
                    # delete old element
                    self.elements.pop(index).purge()
                    # generate two new elements and insert at the position of the old element
                    self.elements.insert(
                        index, TriangleElement2d([neighbor_node, node_m, node_a]))
                    self.elements.insert(
                        index+1, TriangleElement2d([node_b, node_m, neighbor_node]))
                # skip refinement of the newly created elements
                i += 2
            i += 1
