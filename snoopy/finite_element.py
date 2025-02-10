import numpy as np
import gmsh

class FiniteElement():
    '''This class provides all functionality for finite elements.
    '''

    def __init__(self, gmsh_code):
        '''Default constructor. It initialized gmsh if this was not done
        already.
        '''
        if not gmsh.isInitialized():
            gmsh.initialize()

        # this is the gmsh code for this type of element
        self.gmsh_code = gmsh_code

        # get the number of nodes
        self.num_nodes = self.get_number_of_nodes()

    def get_number_of_nodes(self):
        '''Get the number of nodes of this element.
        
        :return:
            The number of nodes as integer.
        '''
        # evaluate the basis functions
        _, basis, _ = gmsh.model.mesh.getBasisFunctions(self.gmsh_code, np.array([0., 0., 0.]), "Lagrange")

        # the number of nodes
        return len(basis)

    def evaluate_basis(self, q):
        '''Evaluate the basis functions.
        
        :param q:
            The local coordinates in an array with M*3 elements, where
            M is the number of points. This is the default gmsh style, where
            all quadrature points are always given in u, v, w, i.e. 3D local
            coordinates.

        :return:
            The six basis functions evaluated at the M points.
            i.e. an (M x K) array where K is the number of nodes.
        '''
        
        # this is the number of evaluation points
        M = np.int32(len(q)/3)

        # evaluate the basis functions
        _, basis, _ = gmsh.model.mesh.getBasisFunctions(self.gmsh_code, q, "Lagrange")

        # reshape
        basis.shape = (M, self.num_nodes)

        # return
        return basis

    def evaluate_basis_derivative(self, q):
        '''Evaluate the derivatives of the basis functions.
        
        :param q:
            The local coordinates in an array with M*3 elements, where
            M is the number of points. This is the default gmsh style, where
            all quadrature points are always given in u, v, w, i.e. 3D local
            coordinates.

        :return:
            The eight basis function derivatives evaluated at the M points.
            i.e. an (M x K x 3) array. Where K is the number of nodes.
        '''

        # this is the number of evaluation points
        M = np.int32(len(q)/3)

        # evaluate the basis functions
        _, basis, _ = gmsh.model.mesh.getBasisFunctions(self.gmsh_code, q, "GradLagrange")

        # reshape
        basis.shape = (M, self.num_nodes, 3)

        # return
        return basis

    def evaluate(self, c, nodes, basis):
        '''Evaluate the positions given the evaluated basis
        functions at some local coordinates.
        
        :param c:
            The node connectivity of this element.
        
        :param nodes:
            The nodes.

        :param basis:
            The basis functions evaluated at the local coordinates.

        :return:
            The positions at the M evaluation points in an M x 3 array.
        '''

        # the number of evaluation points
        num_eval = basis.shape[0]

        # return values
        ret_val = np.zeros((num_eval, 3))

        # add basis functions
        for i in range(self.num_nodes):
                        
            ret_val[:, 0] += basis[:, i]*nodes[c[i], 0]
            ret_val[:, 1] += basis[:, i]*nodes[c[i], 1]
            ret_val[:, 2] += basis[:, i]*nodes[c[i], 2]

        return ret_val

    def evaluate_derivative(self, c, nodes, basis_der):
        '''Evaluate the spatial derivative given the evaluated basis
        functions at some local coordinates.
        
        :param c:
            The node connectivity of this element.
        
        :param nodes:
            The nodes.

        :param basis:
            The basis functions evaluated at the local coordinates.

        :return:
            The positions at the M evaluation points in an M x 3 array.
        '''

        # the number of evaluation points
        num_eval = basis_der.shape[0]

        # return values
        ret_val = np.zeros((num_eval, 3, 2))

        # add basis functions
        for i in range(self.num_nodes):

            ret_val[:, 0, 0] += basis_der[:, i, 0]*nodes[c[i], 0]
            ret_val[:, 1, 0] += basis_der[:, i, 0]*nodes[c[i], 1]
            ret_val[:, 2, 0] += basis_der[:, i, 0]*nodes[c[i], 2]

            ret_val[:, 0, 1] += basis_der[:, i, 1]*nodes[c[i], 0]
            ret_val[:, 1, 1] += basis_der[:, i, 1]*nodes[c[i], 1]
            ret_val[:, 2, 1] += basis_der[:, i, 1]*nodes[c[i], 2]

        return ret_val

    def get_quadrature_rule(self, order):
        '''Get the quadrature nodes and weigths for
        this element.
        
        :param order:
            The order of the quadrature rule.
            
        :retrurn:
            The array of M weights and the local coordinates of the points in an 3*M array.
        '''

        q, weights = gmsh.model.mesh.getIntegrationPoints(self.gmsh_code, "Gauss" + str(order))

        return weights, q
        
    def get_edge_basis_functions(self, mesh, loc, element_type, element_order):
        '''Get the edge elements at the integration points.

        :param mesh:
            A gmsh mesh object.
        
        :param loc:
            The integration local coordinates

        :param element_type:
            The type of the finite element (gmsh).

        :param element_order:
            The element order.

        :return:
            A numpy array with all curls of basis functions. Also the number of orientations.
        '''

        # this is the function type string
        function_type = 'HcurlLegendre' + str(element_order-1)

        # evaluate the edge basis functions at the integration points on the reference element
        _, basis_functions, num_orinentations = mesh.getBasisFunctions(element_type, loc.flatten(), function_type)

        return basis_functions, num_orinentations

    def get_curl_edge_basis_functions(self, mesh, loc, element_type, element_order):
        '''Get the curl of the edge elements at the integration points.

        :param mesh:
            A gmsh mesh object.
        
        :param loc:
            The integration local coordinates

        :param element_type:
            The type of the finite element (gmsh).

        :param element_order:
            The element order.

        :return:
            A numpy array with all curls of basis functions. Also the number of orientations.
        '''

        # this is the function type string
        function_type = 'CurlHcurlLegendre' + str(element_order-1)

        # evaluate the edge basis functions at the integration points on the reference element
        _, basis_functions, num_orinentations = mesh.getBasisFunctions(element_type, loc.flatten(), function_type)

        return basis_functions, num_orinentations

    def compute_J(self, element, nodes, d_phi, dim=3):
        '''Compute the Jacobian matrix,
        given the connectivity of a finite element and the nodal coordinates,
        as well as the derivatives of the element basis functions.

        :param element:
            The connectivity of the element.

        :param nodes:
            The (all) nodal coordinates.

        :param d_phi:
            The derivatives of the shape functions of the finite element.

        :param dim:
            The dimension of the finite element. Default = 3.

        :return:
            The Jacobian in an (M x 3 x dim) numpy array.
        '''

        # this is the number of nodes for the finite element
        num_nodes = len(element)

        # this is the number of evaluation points
        num_eval = d_phi.shape[0]

        # this is the return data
        J = np.zeros((num_eval, 3, dim))

        # fill it
        for i in range(num_nodes):
            for j in range(3):
                for k in range(dim):
                    J[:, j, k] += nodes[element[i], j]*d_phi[:, i, k]

        return J

    def compute_J_inv(self, J):
        '''Invert a numpy array of Jacobian matrices.

        :param J:
            A numpy array of Jacobian matrices (M x 3 x 3).

        :return:
            The inverse of these matrices.
        '''

        # the number of evaluation points
        num_eval = J.shape[0]

        # the return data
        inv_J = 0.0*J

        for i in range(num_eval):
            inv_J[i, :, :] = np.linalg.inv(J[i, :, :])

        return inv_J

    def compute_J_det(self, J, dim=3):
        '''Compute the determinants of an array of Jacobian matrices. 

        :param J:
            A numpy array of Jacobian matrices (M x 3 x 3).

        :param dim:
            The dimension of the finite element. Default = 3.

        :return:
            The inverse of these matrices.
        '''

        # the number of evaluation points
        num_eval = J.shape[0]

        # the return data
        det_J = np.zeros((num_eval, ))

        if dim == 3:

            for i in range(num_eval):
                det_J[i] = np.linalg.det(J[i, :, :])

        elif dim == 2:
            # we compute the surface element
            for i in range(num_eval):
                det_J[i] = np.linalg.norm(np.cross(J[i, :, 0], J[i, :, 1]))

        else:
            print('Error! Unknown dimension {}'.format(dim))

        return det_J