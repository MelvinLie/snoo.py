import numpy as np
import gmsh
from tqdm import tqdm
from scipy.sparse import csr_array
import matplotlib.pyplot as plt
import pandas as pd

from . import fem_c_mod as fem_c
from .finite_element import FiniteElement
from .mesh_tools import get_vector_basis_mesh_info
from .mesh_tools import get_vector_basis_orientations
from .mesh_tools import get_mesh_info
from .mesh_tools import get_global_ids
from .mesh_tools import get_num_edge_dofs

class GradGradAssembler():

    def __init__(self, mesh, domain_tags):
        '''Default constructor.
        
        :param mesh:
            A gmsh mesh object.

        :param domain_tags:
            The tags for the domains.

        :return:
            None.
        '''

        # The nodes are not sorted correctly. I don't know why...
        # But we need to get them like this:
        node_tags, _, _ = mesh.getNodes()
        num_nodes = len(node_tags)
        node_tags = np.unique(node_tags)

        # we now make an array of unique mesh nodes.
        self.nodes = np.zeros((num_nodes, 3))

        for i in range(num_nodes):
            self.nodes[i, :] = mesh.getNode(node_tags[i])[0]
        
        # we allocate lists for the cell connectivities and types
        # for all materials
        self.num_materials = len(domain_tags)
        self.cell_types = []
        self.cells = []
        
        for i in domain_tags:
            
            # get the elements
            c_types_tmp, _, cells_tmp = gmsh.model.mesh.getElements(3, i)

            # append to the list
            self.cell_types.append(c_types_tmp)
            self.cells.append(cells_tmp)
            
        # the number of nodes
        num_nodes = self.nodes.shape[0]


        # cell_mat = self.cells[0] - 1
        # cell_mat = np.array(cell_mat, dtype=np.int32)
        # cell_mat.shape = (np.int32(len(self.cells[0])/4), 4)

        # edges = [[0, 1],
        #          [1, 2],
        #          [2, 0],
        #          [0, 3],
        #          [1, 3],
        #          [3, 3]]

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for i, c in enumerate(cell_mat):

        #     for e in edges:
        #         ax.plot([self.nodes[c[e[0]], 0], self.nodes[c[e[1]], 0]],
        #                 [self.nodes[c[e[0]], 1], self.nodes[c[e[1]], 1]],
        #                 [self.nodes[c[e[0]], 2], self.nodes[c[e[1]], 2]], '--o' ,color='k')
                
        # plt.show()

        return None
    

    def setup_mesh(self, nodes, cells, cell_types):
        """Setup the mesh from external information.

        :param nodes:
            The nodal coordinates.

        :param cells:
            The mesh connectivity list.

        :param cell_types.
            The mesh cell types list.

        :return:
            None.
        """

        self.nodes = nodes
        self.cells = cells
        self.cell_types = cell_types

        return None
    
    def get_num_dofs(self):
        '''Get the total number of degrees of freedom.
        
        :return:
            An integer specifying the number if DoFs.
        '''
        return self.nodes.shape[0]
    
    def get_quadrature_rule(self, gmsh_code, order):
        '''Get the quadrature points and nodes for a
        finite element of certain type.

        :param gmsh_code:
            The cell type gmsh code.

        :param order:
            The quadrature order.

        :return:
            The points q and the weights w.
        '''
        q, w = gmsh.model.mesh.getIntegrationPoints(gmsh_code, "Gauss" + str(order))
        
        return q, w
    
    def compute_grad_phi(self, c, x, d_phi, inv_J):
        '''Compute the magnetic field in a finite element
        with connectivity c. The vector x stores the nodal 
        values of the solution.
        The array grad_phi stores the gradients of the FEM
        basis functions.

        :param c:
            The node connectivity of the finite element.

        :param x:
            The solution vector.

        :param d_phi:
            The derivatives of all basis functions.

        :param inv_J:
            The inverse of the Jacobian matrix of the transformation.

        :return:
            The field vectors H, as well as the magnitude H_mag.
        '''

        # the number of evaluation points
        num_eval = d_phi.shape[0]

        # allocate the return array
        grad_phi = np.zeros((num_eval, 3))

        # evaluate
        grad_phi[:, 0] = d_phi[:, :, 0] @ x[c]
        grad_phi[:, 1] = d_phi[:, :, 1] @ x[c]
        grad_phi[:, 2] = d_phi[:, :, 2] @ x[c]


        # transform it to the global domain
        for m in range(num_eval):
            grad_phi[m, :] = inv_J[m, :, :].T @ grad_phi[m, :]


        # return also the magnitude
        mag = np.linalg.norm(grad_phi, axis=1)

        return grad_phi, mag
        
    def compute_stiffness_matrix(self, x, mu_list, quad_order=8):
        '''Compute the stiffness matrix for the magnetic scalar potential for
        nonlinear problems.
        This implementation is just to test the equations. We do not focus
        on performance at this point. A fast version of this code will be
        implemented in an optimized C++ code.

        :param x:
            The solution vector.
            
        :param mu_list:
            The nonlinear permeability functions for each domain in a list.

        :param quad_order:
            The quadrature order. Default 8.

        :return:
            The stiffness matrix.
        '''

        print('assembling the stiffness matrix (scalar potential) (py)...')

        # the total number of nodes
        num_nodes = self.nodes.shape[0]

        # number of cell types
        #num_cell_types = len(self.cell_types)

        # make the sparse stiffness matrix
        K = csr_array((num_nodes, num_nodes))
    
        # loop over all materials
        for n in range(self.num_materials):

            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get also the permeability in this domain
            mu_fcn = mu_list[n]
            
            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the quadrature nodes and weights
                q, w = self.get_quadrature_rule(ct, quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the shape functions at these points (not needed)
                # phi = finite_element.evaluate_basis(q)

                # evaluate the derivatives at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # the number of triplets
                num_triplets = num_el*el_nodes*el_nodes

                # allocate the space for the triplet list
                ij = np.zeros((num_triplets, 2), dtype=np.int32)
                vals = np.zeros((num_triplets, ))

                # a triplet counter
                t_cnt = 0

                # loop over all elements
                for e_i in tqdm(range(num_el)):

                    # gmsh starts counting at 1
                    e = cells[i][e_i*el_nodes:(e_i+1)*el_nodes] - 1

                    # compute the Jacobian at the integration points
                    # inv_J is a numpy array of dimension (M x 3 x 3) where
                    # the inverse of the Jacobians are stored in the second and
                    # third dimensions
                    # M is the number of integration points
                    # det_J is an (M x 0) numpy array which stores the M values of det(J)
                    J = finite_element.compute_J(e, self.nodes, d_phi)
                    
                    inv_J = finite_element.compute_J_inv(J)
                    det_J = finite_element.compute_J_det(J)


                    _, H_mag = self.compute_grad_phi(e, x, d_phi, inv_J)

                    # compute the permebility
                    mu = mu_fcn(H_mag)

                    # loop over all combinations of basis functions
                    for j in range(el_nodes):
                        for k in range(j, el_nodes):

                            # this is the integration value
                            int_val = 0.0

                            # gaussian integration
                            for m in range(num_quad_pts):

                                # transform the gradients
                                grad_u = inv_J[m, :, :].T @ d_phi[m, j, :]
                                grad_v = inv_J[m, :, :].T @ d_phi[m, k, :]

                                # increment the integration value
                                int_val += mu[m]*det_J[m]*w[m]*np.sum(grad_u*grad_v)

                            # sparse matrix version

                            # fill the triplet list
                            ij[t_cnt, 0] = e[j]
                            ij[t_cnt, 1] = e[k]
                            vals[t_cnt] = int_val
                            t_cnt += 1

                            # apply symmetry
                            if j != k:
                                ij[t_cnt, 0] = e[k]
                                ij[t_cnt, 1] = e[j]
                                vals[t_cnt] = int_val
                                t_cnt += 1

                            # dense matrix version

                            # fill the matrix
                            # K_dense[e[j], e[k]] += int_val

                            # apply symmetry
                            # if j != k:
                            #     K_dense[e[k], e[j]] += int_val

                # print('number of triplets = {}'.format(len(vals)))

                # make the sparse matrix
                K += csr_array((vals, (ij[:, 0], ij[:, 1])), shape=(num_nodes, num_nodes))

        return K
    

    def compute_stiffness_matrix_red(self, H, mu_list, quad_order=8):
        '''Compute the stiffness matrix for the magnetic scalar potential for
        nonlinear problems.
        This implementation is just to test the equations. We do not focus
        on performance at this point. A fast version of this code will be
        implemented in an optimized C++ code.

        :param H:
            The impressed field.
            
        :param mu_list:
            The nonlinear permeability functions for each domain in a list.

        :param quad_order:
            The quadrature order. Default 8.

        :return:
            The stiffness matrix.
        '''

        print('assembling the stiffness matrix (scalar potential) (py)...')

        # the total number of nodes
        num_nodes = self.nodes.shape[0]

        # number of cell types
        #num_cell_types = len(self.cell_types)

        # make the sparse stiffness matrix
        K = csr_array((num_nodes, num_nodes))
    
        # a global element counter (all materials)
        glob_el_cnt = 0

        # loop over all materials
        for n in range(self.num_materials):

            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get also the permeability in this domain
            mu_fcn = mu_list[n]
            
            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the quadrature nodes and weights
                q, w = self.get_quadrature_rule(ct, quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the shape functions at these points (not needed)
                # phi = finite_element.evaluate_basis(q)

                # evaluate the derivatives at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # the number of triplets
                num_triplets = num_el*el_nodes*el_nodes

                # allocate the space for the triplet list
                ij = np.zeros((num_triplets, 2), dtype=np.int32)
                vals = np.zeros((num_triplets, ))

                # a triplet counter
                t_cnt = 0

                # loop over all elements
                for e_i in tqdm(range(num_el)):

                    # gmsh starts counting at 1
                    e = cells[i][e_i*el_nodes:(e_i+1)*el_nodes] - 1

                    # compute the Jacobian at the integration points
                    # inv_J is a numpy array of dimension (M x 3 x 3) where
                    # the inverse of the Jacobians are stored in the second and
                    # third dimensions
                    # M is the number of integration points
                    # det_J is an (M x 0) numpy array which stores the M values of det(J)
                    J = finite_element.compute_J(e, self.nodes, d_phi)
                    
                    inv_J = finite_element.compute_J_inv(J)
                    det_J = finite_element.compute_J_det(J)

                    # get the field vectors
                    this_H = H[glob_el_cnt*num_quad_pts:(glob_el_cnt+1)*num_quad_pts, :]

                    # compute also the field vector magnitude !!!!!!!!
                    H_mag = np.linalg.norm(this_H, axis=1)

                    # compute the permebility
                    mu = mu_fcn(H_mag)

                    # loop over all combinations of basis functions
                    for j in range(el_nodes):
                        for k in range(j, el_nodes):

                            # this is the integration value
                            int_val = 0.0

                            # gaussian integration
                            for m in range(num_quad_pts):

                                # transform the gradients
                                grad_u = inv_J[m, :, :].T @ d_phi[m, j, :]
                                grad_v = inv_J[m, :, :].T @ d_phi[m, k, :]

                                # increment the integration value
                                int_val += mu[m]*det_J[m]*w[m]*np.sum(grad_u*grad_v)

                            # sparse matrix version

                            # fill the triplet list
                            ij[t_cnt, 0] = e[j]
                            ij[t_cnt, 1] = e[k]
                            vals[t_cnt] = int_val
                            t_cnt += 1

                            # apply symmetry
                            if j != k:
                                ij[t_cnt, 0] = e[k]
                                ij[t_cnt, 1] = e[j]
                                vals[t_cnt] = int_val
                                t_cnt += 1

                            # dense matrix version

                            # fill the matrix
                            # K_dense[e[j], e[k]] += int_val

                            # apply symmetry
                            # if j != k:
                            #     K_dense[e[k], e[j]] += int_val

                # print('number of triplets = {}'.format(len(vals)))

                # make the sparse matrix
                K += csr_array((vals, (ij[:, 0], ij[:, 1])), shape=(num_nodes, num_nodes))

        return K
    
    def compute_stiffness_and_jacobi_matrix(self, x, mu_list, quad_order=8):
        '''Compute the stiffness and the jacobi matrix for the magnetic scalar potential for
        nonlinear problems.
        This implementation is just to test the equations. We do not focus
        on performance at this point. A fast version of this code will be
        implemented in an optimized C++ code.

        :param x:
            The solution vector.
            
        :param mu_list:
            A list of permeability classes for each material.

        :param quad_order:
            The quadrature order. Default 8.

        :return:
            The stiffness matrix.
        '''

        print('assembling the stiffness and jacobi matrix (scalar potential) (py)...')

        # the total number of nodes
        num_nodes = self.nodes.shape[0]

        # number of cell types
        #num_cell_types = len(self.cell_types)

        # make the sparse stiffness matrix
        K = csr_array((num_nodes, num_nodes))
    
        # make the sparse Jacobi matrix
        dK = csr_array((num_nodes, num_nodes))

        # loop over all materials
        for n in range(self.num_materials):

            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get also the permeability in this domain
            permeability = mu_list[n]

            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the quadrature nodes and weights
                q, w = self.get_quadrature_rule(ct, quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the shape functions at these points (not needed)
                # phi = finite_element.evaluate_basis(q)

                # evaluate the derivatives at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # the number of triplets
                num_triplets = num_el*el_nodes*el_nodes

                # allocate the space for the triplet list (K)
                ij = np.zeros((num_triplets, 2), dtype=np.int32)
                vals_K = np.zeros((num_triplets, ))

                # allocate the space for the triplet list (J)
                vals_dK = np.zeros((num_triplets, ))

                # a triplet counter
                t_cnt = 0

                # loop over all elements
                for e_i in tqdm(range(num_el)):

                    # gmsh starts counting at 1
                    e = cells[i][e_i*el_nodes:(e_i+1)*el_nodes] - 1

                    # compute the Jacobian at the integration points
                    # inv_J is a numpy array of dimension (M x 3 x 3) where
                    # the inverse of the Jacobians are stored in the second and
                    # third dimensions
                    # M is the number of integration points
                    # det_J is an (M x 0) numpy array which stores the M values of det(J)
                    J = finite_element.compute_J(e, self.nodes, d_phi)
                    
                    inv_J = finite_element.compute_J_inv(J)
                    det_J = finite_element.compute_J_det(J)

                    # compute also the field vector
                    grad_phi, H_mag = self.compute_grad_phi(e, x, d_phi, inv_J)

                    # compute the permebility
                    mu = permeability.evaluate_mu(H_mag)


                    # compute the permebility derivative
                    d_mu = permeability.evaluate_mu_derivative(H_mag)

                    # loop over all combinations of basis functions
                    for j in range(el_nodes):
                        for k in range(j, el_nodes):

                            # this is the integration value (K)
                            int_val_K = 0.0

                            # this is the integration value (dK)
                            int_val_dK = 0.0       


                            # gaussian integration
                            for m in range(num_quad_pts):

                                # transform the gradients
                                grad_u = inv_J[m, :, :].T @ d_phi[m, j, :]
                                grad_v = inv_J[m, :, :].T @ d_phi[m, k, :]

                                # increment the integration value
                                int_val_K += mu[m]*det_J[m]*w[m]*np.sum(grad_u*grad_v)

                                # increment the integration value
                                if H_mag[m] >= 1e-14:
                                    
                                    int_val_dK += d_mu[m]*det_J[m]*w[m]*np.sum(grad_u*grad_phi[m, :])*np.sum(grad_v*grad_phi[m, :])/H_mag[m]
                                    
                            # sparse matrix version

                            # fill the triplet list
                            ij[t_cnt, 0] = e[j]
                            ij[t_cnt, 1] = e[k]
                            vals_K[t_cnt] = int_val_K
                            vals_dK[t_cnt] = int_val_dK
                            t_cnt += 1

                            # apply symmetry
                            if j != k:
                                ij[t_cnt, 0] = e[k]
                                ij[t_cnt, 1] = e[j]
                                vals_K[t_cnt] = int_val_K
                                vals_dK[t_cnt] = int_val_dK
                                t_cnt += 1

                            # dense matrix version

                            # fill the matrix
                            # K_dense[e[j], e[k]] += int_val

                            # apply symmetry
                            # if j != k:
                            #     K_dense[e[k], e[j]] += int_val

                # print('number of triplets = {}'.format(len(vals)))

                # make the sparse matrix
                K += csr_array((vals_K, (ij[:, 0], ij[:, 1])), shape=(num_nodes, num_nodes))
                dK += csr_array((vals_dK, (ij[:, 0], ij[:, 1])), shape=(num_nodes, num_nodes))

            # add K and dK
            dK += K

        return K, dK

    def compute_rhs_and_jacobi_matrix(self, H, mu_list, quad_order=8):
        '''Compute the rhs and the jacobi matrix for the magnetic scalar potential for
        nonlinear problems. The H field vectors may include exterior field so that 
        this function can be used for the reduced vector potential formulation.
        
        This implementation is just to test the equations. We do not focus
        on performance at this point. A fast version of this code will be
        implemented in an optimized C++ code.

        :param x:
            The solution vector.
            
        :param mu_list:
            A list of permeability classes for each material.

        :param quad_order:
            The quadrature order. Default 8.

        :param H_ext:
            An external field for the reduced scalar potential formulation.
            If this array has zero rows (default), the external field is ignored.
            Otherwise, the number of rows must match the number of integration points.

        :return:
            The right hand side vector as well as the jacobian matrix.
        '''

        print('assembling the stiffness and jacobi matrix (scalar potential) (py)...')

        # the total number of nodes
        num_nodes = self.nodes.shape[0]

        # number of cell types
        #num_cell_types = len(self.cell_types)

        # make the rhs vector
        rhs = np.zeros((num_nodes, ))

        # make the sparse Jacobi matrix
        K = csr_array((num_nodes, num_nodes))
    
        # make the sparse Jacobi matrix
        dK = csr_array((num_nodes, num_nodes))

        # a global element counter (all materials)
        glob_el_cnt = 0

        # loop over all materials
        for n in range(self.num_materials):

            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get also the permeability in this domain
            permeability = mu_list[n]

            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the quadrature nodes and weights
                q, w = self.get_quadrature_rule(ct, quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the shape functions at these points (not needed)
                # phi = finite_element.evaluate_basis(q)

                # evaluate the derivatives at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # the number of triplets
                num_triplets = num_el*el_nodes*el_nodes

                # allocate the space for the triplet list (K)
                ij = np.zeros((num_triplets, 2), dtype=np.int32)
                vals_K = np.zeros((num_triplets, ))

                # allocate the space for the triplet list (J)
                vals_dK = np.zeros((num_triplets, ))

                # a triplet counter
                t_cnt = 0

                # loop over all elements
                for e_i in tqdm(range(num_el)):

                    # gmsh starts counting at 1
                    e = cells[i][e_i*el_nodes:(e_i+1)*el_nodes] - 1

                    # compute the Jacobian at the integration points
                    # inv_J is a numpy array of dimension (M x 3 x 3) where
                    # the inverse of the Jacobians are stored in the second and
                    # third dimensions
                    # M is the number of integration points
                    # det_J is an (M x 0) numpy array which stores the M values of det(J)
                    J = finite_element.compute_J(e, self.nodes, d_phi)
                    
                    inv_J = finite_element.compute_J_inv(J)
                    det_J = finite_element.compute_J_det(J)

                    # get the field vectors
                    this_H = H[glob_el_cnt*num_quad_pts:(glob_el_cnt+1)*num_quad_pts, :]

                    # compute also the field vector magnitude
                    H_mag = np.linalg.norm(this_H, axis=1)

                    # compute the permebility
                    mu = permeability.evaluate_mu(H_mag)

                    # compute the permebility derivative
                    d_mu = permeability.evaluate_mu_derivative(H_mag)

                    # loop over all combinations of basis functions
                    for j in range(el_nodes):
                        for k in range(j, el_nodes):

                            # this is the integration value (K)
                            int_val_rhs = 0.0

                            # this is the integration value (K)
                            int_val_K = 0.0

                            # this is the integration value (dK)
                            int_val_dK = 0.0       


                            # gaussian integration
                            for m in range(num_quad_pts):

                                # transform the gradients
                                grad_u = inv_J[m, :, :].T @ d_phi[m, j, :]
                                grad_v = inv_J[m, :, :].T @ d_phi[m, k, :]

                                # increment the integration value
                                int_val_rhs += mu[m]*det_J[m]*w[m]*np.sum(this_H[m, :]*grad_u)

                                # increment the integration value
                                int_val_K += mu[m]*det_J[m]*w[m]*np.sum(grad_u*grad_v)

                                # increment the integration value
                                if H_mag[m] >= 1e-14:
                                    
                                    int_val_dK += d_mu[m]*det_J[m]*w[m]*np.sum(grad_u*this_H[m, :])*np.sum(grad_v*this_H[m, :])/H_mag[m]
                                    
                            # sparse matrix version

                            # fill the triplet list
                            ij[t_cnt, 0] = e[j]
                            ij[t_cnt, 1] = e[k]
                            rhs[e[j]] += int_val_rhs
                            vals_K[t_cnt] = int_val_K
                            vals_dK[t_cnt] = int_val_dK
                            t_cnt += 1

                            # apply symmetry
                            if j != k:
                                ij[t_cnt, 0] = e[k]
                                ij[t_cnt, 1] = e[j]
                                rhs[e[k]] += int_val_rhs
                                vals_K[t_cnt] = int_val_K
                                vals_dK[t_cnt] = int_val_dK
                                t_cnt += 1

                    # increment the global element counter
                    glob_el_cnt += 1

                # print('number of triplets = {}'.format(len(vals)))

                # make the sparse matrix
                K += csr_array((vals_K, (ij[:, 0], ij[:, 1])), shape=(num_nodes, num_nodes))
                dK += csr_array((vals_dK, (ij[:, 0], ij[:, 1])), shape=(num_nodes, num_nodes))

            # add K and dK
            dK += K

        return rhs, dK    

    def compute_rhs(self, H, mu_list, quad_order=8):
        '''Compute the rhs and the jacobi matrix for the magnetic scalar potential for
        nonlinear problems. The H field vectors may include exterior field so that 
        this function can be used for the reduced vector potential formulation.
        
        This implementation is just to test the equations. We do not focus
        on performance at this point. A fast version of this code will be
        implemented in an optimized C++ code.

        :param x:
            The solution vector.
            
        :param mu_list:
            A list of permeability classes for each material.

        :param quad_order:
            The quadrature order. Default 8.

        :param H_ext:
            An external field for the reduced scalar potential formulation.
            If this array has zero rows (default), the external field is ignored.
            Otherwise, the number of rows must match the number of integration points.

        :return:
            The right hand side vector as well as the jacobian matrix.
        '''

        print('assembling the right hand side for the reduced scalar potential formulation (py)...')

        # the total number of nodes
        num_nodes = self.nodes.shape[0]

        # number of cell types
        #num_cell_types = len(self.cell_types)

        # make the rhs vector
        rhs = np.zeros((num_nodes, ))

        # make the sparse Jacobi matrix
        K = csr_array((num_nodes, num_nodes))
    
        # make the sparse Jacobi matrix
        dK = csr_array((num_nodes, num_nodes))

        # a global element counter (all materials)
        glob_el_cnt = 0

        # loop over all materials
        for n in range(self.num_materials):

            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get also the permeability in this domain
            permeability = mu_list[n]

            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the quadrature nodes and weights
                q, w = self.get_quadrature_rule(ct, quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the shape functions at these points (not needed)
                # phi = finite_element.evaluate_basis(q)

                # evaluate the derivatives at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # the number of triplets
                num_triplets = num_el*el_nodes*el_nodes

                # allocate the space for the triplet list (K)
                ij = np.zeros((num_triplets, 2), dtype=np.int32)
                vals_K = np.zeros((num_triplets, ))

                # allocate the space for the triplet list (J)
                vals_dK = np.zeros((num_triplets, ))

                # a triplet counter
                t_cnt = 0

                # loop over all elements
                for e_i in tqdm(range(num_el)):

                    # gmsh starts counting at 1
                    e = cells[i][e_i*el_nodes:(e_i+1)*el_nodes] - 1

                    # compute the Jacobian at the integration points
                    # inv_J is a numpy array of dimension (M x 3 x 3) where
                    # the inverse of the Jacobians are stored in the second and
                    # third dimensions
                    # M is the number of integration points
                    # det_J is an (M x 0) numpy array which stores the M values of det(J)
                    J = finite_element.compute_J(e, self.nodes, d_phi)
                    
                    inv_J = finite_element.compute_J_inv(J)
                    det_J = finite_element.compute_J_det(J)

                    # get the field vectors
                    this_H = H[glob_el_cnt*num_quad_pts:(glob_el_cnt+1)*num_quad_pts, :]

                    # compute also the field vector magnitude
                    H_mag = np.linalg.norm(this_H, axis=1)

                    # compute the permebility
                    mu = permeability.evaluate_mu(H_mag)

                    # loop over all combinations of basis functions
                    for j in range(el_nodes):

                        # this is the integration value (K)
                        int_val_rhs = 0.0

                        # gaussian integration
                        for m in range(num_quad_pts):

                            # transform the gradients
                            grad_u = inv_J[m, :, :].T @ d_phi[m, j, :]

                            # increment the integration value
                            int_val_rhs += mu[m]*det_J[m]*w[m]*np.sum(this_H[m, :]*grad_u)



                        rhs[e[j]] += int_val_rhs

                    # increment the global element counter
                    glob_el_cnt += 1

                # print('number of triplets = {}'.format(len(vals)))

        return rhs    

    def compute_stiffness_and_jacobi_matrix_c(self, x, mu_list, quad_order=8):
        '''Compute the stiffness and the jacobi matrix for the magnetic scalar potential for
        nonlinear problems.
        This implementation is the fast version of the above.
        
        :param x:
            The solution vector.
            
        :param mu_list:
            A list of permeability classes for each material.

        :param quad_order:
            The quadrature order. Default 8.

        :return:
            The stiffness matrix.
        '''

        # print('assembling the stiffness and jacobi matrix (scalar potential) (py)...')

        # the total number of nodes
        num_nodes = self.nodes.shape[0]

        # number of cell types
        #num_cell_types = len(self.cell_types)

        # make the sparse stiffness matrix
        K = csr_array((num_nodes, num_nodes))
    
        # make the sparse Jacobi matrix
        dK = csr_array((num_nodes, num_nodes))

        # loop over all materials
        for n in range(len(mu_list)):

            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get also the permeability in this domain
            permeability = mu_list[n]

            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr of element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the quadrature nodes and weights
                q, w = self.get_quadrature_rule(ct, quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the shape functions at these points (not needed)
                # phi = finite_element.evaluate_basis(q)

                # evaluate the derivatives at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # call the cpp code (gmsh starts counting nodes at 1)
                my_K, my_dK = fem_c.compute_stiffness_and_jacobi_matrix(self.nodes, cells[i] - 1, d_phi, x, permeability, q, w)

                # increment the sparse matrix
                K += my_K
                dK += my_dK
                
            # add K and dK
            dK += K

        return K, dK
    
    def get_total_number_of_elements(self):
        '''Get the total number of elements in this FEM mesh.
        
        :return:
            The number of elements as integer.
            
        '''

        # the counter
        el_cnt = 0

        for n in range(self.num_materials):
            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]
            # loop over cell types
            for i, ct in enumerate(cell_types):
                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # increment the element counter
                el_cnt += num_el

        return el_cnt

    def get_total_number_of_evaluations(self, quad_order):
        '''Get the total number of evaluations in the FEM mesh for
        a certain quadrature order.

        :param quad_order:
            The quadrature order.
        
        :return:
            The number of elements as integer.
            
        '''

        # the counter
        eval_cnt = 0

        for n in range(self.num_materials):
            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]
            # loop over cell types
            for i, ct in enumerate(cell_types):
                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the quadrature nodes and weights
                _, w = self.get_quadrature_rule(ct, quad_order)

                # increment the element counter
                eval_cnt += num_el*len(w)

        return eval_cnt

    def compute_field_evaluation_matrix(self, quad_order=3):
        '''Compute the matrix to evaluate the field in the fem mesh.

        :param quad_order:
            The quadrature order. Default 8.

        :return:
            The sparse evaluation matrix.
        '''

        # the total number of nodes
        num_nodes = self.nodes.shape[0]

        # the total number of fem evaluations
        num_eval = self.get_total_number_of_evaluations(quad_order)

        # make space for the matrix
        X = csr_array((3*num_eval, num_nodes))

        # make space for the return points and field vectors
        points = np.zeros((0, 3))

        # loop over all materials
        for n in range(self.num_materials):
            
            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the quadrature nodes and weights
                q, _ = self.matrix_factory.get_quadrature_rule(ct, quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the shape functions at these points
                phi = finite_element.evaluate_basis(q)

                # evaluate the derivatives at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # make space for the evaluation points of this finite element type
                my_points = np.zeros((num_quad_pts*num_el, 3))

                # make space for the magnetic field of this finite element type
                my_H = np.zeros((num_quad_pts*num_el, 3))

                # call the cpp code (gmsh starts counting nodes at 1)

                # TO BE DONE...
                # my_X = fem_c.compute_field_evaluation_matrix(self.nodes, cells[i] - 1, d_phi, q)

                # increment matrix
                # X += my_X

        return X