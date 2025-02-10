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

def recover_full_solution(u, u_g, b):
    '''Given the fem solution and the lifting vector,
    as well as the boundary node list, recover the 
    full FEM solution vector (scalar).

    :param u:
        The FEM solution vector (reduced).
    
    :param u_g:
        The lifting vector.

    :return:
        The full FEM solution.
    '''
    # the total number of nodes
    num_nodes = len(u_g)

    # the indices of the remaining nodes.
    indx_rem = np.linspace(0, num_nodes-1, num_nodes, dtype=np.int32)

    # delete the boundary nodes
    indx_rem = np.delete(indx_rem, b)

    # copy the lifting vector
    u_ret = u_g.copy()

    # fill it
    u_ret[indx_rem] = u

    # return
    return u_ret

def delete_rows_sparse(M, rows):
    '''Delete some rows of a sparse matrix.

    :param M:
        The sparse matrix.

    :param rows:
        The rows to delete.

    :return:
        The resulting sparse matrix.
    '''

    # this is the number of rows of the matrix
    num_rows = M.shape[0]

    mask = np.ones(num_rows, dtype=bool)
    mask[rows] = False

    return M[np.flatnonzero(mask), :]


def delete_cols_sparse(M, cols):
    '''Delete some columns of a sparse matrix.

    :param M:
        The sparse matrix.

    :param cols:
        The columns to delete.

    :return:
        The resulting sparse matrix.
    '''

    # this is the number of cols of the matrix
    num_cols = M.shape[1]

    mask = np.ones(num_cols, dtype=bool)
    mask[cols] = False

    return M[:, np.flatnonzero(mask)]


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

    def compute_field(self, x, quad_order=3):
        '''Compute the field in the fem mesh.

        :param x:
            The solution vector.
    
        :param quad_order:
            The quadrature order. Default 8.

        :return:
            The evaluation points and the field vectors.
        '''
        
        # make space for the return points and field vectors
        points = np.zeros((0, 3))
        H = np.zeros((0, 3))
        B = np.zeros((0, 3))

        # get the mesh information
        nodes = self.nodes
    
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
                q, _ = self.get_quadrature_rule(ct, quad_order)

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

                # loop over all elements
                for e_i in range(num_el):

                    # gmsh starts counting at 1
                    e = cells[i][e_i*el_nodes:(e_i+1)*el_nodes] - 1

                    # evaluate this finite element for the global position
                    my_points[e_i*num_quad_pts:(e_i+1)*num_quad_pts, :] = finite_element.evaluate(e, nodes, phi)

                    # compute the Jacobian at the integration points
                    J = finite_element.compute_J(e, nodes, d_phi)

                    # we also need the inverse to transform
                    inv_J = finite_element.compute_J_inv(J)

                    # compute also the field vector
                    grad_phi, _ = self.compute_grad_phi(e, x, d_phi, inv_J)

                    for m in range(num_quad_pts):
                        # transform the gradient
                        my_H[e_i*num_quad_pts+m, :] = -1.0*grad_phi[m, :]

                # append to the return arrays
                points = np.append(points, my_points, axis=0)
                H = np.append(H, my_H, axis=0)

        return points, H
        

# class CurlCurlAssembler():

#     def __init__(self, mesh, volume_tags, material_list):
#         '''Default constructor.
        
#         :param mesh:
#             A gmsh mesh object.

#         :param volume_tags:
#             The gmsh tags for the volumes.
            
#         :param material_list:
#             A list of material properties.

#         :return:
#             None.
#         '''

#         # take the mesh
#         self.mesh = mesh

#         # set the element order
#         self.element_order = 1

#         # get some mesh info
#         _, _, num_edges, elementTags = get_mesh_info(self.mesh)

#         # the total number of dofs
#         self.num_dofs = num_edges

#         # get the materials list
#         self.material_list = material_list

#         # The nodes are not sorted correctly. I don't know why...
#         # But we need to get them like this:
#         node_tags, _, _ = mesh.getNodes()
#         num_nodes = len(node_tags)
#         node_tags = np.unique(node_tags)

#         # we now make an array of unique mesh nodes.
#         self.nodes = np.zeros((num_nodes, 3))

#         for i in range(num_nodes):
#             self.nodes[i, :] = mesh.getNode(node_tags[i])[0]
        
        
#         # the number of dofs per element. This is hardcoded at the momement.
#         # Only 1st order approximation is allowed
#         num_dof_el = get_num_edge_dofs(self.element_order)

#         # this is the function type string This is hardcoded at the momement.
#         # Only 1st order approximation is allowed
#         function_type = 'HcurlLegendre' + str(self.element_order-1)

#         # we allocate lists for the cell connectivities and types
#         # for all materials
#         self.num_materials = len(material_list)
#         self.cell_types = []
#         self.cells = []
#         self.global_ids = []
#         self.cell_tags = []

#         print('volume_tags = {}'.format(volume_tags))

#         if len(volume_tags) == 0:

#             # we take all materials
#             self.num_materials = 1

#             # get the elements
#             c_types_tmp, cell_tags, cells_tmp = gmsh.model.mesh.getElements(3, -1)

#             # append to the list
#             self.cell_types.append(c_types_tmp)
#             self.cells.append(cells_tmp)
#             self.cell_tags.append(cell_tags[0])

#             # the number of elements of this material
#             num_el = len(cell_tags[0])

#             # append to the global ids
#             self.global_ids.append(np.zeros((num_el, num_dof_el), dtype=np.int32))

#             # loop over the elements
#             for e in range(num_el):

#                 typeKeys, entityKeys, _ = mesh.getKeysForElement(cell_tags[0][e], function_type, returnCoord=False)
#                 self.global_ids[0][e, :] = entityKeys
        

#         else:
#             for i, tag in enumerate(volume_tags):
                
#                 # get the elements
#                 c_types_tmp, cell_tags, cells_tmp = gmsh.model.mesh.getElements(3, tag)

#                 # append to the list
#                 self.cell_types.append(c_types_tmp)
#                 self.cells.append(cells_tmp)
#                 self.cell_tags.append(cell_tags[0])
                
#                 # the number of elements of this material
#                 num_el = len(cell_tags[0])

#                 # append to the global ids
#                 self.global_ids.append(np.zeros((num_el, num_dof_el), dtype=np.int32))

#                 # loop over the elements
#                 for e in range(num_el):

#                     typeKeys, entityKeys, _ = mesh.getKeysForElement(cell_tags[0][e], function_type, returnCoord=False)
#                     self.global_ids[i][e, :] = entityKeys

#         # the number of nodes
#         num_nodes = self.nodes.shape[0]

#         return None
    
#     def setup_mesh(self, nodes, cells, cell_types):
#         """Setup the mesh from external information.

#         :param nodes:
#             The nodal coordinates.

#         :param cells:
#             The mesh connectivity list.

#         :param cell_types.
#             The mesh cell types list.

#         :return:
#             None.
#         """

#         self.nodes = nodes
#         self.cells = cells
#         self.cell_types = cell_types

#         return None
    

#     def compute_dF_curl_w(self, curl_w_hat, jacobians):

#         #number of quadrature points
#         num_points = np.int32(jacobians.shape[0]/9)

#         #number of basis functions
#         num_basis_fcns = np.int32(curl_w_hat.shape[0] / 3 / num_points)

#         #return values
#         dFcw = np.zeros((3,num_basis_fcns,num_points))

#         #temporal container for curl w
#         curl_w_tmp = np.zeros((3,num_basis_fcns))

#         for i in range(num_points):

#             curl_w_tmp[0,:] = curl_w_hat[0 + i*3*num_basis_fcns:(i+1)*3*num_basis_fcns:3]
#             curl_w_tmp[1,:] = curl_w_hat[1 + i*3*num_basis_fcns:(i+1)*3*num_basis_fcns:3]
#             curl_w_tmp[2,:] = curl_w_hat[2 + i*3*num_basis_fcns:(i+1)*3*num_basis_fcns:3]

#             dFcw[:,:,i] = np.array([[ jacobians[i*9]   , jacobians[i*9+3] , jacobians[i*9+6]],
#                                     [ jacobians[i*9+1] , jacobians[i*9+4] , jacobians[i*9+7]],
#                                     [ jacobians[i*9+2] , jacobians[i*9+5] , jacobians[i*9+8]]]) @ curl_w_tmp

#         return dFcw


#     def compute_stiffness_matrix(self, quad_order=8):
#         '''Compute the stiffness matrix using edge elements.
#         Can at the moment only handle meshes of a single element type.

#         :param quad_order:
#             The order of the quadrature rule.

#         :return:
#             The sparse stiffness matrix.
#         '''


#         print('compute stiffness matrix curl curl...')

#         # make the sparse stiffness matrix
#         K = csr_array((self.num_dofs, self.num_dofs))
    
#         # loop over all materials
#         for n in range(self.num_materials):
            
#             # get the geometry info of this volume
#             cell_types = self.cell_types[n]
#             cells = self.cells[n]

#             # get also the permeability in this domain
#             mat_prop = self.material_list[n]
            
#             # loop over cell types
#             for i, ct in enumerate(cell_types):

#                 # make the finite element
#                 finite_element = FiniteElement(ct)

#                 # the numebr fo element nodes
#                 el_nodes = finite_element.get_number_of_nodes()

#                 # the number of elements of this type
#                 num_el = np.int32(len(cells[i]) / el_nodes)

#                 # get the mesh connectivity
#                 c = cells[i].copy()
#                 c.shape = (num_el, el_nodes)

#                 # get the quadrature nodes and weights
#                 w, q = finite_element.get_quadrature_rule(quad_order)

#                 # the number of quadrature points
#                 num_quad_pts = np.int32(len(q) / 3)

#                 # evaluate the curls of the edge shape functions at the integration points
#                 curls, num_orientations = finite_element.get_curl_edge_basis_functions(self.mesh, q, ct, self.element_order)

#                 # evaluate the derivatives of the lagrange basis functions at these points
#                 d_phi = finite_element.evaluate_basis_derivative(q)

#                 # get the jacobians, determinants and orientations
#                 # _, _, _, orientations = get_vector_basis_mesh_info(self.mesh, q)
#                 jacobians, determinants, coordinates, orientations = get_vector_basis_mesh_info(self.mesh, q)

#                 # get the global ids
#                 glob_ids = self.global_ids[n]

#                 # the number of DoFs for the edge elements per finite element
#                 num_el_dofs = get_num_edge_dofs(self.element_order)

#                 # the number of triplets
#                 num_triplets = num_el*num_el_dofs*num_el_dofs

#                 # allocate the space for the triplet list
#                 ij = np.zeros((num_triplets, 2), dtype=np.int32)
#                 vals = np.zeros((num_triplets, ))

#                 # a triplet counter
#                 t_cnt = 0

#                 # loop over the finite elements
#                 for j, e in enumerate(c):

#                     # compute the Jacobian at the integration points
#                     # inv_J is a numpy array of dimension (M x 3 x 3) where
#                     # the inverse of the Jacobians are stored in the second and
#                     # third dimensions
#                     # M is the number of integration points
#                     # det_J is an (M x 0) numpy array which stores the M values of det(J)
#                     # J = finite_element.compute_J(e - 1, self.nodes, d_phi)
                    
#                     # inv_J = finite_element.compute_J_inv(J)
#                     # det_J = finite_element.compute_J_det(J)

#                     # this offset is the first index in the shape functions list for this orientation
#                     offset_shape_function = 3*num_quad_pts*num_el_dofs*orientations[0, j]

#                     # get the curls of the basis functions with this orientations
#                     curl_w_hat = curls[3*orientations[0, j]*num_quad_pts*num_el_dofs:3*(orientations[0, j]+1)*num_quad_pts*num_el_dofs]

#                     # print('Jac 1 = {}'.format(jacobians[j, :9]))
#                     # print('Jac 2 = {}'.format(J[0, :, :].T.flatten()))

#                     # compute the products dF.curl(w_hat)
#                     dFcw = self.compute_dF_curl_w(curl_w_hat, jacobians[j, :])
#                     # dFcw = self.compute_dF_curl_w(curl_w_hat, J.flatten())

#                     # to do:
#                     # compute nu

#                     #compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
#                     for k in range(num_el_dofs):

#                         # curl_w_hat_k = np.array([])
#                         for l  in range(k, num_el_dofs):
                            
#                             # curl_w_hat_l = np.array([])

#                             # compute v.T A v
#                             vAv = np.sum(dFcw[:, k, :] * dFcw[:, l, :], axis = 0)
#                             this_prod = np.sum(vAv*w/determinants[j, :])
#                             # this_prod = np.sum(vAv*w/det_J)


#                             # fill the triplet list
#                             ij[t_cnt, 0] = glob_ids[j, k] - 1
#                             ij[t_cnt, 1] = glob_ids[j, l] - 1
#                             vals[t_cnt] = this_prod
#                             t_cnt += 1

#                             # apply symmetry
#                             if k != l:
#                                 ij[t_cnt, 0] = glob_ids[j, l] - 1
#                                 ij[t_cnt, 1] = glob_ids[j, k] - 1
#                                 vals[t_cnt] = this_prod
#                                 t_cnt += 1

#             K += csr_array((vals, (ij[:, 0], ij[:, 1])), shape=(self.num_dofs, self.num_dofs))

#         return K
    
#     def compute_stiffness_matrix_alt(self, quad_order=8):
#         '''Compute the stiffness matrix using edge elements.
#         Can at the moment only handle meshes of a single element type.

#         :param quad_order:
#             The order of the quadrature rule.

#         :return:
#             The sparse stiffness matrix.
#         '''


#         print('compute stiffness matrix curl curl...')

#         # all triplets
#         all_ij = np.zeros((0, 2), dtype=np.int64)
#         all_vals = np.zeros((0, ), dtype=float)

#         # make the sparse stiffness matrix
#         K = csr_array((self.num_dofs, self.num_dofs))
    
#         # loop over all materials
#         for n in range(self.num_materials):
            
#             # get the geometry info of this volume
#             cell_types = self.cell_types[n]
#             cells = self.cells[n]

#             # get the global orientations of the finite elements
#             orientations = get_vector_basis_orientations(self.mesh,
#                                                          self.cell_tags[n],
#                                                          element_spec='CurlHcurlLegendre' + str(self.element_order-1))
            
#             # get also the permeability in this domain
#             mat_prop = self.material_list[n]
            
#             # loop over cell types
#             for i, ct in enumerate(cell_types):

#                 # make the finite element
#                 finite_element = FiniteElement(ct)

#                 # the numebr fo element nodes
#                 el_nodes = finite_element.get_number_of_nodes()

#                 # the number of elements of this type
#                 num_el = np.int32(len(cells[i]) / el_nodes)

#                 # get the mesh connectivity
#                 c = cells[i].copy()
#                 c.shape = (num_el, el_nodes)

#                 # get the quadrature nodes and weights
#                 w, q = finite_element.get_quadrature_rule(quad_order)

#                 # the number of quadrature points
#                 num_quad_pts = np.int32(len(q) / 3)

#                 # evaluate the curls of the edge shape functions at the integration points
#                 curls, num_orientations = finite_element.get_curl_edge_basis_functions(self.mesh, q, ct, self.element_order)

#                 # evaluate the derivatives of the lagrange basis functions at these points
#                 d_phi = finite_element.evaluate_basis_derivative(q)

#                 # get the global ids
#                 glob_ids = self.global_ids[n]

#                 # the number of DoFs for the edge elements per finite element
#                 num_el_dofs = get_num_edge_dofs(self.element_order)

#                 # the number of triplets
#                 num_triplets = num_el*num_el_dofs*num_el_dofs

#                 # allocate the space for the triplet list
#                 ij = np.zeros((num_triplets, 2), dtype=np.int32)
#                 vals = np.zeros((num_triplets, ))

#                 # a triplet counter
#                 t_cnt = 0

#                 # loop over the finite elements
#                 for j, e in enumerate(c):


#                     # compute the Jacobian at the integration points
#                     # inv_J is a numpy array of dimension (M x 3 x 3) where
#                     # the inverse of the Jacobians are stored in the second and
#                     # third dimensions
#                     # M is the number of integration points
#                     # det_J is an (M x 0) numpy array which stores the M values of det(J)
#                     J = finite_element.compute_J(e - 1, self.nodes, d_phi)
                    
#                     # inv_J = finite_element.compute_J_inv(J)
#                     det_J = finite_element.compute_J_det(J)

#                     # this offset is the first index in the shape functions list for this orientation
#                     offset_shape_function = 3*num_quad_pts*num_el_dofs*orientations[j]

#                     # get the curls of the basis functions with this orientations
#                     curl_w_hat = curls[3*orientations[j]*num_quad_pts*num_el_dofs:3*(orientations[j]+1)*num_quad_pts*num_el_dofs]

#                     # to do:
#                     # compute nu

#                     # if 166 in glob_ids[j, :] :
#                     #     print('e = {}'.format(e))
#                     #     print('curl_w_hat = {}'.format(curl_w_hat))
#                     #     print('glob_ids[j, :] = {}'.format(glob_ids[j, :]))
#                     #     print('det_J = {}'.format(det_J))

#                     #compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
#                     for k in range(num_el_dofs):

#                         for l  in range(k, num_el_dofs):
                            
#                             # the integration value
#                             int_val = 0.0

#                             for m in range(num_quad_pts):

                                
#                                 curl_w_hat_k = np.array([curl_w_hat[3*(m*num_el_dofs + k)     ],
#                                                          curl_w_hat[3*(m*num_el_dofs + k) + 1 ],
#                                                          curl_w_hat[3*(m*num_el_dofs + k) + 2 ]])

#                                 curl_w_hat_l = np.array([curl_w_hat[3*(m*num_el_dofs + l)     ],
#                                                          curl_w_hat[3*(m*num_el_dofs + l) + 1 ],
#                                                          curl_w_hat[3*(m*num_el_dofs + l) + 2 ]])
                                
#                                 dFcw_k = J[m, :, :] @ curl_w_hat_k
#                                 dFcw_l = J[m, :, :] @ curl_w_hat_l

#                                 int_val += np.sum(dFcw_k*dFcw_l)*w[m]/det_J[m]


#                             # fill the triplet list
#                             ij[t_cnt, 0] = glob_ids[j, k] - 1
#                             ij[t_cnt, 1] = glob_ids[j, l] - 1
#                             vals[t_cnt] = int_val
#                             t_cnt += 1

#                             # apply symmetry
#                             if k != l:
#                                 ij[t_cnt, 0] = glob_ids[j, l] - 1
#                                 ij[t_cnt, 1] = glob_ids[j, k] - 1
#                                 vals[t_cnt] = int_val
#                                 t_cnt += 1

#             all_ij = np.append(all_ij, ij, axis=0)
#             all_vals = np.append(all_vals, vals)

#         K = csr_array((all_vals, (all_ij[:, 0], all_ij[:, 1])), shape=(self.num_dofs, self.num_dofs))

#         return K

#     def compute_stiffness_and_jacobian_matrix(self, x, quad_order=8):
#         '''Compute the stiffness matrix using edge elements.
#         Can at the moment only handle meshes of a single element type.

#         :param x:
#             The solution vector at which the jacobian is calculated.

#         :param quad_order:
#             The order of the quadrature rule.

#         :return:
#             The sparse stiffness matrix.
#         '''


#         print('compute stiffness and jacobian matrix curl curl...')

#         # all triplets
#         all_ij = np.zeros((0, 2), dtype=np.int64)
#         all_vals = np.zeros((0, ), dtype=float)

#         # make the sparse stiffness matrix
#         K = csr_array((self.num_dofs, self.num_dofs))
    
#         # make the sparse jacobi matrix
#         dK = csr_array((self.num_dofs, self.num_dofs))

#         # loop over all materials
#         for n in range(self.num_materials):
            
#             # get the geometry info of this volume
#             cell_types = self.cell_types[n]
#             cells = self.cells[n]

#             # get the global orientations of the finite elements
#             orientations = get_vector_basis_orientations(self.mesh,
#                                                          self.cell_tags[n],
#                                                          element_spec='CurlHcurlLegendre' + str(self.element_order-1))
            
#             # get also the permeability in this domain
#             reluctance = self.material_list[n]
            
#             # loop over cell types
#             for i, ct in enumerate(cell_types):

#                 # make the finite element
#                 finite_element = FiniteElement(ct)

#                 # the numebr fo element nodes
#                 el_nodes = finite_element.get_number_of_nodes()

#                 # the number of elements of this type
#                 num_el = np.int32(len(cells[i]) / el_nodes)

#                 # get the mesh connectivity
#                 c = cells[i].copy()
#                 c.shape = (num_el, el_nodes)

#                 # get the quadrature nodes and weights
#                 w, q = finite_element.get_quadrature_rule(quad_order)

#                 # the number of quadrature points
#                 num_quad_pts = np.int32(len(q) / 3)

#                 # evaluate the curls of the edge shape functions at the integration points
#                 curls, num_orientations = finite_element.get_curl_edge_basis_functions(self.mesh, q, ct, self.element_order)

#                 # evaluate the derivatives of the lagrange basis functions at these points
#                 d_phi = finite_element.evaluate_basis_derivative(q)

#                 # get the global ids
#                 glob_ids = self.global_ids[n]

#                 # the number of DoFs for the edge elements per finite element
#                 num_el_dofs = get_num_edge_dofs(self.element_order)

#                 # the number of triplets
#                 num_triplets = num_el*num_el_dofs*num_el_dofs

#                 # allocate the space for the triplet list for K
#                 ij = np.zeros((num_triplets, 2), dtype=np.int32)
#                 vals_K = np.zeros((num_triplets, ))

#                 # allocate the space for the triplet list (J)
#                 vals_dK = np.zeros((num_triplets, ))

#                 # a triplet counter
#                 t_cnt = 0

#                 # loop over the finite elements
#                 for j, e in enumerate(c):


#                     # compute the Jacobian at the integration points
#                     # inv_J is a numpy array of dimension (M x 3 x 3) where
#                     # the inverse of the Jacobians are stored in the second and
#                     # third dimensions
#                     # M is the number of integration points
#                     # det_J is an (M x 0) numpy array which stores the M values of det(J)
#                     J = finite_element.compute_J(e - 1, self.nodes, d_phi)
                    
#                     # inv_J = finite_element.compute_J_inv(J)
#                     det_J = finite_element.compute_J_det(J)

#                     # this offset is the first index in the shape functions list for this orientation
#                     offset_shape_function = 3*num_quad_pts*num_el_dofs*orientations[j]

#                     # get the curls of the basis functions with this orientations
#                     curl_w_hat = curls[3*orientations[j]*num_quad_pts*num_el_dofs:3*(orientations[j]+1)*num_quad_pts*num_el_dofs]

#                     # compute the B field vector
#                     #B_vec, B_mag = self.compute_grad_phi(e, x, d_phi, inv_J)

#                     # compute the reluctance
#                     #nu = reluctance.evaluate_mu(B_mag)

#                     # compute the permebility derivative
#                     #d_mu = reluctance.evaluate_mu_derivative(B_mag)

#                     #compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
#                     for k in range(num_el_dofs):

#                         for l  in range(k, num_el_dofs):
                            
#                             # the integration value
#                             int_val = 0.0

#                             for m in range(num_quad_pts):

                                
#                                 curl_w_hat_k = np.array([curl_w_hat[3*(m*num_el_dofs + k)     ],
#                                                          curl_w_hat[3*(m*num_el_dofs + k) + 1 ],
#                                                          curl_w_hat[3*(m*num_el_dofs + k) + 2 ]])

#                                 curl_w_hat_l = np.array([curl_w_hat[3*(m*num_el_dofs + l)     ],
#                                                          curl_w_hat[3*(m*num_el_dofs + l) + 1 ],
#                                                          curl_w_hat[3*(m*num_el_dofs + l) + 2 ]])
                                
#                                 dFcw_k = J[m, :, :] @ curl_w_hat_k
#                                 dFcw_l = J[m, :, :] @ curl_w_hat_l

#                                 int_val_K += nu[m]*np.sum(dFcw_k*dFcw_l)*w[m]/det_J[m]

#                                 # increment the integration value
#                                 if B_mag[m] >= 1e-14:
#                                     int_val_dK += d_nu[m]*np.sum(dFcw_k*B_vec[m, :])*np.sum(dFcw_l*B_vec[m, :])*w[m]/det_J[m]/B_mag[m]

#                             # fill the triplet list
#                             ij[t_cnt, 0] = glob_ids[j, k] - 1
#                             ij[t_cnt, 1] = glob_ids[j, l] - 1
#                             vals_K[t_cnt] = int_val_K
#                             vals_dK[t_cnt] = int_val_dK

#                             t_cnt += 1

#                             # apply symmetry
#                             if k != l:
#                                 ij[t_cnt, 0] = glob_ids[j, l] - 1
#                                 ij[t_cnt, 1] = glob_ids[j, k] - 1
#                                 vals_K[t_cnt] = int_val_K
#                                 vals_dK[t_cnt] = int_val_dK

#                                 t_cnt += 1

#             all_ij = np.append(all_ij, ij, axis=0)
#             all_vals_K = np.append(all_vals_K, vals_K)
#             all_vals_dK = np.append(all_vals_dK, vals_dK)

#         K = csr_array((all_vals_K, (all_ij[:, 0], all_ij[:, 1])), shape=(self.num_dofs, self.num_dofs))
#         dK = csr_array((all_vals_dK, (all_ij[:, 0], all_ij[:, 1])), shape=(self.num_dofs, self.num_dofs))

#         return K, dK + K

#     def compute_fcn_rhs(self, J_fcn, quad_order=8):
#         '''Compute the rhs for solving a magnetostatic problem using edge elements.
#         Can at the moment only handle meshes of a single element type.
#         This function is based on a current density function J_fcn.
        
#         :param J_fcn:
#             A function specifying the rhs.
        
#         :param quad_order:
#             The order of the quadrature rule.

#         :return:
#             The sparse stiffness matrix.
#         '''

#         print('compute rhs...')

#         # make the sparse stiffness matrix
#         rhs = np.zeros((self.num_dofs, ))

        
#         # loop over all materials
#         for n in range(self.num_materials):
            
#             # get the geometry info of this volume
#             cell_types = self.cell_types[n]
#             cells = self.cells[n]

#             # get the global orientations of the finite elements
#             orientations = get_vector_basis_orientations(self.mesh,
#                                                          self.cell_tags[n],
#                                                          element_spec='HcurlLegendre' + str(self.element_order-1))    

#             # get also the permeability in this domain
#             mat_prop = self.material_list[n]
            
#             # loop over cell types
#             for i, ct in enumerate(cell_types):

#                 # make the finite element
#                 finite_element = FiniteElement(ct)

#                 # the numebr fo element nodes
#                 el_nodes = finite_element.get_number_of_nodes()

#                 # the number of elements of this type
#                 num_el = np.int32(len(cells[i]) / el_nodes)

#                 # get the mesh connectivity
#                 c = cells[i].copy()
#                 c.shape = (num_el, el_nodes)

#                 # get the quadrature nodes and weights
#                 w, q = finite_element.get_quadrature_rule(quad_order)

#                 # the number of quadrature points
#                 num_quad_pts = np.int32(len(q) / 3)

#                 # evaluate the edge shape functions at the integration points
#                 basis, num_orientations = finite_element.get_edge_basis_functions(self.mesh, q, ct, self.element_order)

#                 # evaluate the shape functions at these points (not needed)
#                 phi = finite_element.evaluate_basis(q)

#                 # evaluate the derivatives of the lagrange basis functions at these points
#                 d_phi = finite_element.evaluate_basis_derivative(q)

#                 # get the global ids
#                 glob_ids = self.global_ids[n]

#                 # the number of DoFs for the edge elements per finite element
#                 num_el_dofs = get_num_edge_dofs(self.element_order)

                    
#                 # loop over the finite elements
#                 for j, e in enumerate(c):

#                     # compute the Jacobian at the integration points
#                     # inv_J is a numpy array of dimension (M x 3 x 3) where
#                     # the inverse of the Jacobians are stored in the second and
#                     # third dimensions
#                     # M is the number of integration points
#                     # det_J is an (M x 0) numpy array which stores the M values of det(J)
#                     J = finite_element.compute_J(e - 1, self.nodes, d_phi)
                
#                     inv_J = finite_element.compute_J_inv(J)
#                     det_J = finite_element.compute_J_det(J)

#                     # this offset is the first index in the shape functions list for this orientation
#                     offset_shape_function = 3*num_quad_pts*num_el_dofs*orientations[j]

#                     # get the curls of the basis functions with this orientations
#                     w_hat = basis[3*orientations[j]*num_quad_pts*num_el_dofs:3*(orientations[j]+1)*num_quad_pts*num_el_dofs]
                    

#                     # evaluate this finite element for the global position
#                     points = finite_element.evaluate(e - 1, self.nodes, phi)

#                     # evaluate current density
#                     j_eval = J_fcn(points)

#                     # compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
#                     for k in range(num_el_dofs):

#                         # the integration value
#                         int_val = 0.0

#                         for m in range(num_quad_pts):

#                             # the basis function in local coordinates
#                             w_hat_k = np.array([w_hat[3*(m*num_el_dofs + k)     ],
#                                                      w_hat[3*(m*num_el_dofs + k) + 1 ],
#                                                      w_hat[3*(m*num_el_dofs + k) + 2 ]])

#                             # the basis function in global coordinates
#                             dFcw_k = inv_J[m, :, :].T @ w_hat_k

                            
#                             # the product J.w_tilde
#                             rhs[glob_ids[j, k] - 1] += np.sum(dFcw_k*j_eval[m, :])*w[m]*det_J[m]
    


#         return rhs

#     def compute_grad_phi(self, c, x, d_phi, inv_J):
#         '''Compute the magnetic field in a finite element
#         with connectivity c. The vector x stores the nodal 
#         values of the solution.
#         The array grad_phi stores the gradients of the FEM
#         basis functions.

#         :param c:
#             The node connectivity of the finite element.

#         :param x:
#             The solution vector.

#         :param d_phi:
#             The derivatives of all basis functions.

#         :param inv_J:
#             The inverse of the Jacobian matrix of the transformation.

#         :return:
#             The field vectors H, as well as the magnitude H_mag.
#         '''

#         # the number of evaluation points
#         num_eval = d_phi.shape[0]

#         # allocate the return array
#         grad_phi = np.zeros((num_eval, 3))

#         # evaluate
#         grad_phi[:, 0] = d_phi[:, :, 0] @ x[c]
#         grad_phi[:, 1] = d_phi[:, :, 1] @ x[c]
#         grad_phi[:, 2] = d_phi[:, :, 2] @ x[c]


#         # transform it to the global domain
#         for m in range(num_eval):
#             grad_phi[m, :] = inv_J[m, :, :].T @ grad_phi[m, :]


#         # return also the magnitude
#         mag = np.linalg.norm(grad_phi, axis=1)

#         return grad_phi, mag

#     def compute_B_in_element(self, glob_ids, x, curl_w_hat, J, det_J):
#         '''Compute the magnetic flux density in a finite element, given the solution vector
#         and the curls of the basis functions.

#         :param glob_ids:
#             The cell global edge basis function identifiers.

#         :param x:
#             The solution vector.

#         :param curl_w_hat:
#             The curl of the basis functions evaluated at the interpolation points.

#         :param J:
#             The jacobian matrices evaluated at the interpolation points.

#         :param det_J:
#             The determinants of the jacobians evaluated at the interpolation points.

#         :return:
#             The B field vectors at the interpolation points.
#         '''

#         # the number of interpolation points
#         num_points = len(det_J)

#         # the number of basis functions
#         num_el_dofs = len(curl_w_hat)/num_points/3

#         # make space for the return vector
#         B_ret = np.zeros((num_points, 3))

#         #compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)

#         for m in range(num_points):

#             for k in range(num_el_dofs):

#                 # the basis function in local coordinates
#                 curl_w_hat_k = np.array([curl_w_hat[3*(m*num_el_dofs + k)     ],
#                                          curl_w_hat[3*(m*num_el_dofs + k) + 1 ],
#                                          curl_w_hat[3*(m*num_el_dofs + k) + 2 ]])

#                 # the B field vector in global coordinates
#                 B_ret[m, :] += J[m, :, :] @ curl_w_hat_k * x[glob_ids[k] - 1] / det_J[m]

#         return B_ret

    
#     def compute_rhs_electric_potential(self, x_p, quad_order=8, select_mat='all'):
#         '''Compute the rhs for solving a magnetostatic problem using edge elements.
#         Can at the moment only handle meshes of a single element type.
#         This function is based on an electric potential discretized on the same mesh.
        
#         :param x_p:
#             The nodal values for the electric potential.
        
#         :param quad_order:
#             The order of the quadrature rule.

#         :param select_mat:
#             Select the material. Default 'all' means all materials.

#         :return:
#             The sparse stiffness matrix.
#         '''

#         print('compute rhs...')

#         # check if material selection is a string
#         if isinstance(select_mat, str):
#             if select_mat == 'all':
#                 materials = range(self.num_mat)
#         else:
#             materials = select_mat

#         # make the sparse stiffness matrix
#         rhs = np.zeros((self.num_dofs, ))
    
#         # loop over all materials
#         for n in materials:
            
#             # get the geometry info of this volume
#             cell_types = self.cell_types[n]
#             cells = self.cells[n]

#             # get also the permeability in this domain
#             mat_prop = self.material_list[n]
            
#             # get the global orientations of the finite elements
#             orientations = get_vector_basis_orientations(self.mesh,
#                                                          self.cell_tags[n],
#                                                          element_spec='HcurlLegendre' + str(self.element_order-1))    
            
#             # loop over cell types
#             for i, ct in enumerate(cell_types):

#                 # make the finite element
#                 finite_element = FiniteElement(ct)

#                 # the numebr fo element nodes
#                 el_nodes = finite_element.get_number_of_nodes()

#                 # the number of elements of this type
#                 num_el = np.int32(len(cells[i]) / el_nodes)

#                 # get the mesh connectivity
#                 c = cells[i].copy()
#                 c.shape = (num_el, el_nodes)

#                 # get the quadrature nodes and weights
#                 w, q = finite_element.get_quadrature_rule(quad_order)

#                 # the number of quadrature points
#                 num_quad_pts = np.int32(len(q) / 3)

#                 # evaluate the edge shape functions at the integration points
#                 basis, num_orientations = finite_element.get_edge_basis_functions(self.mesh, q, ct, self.element_order)

#                 # evaluate the shape functions at these points (not needed)
#                 phi = finite_element.evaluate_basis(q)

#                 # evaluate the derivatives of the lagrange basis functions at these points
#                 d_phi = finite_element.evaluate_basis_derivative(q)

#                 # get the global ids
#                 glob_ids = self.global_ids[n]

#                 # the number of DoFs for the edge elements per finite element
#                 num_el_dofs = get_num_edge_dofs(self.element_order)

#                 # loop over the finite elements
#                 for j, e in enumerate(c):


#                     # compute the Jacobian at the integration points
#                     # inv_J is a numpy array of dimension (M x 3 x 3) where
#                     # the inverse of the Jacobians are stored in the second and
#                     # third dimensions
#                     # M is the number of integration points
#                     # det_J is an (M x 0) numpy array which stores the M values of det(J)
#                     J = finite_element.compute_J(e - 1, self.nodes, d_phi)
                
#                     inv_J = finite_element.compute_J_inv(J)
#                     det_J = finite_element.compute_J_det(J)

#                     # this offset is the first index in the shape functions list for this orientation
#                     offset_shape_function = 3*num_quad_pts*num_el_dofs*orientations[j]

#                     # get the curls of the basis functions with this orientations
#                     w_hat = basis[3*orientations[j]*num_quad_pts*num_el_dofs:3*(orientations[j]+1)*num_quad_pts*num_el_dofs]
                    
#                     # evaluate this finite element for the global position
#                     points = finite_element.evaluate(e - 1, self.nodes, phi)

#                     # evaluate current density
#                     j_eval, _ = self.compute_grad_phi(e - 1, x_p, d_phi, inv_J)

#                     # compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
#                     for k in range(num_el_dofs):

#                         # the integration value
#                         int_val = 0.0

#                         for m in range(num_quad_pts):

#                             # the basis function in local coordinates
#                             w_hat_k = np.array([w_hat[3*(m*num_el_dofs + k)     ],
#                                                      w_hat[3*(m*num_el_dofs + k) + 1 ],
#                                                      w_hat[3*(m*num_el_dofs + k) + 2 ]])

#                             # the basis function in global coordinates
#                             dFcw_k = inv_J[m, :, :].T @ w_hat_k

                            
#                             # the product J.w_tilde
#                             rhs[glob_ids[j, k] - 1] += np.sum(dFcw_k*j_eval[m, :])*w[m]*det_J[m]
    

#         return rhs


#     def get_quadrature_points(self, quad_order=3):
#         '''Get the quadrature points in the domain.

#         :param quad_order:
#             The order of the quadrature rule.

#         :return:
#             The points in an (M x 3) array.

#         '''

#         # make the return field evaluation points
#         points_ret = np.zeros((0, 3))

#         # marker for the domains
#         marker = np.zeros((0, ))

#         # loop over all materials
#         for n in range(self.num_materials):
            
#             # get the geometry info of this volume
#             cell_types = self.cell_types[n]
#             cells = self.cells[n]

#             # get also the permeability in this domain
#             mat_prop = self.material_list[n]
            
#             # loop over cell types
#             for i, ct in enumerate(cell_types):

#                 # make the finite element
#                 finite_element = FiniteElement(ct)

#                 # the numebr fo element nodes
#                 el_nodes = finite_element.get_number_of_nodes()

#                 # the number of elements of this type
#                 num_el = np.int32(len(cells[i]) / el_nodes)

#                 # get the mesh connectivity
#                 c = cells[i].copy()
#                 c.shape = (num_el, el_nodes)

#                 # get the quadrature nodes and weights
#                 w, q = finite_element.get_quadrature_rule(quad_order)

#                 # the number of quadrature points
#                 num_quad_pts = np.int32(len(q) / 3)

#                 # evaluate the shape functions at these points (not needed)
#                 phi = finite_element.evaluate_basis(q)

#                 # field evaluation points for this material
#                 this_points = np.zeros((num_el*num_quad_pts, 3))

#                 # loop over the finite elements
#                 for j, e in enumerate(c):

#                     # evaluate this finite element for the global position
#                     this_points[j*num_quad_pts:(j+1)*num_quad_pts, :] = finite_element.evaluate(e - 1, self.nodes, phi)

#                 points_ret = np.append(points_ret, this_points, axis=0)
#                 marker = np.append(marker, n*np.ones((this_points.shape[0], )), axis=0)
    
#         return points_ret, marker

#     def compute_B(self, x, quad_order=8):
#         '''Compute the B for a given solution vector.
        
#         :param x:
#             The solution vector.
        
#         :param quad_order:
#             The order of the quadrature rule.

#         :return:
#             The sparse stiffness matrix.
#         '''

#         print('compute B...')
#         print('Ho...')

#         # make the B field return vector
#         B_ret = np.zeros((0, 3))

#         # make the return field evaluation points
#         points_ret = np.zeros((0, 3))

#         # loop over all materials
#         for n in range(self.num_materials):
            
#             # get the geometry info of this volume
#             cell_types = self.cell_types[n]
#             cells = self.cells[n]

#             # get the global orientations of the finite elements
#             orientations = get_vector_basis_orientations(self.mesh,
#                                                          self.cell_tags[n],
#                                                          element_spec='CurlHcurlLegendre' + str(self.element_order-1))

#             # get also the permeability in this domain
#             mat_prop = self.material_list[n]
            
#             # loop over cell types
#             for i, ct in enumerate(cell_types):

#                 # make the finite element
#                 finite_element = FiniteElement(ct)

#                 # the numebr fo element nodes
#                 el_nodes = finite_element.get_number_of_nodes()

#                 # the number of elements of this type
#                 num_el = np.int32(len(cells[i]) / el_nodes)

#                 # get the mesh connectivity
#                 c = cells[i].copy()
#                 c.shape = (num_el, el_nodes)

#                 # get the quadrature nodes and weights
#                 w, q = finite_element.get_quadrature_rule(quad_order)

#                 # the number of quadrature points
#                 num_quad_pts = np.int32(len(q) / 3)

#                 # evaluate the curls of the edge shape functions at the integration points
#                 curls, num_orientations = finite_element.get_curl_edge_basis_functions(self.mesh, q, ct, self.element_order)

#                 # evaluate the shape functions at these points (not needed)
#                 phi = finite_element.evaluate_basis(q)

#                 # evaluate the derivatives of the lagrange basis functions at these points
#                 d_phi = finite_element.evaluate_basis_derivative(q)

#                 # get the global ids
#                 glob_ids = self.global_ids[n]

#                 # the number of DoFs for the edge elements per finite element
#                 num_el_dofs = get_num_edge_dofs(self.element_order)

#                 # B field vectors for this material
#                 this_B = np.zeros((num_el*num_quad_pts, 3))

#                 # field evaluation points for this material
#                 this_points = np.zeros((num_el*num_quad_pts, 3))

#                 # loop over the finite elements
#                 for j, e in enumerate(c):


#                     # compute the Jacobian at the integration points
#                     # inv_J is a numpy array of dimension (M x 3 x 3) where
#                     # the inverse of the Jacobians are stored in the second and
#                     # third dimensions
#                     # M is the number of integration points
#                     # det_J is an (M x 0) numpy array which stores the M values of det(J)
#                     J = finite_element.compute_J(e - 1, self.nodes, d_phi)
                    
#                     inv_J = finite_element.compute_J_inv(J)
#                     det_J = finite_element.compute_J_det(J)

#                     # this offset is the first index in the shape functions list for this orientation
#                     offset_shape_function = 3*num_quad_pts*num_el_dofs*orientations[j]

#                     # get the  basis functions with this orientations
#                     curl_w_hat = curls[3*orientations[j]*num_quad_pts*num_el_dofs:3*(orientations[j]+1)*num_quad_pts*num_el_dofs]

#                     # evaluate this finite element for the global position
#                     this_points[j*num_quad_pts:(j+1)*num_quad_pts, :] = finite_element.evaluate(e - 1, self.nodes, phi)

#                     #compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
#                     for k in range(num_el_dofs):

#                         # the integration value
#                         int_val = 0.0

#                         for m in range(num_quad_pts):

#                             # the basis function in local coordinates
#                             curl_w_hat_k = np.array([curl_w_hat[3*(m*num_el_dofs + k)     ],
#                                                 curl_w_hat[3*(m*num_el_dofs + k) + 1 ],
#                                                 curl_w_hat[3*(m*num_el_dofs + k) + 2 ]])

#                             # the basis function in global coordinates
#                             this_B[j*num_quad_pts+m, :] += J[m, :, :] @ curl_w_hat_k * x[glob_ids[j, k] - 1] / det_J[m]


#                     B_test = self.compute_B_in_element(glob_ids[j, :], x, curl_w_hat, J, det_J)

#                     if j == 0:
#                         print('B_test = {}'.format(B_test))
#                         print('this_B = {}'.format(this_B[j*num_quad_pts:(j+1)*num_quad_pts, :]))

#                 B_ret = np.append(B_ret, this_B, axis=0)
#                 points_ret = np.append(points_ret, this_points, axis=0)
    
#         return points_ret, B_ret
    

#     def compute_A(self, x, quad_order=8):
#         '''Compute A for a given solution vector.
        
#         :param x:
#             The solution vector.
        
#         :param quad_order:
#             The order of the quadrature rule.

#         :return:
#             The sparse stiffness matrix.
#         '''

#         print('compute A...')

#         # make the A field return vector
#         A_ret = np.zeros((0, 3))

#         # make the return field evaluation points
#         points_ret = np.zeros((0, 3))

#         # loop over all materials
#         for n in range(self.num_materials):
            
#             # get the geometry info of this volume
#             cell_types = self.cell_types[n]
#             cells = self.cells[n]

#             # get the global orientations of the finite elements
#             orientations = get_vector_basis_orientations(self.mesh,
#                                                          self.cell_tags[n],
#                                                          element_spec='HcurlLegendre' + str(self.element_order-1))

#             # get also the permeability in this domain
#             mat_prop = self.material_list[n]
            
#             # loop over cell types
#             for i, ct in enumerate(cell_types):

#                 # make the finite element
#                 finite_element = FiniteElement(ct)

#                 # the numebr fo element nodes
#                 el_nodes = finite_element.get_number_of_nodes()

#                 # the number of elements of this type
#                 num_el = np.int32(len(cells[i]) / el_nodes)

#                 # get the mesh connectivity
#                 c = cells[i].copy()
#                 c.shape = (num_el, el_nodes)

#                 # get the quadrature nodes and weights
#                 w, q = finite_element.get_quadrature_rule(quad_order)

#                 # the number of quadrature points
#                 num_quad_pts = np.int32(len(q) / 3)

#                 # evaluate the curls of the edge shape functions at the integration points
#                 basis, num_orientations = finite_element.get_edge_basis_functions(self.mesh, q, ct, self.element_order)

#                 # evaluate the shape functions at these points (not needed)
#                 phi = finite_element.evaluate_basis(q)

#                 # evaluate the derivatives of the lagrange basis functions at these points
#                 d_phi = finite_element.evaluate_basis_derivative(q)

#                 # get the global ids
#                 glob_ids = self.global_ids[n]

#                 # the number of DoFs for the edge elements per finite element
#                 num_el_dofs = get_num_edge_dofs(self.element_order)

#                 # A field vectors for this material
#                 this_A = np.zeros((num_el*num_quad_pts, 3))

#                 # field evaluation points for this material
#                 this_points = np.zeros((num_el*num_quad_pts, 3))

#                 # loop over the finite elements
#                 for j, e in enumerate(c):


#                     # compute the Jacobian at the integration points
#                     # inv_J is a numpy array of dimension (M x 3 x 3) where
#                     # the inverse of the Jacobians are stored in the second and
#                     # third dimensions
#                     # M is the number of integration points
#                     # det_J is an (M x 0) numpy array which stores the M values of det(J)
#                     J = finite_element.compute_J(e - 1, self.nodes, d_phi)
                    
#                     inv_J = finite_element.compute_J_inv(J)
#                     det_J = finite_element.compute_J_det(J)

#                     # this offset is the first index in the shape functions list for this orientation
#                     offset_shape_function = 3*num_quad_pts*num_el_dofs*orientations[j]

#                     # get the curls of the basis functions with this orientations
#                     w_hat = basis[3*orientations[j]*num_quad_pts*num_el_dofs:3*(orientations[j]+1)*num_quad_pts*num_el_dofs]

#                     # evaluate this finite element for the global position
#                     this_points[j*num_quad_pts:(j+1)*num_quad_pts, :] = finite_element.evaluate(e - 1, self.nodes, phi)

#                     #compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
#                     for k in range(num_el_dofs):

#                         # the integration value
#                         int_val = 0.0

#                         for m in range(num_quad_pts):

#                             # the basis function in local coordinates
#                             w_hat_k = np.array([w_hat[3*(m*num_el_dofs + k)     ],
#                                                 w_hat[3*(m*num_el_dofs + k) + 1 ],
#                                                 w_hat[3*(m*num_el_dofs + k) + 2 ]])

#                             # the basis function in global coordinates
#                             this_A[j*num_quad_pts+m, :] += inv_J[m, :, :].T @ w_hat_k * x[glob_ids[j, k] - 1]

#                 A_ret = np.append(A_ret, this_A, axis=0)
#                 points_ret = np.append(points_ret, this_points, axis=0)
    
#         return points_ret, A_ret