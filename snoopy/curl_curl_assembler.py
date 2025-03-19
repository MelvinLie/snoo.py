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
from .mesh_tools import get_global_ids_for_entities

class CurlCurlAssembler():

    def __init__(self, mesh, volume_tags, material_list, element_order=1):
        '''Default constructor.
        
        :param mesh:
            A gmsh mesh object.

        :param volume_tags:
            The gmsh tags for the volumes.
            
        :param material_list:
            A list of material properties.

        :param element_order:
            The order of edge finite element.

        :return:
            None.
        '''

        # take the mesh
        self.mesh = mesh

        # set the element order
        self.element_order = element_order

        if element_order > 4:
            print('Warning! Elements order {} not implemented. Using 1'.format(element_order))
            self.element_order = 1

        # get some mesh info
        self.num_dofs, num_faces, num_edges, elementTags = get_mesh_info(self.mesh, element_order)

        # get the materials list
        self.material_list = material_list

        # The nodes are not sorted correctly. I don't know why...
        # But we need to get them like this:
        node_tags, _, _ = mesh.getNodes()
        num_nodes = len(node_tags)
        node_tags = np.unique(node_tags)

        # we now make an array of unique mesh nodes.
        self.nodes = np.zeros((num_nodes, 3))

        for i in range(num_nodes):
            self.nodes[i, :] = mesh.getNode(node_tags[i])[0]
        
        
        # the number of dofs per element.
        num_dof_el = get_num_edge_dofs(self.element_order)

        # this is the function type string
        function_type = 'HcurlLegendre' + str(self.element_order-1)

        # we allocate lists for the cell connectivities and types
        # for all materials
        self.num_materials = len(material_list)
        self.cell_types = []
        self.cells = []
        self.global_ids = []
        self.cell_tags = []


        if len(volume_tags) == 0:

            # we take all materials
            self.num_materials = 1

            # get the elements
            c_types_tmp, cell_tags, cells_tmp = gmsh.model.mesh.getElements(3, -1)

            # append to the list
            self.cell_types.append(c_types_tmp)
            self.cells.append(cells_tmp)
            self.cell_tags.append(cell_tags[0])

            # the number of elements of this material
            num_el = len(cell_tags[0])

            # append to the global ids
            self.global_ids.append(np.zeros((num_el, num_dof_el), dtype=np.int32))

            # loop over the elements
            for e in range(num_el):

                typeKeys, entityKeys, _ = mesh.getKeysForElement(cell_tags[0][e], function_type, returnCoord=False)
                self.global_ids[0][e, :] = get_global_ids_for_entities(typeKeys, entityKeys, num_el, num_edges, num_faces, element_order)
        
        else:
            for i, tag in enumerate(volume_tags):
                
                # get the elements
                c_types_tmp, cell_tags, cells_tmp = gmsh.model.mesh.getElements(3, tag)

                # append to the list
                self.cell_types.append(c_types_tmp)
                self.cells.append(cells_tmp)
                self.cell_tags.append(cell_tags[0])
                
                # the number of elements of this material
                num_el = len(cell_tags[0])

                # append to the global ids
                self.global_ids.append(np.zeros((num_el, num_dof_el), dtype=np.int32))

                # loop over the elements
                for e in range(num_el):

                    typeKeys, entityKeys, _ = mesh.getKeysForElement(cell_tags[0][e], function_type, returnCoord=False)

                    self.global_ids[i][e, :] = get_global_ids_for_entities(typeKeys, entityKeys, num_el, num_edges, num_faces, element_order)

        # the number of nodes
        num_nodes = self.nodes.shape[0]

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
    

    def compute_dF_curl_w(self, curl_w_hat, jacobians):

        #number of quadrature points
        num_points = np.int32(jacobians.shape[0]/9)

        #number of basis functions
        num_basis_fcns = np.int32(curl_w_hat.shape[0] / 3 / num_points)

        #return values
        dFcw = np.zeros((3,num_basis_fcns,num_points))

        #temporal container for curl w
        curl_w_tmp = np.zeros((3,num_basis_fcns))

        for i in range(num_points):

            curl_w_tmp[0,:] = curl_w_hat[0 + i*3*num_basis_fcns:(i+1)*3*num_basis_fcns:3]
            curl_w_tmp[1,:] = curl_w_hat[1 + i*3*num_basis_fcns:(i+1)*3*num_basis_fcns:3]
            curl_w_tmp[2,:] = curl_w_hat[2 + i*3*num_basis_fcns:(i+1)*3*num_basis_fcns:3]

            dFcw[:,:,i] = np.array([[ jacobians[i*9]   , jacobians[i*9+3] , jacobians[i*9+6]],
                                    [ jacobians[i*9+1] , jacobians[i*9+4] , jacobians[i*9+7]],
                                    [ jacobians[i*9+2] , jacobians[i*9+5] , jacobians[i*9+8]]]) @ curl_w_tmp

        return dFcw


    def compute_stiffness_matrix(self, quad_order=8):
        '''Compute the stiffness matrix using edge elements.
        Can at the moment only handle meshes of a single element type.

        :param quad_order:
            The order of the quadrature rule.

        :return:
            The sparse stiffness matrix.
        '''


        print('compute stiffness matrix curl curl...')

        # make the sparse stiffness matrix
        K = csr_array((self.num_dofs, self.num_dofs))
    
        # loop over all materials
        for n in range(self.num_materials):
            
            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get also the permeability in this domain
            mat_prop = self.material_list[n]
            
            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the mesh connectivity
                c = cells[i].copy()
                c.shape = (num_el, el_nodes)

                # get the quadrature nodes and weights
                w, q = finite_element.get_quadrature_rule(quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the curls of the edge shape functions at the integration points
                curls, num_orientations = finite_element.get_curl_edge_basis_functions(self.mesh, q, ct, self.element_order)

                # evaluate the derivatives of the lagrange basis functions at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # get the jacobians, determinants and orientations
                # _, _, _, orientations = get_vector_basis_mesh_info(self.mesh, q)
                jacobians, determinants, coordinates, orientations = get_vector_basis_mesh_info(self.mesh, q)

                # get the global ids
                glob_ids = self.global_ids[n]

                # the number of DoFs for the edge elements per finite element
                num_el_dofs = get_num_edge_dofs(self.element_order)

                # the number of triplets
                num_triplets = num_el*num_el_dofs*num_el_dofs

                # allocate the space for the triplet list
                ij = np.zeros((num_triplets, 2), dtype=np.int32)
                vals = np.zeros((num_triplets, ))

                # a triplet counter
                t_cnt = 0

                # loop over the finite elements
                for j, e in enumerate(c):

                    # compute the Jacobian at the integration points
                    # inv_J is a numpy array of dimension (M x 3 x 3) where
                    # the inverse of the Jacobians are stored in the second and
                    # third dimensions
                    # M is the number of integration points
                    # det_J is an (M x 0) numpy array which stores the M values of det(J)
                    # J = finite_element.compute_J(e - 1, self.nodes, d_phi)
                    
                    # inv_J = finite_element.compute_J_inv(J)
                    # det_J = finite_element.compute_J_det(J)

                    # this offset is the first index in the shape functions list for this orientation
                    offset_shape_function = 3*num_quad_pts*num_el_dofs*orientations[0, j]

                    # get the curls of the basis functions with this orientations
                    curl_w_hat = curls[3*orientations[0, j]*num_quad_pts*num_el_dofs:3*(orientations[0, j]+1)*num_quad_pts*num_el_dofs]

                    # print('Jac 1 = {}'.format(jacobians[j, :9]))
                    # print('Jac 2 = {}'.format(J[0, :, :].T.flatten()))

                    # compute the products dF.curl(w_hat)
                    dFcw = self.compute_dF_curl_w(curl_w_hat, jacobians[j, :])
                    # dFcw = self.compute_dF_curl_w(curl_w_hat, J.flatten())

                    # to do:
                    # compute nu

                    #compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
                    for k in range(num_el_dofs):

                        # curl_w_hat_k = np.array([])
                        for l  in range(k, num_el_dofs):
                            
                            # curl_w_hat_l = np.array([])

                            # compute v.T A v
                            vAv = np.sum(dFcw[:, k, :] * dFcw[:, l, :], axis = 0)
                            this_prod = np.sum(vAv*w/determinants[j, :])
                            # this_prod = np.sum(vAv*w/det_J)


                            # fill the triplet list
                            ij[t_cnt, 0] = glob_ids[j, k] - 1
                            ij[t_cnt, 1] = glob_ids[j, l] - 1
                            vals[t_cnt] = this_prod
                            t_cnt += 1

                            # apply symmetry
                            if k != l:
                                ij[t_cnt, 0] = glob_ids[j, l] - 1
                                ij[t_cnt, 1] = glob_ids[j, k] - 1
                                vals[t_cnt] = this_prod
                                t_cnt += 1

            K += csr_array((vals, (ij[:, 0], ij[:, 1])), shape=(self.num_dofs, self.num_dofs))

        return K
    
    def compute_stiffness_matrix_alt(self, quad_order=8):
        '''Compute the stiffness matrix using edge elements.
        Can at the moment only handle meshes of a single element type.

        :param quad_order:
            The order of the quadrature rule.

        :return:
            The sparse stiffness matrix.
        '''


        print('compute stiffness matrix curl curl...')

        # all triplets
        all_ij = np.zeros((0, 2), dtype=np.int64)
        all_vals = np.zeros((0, ), dtype=float)

        # make the sparse stiffness matrix
        K = csr_array((self.num_dofs, self.num_dofs))
    
        # loop over all materials
        for n in range(self.num_materials):
            
            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get the global orientations of the finite elements
            orientations = get_vector_basis_orientations(self.mesh,
                                                         self.cell_tags[n],
                                                         element_spec='CurlHcurlLegendre' + str(self.element_order-1))
            
            # get also the permeability in this domain
            mat_prop = self.material_list[n]
            
            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the mesh connectivity
                c = cells[i].copy()
                c.shape = (num_el, el_nodes)

                # get the quadrature nodes and weights
                w, q = finite_element.get_quadrature_rule(quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the curls of the edge shape functions at the integration points
                curls, num_orientations = finite_element.get_curl_edge_basis_functions(self.mesh, q, ct, self.element_order)

                # evaluate the derivatives of the lagrange basis functions at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # get the global ids
                glob_ids = self.global_ids[n]

                # the number of DoFs for the edge elements per finite element
                num_el_dofs = get_num_edge_dofs(self.element_order)

                # the number of triplets
                num_triplets = num_el*num_el_dofs*num_el_dofs

                # allocate the space for the triplet list
                ij = np.zeros((num_triplets, 2), dtype=np.int32)
                vals = np.zeros((num_triplets, ))

                # a triplet counter
                t_cnt = 0

                # loop over the finite elements
                for j, e in enumerate(c):


                    # compute the Jacobian at the integration points
                    # inv_J is a numpy array of dimension (M x 3 x 3) where
                    # the inverse of the Jacobians are stored in the second and
                    # third dimensions
                    # M is the number of integration points
                    # det_J is an (M x 0) numpy array which stores the M values of det(J)
                    J = finite_element.compute_J(e - 1, self.nodes, d_phi)
                    
                    # inv_J = finite_element.compute_J_inv(J)
                    det_J = finite_element.compute_J_det(J)

                    # this offset is the first index in the shape functions list for this orientation
                    offset_shape_function = 3*num_quad_pts*num_el_dofs*orientations[j]

                    # get the curls of the basis functions with this orientations
                    curl_w_hat = curls[3*orientations[j]*num_quad_pts*num_el_dofs:3*(orientations[j]+1)*num_quad_pts*num_el_dofs]

                    # to do:
                    # compute nu

                    # if 166 in glob_ids[j, :] :
                    #     print('e = {}'.format(e))
                    #     print('curl_w_hat = {}'.format(curl_w_hat))
                    #     print('glob_ids[j, :] = {}'.format(glob_ids[j, :]))
                    #     print('det_J = {}'.format(det_J))

                    #compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
                    for k in range(num_el_dofs):

                        for l  in range(k, num_el_dofs):
                            
                            # the integration value
                            int_val = 0.0

                            for m in range(num_quad_pts):

                                
                                curl_w_hat_k = np.array([curl_w_hat[3*(m*num_el_dofs + k)     ],
                                                         curl_w_hat[3*(m*num_el_dofs + k) + 1 ],
                                                         curl_w_hat[3*(m*num_el_dofs + k) + 2 ]])

                                curl_w_hat_l = np.array([curl_w_hat[3*(m*num_el_dofs + l)     ],
                                                         curl_w_hat[3*(m*num_el_dofs + l) + 1 ],
                                                         curl_w_hat[3*(m*num_el_dofs + l) + 2 ]])
                                
                                dFcw_k = J[m, :, :] @ curl_w_hat_k
                                dFcw_l = J[m, :, :] @ curl_w_hat_l

                                int_val += np.sum(dFcw_k*dFcw_l)*w[m]/det_J[m]*mat_prop


                            # fill the triplet list
                            ij[t_cnt, 0] = glob_ids[j, k] - 1
                            ij[t_cnt, 1] = glob_ids[j, l] - 1
                            vals[t_cnt] = int_val
                            t_cnt += 1

                            # apply symmetry
                            if k != l:
                                ij[t_cnt, 0] = glob_ids[j, l] - 1
                                ij[t_cnt, 1] = glob_ids[j, k] - 1
                                vals[t_cnt] = int_val
                                t_cnt += 1

            all_ij = np.append(all_ij, ij, axis=0)
            all_vals = np.append(all_vals, vals)

        K = csr_array((all_vals, (all_ij[:, 0], all_ij[:, 1])), shape=(self.num_dofs, self.num_dofs))

        return K


    def compute_stiffness_matrix_alt_2(self, quad_order=8):
        '''Compute the stiffness matrix using edge elements.
        Can at the moment only handle meshes of a single element type.

        :param quad_order:
            The order of the quadrature rule.

        :return:
            The sparse stiffness matrix.
        '''


        print('compute stiffness matrix curl curl...')

        # all triplets
        all_ij = np.zeros((0, 2), dtype=np.int64)
        all_vals = np.zeros((0, ), dtype=float)

        # make the sparse stiffness matrix
        K = csr_array((self.num_dofs, self.num_dofs))
    
        # loop over all materials
        for n in range(self.num_materials):
            
            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get the global orientations of the finite elements
            orientations = get_vector_basis_orientations(self.mesh,
                                                         self.cell_tags[n],
                                                         element_spec='CurlHcurlLegendre' + str(self.element_order-1))
            
            # get also the permeability in this domain
            reluctance = self.material_list[n]

            nu = reluctance.evaluate_nu(0.0)
            
            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the mesh connectivity
                c = cells[i].copy()
                c.shape = (num_el, el_nodes)

                # get the quadrature nodes and weights
                w, q = finite_element.get_quadrature_rule(quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the curls of the edge shape functions at the integration points
                curls, num_orientations = finite_element.get_curl_edge_basis_functions(self.mesh, q, ct, self.element_order)

                # evaluate the derivatives of the lagrange basis functions at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # get the global ids
                glob_ids = self.global_ids[n]

                # the number of DoFs for the edge elements per finite element
                num_el_dofs = get_num_edge_dofs(self.element_order)

                # the number of triplets
                num_triplets = num_el*num_el_dofs*num_el_dofs

                # allocate the space for the triplet list
                ij = np.zeros((num_triplets, 2), dtype=np.int32)
                vals = np.zeros((num_triplets, ))

                # a triplet counter
                t_cnt = 0

                # loop over the finite elements
                for j, e in enumerate(c):


                    # compute the Jacobian at the integration points
                    # inv_J is a numpy array of dimension (M x 3 x 3) where
                    # the inverse of the Jacobians are stored in the second and
                    # third dimensions
                    # M is the number of integration points
                    # det_J is an (M x 0) numpy array which stores the M values of det(J)
                    J = finite_element.compute_J(e - 1, self.nodes, d_phi)
                    
                    # inv_J = finite_element.compute_J_inv(J)
                    det_J = finite_element.compute_J_det(J)

                    # this offset is the first index in the shape functions list for this orientation
                    offset_shape_function = 3*num_quad_pts*num_el_dofs*orientations[j]

                    # get the curls of the basis functions with this orientations
                    curl_w_hat = curls[3*orientations[j]*num_quad_pts*num_el_dofs:3*(orientations[j]+1)*num_quad_pts*num_el_dofs]

                    # to do:
                    # compute nu

                    # if 166 in glob_ids[j, :] :
                    #     print('e = {}'.format(e))
                    #     print('curl_w_hat = {}'.format(curl_w_hat))
                    #     print('glob_ids[j, :] = {}'.format(glob_ids[j, :]))
                    #     print('det_J = {}'.format(det_J))

                    #compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
                    for k in range(num_el_dofs):

                        for l  in range(k, num_el_dofs):
                            
                            # the integration value
                            int_val = 0.0

                            for m in range(num_quad_pts):

                                
                                curl_w_hat_k = np.array([curl_w_hat[3*(m*num_el_dofs + k)     ],
                                                         curl_w_hat[3*(m*num_el_dofs + k) + 1 ],
                                                         curl_w_hat[3*(m*num_el_dofs + k) + 2 ]])

                                curl_w_hat_l = np.array([curl_w_hat[3*(m*num_el_dofs + l)     ],
                                                         curl_w_hat[3*(m*num_el_dofs + l) + 1 ],
                                                         curl_w_hat[3*(m*num_el_dofs + l) + 2 ]])
                                
                                dFcw_k = J[m, :, :] @ curl_w_hat_k
                                dFcw_l = J[m, :, :] @ curl_w_hat_l

                                int_val += np.sum(dFcw_k*dFcw_l)*w[m]/det_J[m]*nu


                            # fill the triplet list
                            ij[t_cnt, 0] = glob_ids[j, k] - 1
                            ij[t_cnt, 1] = glob_ids[j, l] - 1
                            vals[t_cnt] = int_val
                            t_cnt += 1

                            # apply symmetry
                            if k != l:
                                ij[t_cnt, 0] = glob_ids[j, l] - 1
                                ij[t_cnt, 1] = glob_ids[j, k] - 1
                                vals[t_cnt] = int_val
                                t_cnt += 1

            all_ij = np.append(all_ij, ij, axis=0)
            all_vals = np.append(all_vals, vals)

        K = csr_array((all_vals, (all_ij[:, 0], all_ij[:, 1])), shape=(self.num_dofs, self.num_dofs))

        return K

    def compute_stiffness_and_jacobian_matrix(self, x, quad_order=3):
        '''Compute the stiffness matrix using edge elements.
        Can at the moment only handle meshes of a single element type.

        :param x:
            The solution vector at which the jacobian is calculated.

        :param quad_order:
            The order of the quadrature rule.

        :return:
            The sparse stiffness matrix.
        '''


        print('compute stiffness and jacobian matrix curl curl...')

        # all triplets
        all_ij = np.zeros((0, 2), dtype=np.int64)
        all_vals_K = np.zeros((0, ), dtype=float)
        all_vals_dK = np.zeros((0, ), dtype=float)

        # make the sparse stiffness matrix
        K = csr_array((self.num_dofs, self.num_dofs))
    
        # make the sparse jacobi matrix
        dK = csr_array((self.num_dofs, self.num_dofs))

        # loop over all materials
        for n in range(self.num_materials):
            
            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get the global orientations of the finite elements
            orientations = get_vector_basis_orientations(self.mesh,
                                                         self.cell_tags[n],
                                                         element_spec='CurlHcurlLegendre' + str(self.element_order-1))
            
            # get also the permeability in this domain
            reluctance = self.material_list[n]
            
            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the mesh connectivity
                c = cells[i].copy()
                c.shape = (num_el, el_nodes)

                # get the quadrature nodes and weights
                w, q = finite_element.get_quadrature_rule(quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the curls of the edge shape functions at the integration points
                curls, num_orientations = finite_element.get_curl_edge_basis_functions(self.mesh, q, ct, self.element_order)

                # evaluate the derivatives of the lagrange basis functions at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # get the global ids
                glob_ids = self.global_ids[n]

                # the number of DoFs for the edge elements per finite element
                num_el_dofs = get_num_edge_dofs(self.element_order)

                # the number of triplets
                num_triplets = num_el*num_el_dofs*num_el_dofs

                # allocate the space for the triplet list for K
                ij = np.zeros((num_triplets, 2), dtype=np.int32)
                vals_K = np.zeros((num_triplets, ))

                # allocate the space for the triplet list (J)
                vals_dK = np.zeros((num_triplets, ))

                # a triplet counter
                t_cnt = 0

                # loop over the finite elements
                for j, e in enumerate(c):


                    # compute the Jacobian at the integration points
                    # inv_J is a numpy array of dimension (M x 3 x 3) where
                    # the inverse of the Jacobians are stored in the second and
                    # third dimensions
                    # M is the number of integration points
                    # det_J is an (M x 0) numpy array which stores the M values of det(J)
                    J = finite_element.compute_J(e - 1, self.nodes, d_phi)
                    
                    # inv_J = finite_element.compute_J_inv(J)
                    det_J = finite_element.compute_J_det(J)

                    # this offset is the first index in the shape functions list for this orientation
                    offset_shape_function = 3*num_quad_pts*num_el_dofs*orientations[j]

                    # get the curls of the basis functions with this orientations
                    curl_w_hat = curls[3*orientations[j]*num_quad_pts*num_el_dofs:3*(orientations[j]+1)*num_quad_pts*num_el_dofs]

                    B_vec = self.compute_B_in_element(glob_ids[j, :], x, curl_w_hat, J, det_J)

                    # compute also the norm
                    B_mag = np.linalg.norm(B_vec, axis=1)

                    # compute the reluctance
                    nu = reluctance.evaluate_nu(B_mag)

                    # compute the permebility derivative
                    d_nu = reluctance.evaluate_nu_derivative(B_mag)

                    #compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
                    for k in range(num_el_dofs):

                        for l  in range(k, num_el_dofs):
                            
                            # the integration values
                            int_val_K = 0.0
                            int_val_dK = 0.0

                            for m in range(num_quad_pts):

                                
                                curl_w_hat_k = np.array([curl_w_hat[3*(m*num_el_dofs + k)     ],
                                                         curl_w_hat[3*(m*num_el_dofs + k) + 1 ],
                                                         curl_w_hat[3*(m*num_el_dofs + k) + 2 ]])

                                curl_w_hat_l = np.array([curl_w_hat[3*(m*num_el_dofs + l)     ],
                                                         curl_w_hat[3*(m*num_el_dofs + l) + 1 ],
                                                         curl_w_hat[3*(m*num_el_dofs + l) + 2 ]])
                                
                                dFcw_k = J[m, :, :] @ curl_w_hat_k
                                dFcw_l = J[m, :, :] @ curl_w_hat_l

                                int_val_K += nu[m]*np.sum(dFcw_k*dFcw_l)*w[m]/det_J[m]

                                # increment the integration value
                                if B_mag[m] >= 1e-14:
                                    int_val_dK += d_nu[m]*np.sum(dFcw_k*B_vec[m, :])*np.sum(dFcw_l*B_vec[m, :])*w[m]/det_J[m]/B_mag[m]

                            # fill the triplet list
                            ij[t_cnt, 0] = glob_ids[j, k] - 1
                            ij[t_cnt, 1] = glob_ids[j, l] - 1
                            vals_K[t_cnt] = int_val_K
                            vals_dK[t_cnt] = int_val_dK

                            t_cnt += 1

                            # apply symmetry
                            if k != l:
                                ij[t_cnt, 0] = glob_ids[j, l] - 1
                                ij[t_cnt, 1] = glob_ids[j, k] - 1
                                vals_K[t_cnt] = int_val_K
                                vals_dK[t_cnt] = int_val_dK

                                t_cnt += 1

            all_ij = np.append(all_ij, ij, axis=0)
            all_vals_K = np.append(all_vals_K, vals_K)
            all_vals_dK = np.append(all_vals_dK, vals_dK)

        K = csr_array((all_vals_K, (all_ij[:, 0], all_ij[:, 1])), shape=(self.num_dofs, self.num_dofs))
        dK = csr_array((all_vals_dK, (all_ij[:, 0], all_ij[:, 1])), shape=(self.num_dofs, self.num_dofs))

        return K, dK + K

    def compute_stiffness_and_jacobi_matrix_c(self, x, quad_order=2, source_fields=[]):
        '''Compute the stiffness and the jacobi matrix for the magnetic vector potential formulation for
        nonlinear problems.
        This implementation is the fast version of the above.
        
        :param x:
            The solution vector.

        :param quad_order:
            The quadrature order. Default 2.
    
        :param source_fields:
            A list of numpy arrays, one for each domain. The list specifies the source fields
            for the reduced vector potential formulation.
            If the list is empty, the source field is ignored.

        :return:
            The stiffness matrix.
        '''

        # all triplets
        all_ij = np.zeros((0, 2), dtype=np.int64)
        all_vals_K = np.zeros((0, ), dtype=float)
        all_vals_dK = np.zeros((0, ), dtype=float)

        # make the sparse stiffness matrix
        K = csr_array((self.num_dofs, self.num_dofs))
    
        # make the sparse jacobi matrix
        dK = csr_array((self.num_dofs, self.num_dofs))

        # check if source field list is valid
        use_source_fields = False
        if len(source_fields) == self.num_materials:
            use_source_fields = True

        # make space for the rhs
        rhs = np.zeros((self.num_dofs, ))

        # loop over all materials
        for n in range(self.num_materials):
            
            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get the global orientations of the finite elements
            orientations = get_vector_basis_orientations(self.mesh,
                                                         self.cell_tags[n],
                                                         element_spec='CurlHcurlLegendre' + str(self.element_order-1))
            
            # get also the permeability in this domain
            reluctance = self.material_list[n]
            
            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the mesh connectivity
                c = cells[i].copy()
                c.shape = (num_el, el_nodes)

                # get the quadrature nodes and weights
                w, q = finite_element.get_quadrature_rule(quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the curls of the edge shape functions at the integration points
                curls, num_orientations = finite_element.get_curl_edge_basis_functions(self.mesh, q, ct, self.element_order)

                # evaluate the derivatives of the lagrange basis functions at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # get the global ids
                glob_ids = self.global_ids[n]

                # the number of DoFs for the edge elements per finite element
                num_el_dofs = get_num_edge_dofs(self.element_order)

                # the number of triplets
                num_triplets = num_el*num_el_dofs*num_el_dofs

                if use_source_fields:
                    if len(source_fields[n]) == 0:
                        B_s = np.zeros((0, 3))
                    else:
                        B_s = source_fields[n]
                else:
                    B_s = np.zeros((0, 3))

                # call the cpp code (gmsh starts counting nodes at 1)
                this_K, this_dK, this_rhs = fem_c.compute_stiffness_and_jacobi_matrix_curl_curl(self.nodes,
                                                                            cells[i] - 1,
                                                                            glob_ids - 1,
                                                                            curls,
                                                                            d_phi,
                                                                            orientations,
                                                                            x,
                                                                            B_s,
                                                                            reluctance,
                                                                            q,
                                                                            w,
                                                                            self.num_dofs,
                                                                            num_el_dofs)
                K += this_K
                dK += this_dK
                rhs += this_rhs

        return K, dK + K, rhs

    def compute_fcn_rhs(self, J_fcn, quad_order=8):
        '''Compute the rhs for solving a magnetostatic problem using edge elements.
        Can at the moment only handle meshes of a single element type.
        This function is based on a current density function J_fcn.
        
        :param J_fcn:
            A function specifying the rhs.
        
        :param quad_order:
            The order of the quadrature rule.

        :return:
            The sparse stiffness matrix.
        '''

        print('compute rhs...')

        # make the sparse stiffness matrix
        rhs = np.zeros((self.num_dofs, ))

        
        # loop over all materials
        for n in range(self.num_materials):
            
            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get the global orientations of the finite elements
            orientations = get_vector_basis_orientations(self.mesh,
                                                         self.cell_tags[n],
                                                         element_spec='HcurlLegendre' + str(self.element_order-1))    

            # get also the permeability in this domain
            mat_prop = self.material_list[n]
            
            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the mesh connectivity
                c = cells[i].copy()
                c.shape = (num_el, el_nodes)

                # get the quadrature nodes and weights
                w, q = finite_element.get_quadrature_rule(quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the edge shape functions at the integration points
                basis, num_orientations = finite_element.get_edge_basis_functions(self.mesh, q, ct, self.element_order)

                # evaluate the shape functions at these points (not needed)
                phi = finite_element.evaluate_basis(q)

                # evaluate the derivatives of the lagrange basis functions at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # get the global ids
                glob_ids = self.global_ids[n]

                # the number of DoFs for the edge elements per finite element
                num_el_dofs = get_num_edge_dofs(self.element_order)

                    
                # loop over the finite elements
                for j, e in enumerate(c):

                    # compute the Jacobian at the integration points
                    # inv_J is a numpy array of dimension (M x 3 x 3) where
                    # the inverse of the Jacobians are stored in the second and
                    # third dimensions
                    # M is the number of integration points
                    # det_J is an (M x 0) numpy array which stores the M values of det(J)
                    J = finite_element.compute_J(e - 1, self.nodes, d_phi)
                
                    inv_J = finite_element.compute_J_inv(J)
                    det_J = finite_element.compute_J_det(J)

                    # this offset is the first index in the shape functions list for this orientation
                    offset_shape_function = 3*num_quad_pts*num_el_dofs*orientations[j]

                    # get the curls of the basis functions with this orientations
                    w_hat = basis[3*orientations[j]*num_quad_pts*num_el_dofs:3*(orientations[j]+1)*num_quad_pts*num_el_dofs]
                    

                    # evaluate this finite element for the global position
                    points = finite_element.evaluate(e - 1, self.nodes, phi)

                    # evaluate current density
                    j_eval = J_fcn(points)

                    # compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
                    for k in range(num_el_dofs):

                        # the integration value
                        int_val = 0.0

                        for m in range(num_quad_pts):

                            # the basis function in local coordinates
                            w_hat_k = np.array([w_hat[3*(m*num_el_dofs + k)     ],
                                                     w_hat[3*(m*num_el_dofs + k) + 1 ],
                                                     w_hat[3*(m*num_el_dofs + k) + 2 ]])

                            # the basis function in global coordinates
                            dFcw_k = inv_J[m, :, :].T @ w_hat_k

                            
                            # the product J.w_tilde
                            rhs[glob_ids[j, k] - 1] += np.sum(dFcw_k*j_eval[m, :])*w[m]*det_J[m]
    


        return rhs

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

    def compute_B_in_element(self, glob_ids, x, curl_w_hat, J, det_J):
        '''Compute the magnetic flux density in a finite element, given the solution vector
        and the curls of the basis functions.

        :param glob_ids:
            The cell global edge basis function identifiers.

        :param x:
            The solution vector.

        :param curl_w_hat:
            The curl of the basis functions evaluated at the interpolation points.

        :param J:
            The jacobian matrices evaluated at the interpolation points.

        :param det_J:
            The determinants of the jacobians evaluated at the interpolation points.

        :return:
            The B field vectors at the interpolation points.
        '''

        # the number of interpolation points
        num_points = len(det_J)

        # the number of basis functions
        num_el_dofs = int(len(curl_w_hat)/num_points/3)

        # make space for the return vector
        B_ret = np.zeros((num_points, 3))

        #compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)

        for m in range(num_points):

            for k in range(num_el_dofs):

                # the basis function in local coordinates
                curl_w_hat_k = np.array([curl_w_hat[3*(m*num_el_dofs + k)     ],
                                         curl_w_hat[3*(m*num_el_dofs + k) + 1 ],
                                         curl_w_hat[3*(m*num_el_dofs + k) + 2 ]])



                # the B field vector in global coordinates
                B_ret[m, :] += J[m, :, :] @ curl_w_hat_k * x[glob_ids[k] - 1] / det_J[m]

        return B_ret
    
    def compute_rhs_electric_potential(self, x_p, quad_order=8, select_mat='all'):
        '''Compute the rhs for solving a magnetostatic problem using edge elements.
        Can at the moment only handle meshes of a single element type.
        This function is based on an electric potential discretized on the same mesh.
        
        :param x_p:
            The nodal values for the electric potential.
        
        :param quad_order:
            The order of the quadrature rule.

        :param select_mat:
            Select the material. Default 'all' means all materials.

        :return:
            The sparse stiffness matrix.
        '''

        print('compute rhs...')

        # check if material selection is a string
        if isinstance(select_mat, str):
            if select_mat == 'all':
                materials = range(self.num_mat)
        else:
            materials = select_mat

        # make the sparse stiffness matrix
        rhs = np.zeros((self.num_dofs, ))
    
        # loop over all materials
        for n in materials:
            
            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get also the permeability in this domain
            mat_prop = self.material_list[n]
            
            # get the global orientations of the finite elements
            orientations = get_vector_basis_orientations(self.mesh,
                                                         self.cell_tags[n],
                                                         element_spec='HcurlLegendre' + str(self.element_order-1))    
            
            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the mesh connectivity
                c = cells[i].copy()
                c.shape = (num_el, el_nodes)

                # get the quadrature nodes and weights
                w, q = finite_element.get_quadrature_rule(quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the edge shape functions at the integration points
                basis, num_orientations = finite_element.get_edge_basis_functions(self.mesh, q, ct, self.element_order)

                # evaluate the shape functions at these points (not needed)
                phi = finite_element.evaluate_basis(q)

                # evaluate the derivatives of the lagrange basis functions at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # get the global ids
                glob_ids = self.global_ids[n]

                # the number of DoFs for the edge elements per finite element
                num_el_dofs = get_num_edge_dofs(self.element_order)

                # loop over the finite elements
                for j, e in enumerate(c):


                    # compute the Jacobian at the integration points
                    # inv_J is a numpy array of dimension (M x 3 x 3) where
                    # the inverse of the Jacobians are stored in the second and
                    # third dimensions
                    # M is the number of integration points
                    # det_J is an (M x 0) numpy array which stores the M values of det(J)
                    J = finite_element.compute_J(e - 1, self.nodes, d_phi)
                
                    inv_J = finite_element.compute_J_inv(J)
                    det_J = finite_element.compute_J_det(J)

                    # this offset is the first index in the shape functions list for this orientation
                    offset_shape_function = 3*num_quad_pts*num_el_dofs*orientations[j]

                    # get the curls of the basis functions with this orientations
                    w_hat = basis[3*orientations[j]*num_quad_pts*num_el_dofs:3*(orientations[j]+1)*num_quad_pts*num_el_dofs]
                    
                    # evaluate this finite element for the global position
                    points = finite_element.evaluate(e - 1, self.nodes, phi)

                    # evaluate current density
                    j_eval, _ = self.compute_grad_phi(e - 1, x_p, d_phi, inv_J)

                    # compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
                    for k in range(num_el_dofs):

                        # the integration value
                        int_val = 0.0

                        for m in range(num_quad_pts):

                            # the basis function in local coordinates
                            w_hat_k = np.array([w_hat[3*(m*num_el_dofs + k)     ],
                                                     w_hat[3*(m*num_el_dofs + k) + 1 ],
                                                     w_hat[3*(m*num_el_dofs + k) + 2 ]])

                            # the basis function in global coordinates
                            dFcw_k = inv_J[m, :, :].T @ w_hat_k

                            
                            # the product J.w_tilde
                            rhs[glob_ids[j, k] - 1] += np.sum(dFcw_k*j_eval[m, :])*w[m]*det_J[m]
    

        return rhs

    def compute_rhs_reduced_vector_potential(self, source_fields, domain_ids, quad_order, x=np.array([])):
        '''Compute the rhs for solving a magnetostatic problem using edge elements and the reduced
        vector potential formulation.

        :param source_fields:
            A list of numpy arrays with the precomputed source fields (flux densities) at the integration points.
        
        :param domain_ids:
            A list of integers specifying the domain identifyers.

        :param quad_order:
            The order of the quadrature rule.
        
        :param x:
            The initial solution. If empty, x=0 is assumed.

        :return:
            A numpy array with the right hand side.
        '''

        print('compute rhs...')

        # make the sparse stiffness matrix
        rhs = np.zeros((self.num_dofs, ))

        if len(x) == 0:
            x = np.zeros((self.num_dofs, ))
    
        # loop over all materials
        for i_n, n in enumerate(domain_ids):
            
            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get also the permeability in this domain
            mat_prop = self.material_list[n]
            
            # get the global orientations of the finite elements
            orientations = get_vector_basis_orientations(self.mesh,
                                                         self.cell_tags[n],
                                                         element_spec='CurlHcurlLegendre' + str(self.element_order-1))    
            
            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the mesh connectivity
                c = cells[i].copy()
                c.shape = (num_el, el_nodes)

                # get the quadrature nodes and weights
                w, q = finite_element.get_quadrature_rule(quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the curls of the edge shape functions at the integration points
                curls, num_orientations = finite_element.get_curl_edge_basis_functions(self.mesh, q, ct, self.element_order)

                # evaluate the shape functions at these points (not needed)
                phi = finite_element.evaluate_basis(q)

                # evaluate the derivatives of the lagrange basis functions at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # get the global ids
                glob_ids = self.global_ids[n]

                # the number of DoFs for the edge elements per finite element
                num_el_dofs = get_num_edge_dofs(self.element_order)

                # get the values of the source field
                H_s = source_fields[i_n]/4/np.pi*1e7

                # loop over the finite elements
                for j, e in enumerate(c):


                    # compute the Jacobian at the integration points
                    # inv_J is a numpy array of dimension (M x 3 x 3) where
                    # the inverse of the Jacobians are stored in the second and
                    # third dimensions
                    # M is the number of integration points
                    # det_J is an (M x 0) numpy array which stores the M values of det(J)
                    J = finite_element.compute_J(e - 1, self.nodes, d_phi)
                
                    inv_J = finite_element.compute_J_inv(J)
                    det_J = finite_element.compute_J_det(J)

                    # this offset is the first index in the shape functions list for this orientation
                    offset_shape_function = 3*num_quad_pts*num_el_dofs*orientations[j]

                    # get the curls of the basis functions with this orientations
                    curl_w_hat = curls[3*orientations[j]*num_quad_pts*num_el_dofs:3*(orientations[j]+1)*num_quad_pts*num_el_dofs]
                    # print(' j = {}'.format(j))

                    # compute the curl of Ar in the finite element
                    curl_Ar = self.compute_B_in_element(glob_ids[j, :], x, curl_w_hat, J, det_J)

                    # compute the B field
                    B_vec = curl_Ar + 4*np.pi*1e-7*H_s[j*num_quad_pts:(j+1)*num_quad_pts, :]

                    # compute also the norm
                    B_mag = np.linalg.norm(B_vec, axis=1)

                    # compute the reluctance
                    nu = mat_prop.evaluate_nu(B_mag)

                    # compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
                    for k in range(num_el_dofs):

                        # the integration value
                        int_val = 0.0

                        for m in range(num_quad_pts):

                            # the curl of the basis function in local coordinates
                            curl_w_hat_k = np.array([curl_w_hat[3*(m*num_el_dofs + k)     ],
                                                     curl_w_hat[3*(m*num_el_dofs + k) + 1 ],
                                                     curl_w_hat[3*(m*num_el_dofs + k) + 2 ]])

                            # the curl of the basis function in global coordinates
                            curl_w_k = J[m, :, :] @ curl_w_hat_k / det_J[m]

                            # the product J.w_tilde
                            rhs[glob_ids[j, k] - 1] += (1.0 - nu[m]*4*np.pi*1e-7)*np.sum(curl_w_k*H_s[j*num_quad_pts+m, :])*w[m]*det_J[m]
    

        return rhs
    
    def compute_rhs_reduced_vector_potential_c(self, source_fields, domain_ids, quad_order, x=np.array([])):
        '''Compute the rhs for solving a magnetostatic problem using edge elements and the reduced
        vector potential formulation. This code runs the fast C code implementation of the above.
        
        :param source_fields:
            A list of numpy arrays with the precomputed source fields (flux densities) at the integration points.
        
        :param domain_ids:
            A list of integers specifying the domain identifyers.

        :param quad_order:
            The order of the quadrature rule.

        :param x:
            The initial solution. If empty, x=0 is assumed.

        :return:
            A numpy array with the right hand side.
        '''

        print('compute rhs...')

        # make the sparse stiffness matrix
        rhs = np.zeros((self.num_dofs, ))
    
        if len(x) == 0:
            x = np.zeros((self.num_dofs, ))

        # loop over all materials
        for i_n, n in enumerate(domain_ids):
            
            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get also the permeability in this domain
            reluctance = self.material_list[n]
            
            # get the global orientations of the finite elements
            orientations = get_vector_basis_orientations(self.mesh,
                                                         self.cell_tags[n],
                                                         element_spec='CurlHcurlLegendre' + str(self.element_order-1))    
            
            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the mesh connectivity
                c = cells[i].copy()
                c.shape = (num_el, el_nodes)

                # get the quadrature nodes and weights
                w, q = finite_element.get_quadrature_rule(quad_order)

                # evaluate the curls of the edge shape functions at the integration points
                curls, num_orientations = finite_element.get_curl_edge_basis_functions(self.mesh, q, ct, self.element_order)

                # evaluate the shape functions at these points (not needed)
                phi = finite_element.evaluate_basis(q)

                # evaluate the derivatives of the lagrange basis functions at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # get the global ids
                glob_ids = self.global_ids[n]

                # the number of DoFs for the edge elements per finite element
                num_el_dofs = get_num_edge_dofs(self.element_order)

                # launch the c code assembler
                rhs += fem_c.compute_rhs_Hcurl_red_c(self.nodes,
                                                    cells[i] - 1,
                                                    glob_ids - 1,
                                                    curls,
                                                    d_phi,
                                                    orientations,
                                                    source_fields[i_n],
                                                    reluctance,
                                                    x, 
                                                    q,
                                                    w,
                                                    self.num_dofs,
                                                    num_el_dofs)
                
                

        return rhs

    def get_quadrature_points(self, quad_order=3, domain_ids=[]):
        '''Get the quadrature points in the domain.

        :param quad_order:
            The order of the quadrature rule.

        :param domain_ids:
            A list of domain identifyers. If empty all domains are evaluated.

        :return:
            A list with the points in an (M x 3) array for each domain.

        '''

        # make the return field evaluation points
        points_ret = []

        # marker for the domains
        marker = np.zeros((0, ))

        # the list of domain indices
        n_list = range(self.num_materials)

        if len(domain_ids) > 0:
            n_list = domain_ids

        # loop over all materials
        for n in n_list:
            
            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get also the permeability in this domain
            mat_prop = self.material_list[n]
            
            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the mesh connectivity
                c = cells[i].copy()
                c.shape = (num_el, el_nodes)

                # get the quadrature nodes and weights
                w, q = finite_element.get_quadrature_rule(quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the shape functions at these points (not needed)
                phi = finite_element.evaluate_basis(q)

                # field evaluation points for this material
                this_points = np.zeros((num_el*num_quad_pts, 3))

                # loop over the finite elements
                for j, e in enumerate(c):

                    # evaluate this finite element for the global position
                    this_points[j*num_quad_pts:(j+1)*num_quad_pts, :] = finite_element.evaluate(e - 1, self.nodes, phi)

                points_ret.append(this_points)
    
        return points_ret

    def compute_B_py(self, x, quad_order=8):
        '''Compute the B for a given solution vector in python (slow)
        
        :param x:
            The solution vector.
        
        :param quad_order:
            The order of the quadrature rule.

        :return:
            The sparse stiffness matrix.
        '''

        print('compute B...')

        # make the B field return vector
        B_ret = np.zeros((0, 3))

        # make the return field evaluation points
        points_ret = np.zeros((0, 3))

        # loop over all materials
        for n in range(self.num_materials):
            
            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get the global orientations of the finite elements
            orientations = get_vector_basis_orientations(self.mesh,
                                                         self.cell_tags[n],
                                                         element_spec='CurlHcurlLegendre' + str(self.element_order-1))

            # get also the permeability in this domain
            mat_prop = self.material_list[n]
            
            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the mesh connectivity
                c = cells[i].copy()
                c.shape = (num_el, el_nodes)

                # get the quadrature nodes and weights
                w, q = finite_element.get_quadrature_rule(quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the curls of the edge shape functions at the integration points
                curls, num_orientations = finite_element.get_curl_edge_basis_functions(self.mesh, q, ct, self.element_order)

                # evaluate the shape functions at these points (not needed)
                phi = finite_element.evaluate_basis(q)

                # evaluate the derivatives of the lagrange basis functions at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # get the global ids
                glob_ids = self.global_ids[n]

                # the number of DoFs for the edge elements per finite element
                num_el_dofs = get_num_edge_dofs(self.element_order)

                # B field vectors for this material
                this_B = np.zeros((num_el*num_quad_pts, 3))

                # field evaluation points for this material
                this_points = np.zeros((num_el*num_quad_pts, 3))

                # loop over the finite elements
                for j, e in enumerate(c):


                    # compute the Jacobian at the integration points
                    # inv_J is a numpy array of dimension (M x 3 x 3) where
                    # the inverse of the Jacobians are stored in the second and
                    # third dimensions
                    # M is the number of integration points
                    # det_J is an (M x 0) numpy array which stores the M values of det(J)
                    J = finite_element.compute_J(e - 1, self.nodes, d_phi)

                    inv_J = finite_element.compute_J_inv(J)
                    det_J = finite_element.compute_J_det(J)

                    # this offset is the first index in the shape functions list for this orientation
                    offset_shape_function = 3*num_quad_pts*num_el_dofs*orientations[j]

                    # get the  basis functions with this orientations
                    curl_w_hat = curls[3*orientations[j]*num_quad_pts*num_el_dofs:3*(orientations[j]+1)*num_quad_pts*num_el_dofs]

                    # evaluate this finite element for the global position
                    this_points[j*num_quad_pts:(j+1)*num_quad_pts, :] = finite_element.evaluate(e - 1, self.nodes, phi)

                    #compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
                    for k in range(num_el_dofs):

                        # the integration value
                        int_val = 0.0

                        for m in range(num_quad_pts):

                            # the basis function in local coordinates
                            curl_w_hat_k = np.array([curl_w_hat[3*(m*num_el_dofs + k)     ],
                                                curl_w_hat[3*(m*num_el_dofs + k) + 1 ],
                                                curl_w_hat[3*(m*num_el_dofs + k) + 2 ]])

                            # the basis function in global coordinates
                            this_B[j*num_quad_pts+m, :] += J[m, :, :] @ curl_w_hat_k * x[glob_ids[j, k] - 1] / det_J[m]

                    

                B_ret = np.append(B_ret, this_B, axis=0)
                points_ret = np.append(points_ret, this_points, axis=0)
    
        return points_ret, B_ret


    def compute_B(self, x, quad_order=8):
        '''Compute the B for a given solution vector.
        
        :param x:
            The solution vector.
        
        :param quad_order:
            The order of the quadrature rule.

        :return:
            The sparse stiffness matrix.
        '''

        print('compute B...')

        # make the B field return vector
        B = np.zeros((0, 3))

        # make the return field evaluation points
        points = np.zeros((0, 3))

        # loop over all materials
        for n in range(self.num_materials):
            
            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get the global orientations of the finite elements
            orientations = get_vector_basis_orientations(self.mesh,
                                                         self.cell_tags[n],
                                                         element_spec='CurlHcurlLegendre' + str(self.element_order-1))

            # get also the permeability in this domain
            mat_prop = self.material_list[n]
            
            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the mesh connectivity
                c = cells[i].copy()
                c.shape = (num_el, el_nodes)

                # get the quadrature nodes and weights
                w, q = finite_element.get_quadrature_rule(quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the curls of the edge shape functions at the integration points
                curls, num_orientations = finite_element.get_curl_edge_basis_functions(self.mesh, q, ct, self.element_order)

                # evaluate the shape functions at these points (not needed)
                phi = finite_element.evaluate_basis(q)

                # evaluate the derivatives of the lagrange basis functions at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # get the global ids
                glob_ids = self.global_ids[n]

                # the number of DoFs for the edge elements per finite element
                num_el_dofs = get_num_edge_dofs(self.element_order)

                # evaluate the solution           
                this_points, this_B = fem_c.compute_B_Hcurl(self.nodes, cells[i] - 1, glob_ids - 1,
                                                  curls, phi, d_phi, orientations, x, q, w, self.num_dofs, num_el_dofs)
    
                points = np.append(points, this_points, axis=0)
                B = np.append(B, this_B, axis=0)

        return points, B

    def compute_A(self, x, quad_order=8):
        '''Compute A for a given solution vector.
        
        :param x:
            The solution vector.
        
        :param quad_order:
            The order of the quadrature rule.

        :return:
            The sparse stiffness matrix.
        '''

        print('compute A...')

        # make the A field return vector
        A_ret = np.zeros((0, 3))

        # make the return field evaluation points
        points_ret = np.zeros((0, 3))

        # loop over all materials
        for n in range(self.num_materials):
            
            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get the global orientations of the finite elements
            orientations = get_vector_basis_orientations(self.mesh,
                                                         self.cell_tags[n],
                                                         element_spec='HcurlLegendre' + str(self.element_order-1))

            # get also the permeability in this domain
            mat_prop = self.material_list[n]
            
            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the mesh connectivity
                c = cells[i].copy()
                c.shape = (num_el, el_nodes)

                # get the quadrature nodes and weights
                w, q = finite_element.get_quadrature_rule(quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the curls of the edge shape functions at the integration points
                basis, num_orientations = finite_element.get_edge_basis_functions(self.mesh, q, ct, self.element_order)

                # evaluate the shape functions at these points (not needed)
                phi = finite_element.evaluate_basis(q)

                # evaluate the derivatives of the lagrange basis functions at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # get the global ids
                glob_ids = self.global_ids[n]

                # the number of DoFs for the edge elements per finite element
                num_el_dofs = get_num_edge_dofs(self.element_order)

                # A field vectors for this material
                this_A = np.zeros((num_el*num_quad_pts, 3))

                # field evaluation points for this material
                this_points = np.zeros((num_el*num_quad_pts, 3))

                # loop over the finite elements
                for j, e in enumerate(c):


                    # compute the Jacobian at the integration points
                    # inv_J is a numpy array of dimension (M x 3 x 3) where
                    # the inverse of the Jacobians are stored in the second and
                    # third dimensions
                    # M is the number of integration points
                    # det_J is an (M x 0) numpy array which stores the M values of det(J)
                    J = finite_element.compute_J(e - 1, self.nodes, d_phi)
                    
                    inv_J = finite_element.compute_J_inv(J)
                    det_J = finite_element.compute_J_det(J)

                    # this offset is the first index in the shape functions list for this orientation
                    offset_shape_function = 3*num_quad_pts*num_el_dofs*orientations[j]

                    # get the curls of the basis functions with this orientations
                    w_hat = basis[3*orientations[j]*num_quad_pts*num_el_dofs:3*(orientations[j]+1)*num_quad_pts*num_el_dofs]

                    # evaluate this finite element for the global position
                    this_points[j*num_quad_pts:(j+1)*num_quad_pts, :] = finite_element.evaluate(e - 1, self.nodes, phi)

                    #compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
                    for k in range(num_el_dofs):

                        # the integration value
                        int_val = 0.0

                        for m in range(num_quad_pts):

                            # the basis function in local coordinates
                            w_hat_k = np.array([w_hat[3*(m*num_el_dofs + k)     ],
                                                w_hat[3*(m*num_el_dofs + k) + 1 ],
                                                w_hat[3*(m*num_el_dofs + k) + 2 ]])

                            # the basis function in global coordinates
                            this_A[j*num_quad_pts+m, :] += inv_J[m, :, :].T @ w_hat_k * x[glob_ids[j, k] - 1]

                A_ret = np.append(A_ret, this_A, axis=0)
                points_ret = np.append(points_ret, this_points, axis=0)
    
        return points_ret, A_ret
    
    def compute_stiffness_and_jacobia_matrix(self, x_0, quad_order=8):
        '''Compute the stiffness and the jacobi matrix using edge elements.
        Can at the moment only handle meshes of a single element type.

        :param x_0:
            The soluion vector at which the jacobia matrix is approximated.

        :param quad_order:
            The order of the quadrature rule.

        :return:
            The sparse stiffness matrix.
        '''


        print('compute stiffness and jacobi matrix curl curl...')

        # all triplets
        all_ij = np.zeros((0, 2), dtype=np.int64)
        all_vals = np.zeros((0, ), dtype=float)

        # make the sparse stiffness matrix
        K = csr_array((self.num_dofs, self.num_dofs))

        # make the sparse jacobi matrix
        dK = csr_array((self.num_dofs, self.num_dofs))

        # loop over all materials
        for n in range(self.num_materials):
            
            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get the global orientations of the finite elements
            orientations = get_vector_basis_orientations(self.mesh,
                                                         self.cell_tags[n],
                                                         element_spec='CurlHcurlLegendre' + str(self.element_order-1))
            
            # get also the permeability in this domain
            mat_prop = self.material_list[n]
            
            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the mesh connectivity
                c = cells[i].copy()
                c.shape = (num_el, el_nodes)

                # get the quadrature nodes and weights
                w, q = finite_element.get_quadrature_rule(quad_order)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the curls of the edge shape functions at the integration points
                curls, num_orientations = finite_element.get_curl_edge_basis_functions(self.mesh, q, ct, self.element_order)

                # evaluate the derivatives of the lagrange basis functions at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # get the global ids
                glob_ids = self.global_ids[n]

                # the number of DoFs for the edge elements per finite element
                num_el_dofs = get_num_edge_dofs(self.element_order)

                # the number of triplets
                num_triplets = num_el*num_el_dofs*num_el_dofs

                # allocate the space for the triplet list
                ij = np.zeros((num_triplets, 2), dtype=np.int32)
                vals = np.zeros((num_triplets, ))

                # a triplet counter
                t_cnt = 0

                # loop over the finite elements
                for j, e in enumerate(c):


                    # compute the Jacobian at the integration points
                    # inv_J is a numpy array of dimension (M x 3 x 3) where
                    # the inverse of the Jacobians are stored in the second and
                    # third dimensions
                    # M is the number of integration points
                    # det_J is an (M x 0) numpy array which stores the M values of det(J)
                    J = finite_element.compute_J(e - 1, self.nodes, d_phi)
                    
                    # inv_J = finite_element.compute_J_inv(J)
                    det_J = finite_element.compute_J_det(J)

                    # this offset is the first index in the shape functions list for this orientation
                    offset_shape_function = 3*num_quad_pts*num_el_dofs*orientations[j]

                    # get the curls of the basis functions with this orientations
                    curl_w_hat = curls[3*orientations[j]*num_quad_pts*num_el_dofs:3*(orientations[j]+1)*num_quad_pts*num_el_dofs]

                    # to do:
                    # compute nu

                    # if 166 in glob_ids[j, :] :
                    #     print('e = {}'.format(e))
                    #     print('curl_w_hat = {}'.format(curl_w_hat))
                    #     print('glob_ids[j, :] = {}'.format(glob_ids[j, :]))
                    #     print('det_J = {}'.format(det_J))

                    #compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
                    for k in range(num_el_dofs):

                        for l  in range(k, num_el_dofs):
                            
                            # the integration value
                            int_val = 0.0

                            for m in range(num_quad_pts):

                                
                                curl_w_hat_k = np.array([curl_w_hat[3*(m*num_el_dofs + k)     ],
                                                         curl_w_hat[3*(m*num_el_dofs + k) + 1 ],
                                                         curl_w_hat[3*(m*num_el_dofs + k) + 2 ]])

                                curl_w_hat_l = np.array([curl_w_hat[3*(m*num_el_dofs + l)     ],
                                                         curl_w_hat[3*(m*num_el_dofs + l) + 1 ],
                                                         curl_w_hat[3*(m*num_el_dofs + l) + 2 ]])
                                
                                dFcw_k = J[m, :, :] @ curl_w_hat_k
                                dFcw_l = J[m, :, :] @ curl_w_hat_l

                                int_val += np.sum(dFcw_k*dFcw_l)*w[m]/det_J[m]*mat_prop


                            # fill the triplet list
                            ij[t_cnt, 0] = glob_ids[j, k] - 1
                            ij[t_cnt, 1] = glob_ids[j, l] - 1
                            vals[t_cnt] = int_val
                            t_cnt += 1

                            # apply symmetry
                            if k != l:
                                ij[t_cnt, 0] = glob_ids[j, l] - 1
                                ij[t_cnt, 1] = glob_ids[j, k] - 1
                                vals[t_cnt] = int_val
                                t_cnt += 1

            all_ij = np.append(all_ij, ij, axis=0)
            all_vals = np.append(all_vals, vals)

        K = csr_array((all_vals, (all_ij[:, 0], all_ij[:, 1])), shape=(self.num_dofs, self.num_dofs))

        return K

    def compute_nodal_fields(self, x, mat_ids):
        '''Compute the fields at the mesh nodes by element averaging.
        At the moment we consider only lowest order FEM solutions, where
        the fields in the elements are constant.

        :param x:
            The soluion vector.

        :param mat_id:
            The material identifyers (list of integers).

        :return:
            The sparse stiffness matrix.
        '''

        # get the number of nodes
        num_nodes = self.nodes.shape[0]

        # the B fields at the nodes
        B_ret = np.zeros((num_nodes, 3))

        # the node multiplicity
        multiplicity = np.zeros((num_nodes, ), dtype=np.int64)

        # loop over all materials
        for n in mat_ids:

            # get the geometry info of this volume
            cell_types = self.cell_types[n]
            cells = self.cells[n]

            # get the global orientations of the finite elements
            orientations = get_vector_basis_orientations(self.mesh,
                                                         self.cell_tags[n],
                                                         element_spec='CurlHcurlLegendre' + str(self.element_order-1))

            # loop over cell types
            for i, ct in enumerate(cell_types):

                # make the finite element
                finite_element = FiniteElement(ct)

                # the numebr fo element nodes
                el_nodes = finite_element.get_number_of_nodes()

                # the number of elements of this type
                num_el = np.int32(len(cells[i]) / el_nodes)

                # get the mesh connectivity
                c = cells[i].copy()
                c.shape = (num_el, el_nodes)

                # get the quadrature nodes and weights
                _, q = finite_element.get_quadrature_rule(0)

                # the number of quadrature points
                num_quad_pts = np.int32(len(q) / 3)

                # evaluate the curls of the edge shape functions at the integration points
                curls, _ = finite_element.get_curl_edge_basis_functions(self.mesh, q, ct, self.element_order)

                # evaluate the derivatives of the lagrange basis functions at these points
                d_phi = finite_element.evaluate_basis_derivative(q)

                # get the global ids
                glob_ids = self.global_ids[n]

                # the number of DoFs for the edge elements per finite element
                num_el_dofs = get_num_edge_dofs(self.element_order)

                # loop over the finite elements
                for j, e in enumerate(c):

                    J = finite_element.compute_J(e - 1, self.nodes, d_phi)
                    det_J = finite_element.compute_J_det(J)

                    # get the  basis functions with this orientations
                    curl_w_hat = curls[3*orientations[j]*num_quad_pts*num_el_dofs:3*(orientations[j]+1)*num_quad_pts*num_el_dofs]

                    # the B field vector in the center of this element
                    this_B = 0

                    #compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
                    for k in range(num_el_dofs):

                        # the basis function in local coordinates
                        curl_w_hat_k = np.array([curl_w_hat[3*k],
                                                curl_w_hat[3*k + 1],
                                                curl_w_hat[3*k + 2]])

                        # the basis function in global coordinates
                        this_B += J[0, :, :] @ curl_w_hat_k * x[glob_ids[j, k] - 1] / det_J[0]

                    B_ret[e-1, 0] += this_B[0]
                    B_ret[e-1, 1] += this_B[1]
                    B_ret[e-1, 2] += this_B[2]
                    multiplicity[e-1] += 1


        # average over the elements
        mask = multiplicity != 0

        B_ret[mask, 0] /= multiplicity[mask]
        B_ret[mask, 1] /= multiplicity[mask]
        B_ret[mask, 2] /= multiplicity[mask]
        
        return B_ret