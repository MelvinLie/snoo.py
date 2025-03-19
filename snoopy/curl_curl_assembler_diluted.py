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

class CurlCurlAssemblerDiluted():

    def __init__(self, mesh, volume_tags, material_list, dilutions_list, element_order=1):
        '''Default constructor.
        
        :param mesh:
            A gmsh mesh object.

        :param volume_tags:
            The gmsh tags for the volumes.
            
        :param material_list:
            A list of material properties.

        :param dilutions_list:
            A list of dilution objects.

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

        self.dilutions = dilutions_list

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
            The stiffness matrix, the Jacobi matrix and the right hand side.
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

                # evaluate the basis functions
                phi = finite_element.evaluate_basis(q)

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

                if self.dilutions[n] == None:
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
                else:
                    # call the cpp code (gmsh starts counting nodes at 1)
                    this_K, this_dK, this_rhs = fem_c.compute_stiffness_and_jacobi_matrix_curl_curl_diluted(self.nodes,
                                                                                cells[i] - 1,
                                                                                glob_ids - 1,
                                                                                curls,
                                                                                phi,
                                                                                d_phi,
                                                                                orientations,
                                                                                x,
                                                                                B_s,
                                                                                reluctance,
                                                                                self.dilutions[n],
                                                                                q,
                                                                                w,
                                                                                self.num_dofs,
                                                                                num_el_dofs)
                    
                K += this_K
                dK += this_dK
                rhs += this_rhs

        return K, dK + K, rhs

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