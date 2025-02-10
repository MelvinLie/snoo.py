import numpy as np
import gmsh
import pyvista as pv

from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg
from scipy.sparse import diags

import matplotlib.pyplot as plt

from .matrix_assembly import GradGradAssembler
from .matrix_assembly import delete_rows_sparse
from .matrix_assembly import delete_cols_sparse
from .matrix_assembly import recover_full_solution

from .materials import ConstantPermeability

class LaplaceSubdomainSolver():

    def __init__(self, mesh, subdomain_tag, terminal_groups, quad_order=4):
        '''Setup the solver with the geometry info.
        
        :param mesh:
            The gmsh mesh object.
            
        :param subdomain_tag:
            The tag of the subdomain.

        :param terminal_groups:
            The physical groups for the two terminals in a tuple.

        :param quad_order:
            The order of the quadrature rule.

        :return:
            None
        '''

        # initialize gmsh if not done already
        if not gmsh.isInitialized():
            gmsh.initialize()

        # setup the matrix factory
        self.matrix_factory = GradGradAssembler(mesh, [subdomain_tag])

        # take in also the terminal ids
        self.terminal_1 = terminal_groups[0]
        self.terminal_2 = terminal_groups[1]

        # setup the boundary condition
        self.setup_boundary_conditions()

        # the subdomain mask
        self.subdomain_mask = self.get_subdomain_mask(subdomain_tag)

        # the number of DoFs
        nodes_all = self.matrix_factory.nodes

        # number of all nodes
        self.num_dofs_all = nodes_all.shape[0]

        # nodes of the subdomain
        nodes_subdom = nodes_all[self.subdomain_mask]

        # the number of degrees of freedom
        self.num_dofs = nodes_subdom.shape[0]

        # set the order of the quadrature rule
        self.quad_order = quad_order

    def setup_boundary_conditions(self):
        '''Setup the boundary conditions.
        
        :return:
            None.
        
        '''

        # get the coordinates for the terminal 1
        node_tags_1, coords_1, _ = gmsh.model.mesh.getNodes(2, self.terminal_1, includeBoundary=True)
        coords_1.shape = (np.int32(len(coords_1)/3), 3)

        # get the coordinates for the terminal 2
        node_tags_2, coords_2, _ = gmsh.model.mesh.getNodes(2, self.terminal_2, includeBoundary=True)
        coords_2.shape = (np.int32(len(coords_2)/3), 3)

        # find the terminal nodes
        nodes = self.matrix_factory.nodes

        self.tags_term_1 = np.zeros((coords_1.shape[0], ), dtype=np.int32)
        self.tags_term_2 = np.zeros((coords_2.shape[0], ), dtype=np.int32)


        for i, coords in enumerate(coords_1):
            # search for this nodes in my nodes
            diff_x = coords[0] - nodes[:, 0]
            diff_y = coords[1] - nodes[:, 1]
            diff_z = coords[2] - nodes[:, 2]
            # the distance
            dist = np.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
            # get the closest one
            self.tags_term_1[i] = np.argmin(dist)
        
        for i, coords in enumerate(coords_2):
            # search for this nodes in my nodes
            diff_x = coords[0] - nodes[:, 0]
            diff_y = coords[1] - nodes[:, 1]
            diff_z = coords[2] - nodes[:, 2]
            # the distance
            dist = np.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
            # get the closest one
            self.tags_term_2[i] = np.argmin(dist)


        self.term_tags = np.append(self.tags_term_1, self.tags_term_2, axis=0)

    def solve(self, Phi):
        '''Solve the problem.
        
        :param Phi:
            The potential difference.

        :return:
            The solution vector.
        '''

        # get the lifting vector
        x_l = self.make_lifting_vector(Phi)

        # compute stiffness matrix and jacobian
        K, _ = self.matrix_factory.compute_stiffness_and_jacobi_matrix_c(0.0*x_l, [ConstantPermeability(1.0)], quad_order=self.quad_order)

        # compute right hand side
        rhs =  - K @ x_l

        # reduce equation system (impose boundary conditions)
        K_c = delete_rows_sparse(K, self.term_tags)
        K_c = delete_cols_sparse(K_c, self.term_tags)
        rhs_c = np.delete(rhs, self.term_tags)
        mask_c = np.delete(self.subdomain_mask, self.term_tags)

        num_dof_red = len(rhs_c)

        # reduce equation system (to subdomain)
        K_c = K_c[mask_c, :]
        K_c = K_c[:, mask_c]
        rhs_c = rhs_c[mask_c]

        # solve
        x_c = spsolve(K_c, rhs_c)

        x = np.zeros((num_dof_red, ))
        x[mask_c] = x_c

        # recover full x_np1 vector
        x = recover_full_solution(x, x_l, self.term_tags)

        return x

    def get_subdomain_mask(self, subdomain_tag):
        '''Solve the problem.

        :param subdomain_tag:
            The tag of the subdomain.

        :return:
            The mask used to mask out the subdomain.        
        '''

        # get the coordinates for this subdomain
        node_tags, coords, _ = gmsh.model.mesh.getNodes(3, subdomain_tag, includeBoundary=True)
        coords.shape = (np.int32(len(coords)/3), 3)

        # get all nodes in the mesh
        nodes = self.matrix_factory.nodes

        # the return mask
        ret_mask = np.zeros((nodes.shape[0], ), dtype=np.bool_)

        for i, n in enumerate(nodes):

            # search for this nodes in my nodes
            diff_x = coords[:, 0] - n[0]
            diff_y = coords[:, 1] - n[1]
            diff_z = coords[:, 2] - n[2]

            # the distance
            dist = np.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)


            if min(dist) < 1e-14:
                ret_mask[i] = True

        return ret_mask
    
    def make_lifting_vector(self, Phi):
        '''Make the lifting vector.
        
        :param Phi:
            The potential difference.
            
        :return:
            The lifting vector.
        '''

        x_l = np.zeros((self.num_dofs_all, ))
        
        for i in self.tags_term_1:
            x_l[i] = -0.5*Phi

        for i in self.tags_term_2:
            x_l[i] = 0.5*Phi

        return x_l
    