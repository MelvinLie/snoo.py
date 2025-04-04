import numpy as np
import gmsh

from scipy.sparse.linalg import cg
from scipy.sparse import diags

from .curl_curl_assembler import CurlCurlAssembler

from .materials import ConstantReluctance

from .mesh_tools import get_edge_boundary_dofs
from .mesh_tools import get_cotree_dofs

class RedMVPSolver():
    '''This is a solver for the reduced vector potential formulation.
    '''
    def __init__(self, gmsh_model, coil_list, iron_tags,
                 air_tags, reluctances,
                 max_newton_iterations=25,
                 element_order=1,
                 quad_order=6):
        '''Setup the solver with the geometry info.
        
        :param gmsh_model:
            The gmsh model object.
            
        :param coil_list:
            A list of coil objects. They need to have a function 'compute_B(points)'
            for the field evaluation.

        :param iron_tags:
            The tags for the iron domains.

        :param air_tags:
            The tags for the air domains.

        :param reluctances:
            A list with reluctances for the iron domains. In the same order
            than the iron_tags.
        :param element_order:
            The order of the FEM approximation space for the vector potential.

        :param max_newton_iteration:
            The maximum number of Newton iterations.

        :return:
            None
        '''

        # initialize gmsh if not done already
        if not gmsh.isInitialized():
            gmsh.initialize()

        # store the tags for the iron domains
        self.iron_tags = iron_tags

        # store the tags for the iron domains
        self.air_tags = air_tags

        # store the excitation coils
        self.coil_list = coil_list

        # append all domain tags for the vector potential solver
        self.domain_tags = iron_tags + air_tags

        # append all material properties
        material_properties = reluctances + [ConstantReluctance(0.25/np.pi*1e7)]*len(air_tags)

        # the maximum number of Newton iterations
        self.max_newton_iterations = max_newton_iterations

        # store the finite element order
        self.element_order = element_order

        if element_order > 3:
            print('Error! Element order {} not implemented! Choosing 1.')
            element_order = 1

        self.quad_order = quad_order

        # get the dirichlet boundary dofs
        self.dirichlet_dofs = get_edge_boundary_dofs(gmsh_model,
                                                     ["Dirichlet Boundary"],
                                                     element_order)

        # setup the matrix factory for the vector potential solver
        self.curl_curl_factory = CurlCurlAssembler(gmsh_model.mesh,
                                                    self.domain_tags,
                                                    material_properties,
                                                    element_order)

        # the number of dofs for the vector potential
        self.num_dofs_mvp = self.curl_curl_factory.num_dofs


    def get_global_ids(self):
        '''Get the global ids for the field solution.

        :return:
            A (E x N) matrix (int) where E is the number of elements and N is the number
            of edge basis functions per element. 
        '''
        return self.curl_curl_factory.global_ids

    def make_boundary_mask(self):
        '''Make a mask to mask out the boundary degrees of freedom.

        :return:
            The boundary mask.
        '''

        # make the mask
        mask = np.ones((self.num_dofs_mvp, ), dtype=np.bool)

        # set the boundary dofs to fals
        for bc in self.dirichlet_dofs:
            mask[bc-1] = False

        return mask

    def solve(self, tolerance = 1e-4, x_0=np.zeros((0, )), maxiter=-1, apply_gauge=False):
        '''Solve the problem.

        :param tolerance:
            The tolerance for the cg iterations.

        :param xcg_0:
            The initial guess for the cg iterations.
                
        :param maxiter:
            The maximum number of iterations. If -1, the default scipy setting is used.

        :param apply_gauge:
            Set this flag true to enable the tree-cotree gauged formulation.

        :return:
            The solution vector.
        '''


        # get the number of iron and air domains for convinience
        num_iron_domains = len(self.iron_tags)
        num_air_domains = len(self.air_tags)

        # the initial solution vector for the mvp here 0, a linear problem
        if len(x_0) == 0:
            x_0 = np.zeros((self.num_dofs_mvp, ))

        # compute the source fields
        # we only need to compute them in the iron domains
        quad_points = self.curl_curl_factory.get_quadrature_points(self.quad_order,
                                                                   list(range(num_iron_domains)))

        # we evaluate the source field
        source_fields = []

        for i in range(num_iron_domains):
            B_src = 0.0*quad_points[i]
            for coil in self.coil_list:
                B_src += coil.compute_B(quad_points[i])
            source_fields.append(B_src)

        for i in range(num_air_domains):
            source_fields.append(np.zeros((0, 3)))

        # make a boundary mask for the dirichlet boundary condition
        mask = self.make_boundary_mask()

        if apply_gauge:
            tree_dofs, _ = get_cotree_dofs(self.curl_curl_factory.mesh)
            mask[tree_dofs] = False

        # initalize current solution vector
        x_n = x_0.copy()

        print('***********************')
        print('start Newton iterations')
        for i in range(self.max_newton_iterations): 

            # compute the stiffness and Jacobi matrix
            K, J, rhs = self.curl_curl_factory.compute_stiffness_and_jacobi_matrix_c(x_n,
                                                                quad_order=self.quad_order,
                                                                source_fields=source_fields)
            
            # compute right hand side
            b =  J @ x_n - K @ x_n + rhs

            # check if zero excitation
            if max(abs(b)) < 1e-14:
                x_np1 = 0.0*x_n
                print('  zero excitation!')
                break

            # reduce equation system (impose boundary conditions)
            J_c = J[mask, :]
            J_c = J_c[:, mask]

            # jacobi preconditioner
            M = diags(1.0/J_c.diagonal(), shape=J_c.shape)

            # launch cg
            if maxiter < 0:
                xx, _ = cg(J_c, b[mask], atol=tolerance, x0=x_n[mask], M=M)
            else:
                xx, _ = cg(J_c, b[mask], atol=tolerance, x0=x_n[mask], M=M, maxiter=maxiter)

            # we need to get back to the full solution vector
            x_np1 = 0.0*x_n
            x_np1[mask == True] = xx

            # the step
            h = x_np1 - x_n

            print(f"  step {i}")
            print(f"    max relative increment {max(abs(h))/max(abs(x_np1)):.3e}")

            # stop if tolerance is reached
            if max(abs(h)) < tolerance*max(abs(x_np1)):
                print('  tolerance reached!')
                break
            else:
                x_n = x_np1.copy()

        return x_np1
