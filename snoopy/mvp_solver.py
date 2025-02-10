import numpy as np
import gmsh
import pyvista as pv

from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg
from scipy.sparse import diags

import matplotlib.pyplot as plt

from .laplace_subdomain_solver import LaplaceSubdomainSolver
from .curl_curl_assembler import CurlCurlAssembler
from .matrix_assembly import delete_rows_sparse
from .matrix_assembly import delete_cols_sparse
from .matrix_assembly import recover_full_solution

from .materials import ConstantReluctance

from .mesh_tools import get_edge_boundary_dofs
from .mesh_tools import get_cotree_dofs

from .plot_tools import plot_vector_field

class MVPSolver():

    def __init__(self, gmsh_model, coil_tags, iron_tags,
                 air_tags, coil_terminals, reluctances,
                 max_newton_iterations=25,
                 element_order=1,
                 quad_order=-1):
        '''Setup the solver with the geometry info.
        
        :param gmsh_model:
            The gmsh model object.
            
        :param coil_tags:
            The tags for the coil domains.

        :param iron_tags:
            The tags for the iron domains.

        :param air_tags:
            The tags for the air domains.

        :param coil_terminals:
            A list of tuples for the tags of the coil terminals. In the same order
            than the coil_tags.

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

        # initialize the subdomain solvers for the coil domains
        self.coil_sd_solvers = []
        for i, ct in enumerate(coil_tags):
            self.coil_sd_solvers.append(LaplaceSubdomainSolver(gmsh_model.mesh, ct, coil_terminals[i]))

        # append all domain tags for the vector potential solver
        domain_tags = coil_tags + iron_tags + air_tags

        # append all material properties
        material_properties = [ConstantReluctance(0.25/np.pi*1e7)]*len(coil_tags) + reluctances + [ConstantReluctance(0.25/np.pi*1e7)]*len(air_tags)

        # the maximum number of Newton iterations
        self.max_newton_iterations = max_newton_iterations

        # store the finite element order
        self.element_order = element_order

        if (element_order > 3):
            print('Error! Element order {} not implemented! Choosing 1.')
            element_order = 1

        # store the quadrature order
        if quad_order < 0:
            self.quad_order = element_order + 2
        else:
            self.quad_order = quad_order

        # get the dirichlet boundary dofs
        self.dirichlet_dofs = get_edge_boundary_dofs(gmsh_model, ["Dirichlet Boundary"], element_order)

        # setup the matrix factory for the vector potential solver
        self.curl_curl_factory = CurlCurlAssembler(gmsh_model.mesh, domain_tags, material_properties, element_order)

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

    def solve_lin(self, phi_list, tolerance = 1e-4, xcg_0 = np.zeros((0, )), maxiter=-1, apply_gauge=False):
        '''Solve the problem.
        
        :param phi_list:
            The potential differences at the terminals to drive the current.

        :param tolerance:
            The tolerance for the cg iterations.

        :param xcg_0:
            The initial guess for the cg iterations.
            
        :param maxiter:
            The maximum number of iterations. If -1, the default scipy setting is used.

        :return:
            The solution vector.
        '''

        # a list with all scalar potential solution vectors
        sol_sp = []

        # the right hand side for the curl curl equation
        rhs_mvp = np.zeros((self.num_dofs_mvp, ))

        # the initial solution vector for the mvp here 0, a linear problem
        x_0 = np.zeros((self.num_dofs_mvp, ))

        # loop over all coil domain solvers
        for i, sol in enumerate(self.coil_sd_solvers):

            # solve for the electric scalar potential in the subdomain
            sol_sp.append(sol.solve(phi_list[i]))

            # increment the right hand side
            rhs_mvp += self.curl_curl_factory.compute_rhs_electric_potential(sol_sp[-1],
                                                                             quad_order=self.quad_order,
                                                                             select_mat=[i])

        # compute the contribution to the rhs from this coil
        K, _ = self.curl_curl_factory.compute_stiffness_and_jacobi_matrix_c(x_0, quad_order=self.quad_order)

        # make a boundary mask for the dirichlet boundary condition
        mask = self.make_boundary_mask()

        if apply_gauge:
            # get the spanning tree
            tree_dofs, cotree_dofs = get_cotree_dofs(self.curl_curl_factory.mesh)
            mask[tree_dofs] = False

        # apply the boundary condition
        K = K[mask, :]
        K = K[:, mask]
        rhs_mvp = rhs_mvp[mask]

        if len(xcg_0) == len(mask):
            xcg_0 = xcg_0[mask]

        # jacobi preconditioner
        M = diags(1.0/K.diagonal(), shape=K.shape)

        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.plot(rhs_mvp)
        ax = fig.add_subplot(122)
        cntrf = ax.contourf(K.toarray())
        fig.colorbar(cntrf)
        plt.show()

        # solve the problem
        # print("solve by conjugare gradients")
        if maxiter < 0:
            if len(xcg_0) == len(rhs_mvp):
                x, exit_code = cg(K, rhs_mvp, atol=tolerance, x0=xcg_0, M=M)
            else:
                x, exit_code = cg(K, rhs_mvp, M=M)
        else:
            if len(xcg_0) == len(rhs_mvp):
                x, exit_code = cg(K, rhs_mvp, atol=tolerance, x0=xcg_0, M=M, maxiter=maxiter)
            else:
                x, exit_code = cg(K, rhs_mvp, atol=tolerance, M=M, maxiter=maxiter)

        # we need to get back to the full solution vector
        x_long = 0.0*x_0
        x_long[mask == True] = x

        print('exit code = {}'.format(exit_code))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x)
        plt.show()

        return x_long

    def solve_nonlin(self, phi_list, tolerance = 1e-4, x_0=np.zeros((0, )), maxiter=-1, apply_gauge=False, plot_current_density=False):
            '''Solve the problem.
            
            :param phi_list:
                The potential differences at the terminals to drive the current.

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

            # a list with all scalar potential solution vectors
            sol_sp = []

            # the right hand side for the curl curl equation
            rhs_mvp = np.zeros((self.num_dofs_mvp, ))

            # the initial solution vector for the mvp here 0, a linear problem
            if len(x_0) == 0:
                x_0 = np.zeros((self.num_dofs_mvp, ))

            # make a boundary mask for the dirichlet boundary condition
            mask = self.make_boundary_mask()

            if apply_gauge:
                # get the spanning tree
                tree_dofs, cotree_dofs = get_cotree_dofs(self.curl_curl_factory.mesh)
                mask[tree_dofs] = False

            # loop over all coil domain solvers
            for i, sol in enumerate(self.coil_sd_solvers):

                # solve for the electric scalar potential in the subdomain
                sol_sp.append(sol.solve(phi_list[i]))

                # increment the right hand side
                rhs_mvp += self.curl_curl_factory.compute_rhs_electric_potential(sol_sp[-1],
                                                                                 quad_order=self.quad_order,
                                                                                 select_mat=[i])

            if plot_current_density:
                
                pl = pv.Plotter()

                for i, x_sp in enumerate(sol_sp):
                    pp, JJ = self.coil_sd_solvers[i].matrix_factory.compute_field(x_sp)
                
                    plot_vector_field(pl, pp, JJ, title='J_{} in A/m**2'.format(i), mag=1/phi_list[i])
                
                pl.add_axes()
                pl.show_grid()
                pl.show()

            # reduce equation system (impose boundary conditions)
            rhs_mvp_c = rhs_mvp[mask]
            x_0_c = x_0[mask]

            # initalize current solution vector
            x_n = x_0.copy()

            
            print('***********************')
            print('start Newton iterations')
            for i in range(self.max_newton_iterations):
                
                # compute the contribution to the rhs from this coil
                K, J = self.curl_curl_factory.compute_stiffness_and_jacobi_matrix_c(x_n, quad_order=self.quad_order)

                # compute right hand side
                rhs =  J @ x_n - K @ x_n + rhs_mvp

                # reduce equation system (impose boundary conditions)
                J_c = J[mask, :]
                J_c = J_c[:, mask]

                # jacobi preconditioner
                M = diags(1.0/J_c.diagonal(), shape=J_c.shape)

                # launch cg
                if maxiter < 0:
                    xx, exit_code = cg(J_c, rhs[mask], atol=tolerance, x0=x_n[mask], M=M)
                else:
                    xx, exit_code = cg(J_c, rhs[mask], atol=tolerance, x0=x_n[mask], M=M, maxiter=maxiter)

                # we need to get back to the full solution vector
                x_np1 = 0.0*x_n
                x_np1[mask == True] = xx

                # the step
                h = x_np1 - x_n

                print('  step {}'.format(i))
                print('    max relative increment {:.3e}'.format(max(abs(h))/max(abs(x_np1))))

                # stop if tolerance is reached
                if max(abs(h)) < tolerance*max(abs(x_np1)):
                    print('  tolerance reached!')
                    break
                else:
                    x_n = x_np1.copy()

            return x_np1
