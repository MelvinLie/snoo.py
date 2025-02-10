# distutils: sources = [c-algorithms/ctools.c]
# distutils: include_dirs = [c-algorithms/]

import numpy as np
cimport numpy as np
cimport ctools
from cython cimport view
from libc.stdlib cimport malloc, free
from scipy.sparse import csr_array

from .mesh_tools import get_order
from .evaluators import evaluate_field
from .integration_tools import *
from .hexahedral_elements import *

def compute_K_phi_c(p, c, num_quad=8):
    '''Compute the stiffness matrix for the magnetic scalar potential.

    :param p:
        The node coefficients.

    :param c:
        The connectivity.

    :param num_quad:
        The number of quadrature points. Default 8.

    :return:
        The stiffness matrix.
    '''

    # ===========================================
    # Python Part
    # ===========================================

    # the total number of nodes
    num_nodes = p.shape[0]

    # the number of elements
    num_el = c.shape[0]

    # element nodes
    el_nodes = c.shape[1]

    # make space for the discrete operator
    # K = np.zeros((num_nodes, num_nodes))

    # these settings are element type dependent
    if el_nodes == 8:
        print('using 8 noded brick elements (1st oder)')
        order = 1

    elif el_nodes == 20:
        print('using 20 noded brick elements (2nd oder)')
        order = 2

    elif el_nodes == 32:
        print('using 32 noded brick elements (3rd oder)')
        order = 3

    else:
        print('finite element with {} not found!'.format(num_nodes))

    print('assembling the stiffness matrix (scalar laplace) (c)...')

    # get the quadrature nodes and weights
    q, w = get_quadrature_rule(num_quad)

    # get the number of elements in the mesh
    num_el = c.shape[0]

    # evaluate the shape functions at these points (not needed)
    phi = eval_shape_hexahedron(q, order)

    # evaluate the derivatives at these points
    d_phi = eval_gradient_hexahedron(q, order)

    # the required space for the triplet list
    num_triplets = num_el*el_nodes*el_nodes

    # ===========================================
    # Convert python --> C
    # ===========================================
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] p_buff = np.ascontiguousarray(p.flatten(), dtype = np.double)
    cdef double* p_c = <double*> p_buff.data

    # the point data array is structured as follows
    # [ x_1, y_1, z_2, x_2, y_2, z_2, ... ]
        
    cdef np.ndarray[int, ndim=1, mode = 'c'] c_buff = np.ascontiguousarray(c.flatten(), dtype = np.int32)
    cdef int* c_c = <int*> c_buff.data
    
    # the connectivity data is structured as follows, (N-noded element)
    # [ el_1_n_1, el_1_n_2, ... el_1_n_N, el_2_n_1, ...]

    cdef np.ndarray[double, ndim=1, mode = 'c'] q_buff = np.ascontiguousarray(q.flatten(), dtype = np.double)
    cdef double* q_c = <double*> q_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] w_buff = np.ascontiguousarray(w.flatten(), dtype = np.double)
    cdef double* w_c = <double*> w_buff.data
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] d_phi_buff = np.ascontiguousarray(d_phi.flatten(), dtype = np.double)
    cdef double* d_phi_c = <double*> d_phi_buff.data
    
    # the derivative data array is structured as follows:
    # [ d/du phi_1(x_1), d/dv phi_1(x_1), d/dw phi_1(x_1), d/du phi_2(x_1), ... , d/du phi_1(x_2)]
    # where x_i is the ith point, phi_i is the ith basis function and the derivative are d/du, d/dv, d/dw

    cdef np.ndarray[double, ndim=1, mode = 'c'] phi_buff = np.ascontiguousarray(phi.flatten(), dtype = np.double)
    cdef double* phi_c = <double*> phi_buff.data

    # the scalar field data array is structured as follows:
    # [ phi_1(x_1), phi_2(x_1), ... , phi_1(x_2)]
    # where x_i is the ith point, phi_i is the ith basis function

    
    # allocate space for the triplet list
    cdef int* i_c = <int *> malloc( num_triplets * sizeof(int))
    cdef int* j_c = <int *> malloc( num_triplets * sizeof(int))
    cdef double* vals_c = <double *> malloc( num_triplets * sizeof(double))
    cdef ctools.triplet_list* triplets_c = ctools.make_triplet_list(num_triplets, i_c, j_c, vals_c)

    # make a c style mesh object
    cdef ctools.mesh* mesh_c = ctools.make_mesh(num_nodes,
                                             num_el,
                                             el_nodes,
                                             p_c,
                                             c_c)

    # make c style quadrature rule object
    cdef ctools.quad_3D* quad_c = ctools.make_quadrature_rule(len(w),
                                                            q_c,
                                                            w_c);

    # ===========================================
    # Run C code
    # ===========================================

    ctools.compute_K(triplets_c, mesh_c, quad_c, d_phi_c)

    # ===========================================
    # Convert C -> python
    # ===========================================
    cdef view.array i = <int[:num_triplets]> i_c

    cdef view.array j = <int[:num_triplets]> j_c

    cdef view.array vals = <double[:num_triplets]> vals_c

    K = csr_array((vals, (i, j)), shape=(num_nodes, num_nodes))

    # ===========================================
    # Memory management
    # ===========================================
    # free(p_c)
    # free(c_c)
    # free(q_c)
    # free(w_c)
    # free(d_phi_c)
    # free(phi_c)
    # free(mesh_c)
    # free(quad_c)
    # free(triplets_c)


    # ===========================================
    # Return
    # ===========================================
    return K


def compute_K_A_c(p, c, v=np.array([]), vs=np.array([]), num_quad=8):
    '''Compute the stiffness matrix for the magnetic vector potential
    Ansatz.

    :param p:
        The node coefficients.

    :param c:
        The connectivity.

    :param v:
        The reluctivity, per element for the curl curl integral.

    :param vs:
        The reluctivity, per element for the div div integral.

    :param num_quad:
        The number of quadrature points. Default 8.

    :return:
        The stiffness matrix.
    '''

    # ===========================================
    # Python Part
    # ===========================================

    # the total number of nodes
    num_nodes = p.shape[0]

    # the number of elements
    num_el = c.shape[0]

    # element nodes
    el_nodes = c.shape[1]

    # these settings are element type dependent
    if el_nodes == 8:
        print('using 8 noded brick elements (1st oder)')
        order = 1

    elif el_nodes == 20:
        print('using 20 noded brick elements (2nd oder)')
        order = 2

    elif el_nodes == 32:
        print('using 32 noded brick elements (3rd oder)')
        order = 3

    else:
        print('finite element with {} not found!'.format(num_nodes))

    print('assembling the stiffness matrix (vector laplace) (c)...')

    # get the quadrature nodes and weights
    q, w = get_quadrature_rule(num_quad)

    # get the number of elements in the mesh
    num_el = c.shape[0]

    # evaluate the shape functions at these points (not needed)
    phi = eval_shape_hexahedron(q, order)

    # evaluate the derivatives at these points
    d_phi = eval_gradient_hexahedron(q, order)

    # get the curls of the basis functions
    curls = assemble_curl(d_phi)

    # assemble the gradients for this vector space
    grads = assemble_grad(d_phi)

    # the required space for the triplet list
    num_triplets = num_el*el_nodes*el_nodes*9

    # the reluctivity
    if len(v) == 0:
        v = np.ones((c.shape[0]))

    # the reluctivity
    if len(vs) == 0:
        vs = np.ones((c.shape[0]))

    # ===========================================
    # Convert python --> C
    # ===========================================
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] p_buff = np.ascontiguousarray(p.flatten(), dtype = np.double)
    cdef double* p_c = <double*> p_buff.data

    # the point data array is structured as follows
    # [ x_1, y_1, z_2, x_2, y_2, z_2, ... ]
        
    cdef np.ndarray[int, ndim=1, mode = 'c'] c_buff = np.ascontiguousarray(c.flatten(), dtype = np.int32)
    cdef int* c_c = <int*> c_buff.data
    
    # the connectivity data is structured as follows, (N-noded element)
    # [ el_1_n_1, el_1_n_2, ... el_1_n_N, el_2_n_1, ...]

    cdef np.ndarray[double, ndim=1, mode = 'c'] q_buff = np.ascontiguousarray(q.flatten(), dtype = np.double)
    cdef double* q_c = <double*> q_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] w_buff = np.ascontiguousarray(w.flatten(), dtype = np.double)
    cdef double* w_c = <double*> w_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] d_phi_buff = np.ascontiguousarray(d_phi.flatten(), dtype = np.double)
    cdef double* d_phi_c = <double*> d_phi_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] curls_buff = np.ascontiguousarray(curls.flatten(), dtype = np.double)
    cdef double* curls_c = <double*> curls_buff.data
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] grads_buff = np.ascontiguousarray(grads.flatten(), dtype = np.double)
    cdef double* grads_c = <double*> grads_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] phi_buff = np.ascontiguousarray(phi.flatten(), dtype = np.double)
    cdef double* phi_c = <double*> phi_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] v_buff = np.ascontiguousarray(v.flatten(), dtype = np.double)
    cdef double* v_c = <double*> v_buff.data
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] vs_buff = np.ascontiguousarray(vs.flatten(), dtype = np.double)
    cdef double* vs_c = <double*> vs_buff.data

    # allocate space for the triplet list
    cdef int* i_c = <int *> malloc( num_triplets * sizeof(int))
    cdef int* j_c = <int *> malloc( num_triplets * sizeof(int))
    cdef double* vals_c = <double *> malloc( num_triplets * sizeof(double))
    cdef ctools.triplet_list* triplets_c = ctools.make_triplet_list(num_triplets, i_c, j_c, vals_c)

    # make a c style mesh object
    cdef ctools.mesh* mesh_c = ctools.make_mesh(num_nodes,
                                             num_el,
                                             el_nodes,
                                             p_c,
                                             c_c)

    # make c style quadrature rule object
    cdef ctools.quad_3D* quad_c = ctools.make_quadrature_rule(len(w),
                                                            q_c,
                                                            w_c)

    # ===========================================
    # Run C code
    # ===========================================

    ctools.compute_K_A(triplets_c, mesh_c, quad_c, d_phi_c, curls_c, grads_c, v_c, vs_c)

    # ===========================================
    # Convert C -> python
    # ===========================================
    cdef view.array i = <int[:num_triplets]> i_c

    cdef view.array j = <int[:num_triplets]> j_c

    cdef view.array vals = <double[:num_triplets]> vals_c

    K = csr_array((vals, (i, j)), shape=(3*num_nodes, 3*num_nodes))

    # ===========================================
    # Memory management
    # ===========================================
    # free(p_c)
    # free(c_c)
    # free(q_c)
    # free(w_c)
    # free(d_phi_c)
    # free(phi_c)
    # free(mesh_c)
    # free(quad_c)
    # free(triplets_c)


    # ===========================================
    # Return
    # ===========================================
    return K


def compute_K_div(p, c, num_quad=8):
    '''Compute the stiffness matrix of type div(u)div(v).

    :param p:
        The node coefficients.

    :param c:
        The connectivity.

    :param num_quad:
        The number of quadrature points. Default 8.

    :return:
        The stiffness matrix.
    '''

    # ===========================================
    # Python Part
    # ===========================================

    # the total number of nodes
    num_nodes = p.shape[0]

    # the number of elements
    num_el = c.shape[0]

    # element nodes
    el_nodes = c.shape[1]

    # these settings are element type dependent
    if el_nodes == 8:
        print('using 8 noded brick elements (1st oder)')
        order = 1

    elif el_nodes == 20:
        print('using 20 noded brick elements (2nd oder)')
        order = 2

    elif el_nodes == 32:
        print('using 32 noded brick elements (3rd oder)')
        order = 3

    else:
        print('finite element with {} not found!'.format(num_nodes))

    print('assembling the stiffness matrix (div div) (c)...')

    # get the quadrature nodes and weights
    q, w = get_quadrature_rule(num_quad)

    # get the number of elements in the mesh
    num_el = c.shape[0]

    # evaluate the derivatives at these points
    d_phi = eval_gradient_hexahedron(q, order)

    # assemble the gradients for this vector space
    grads = assemble_grad(d_phi)

    # the required space for the triplet list
    num_triplets = num_el*el_nodes*el_nodes*9

    # ===========================================
    # Convert python --> C
    # ===========================================
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] p_buff = np.ascontiguousarray(p.flatten(), dtype = np.double)
    cdef double* p_c = <double*> p_buff.data

    # the point data array is structured as follows
    # [ x_1, y_1, z_2, x_2, y_2, z_2, ... ]
        
    cdef np.ndarray[int, ndim=1, mode = 'c'] c_buff = np.ascontiguousarray(c.flatten(), dtype = np.int32)
    cdef int* c_c = <int*> c_buff.data
    
    # the connectivity data is structured as follows, (N-noded element)
    # [ el_1_n_1, el_1_n_2, ... el_1_n_N, el_2_n_1, ...]

    cdef np.ndarray[double, ndim=1, mode = 'c'] q_buff = np.ascontiguousarray(q.flatten(), dtype = np.double)
    cdef double* q_c = <double*> q_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] w_buff = np.ascontiguousarray(w.flatten(), dtype = np.double)
    cdef double* w_c = <double*> w_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] d_phi_buff = np.ascontiguousarray(d_phi.flatten(), dtype = np.double)
    cdef double* d_phi_c = <double*> d_phi_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] grads_buff = np.ascontiguousarray(grads.flatten(), dtype = np.double)
    cdef double* grads_c = <double*> grads_buff.data

    
    # allocate space for the triplet list
    cdef int* i_c = <int *> malloc( num_triplets * sizeof(int))
    cdef int* j_c = <int *> malloc( num_triplets * sizeof(int))
    cdef double* vals_c = <double *> malloc( num_triplets * sizeof(double))
    cdef ctools.triplet_list* triplets_c = ctools.make_triplet_list(num_triplets, i_c, j_c, vals_c)

    # make a c style mesh object
    cdef ctools.mesh* mesh_c = ctools.make_mesh(num_nodes,
                                             num_el,
                                             el_nodes,
                                             p_c,
                                             c_c)

    # make c style quadrature rule object
    cdef ctools.quad_3D* quad_c = ctools.make_quadrature_rule(len(w),
                                                            q_c,
                                                            w_c)

    # ===========================================
    # Run C code
    # ===========================================

    ctools.compute_K_div(triplets_c, mesh_c, quad_c, d_phi_c, grads_c)

    # ===========================================
    # Convert C -> python
    # ===========================================
    cdef view.array i = <int[:num_triplets]> i_c

    cdef view.array j = <int[:num_triplets]> j_c

    cdef view.array vals = <double[:num_triplets]> vals_c

    K = csr_array((vals, (i, j)), shape=(3*num_nodes, 3*num_nodes))

    # ===========================================
    # Memory management
    # ===========================================
    # free(p_c)
    # free(c_c)
    # free(q_c)
    # free(w_c)
    # free(d_phi_c)
    # free(phi_c)
    # free(mesh_c)
    # free(quad_c)
    # free(triplets_c)


    # ===========================================
    # Return
    # ===========================================
    return K

def compute_rhs_A_c(J_fcn, p, c, quad_order=8):
    '''Compute the right hand side for the magnetic vector potential
    formulation in H1^3, for a given current density vector.
    
    :param J:
        The current density function.

    :param p:
        The nodal coordinates of the mesh.

    :param c:
        The cells of the mesh.

    :param quad_order:
        The order of the quadrature rule.

    :return:
        The Right hand side in a numpy array.
    '''

    # ===========================================
    # Python Part
    # ===========================================

    # the total number of nodes
    num_nodes = p.shape[0]

    # the number of elements
    num_el = c.shape[0]

    # element nodes
    el_nodes = c.shape[1]

    # the number of basis functions
    num_dofs = 3*num_nodes

    # make space for the return
    rhs = np.zeros((num_dofs, ))

    # the ansatz space order
    order = get_order(c)

    # get the quadrature nodes and weights
    q, w = get_quadrature_rule(quad_order, 'Hex')

    # evaluate the shape functions at these points (not needed)
    phi = eval_shape_hexahedron(q, order)
    
    # evaluate the derivatives at these points
    d_phi = eval_gradient_hexahedron(q, order)

    # evaluate the J field
    J_eval = evaluate_field(J_fcn, 3, p, c, q)

    # ===========================================
    # Convert python --> C
    # ===========================================
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] p_buff = np.ascontiguousarray(p.flatten(), dtype = np.double)
    cdef double* p_c = <double*> p_buff.data

    # the point data array is structured as follows
    # [ x_1, y_1, z_2, x_2, y_2, z_2, ... ]
        
    cdef np.ndarray[int, ndim=1, mode = 'c'] c_buff = np.ascontiguousarray(c.flatten(), dtype = np.int32)
    cdef int* c_c = <int*> c_buff.data
    
    # the connectivity data is structured as follows, (N-noded element)
    # [ el_1_n_1, el_1_n_2, ... el_1_n_N, el_2_n_1, ...]

    cdef np.ndarray[double, ndim=1, mode = 'c'] q_buff = np.ascontiguousarray(q.flatten(), dtype = np.double)
    cdef double* q_c = <double*> q_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] w_buff = np.ascontiguousarray(w.flatten(), dtype = np.double)
    cdef double* w_c = <double*> w_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] phi_buff = np.ascontiguousarray(phi.flatten(), dtype = np.double)
    cdef double* phi_c = <double*> phi_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] d_phi_buff = np.ascontiguousarray(d_phi.flatten(), dtype = np.double)
    cdef double* d_phi_c = <double*> d_phi_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] j_buff = np.ascontiguousarray(J_eval.flatten(), dtype = np.double)
    cdef double* j_c = <double*> j_buff.data

    # make a c style mesh object
    cdef ctools.mesh* mesh_c = ctools.make_mesh(num_nodes,
                                             num_el,
                                             el_nodes,
                                             p_c,
                                             c_c)

    # make c style quadrature rule object
    cdef ctools.quad_3D* quad_c = ctools.make_quadrature_rule(len(w),
                                                            q_c,
                                                            w_c)

    # the return vector
    cdef double* rhs_c = <double *> malloc( num_dofs * sizeof(double))

    # ===========================================
    # Run C code
    # ===========================================

    rhs_c = ctools.compute_rhs_A_c(mesh_c, quad_c, phi_c, d_phi_c, j_c)

    # ===========================================
    # Convert C -> python
    # ===========================================
    cdef view.array rhs_array = <double[:num_dofs]> rhs_c

    rhs = np.asarray(rhs_array)

    # ===========================================
    # Return
    # ===========================================
    return rhs