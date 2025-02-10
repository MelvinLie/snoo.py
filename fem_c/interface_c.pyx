# distutils: sources = [fem_c/ctools.c]
# distutils: include_dirs = [fem_c/]

import numpy as np
cimport numpy as np
cimport ctools
from cython cimport view
from libc.stdlib cimport malloc, free
from scipy.sparse import csr_array

def compute_stiffness_and_jacobi_matrix(nodes, cells, d_phi, x, permeability, q, w):
    '''Compute the stiffness matrix for the magnetic scalar potential.

    :param nodes:
        The coefficients of the nodal values.

    :param cells:
        The mesh connectivity.

    :param d_phi:
        The derivatives of the fem basis function on the reference element.

    :param x:
        The solution vector at which the jacobian is computed.

    :param q:
        The quadrature points.

    :param w:
        The quarature weights.
    
    :param permeability:
        A permeability object (see materials.py).

    :return:
        The stiffness matrix and the jacobian.
    '''

    # ===========================================
    # Python Part
    # ===========================================

    # the total number of nodes
    num_nodes = nodes.shape[0]

    # the number of nodes of the finite elements
    el_nodes = d_phi.shape[1]

    # the number of elements
    num_el = np.int32(len(cells) / el_nodes)

    # print('assembling the stiffness and jacobi matrix (scalar laplace) (c)...')

    # the required space for the triplet list
    num_triplets = num_el*el_nodes*el_nodes

    # the spline parameters for the permeability
    spl_t = np.zeros((1, ))
    spl_c = np.zeros((1, ))
    spl_k = 1
    spl_H_min = 0.0
    spl_mu_H_min = 4*np.pi*1e-7
    spl_num_coeffs = 0

    if permeability.get_type_spec() == 0:
        spl_mu_H_min = permeability.evaluate_mu(0.0)
    elif permeability.get_type_spec() == 1:
        spl_t, spl_c, spl_k, spl_H_min, spl_mu_H_min = permeability.get_spline_information()
        spl_num_coeffs = len(spl_t) - spl_k - 1
    else:
        print('Type specifyer of permeability not found!')

    # ===========================================
    # Convert python --> C
    # ===========================================
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] nodes_buff = np.ascontiguousarray(nodes.flatten(), dtype = np.double)
    cdef double* nodes_c = <double*> nodes_buff.data

    # the point data array is structured as follows
    # [ x_1, y_1, z_2, x_2, y_2, z_2, ... ]
        
    cdef np.ndarray[int, ndim=1, mode = 'c'] cells_buff = np.ascontiguousarray(cells.flatten(), dtype = np.int32)
    cdef int* c_c = <int*> cells_buff.data
    
    # the connectivity data is structured as follows, (N-noded element)
    # [ el_1_n_1, el_1_n_2, ... el_1_n_N, el_2_n_1, ...]

    cdef np.ndarray[double, ndim=1, mode = 'c'] x_buff = np.ascontiguousarray(x.flatten(), dtype = np.double)
    cdef double* x_c = <double*> x_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] q_buff = np.ascontiguousarray(q.flatten(), dtype = np.double)
    cdef double* q_c = <double*> q_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] w_buff = np.ascontiguousarray(w.flatten(), dtype = np.double)
    cdef double* w_c = <double*> w_buff.data
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] d_phi_buff = np.ascontiguousarray(d_phi.flatten(), dtype = np.double)
    cdef double* d_phi_c = <double*> d_phi_buff.data
    
    # the derivative data array is structured as follows:
    # [ d/du phi_1(x_1), d/dv phi_1(x_1), d/dw phi_1(x_1), d/du phi_2(x_1), ... , d/du phi_1(x_2)]
    # where x_i is the ith point, phi_i is the ith basis function and the derivative are d/du, d/dv, d/dw
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] spl_t_buff = np.ascontiguousarray(spl_t, dtype = np.double)
    cdef double* spl_t_c = <double*> spl_t_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] spl_c_buff = np.ascontiguousarray(spl_c, dtype = np.double)
    cdef double* spl_c_c = <double*> spl_c_buff.data

    # allocate space for the triplet lists
    cdef int* i_K_c = <int *> malloc( num_triplets * sizeof(int))
    cdef int* j_K_c = <int *> malloc( num_triplets * sizeof(int))
    cdef double* vals_K_c = <double *> malloc( num_triplets * sizeof(double))
    cdef ctools.triplet_list* triplets_K_c = ctools.make_triplet_list(num_triplets, i_K_c, j_K_c, vals_K_c)

    cdef int* i_dK_c = <int *> malloc( num_triplets * sizeof(int))
    cdef int* j_dK_c = <int *> malloc( num_triplets * sizeof(int))
    cdef double* vals_dK_c = <double *> malloc( num_triplets * sizeof(double))
    cdef ctools.triplet_list* triplets_dK_c = ctools.make_triplet_list(num_triplets, i_dK_c, j_dK_c, vals_dK_c)

    # make a permeability struct
    cdef ctools.permeability* perm_c = ctools.make_permeability(permeability.get_type_spec(), spl_k, spl_num_coeffs, spl_t_c, spl_c_c, spl_H_min, spl_mu_H_min)

    # make a c style mesh object
    cdef ctools.mesh* mesh_c = ctools.make_mesh(num_nodes,
                                             num_el,
                                             el_nodes,
                                             nodes_c,
                                             c_c)

    # make c style quadrature rule object
    cdef ctools.quad_3D* quad_c = ctools.make_quadrature_rule(len(w),
                                                                q_c,
                                                                w_c)

    # ===========================================
    # Run C code
    # ===========================================

    ctools.compute_K_dK(triplets_K_c, triplets_dK_c, mesh_c, quad_c, d_phi_c, x_c, perm_c)

    # ===========================================
    # Convert C -> python
    # ===========================================
    cdef view.array i_K = <int[:num_triplets]> i_K_c
    cdef view.array j_K = <int[:num_triplets]> j_K_c
    cdef view.array vals_K = <double[:num_triplets]> vals_K_c

    cdef view.array i_dK = <int[:num_triplets]> i_dK_c
    cdef view.array j_dK = <int[:num_triplets]> j_dK_c
    cdef view.array vals_dK = <double[:num_triplets]> vals_dK_c

    K = csr_array((vals_K, (i_K, j_K)), shape=(num_nodes, num_nodes))
    dK = csr_array((vals_dK, (i_dK, j_dK)), shape=(num_nodes, num_nodes))

    # ===========================================
    # Return
    # ===========================================
    return K, dK

def compute_stiffness_and_jacobi_matrix_curl_curl(nodes, cells, glob_ids, curls, d_phi, orientations, x, B_s, reluctance, q, w, num_dofs, num_el_dofs):
    '''Compute the stiffness matrix and the jacobian for the magnetic vector potential formulation.

    :param nodes:
        The coefficients of the nodal values.

    :param cells:
        The mesh connectivity.

    :param glob_ids:
        The global dof identifiers.

    :param curls:
        The curls evaluated on the basis functions.

    :param d_phi:
        The derivatives of the fem basis function on the reference element.

    :param orientations:
        The global orientations of the finite elements.

    :param x:
        The solution vector at which the jacobian is computed.

    :param B_s:
        An optional source potential. This is useful for the reduced vector potential formulation.
        If this array is empty, it will be ignored.

    :param reluctance:
        The material property. Here reluctance.

    :param q:
        The quadrature points.

    :param w:
        The quarature weights.

    :param num_dofs:
        The number of degrees of freedom.

    :param num_el_dofs:
        The number of DoFs for each finite element.
    
    :return:
        The stiffness matrix and the jacobian.
    '''

    # ===========================================
    # Python Part
    # ===========================================

    # the total number of nodes
    num_nodes = nodes.shape[0]

    # the number of nodes of the finite elements
    el_nodes = d_phi.shape[1]

    # the number of elements
    num_el = np.int32(len(cells) / el_nodes)

    # the required space for the triplet list
    num_triplets = num_el*num_el_dofs*num_el_dofs

    # the spline parameters for the permeability
    spl_x = np.zeros((0, ))
    spl_c = np.zeros((1, ))

    if reluctance.get_type_spec() == 1:
        spl_c, spl_x = reluctance.get_spline_information()
    elif reluctance.get_type_spec() == 0:
        spl_c[0] = reluctance.value
    else:
        print('Type specifyer of reluctance not found!')


    # ===========================================
    # Convert python --> C
    # ===========================================
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] nodes_buff = np.ascontiguousarray(nodes.flatten(), dtype = np.double)
    cdef double* nodes_c = <double*> nodes_buff.data

    # the point data array is structured as follows
    # [ x_1, y_1, z_2, x_2, y_2, z_2, ... ]
        
    cdef np.ndarray[int, ndim=1, mode = 'c'] cells_buff = np.ascontiguousarray(cells.flatten(), dtype = np.int32)
    cdef int* c_c = <int*> cells_buff.data
    
    # the connectivity data is structured as follows, (N-noded element)
    # [ el_1_n_1, el_1_n_2, ... el_1_n_N, el_2_n_1, ...]

    cdef np.ndarray[int, ndim=1, mode = 'c'] glob_ids_buff = np.ascontiguousarray(glob_ids.flatten(), dtype = np.int32)
    cdef int* glob_ids_c = <int*> glob_ids_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] x_buff = np.ascontiguousarray(x.flatten(), dtype = np.double)
    cdef double* x_c = <double*> x_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] q_buff = np.ascontiguousarray(q.flatten(), dtype = np.double)
    cdef double* q_c = <double*> q_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] w_buff = np.ascontiguousarray(w.flatten(), dtype = np.double)
    cdef double* w_c = <double*> w_buff.data
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] curls_buff = np.ascontiguousarray(curls.flatten(), dtype = np.double)
    cdef double* curls_c = <double*> curls_buff.data

    # Add here how curls are organized
    # 
    # 

    cdef np.ndarray[double, ndim=1, mode = 'c'] d_phi_buff = np.ascontiguousarray(d_phi.flatten(), dtype = np.double)
    cdef double* d_phi_c = <double*> d_phi_buff.data
    
    # the derivative data array is structured as follows:
    # [ d/du phi_1(x_1), d/dv phi_1(x_1), d/dw phi_1(x_1), d/du phi_2(x_1), ... , d/du phi_1(x_2)]
    # where x_i is the ith point, phi_i is the ith basis function and the derivative are d/du, d/dv, d/dw

    # the external fields (optional)
    cdef np.ndarray[double, ndim=1, mode = 'c'] B_s_buff = np.ascontiguousarray(B_s.flatten(), dtype = np.double)
    cdef double* B_s_c = <double*> B_s_buff.data
    
    cdef np.ndarray[int, ndim=1, mode = 'c'] orientations_buff = np.ascontiguousarray(orientations.flatten(), dtype = np.int32)
    cdef int* orientations_c = <int*> orientations_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] spl_x_buff = np.ascontiguousarray(spl_x, dtype = np.double)
    cdef double* spl_x_c = <double*> spl_x_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] spl_c_buff = np.ascontiguousarray(spl_c.flatten(), dtype = np.double)
    cdef double* spl_c_c = <double*> spl_c_buff.data

    # allocate space for the triplet lists
    cdef int* i_K_c = <int *> malloc( num_triplets * sizeof(int))
    cdef int* j_K_c = <int *> malloc( num_triplets * sizeof(int))
    cdef double* vals_K_c = <double *> malloc( num_triplets * sizeof(double))
    cdef ctools.triplet_list* triplets_K_c = ctools.make_triplet_list(num_triplets, i_K_c, j_K_c, vals_K_c)

    cdef int* i_dK_c = <int *> malloc( num_triplets * sizeof(int))
    cdef int* j_dK_c = <int *> malloc( num_triplets * sizeof(int))
    cdef double* vals_dK_c = <double *> malloc( num_triplets * sizeof(double))
    cdef ctools.triplet_list* triplets_dK_c = ctools.make_triplet_list(num_triplets, i_dK_c, j_dK_c, vals_dK_c)

    # make a reluctance struct
    cdef ctools.reluctance* rel_c = ctools.make_reluctance(reluctance.get_type_spec(),
                                                           len(spl_x) - 1,
                                                           spl_c_c,
                                                           spl_x_c)

    # make a c style mesh object
    cdef ctools.mesh* mesh_c = ctools.make_mesh(num_nodes,  num_el, el_nodes, nodes_c, c_c)

    # make c style quadrature rule object
    cdef ctools.quad_3D* quad_c = ctools.make_quadrature_rule(len(w), q_c, w_c)

    # allocate the rhs data
    cdef double* rhs_c = <double *> malloc( num_dofs * sizeof(double))

    for i in range(num_dofs):
        rhs_c[i] = 0.0

    # ===========================================
    # Run C code
    # ===========================================

    if len(B_s) == 0:
        ctools.compute_K_dK_Hcurl(triplets_K_c, 
                                  triplets_dK_c,
                                  glob_ids_c,
                                  rel_c,
                                  mesh_c,
                                  quad_c,
                                  curls_c,
                                  d_phi_c,
                                  orientations_c,
                                  x_c)

    else:
        ctools.compute_K_dK_Hcurl_red(triplets_K_c, 
                                  triplets_dK_c,
                                  rhs_c,
                                  glob_ids_c,
                                  rel_c,
                                  mesh_c,
                                  quad_c,
                                  curls_c,
                                  d_phi_c,
                                  orientations_c,
                                  x_c,
                                  B_s_c)

    # ===========================================
    # Convert C -> python
    # ===========================================
    cdef view.array i_K = <int[:num_triplets]> i_K_c
    cdef view.array j_K = <int[:num_triplets]> j_K_c
    cdef view.array vals_K = <double[:num_triplets]> vals_K_c

    cdef view.array i_dK = <int[:num_triplets]> i_dK_c
    cdef view.array j_dK = <int[:num_triplets]> j_dK_c
    cdef view.array vals_dK = <double[:num_triplets]> vals_dK_c

    cdef view.array rhs_array = <double[:num_dofs]> rhs_c

    rhs = np.asarray(rhs_array)
    rhs.shape = (num_dofs, )

    K = csr_array((vals_K, (i_K, j_K)), shape=(num_dofs, num_dofs))
    dK = csr_array((vals_dK, (i_dK, j_dK)), shape=(num_dofs, num_dofs))

    # ===========================================
    # Return
    # ===========================================
    return K, dK, rhs


def compute_rhs_Hcurl_red_c(nodes, cells, glob_ids, curls, d_phi, orientations, B_s, reluctance, x, q, w, num_dofs, num_el_dofs):
    '''Compute the stiffness matrix and the jacobian for the magnetic vector potential formulation.

    :param nodes:
        The coefficients of the nodal values.

    :param cells:
        The mesh connectivity.

    :param glob_ids:
        The global dof identifiers.

    :param curls:
        The curls evaluated on the basis functions.

    :param d_phi:
        The derivatives of the fem basis function on the reference element.

    :param orientations:
        The global orientations of the finite elements.

    :param H_s:
        The source magnetic field. Not the flux density!

    :param x:
        The initial solution.

    :param q:
        The quadrature points.

    :param w:
        The quarature weights.

    :param num_dofs:
        The number of degrees of freedom.

    :param num_el_dofs:
        The number of DoFs for each finite element.
    
    :return:
        The stiffness matrix and the jacobian.
    '''

    # ===========================================
    # Python Part
    # ===========================================

    # the total number of nodes
    num_nodes = nodes.shape[0]

    # the number of nodes of the finite elements
    el_nodes = d_phi.shape[1]

    # the number of elements
    num_el = np.int32(len(cells) / el_nodes)

    # the required space for the triplet list
    num_triplets = num_el*num_el_dofs*num_el_dofs

    # the spline parameters for the permeability
    spl_x = np.zeros((0, ))
    spl_c = np.zeros((1, ))

    if reluctance.get_type_spec() == 1:
        spl_c, spl_x = reluctance.get_spline_information()
    elif reluctance.get_type_spec() == 0:
        spl_c[0] = reluctance.value
    else:
        print('Type specifyer of reluctance not found!')

    # ===========================================
    # Convert python --> C
    # ===========================================
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] nodes_buff = np.ascontiguousarray(nodes.flatten(), dtype = np.double)
    cdef double* nodes_c = <double*> nodes_buff.data

    # the point data array is structured as follows
    # [ x_1, y_1, z_2, x_2, y_2, z_2, ... ]
        
    cdef np.ndarray[int, ndim=1, mode = 'c'] cells_buff = np.ascontiguousarray(cells.flatten(), dtype = np.int32)
    cdef int* c_c = <int*> cells_buff.data
    
    # the connectivity data is structured as follows, (N-noded element)
    # [ el_1_n_1, el_1_n_2, ... el_1_n_N, el_2_n_1, ...]

    cdef np.ndarray[int, ndim=1, mode = 'c'] glob_ids_buff = np.ascontiguousarray(glob_ids.flatten(), dtype = np.int32)
    cdef int* glob_ids_c = <int*> glob_ids_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] x_buff = np.ascontiguousarray(x.flatten(), dtype = np.double)
    cdef double* x_c = <double*> x_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] q_buff = np.ascontiguousarray(q.flatten(), dtype = np.double)
    cdef double* q_c = <double*> q_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] w_buff = np.ascontiguousarray(w.flatten(), dtype = np.double)
    cdef double* w_c = <double*> w_buff.data
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] curls_buff = np.ascontiguousarray(curls.flatten(), dtype = np.double)
    cdef double* curls_c = <double*> curls_buff.data

    # Add here how curls are organized
    # 
    # 

    cdef np.ndarray[double, ndim=1, mode = 'c'] d_phi_buff = np.ascontiguousarray(d_phi.flatten(), dtype = np.double)
    cdef double* d_phi_c = <double*> d_phi_buff.data
    
    # the derivative data array is structured as follows:
    # [ d/du phi_1(x_1), d/dv phi_1(x_1), d/dw phi_1(x_1), d/du phi_2(x_1), ... , d/du phi_1(x_2)]
    # where x_i is the ith point, phi_i is the ith basis function and the derivative are d/du, d/dv, d/dw

    # the external fields (optional)
    cdef np.ndarray[double, ndim=1, mode = 'c'] B_s_buff = np.ascontiguousarray(B_s.flatten(), dtype = np.double)
    cdef double* B_s_c = <double*> B_s_buff.data
    
    cdef np.ndarray[int, ndim=1, mode = 'c'] orientations_buff = np.ascontiguousarray(orientations.flatten(), dtype = np.int32)
    cdef int* orientations_c = <int*> orientations_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] spl_x_buff = np.ascontiguousarray(spl_x, dtype = np.double)
    cdef double* spl_x_c = <double*> spl_x_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] spl_c_buff = np.ascontiguousarray(spl_c.flatten(), dtype = np.double)
    cdef double* spl_c_c = <double*> spl_c_buff.data

    # make a c style mesh object
    cdef ctools.mesh* mesh_c = ctools.make_mesh(num_nodes,  num_el, el_nodes, nodes_c, c_c)

    # make c style quadrature rule object
    cdef ctools.quad_3D* quad_c = ctools.make_quadrature_rule(len(w), q_c, w_c)

    # make a reluctance struct
    cdef ctools.reluctance* rel_c = ctools.make_reluctance(reluctance.get_type_spec(),
                                                           len(spl_x) - 1,
                                                           spl_c_c,
                                                           spl_x_c)

    # allocate the rhs data
    cdef double* rhs_c = <double *> malloc( num_dofs * sizeof(double))

    for i in range(num_dofs):
        rhs_c[i] = 0.0

    # ===========================================
    # Run C code
    # ===========================================

    ctools.compute_rhs_Hcurl_red(rhs_c,
                                  glob_ids_c,
                                  rel_c,
                                  mesh_c,
                                  quad_c,
                                  curls_c,
                                  d_phi_c,
                                  orientations_c,
                                  B_s_c,
                                  x_c,
                                  num_el_dofs)

    # ===========================================
    # Convert C -> python
    # ===========================================
    cdef view.array rhs_array = <double[:num_dofs]> rhs_c

    rhs = np.asarray(rhs_array)
    rhs.shape = (num_dofs, )

    # ===========================================
    # Clean up
    # ===========================================

    
    # ===========================================
    # Return
    # ===========================================
    return rhs


def compute_B_Hcurl(nodes, cells, glob_ids, curls, phi, d_phi, orientations, x, q, w, num_dofs, num_el_dofs):
    '''Compute the B field vectors at local coordinates in the FEM mesh.

    :param nodes:
        The coefficients of the nodal values.

    :param cells:
        The mesh connectivity.

    :param glob_ids:
        The global dof identifiers.

    :param curls:
        The curls evaluated on the basis functions.

    :param phi:
        The fem basis function on the reference element.

    :param d_phi:
        The derivatives of the fem basis function on the reference element.

    :param orientations:
        The global orientations of the finite elements.

    :param x:
        The solution vector at which the jacobian is computed.

    :param q:
        The quadrature points.

    :param w:
        The quarature weights.

    :param num_dofs:
        The number of degrees of freedom.

    :param num_el_dofs:
        The number of DoFs for each finite element.
    
    :return:
        The B field vectors at the evaluation positions.
    '''

    # ===========================================
    # Python Part
    # ===========================================

    # the total number of nodes
    num_nodes = nodes.shape[0]

    # the number of nodes of the finite elements
    el_nodes = d_phi.shape[1]

    # the number of elements
    num_el = np.int32(len(cells) / el_nodes)

    # the required space for the triplet list
    num_triplets = num_el*num_el_dofs*num_el_dofs
    
    # ===========================================
    # Convert python --> C
    # ===========================================
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] nodes_buff = np.ascontiguousarray(nodes.flatten(), dtype = np.double)
    cdef double* nodes_c = <double*> nodes_buff.data

    # the point data array is structured as follows
    # [ x_1, y_1, z_2, x_2, y_2, z_2, ... ]
        
    cdef np.ndarray[int, ndim=1, mode = 'c'] cells_buff = np.ascontiguousarray(cells.flatten(), dtype = np.int32)
    cdef int* c_c = <int*> cells_buff.data
    
    # the connectivity data is structured as follows, (N-noded element)
    # [ el_1_n_1, el_1_n_2, ... el_1_n_N, el_2_n_1, ...]

    cdef np.ndarray[int, ndim=1, mode = 'c'] glob_ids_buff = np.ascontiguousarray(glob_ids.flatten(), dtype = np.int32)
    cdef int* glob_ids_c = <int*> glob_ids_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] x_buff = np.ascontiguousarray(x.flatten(), dtype = np.double)
    cdef double* x_c = <double*> x_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] q_buff = np.ascontiguousarray(q.flatten(), dtype = np.double)
    cdef double* q_c = <double*> q_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] w_buff = np.ascontiguousarray(w.flatten(), dtype = np.double)
    cdef double* w_c = <double*> w_buff.data
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] curls_buff = np.ascontiguousarray(curls.flatten(), dtype = np.double)
    cdef double* curls_c = <double*> curls_buff.data

    # Add here how curls are organized
    # 
    # 

    cdef np.ndarray[double, ndim=1, mode = 'c'] phi_buff = np.ascontiguousarray(phi.flatten(), dtype = np.double)
    cdef double* phi_c = <double*> phi_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] d_phi_buff = np.ascontiguousarray(d_phi.flatten(), dtype = np.double)
    cdef double* d_phi_c = <double*> d_phi_buff.data
    
    # the derivative data array is structured as follows:
    # [ d/du phi_1(x_1), d/dv phi_1(x_1), d/dw phi_1(x_1), d/du phi_2(x_1), ... , d/du phi_1(x_2)]
    # where x_i is the ith point, phi_i is the ith basis function and the derivative are d/du, d/dv, d/dw
    
    cdef np.ndarray[int, ndim=1, mode = 'c'] orientations_buff = np.ascontiguousarray(orientations.flatten(), dtype = np.int32)
    cdef int* orientations_c = <int*> orientations_buff.data

    # allocate space for the B field vectors
    cdef double* B_c = <double *> malloc( 3 * num_el * len(w) * sizeof(double))

    # allocate space for the B fields
    cdef double* points_c = <double *> malloc( 3 * num_el * len(w) * sizeof(double))

    # make a c style mesh object
    cdef ctools.mesh* mesh_c = ctools.make_mesh(num_nodes,  num_el, el_nodes, nodes_c, c_c)

    # make c style quadrature rule object
    cdef ctools.quad_3D* quad_c = ctools.make_quadrature_rule(len(w), q_c, w_c)

    # ===========================================
    # Run C code
    # ===========================================

    ctools.compute_B_Hcurl(points_c, B_c, num_el_dofs, glob_ids_c, mesh_c,
                            quad_c, curls_c, phi_c, d_phi_c, orientations_c, x_c)

    # ===========================================
    # Convert C -> python
    # ===========================================
    cdef view.array B_array = <double[:3*num_el*len(w)]> B_c
    cdef view.array points_array = <double[:3*num_el*len(w)]> points_c

    points = np.asarray(points_array)
    points.shape = (num_el*len(w), 3)

    B = np.asarray(B_array)
    B.shape = (num_el*len(w), 3)

    # ===========================================
    # Return
    # ===========================================
    return points, B


def compute_B_line_segs_cpp(src, tar, current, radius):
    '''Launch the C code to compute the B field for given
    sources, targets and current.

    :param src:
        The sources in an N x 6 numpy array.

    :param tar:
        The targets in an M x 3 numpy array.

    :param current:
        The strand current.

    :param radius:
        The strand radius.

    :return:
        The B fields in an M x 3 numpy array. 
    '''

    # get the number of sources
    num_src = src.shape[0]
    
    if src.shape[1] != 6:
        print("Error! The source array must be of dimension N x 6!")
        return -1 
    
    # get the number of targets
    num_tar = tar.shape[0]
    if tar.shape[1] != 3:
        print("Error! The target array must be of dimension M x 3!")
        return -1 

    # ====================================
    # Convert python -> C
    # ====================================


    # make c type data pointers
    cdef np.ndarray[double, ndim=1, mode = 'c'] src_buff = np.ascontiguousarray(src.flatten(), dtype = np.double)
    cdef double* src_c = <double*> src_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] tar_buff = np.ascontiguousarray(tar.flatten(), dtype = np.double)
    cdef double* tar_c = <double*> tar_buff.data
        
    # ====================================
    # run cpp code
    # ====================================


    cdef double *B_c = <double *> malloc(3*num_tar*sizeof(double))
    ctools.compute_B_line_segs(B_c, src_c, tar_c, current, radius, num_src, num_tar)


    # ===========================================
    # Convert C -> python
    # ===========================================
    
    cdef view.array B_array = <double[:3*num_tar]> B_c

    B = np.asarray(B_array)
    B.shape = (num_tar, 3)

    return B.copy()