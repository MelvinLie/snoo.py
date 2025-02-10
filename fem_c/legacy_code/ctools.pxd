cdef extern from "ctools.h":

	ctypedef struct mesh:
		pass

	ctypedef struct triplet_list:
		pass

	ctypedef struct quad_3D:
		pass

	triplet_list* make_triplet_list(int number_of_triplets,
                                    int *row_ptr,
                                    int *col_ptr,
                                    double *vals)

	mesh* make_mesh(int num_nodes,
                       int num_elements,
                       int num_nodes_per_element,
                       double *nodes,
                       int *cells)

	quad_3D* make_quadrature_rule(int number_of_points,
								  double *points,
								  double *weights)

	void  compute_K(triplet_list *triplets, 
					mesh *msh,
					quad_3D *quad,
					double *d_phi_c)

	void  compute_K_A(triplet_list *triplets, 
					mesh *msh,
					quad_3D *quad,
					double *d_phi_c,
					double *curls_c,
					double *grads_c,
					double *v_c, 
					double *vs_c)

	void compute_K_div(triplet_list *triplets, 
					mesh *msh,
					quad_3D *quad,
					double *d_phi_c,
					double *grads_c)

	void compute_jacobian(double *J,
                      int *node_pos,
                      mesh *msh,
		      		  int num_quad,
                      double *d_phi_c)
    
	double invert_matrix_3x3(double *inv, double *mat)
    
	void invert_jacobians(double *J_det, double *J_inv, double *J, int num_quad)
    
	double kernel_scalar_laplace(double det,
                             double weight,
                             double *J_inv,
                             double *d_phi_1,
                             double *d_phi_2)

	double kernel_curl_curl(double det,
                             double weight,
                             double *J_inv,
                             double *curl_1,
                             double *curl_2,
                             double *grad_1,
                             double *grad_2)
	
	double kernel_div_div(double det,
                             double weight,
                             double *J_inv,
                             double *grad_1,
                             double *grad_2,
                             int row)

	double *compute_rhs_A_c(mesh *msh,
							quad_3D *quad,
							double *phi_c,
							double *d_phi_c,
							double *j_c)