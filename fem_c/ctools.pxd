cdef extern from "ctools.h":

	ctypedef struct mesh:
		pass

	ctypedef struct triplet_list:
		pass

	ctypedef struct quad_3D:
		pass

	ctypedef struct permeability:
		pass

	ctypedef struct reluctance:
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

	permeability* make_permeability(int type_spec,
                                       int degree,
									   int num_control_points,
                                       double *knots,
                                       double *control_points,
                                       double H_min,
                                       double mu_H_min)

	reluctance* make_reluctance(int type_spec,
										int num_intervals,
										double *coefficients,
										double *knots)

	double eval_cox_de_boor(int i, int k, double *t, double x)
	double eval_cox_de_boor_derivative(int i, int k, double *t, double x)

	double eval_basis_spline(double *t, double *c, int k, int n, double x)
	double eval_basis_spline_derivative(double *t, double *c, int k, int n, double x)

	double compute_mu(permeability *perm_c, double H_mag)
	double compute_mu_derivative(permeability *perm_c, double H_mag)

	void  compute_K(triplet_list *triplets, 
					mesh *msh,
					quad_3D *quad,
					double *d_phi_c)
		
	void  compute_K_dK(triplet_list *triplets_K, 
                    triplet_list *triplets_dK,
                    mesh *msh,
                    quad_3D *quad,
                    double *d_phi_c,
					double *x_c,
                    permeability *perm_c)

	
	void  compute_K_dK_Hcurl(triplet_list *triplets_K, 
						triplet_list *triplets_dK,
						int *glob_ids_c,
						reluctance *rel_c,
						mesh *msh,
						quad_3D *quad,
						double *curls_c,
						double *d_phi_c,
						int *orientations_c,
						double *x_c)

	void compute_K_dK_Hcurl_red(triplet_list *triplets_K, 
	                    triplet_list *triplets_dK,
	                    double *rhs_c,
	                    int *glob_ids_c,
	                    reluctance *rel_c,
	                    mesh *msh,
	                    quad_3D *quad,
	                    double *curls_c,
	                    double *d_phi_c,
	                    int *orientations_c,
	                    double *x_c,
	                    double *B_s)

	void compute_rhs_Hcurl_red(double *rhs_c,
						int *glob_ids_c,
                    	reluctance *rel_c,
						mesh *msh,
						quad_3D *quad,
						double *curls_c,
						double *d_phi_c,
						int *orientations_c,
						double *B_s,
                    	double *x_c,
						int num_el_dofs)

	void compute_B_Hcurl(double *points,
						double *B,
						int num_el_dofs,
						int *glob_ids_c,
						mesh *msh,
						quad_3D *quad,
						double *curls_c,
						double *phi_c,
						double *d_phi_c,
						int *orientations_c,
						double *x_c)
						
	void compute_B_line_segs(double* B_ret,
				const double *src_ptr,
				const double *tar_ptr,
				const double current,
				const double rad,
				const int num_src,
				const int num_tar)
