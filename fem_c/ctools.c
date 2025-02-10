#include "ctools.h"
#include <stdio.h>
#include <omp.h>
#include <math.h>

#ifndef M_PI
  #define M_PI 3.141592653589793238462643383279502984
#endif

struct triplet_list* make_triplet_list(int number_of_triplets,
                                       int *row_ptr,
                                       int *col_ptr,
                                       double *vals){

	// allocate the memory for the triplet list
	triplet_list *triplets	= (triplet_list*) calloc(1, sizeof(triplet_list));

	// fill it
	triplets->counter = 0;
	triplets->row = row_ptr;
    triplets->col = col_ptr;
	triplets->vals = vals;
    triplets->num_triplets = number_of_triplets;

	return triplets;

}

struct mesh* make_mesh(int num_nodes,
                       int num_elements,
                       int num_nodes_per_element,
                       double *nodes,
                       int *cells){

	// allocate the memory for the mesh
	mesh *m	= (mesh*) calloc(1, sizeof(mesh));

	// fill it
	m->num_nodes = num_nodes;
	m->num_elements = num_elements;
	m->num_nodes_per_element = num_nodes_per_element;
	m->nodes = nodes;
	m->cells = cells;

	return m;

	}

struct quad_3D* make_quadrature_rule(int number_of_points,
									 double *points,
									 double *weights){

	// allocate the memory for the quadrature rule
	quad_3D *q	= (quad_3D*) calloc(1, sizeof(quad_3D));

	q->num_points = number_of_points;
	q->points = points;
	q->weights = weights;

	return q;

}

struct permeability* make_permeability(int type_spec,
                                       int degree,
                                       int num_control_points,
                                       double *knots,
                                       double *control_points,
                                       double H_min,
                                       double mu_H_min){

	// allocate the memory for the permeability
	permeability *perm	= (permeability*) calloc(1, sizeof(permeability));

	// fill it
	perm->type_spec = type_spec;
	perm->degree = degree;
    perm->num_control_points = num_control_points;
    perm->knots = knots;
    perm->control_points = control_points;
    perm->H_min = H_min;
    perm->mu_H_min = mu_H_min;

	return perm;

}

struct reluctance* make_reluctance(int type_spec,
                                       int num_intervals,
                                       double *coefficients,
                                       double *knots){

	// allocate the memory for the reluctance
	reluctance *rel	= (reluctance*) calloc(1, sizeof(reluctance));

	// fill it
	rel->type_spec = type_spec;
	rel->num_intervals = num_intervals;
	rel->coefficients = coefficients;
	rel->knots = knots;

	return rel;

}

double eval_pchip_interpolator(const int num_intervals, const double *c, const double *x_k, const double x){

    double f = 0;

    if (num_intervals == -1){
        return c[0];
    }
    else{
        // loop over the intervals
        for(int i = 0; i < num_intervals; ++i){


            // check if the point is in the interval
            if ((x_k[i] <= x) && (x_k[i+1] > x)){

                // loop over the coefficients
                for(int j = 0; j < 4; ++j){
                    f += c[num_intervals*j + i]*pow(x - x_k[i], 3-j);
                }
            }

        }

        if (x_k[num_intervals] <= x){
            // loop over the coefficients
            for(int j = 0; j < 4; ++j){
                
                f += c[num_intervals*(j + 1) - 1]*pow(x - x_k[num_intervals-1], 3-j);
            }

        }
    }
    
    
    return f;

}

double eval_pchip_interpolator_derivative(const int num_intervals, const double *c, const double *x_k, const double x){

    double f = 0;

    if (num_intervals == -1){
        return 0.0;
    }
    else{
        // loop over the intervals
        for(int i = 0; i < num_intervals; ++i){


            // check if the point is in the interval
            if ((x_k[i] <= x) && (x_k[i+1] > x)){

                // loop over the coefficients
                for(int j = 0; j < 3; ++j){
                    f += (3-j)*c[num_intervals*j + i]*pow(x - x_k[i], 2-j);
                }
            }

        }

        if (x_k[num_intervals] <= x){
            // loop over the coefficients
            for(int j = 0; j < 3; ++j){
                
                f += (3-j)*c[num_intervals*(j + 1) - 1]*pow(x - x_k[num_intervals-1], 2-j);
            }

        }
    }
    
    
    return f;

}

double eval_cox_de_boor(int i, int k, double *t, double x) {
    /**
        * Evaluate Cox de Boor's algorithm for the computation of BSpline basis functions.
        *  
        * @param i evaluate the i'th basis function (integer)
        * @param k the spline degree (integer)
        * @param t the knot vector (double-array)
        * @param x the evaluation point (double)
        * @return The spline basis function value (double).
    */

    // Base case: k == 0
    if (k == 0) {
        if (t[i] <= x && x < t[i + 1]) {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    // Recursively calculate the basis functions
    double a = 0.0, b = 0.0;

    // First term
    if (t[i + k] != t[i]) {
        a = (x - t[i]) / (t[i + k] - t[i]) * eval_cox_de_boor(i, k - 1, t, x);
    }

    // Second term
    if (t[i + k + 1] != t[i + 1]) {
        b = (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * eval_cox_de_boor(i + 1, k - 1, t, x);
    }

    return a + b;
}

double eval_cox_de_boor_derivative(int i, int k, double *t, double x) {
    /**
        * Evaluate derivative of a BSpline basis function.
        *  
        * @param i evaluate the i'th basis function (integer)
        * @param k the spline degree (integer)
        * @param t the knot vector (double-array)
        * @param x the evaluation point (double)
        * @return The spline basis function derivative value (double).
    */

    // If degree is zero, the derivative is zero because the basis function is constant
    if (k == 0) {
        return 0.0;
    }

    double term1 = 0.0, term2 = 0.0;

    // First term: k / (t[i+k] - t[i]) * N_{i,k-1}(x)
    if (t[i + k] != t[i]) {
        term1 = k / (t[i + k] - t[i]) * eval_cox_de_boor(i, k - 1, t, x);
    }

    // Second term: k / (t[i+k+1] - t[i+1]) * N_{i+1,k-1}(x)
    if (t[i + k + 1] != t[i + 1]) {
        term2 = k / (t[i + k + 1] - t[i + 1]) * eval_cox_de_boor(i + 1, k - 1, t, x);
    }

    return term1 - term2;

}

double eval_basis_spline(double *t, double *c, int k, int n, double x) {
    /**
        * Evaluate a basis spline.
        *  
        * @param t the knot vector (double-array)
        * @param c the coefficient vector (double-array)
        * @param k the spline degree (integer)
        * @param n the number of basis functions (integer)
        * @param x the evaluation point (double)
        * @return The spline basis function value (double).
    */

    double spline_value = 0.0;

    for (int i = 0; i < n; i++) {
        spline_value += c[i] * eval_cox_de_boor(i, k, t, x);
    }

    return spline_value;
}

double eval_basis_spline_derivative(double *t, double *c, int k, int n, double x){
    /**
        * Evaluate the derivative of a basis spline.
        *  
        * @param t the knot vector (double-array)
        * @param c the coefficient vector (double-array)
        * @param k the spline degree (integer)
        * @param n the number of basis functions (integer)
        * @param x the evaluation point (double)
        * @return The spline basis function value (double).
    */

    double spline_derivative = 0.0;

    for (int i = 0; i < n; i++) {
        spline_derivative += c[i] * eval_cox_de_boor_derivative(i, k, t, x);

    }

    return spline_derivative;
}

double compute_mu(permeability *perm_c, double H_mag){

    double x;

    if (perm_c->type_spec == 0 || H_mag < perm_c->H_min){
        return perm_c->mu_H_min;
    }

    x = log10(H_mag);

    return eval_basis_spline(perm_c->knots, perm_c->control_points, perm_c->degree, perm_c->num_control_points, x);

}

double compute_mu_derivative(permeability *perm_c, double H_mag){

    double x, d_mu;

    if (perm_c->type_spec == 0 || H_mag < perm_c->H_min){
        return 0.0;
    }

    x = log10(H_mag);

    d_mu = eval_basis_spline_derivative(perm_c->knots, perm_c->control_points, perm_c->degree, perm_c->num_control_points, x);
    d_mu /= H_mag*log(10.0);

    return d_mu;

}

void compute_H_mag(double *H_mag, double *grad_phi, int num_points){


    for (int i = 0; i < num_points; ++i){

        H_mag[i] = sqrt(grad_phi[3*i]*grad_phi[3*i] + grad_phi[3*i+1]*grad_phi[3*i+1] + grad_phi[3*i+2]*grad_phi[3*i+2]);

    }
    return;

}

void compute_mu_and_derivative(double *mu, double *d_mu, permeability *perm_c, double *H_mag, int num_points){


    for (int i = 0; i < num_points; ++i){

        mu[i] = compute_mu(perm_c, H_mag[i]);
        d_mu[i] = compute_mu_derivative(perm_c, H_mag[i]);
        
    }
    return;

}

void compute_jacobian(double *J,
                      int *node_pos,
                      mesh *msh,
                      int num_quad,
                      double *d_phi_c){
    
    // the Jacobian array is structured as follows:
    // [ d/du x | (u_1,v_1,w_1) , d/dv x | (u_1,v_1,w_1), d/dw x | (u_1,v_1,w_1), ...
    //   d/du y | (u_1,v_1,w_1) , d/dv y | (u_1,v_1,w_1), d/dw y | (u_1,v_1,w_1), ...
    //   d/du z | (u_1,v_1,w_1) , d/dv z | (u_1,v_1,w_1), d/dw z | (u_1,v_1,w_1), ...
    //   d/du x | (u_2,v_2,w_2) , d/dv x | (u_2,v_2,w_2), d/dw x | (u_2,v_2,w_2), ...]
    
    int i;  // the node counter
    int j;	// the counter for the quadrature point
    int k; 	// the counter of the dimension (x,y,z)
    int l; 	// the counter of the parameterization (u,v,w)
    
    // zero the Jacobian
    for (j = 0; j < 9*num_quad; ++j){
        J[j] = 0.;
    }

    // loop over the basis functions
    for (i = 0; i < msh->num_nodes_per_element; ++i){
                
        // loop over the quadrature points
        for(j = 0; j < num_quad; ++j){
            
            // loop over the spatial dimensions x,y,z
            for (k = 0; k < 3; ++k){
                
                // loop over the parametric space u,v,w
                for (l = 0; l < 3; ++l){
                    // increment the jacobian
                    J[j*9 + k*3 + l] += msh->nodes[3*node_pos[i] + k]*d_phi_c[j*msh->num_nodes_per_element*3 + 3*i + l];
                }
            }
        }
    }
    return;
}

void mat_vec(double *result, const double *mat, const double *vec, const int transpose){

    if (transpose == 0){
        result[0] = mat[0]*vec[0] + mat[1]*vec[1] + mat[2]*vec[2];
        result[1] = mat[3]*vec[0] + mat[4]*vec[1] + mat[5]*vec[2];
        result[2] = mat[6]*vec[0] + mat[7]*vec[1] + mat[8]*vec[2];
    }
    else if (transpose == 1){
        result[0] = mat[0]*vec[0] + mat[3]*vec[1] + mat[6]*vec[2];
        result[1] = mat[1]*vec[0] + mat[4]*vec[1] + mat[7]*vec[2];
        result[2] = mat[2]*vec[0] + mat[5]*vec[1] + mat[8]*vec[2];
    }
    else{
        printf("Error! The transpose flag for mat_vec needs to be 0 or 1!\n");
    }

    return;
}

double dot_prod(const double *v1, const double *v2){

    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];

}

double invert_matrix_3x3(double *inv,
                       double *mat){
    
    double det = mat[0] * (mat[4] * mat[8] - mat[7] * mat[5]) -
                 mat[1] * (mat[3] * mat[8] - mat[5] * mat[6]) +
                 mat[2] * (mat[3] * mat[7] - mat[4] * mat[6]);
    
    double invdet = 1 / det;
    
    inv[0] = (mat[4] * mat[8] - mat[7] * mat[5]) * invdet;
    inv[1] = (mat[2] * mat[7] - mat[1] * mat[8]) * invdet;
    inv[2] = (mat[1] * mat[5] - mat[2] * mat[4]) * invdet;
    inv[3] = (mat[5] * mat[6] - mat[3] * mat[8]) * invdet;
    inv[4] = (mat[0] * mat[8] - mat[2] * mat[6]) * invdet;
    inv[5] = (mat[3] * mat[2] - mat[0] * mat[5]) * invdet;
    inv[6] = (mat[3] * mat[7] - mat[6] * mat[4]) * invdet;
    inv[7] = (mat[6] * mat[1] - mat[0] * mat[7]) * invdet;
    inv[8] = (mat[0] * mat[4] - mat[3] * mat[1]) * invdet;
    
    return det;
}


void invert_jacobians(double *J_det, double *J_inv, double *J, int num_quad){
    
    
    int i; // running index
    
    for (i = 0; i < num_quad; ++i){
     
        J_det[i] = invert_matrix_3x3(&J_inv[i*9], &J[i*9]);


    }
    

    return;
    
}

void evaluate_finite_element(double *points, int *node_pos,
                  int num_quad, int num_basis_fcns,
                   double *phi_c, double *nodes){
    
    for(int i = 0; i < num_quad; ++i){

        // zero points
        points[3*i  ] = 0.0;
        points[3*i+1] = 0.0;
        points[3*i+2] = 0.0;

        for(int j = 0; j < num_basis_fcns; ++j){

            points[3*i    ] += phi_c[num_basis_fcns*i + j]*nodes[3*node_pos[j]];
            points[3*i + 1] += phi_c[num_basis_fcns*i + j]*nodes[3*node_pos[j] + 1];
            points[3*i + 2] += phi_c[num_basis_fcns*i + j]*nodes[3*node_pos[j] + 2];


        }

    }
    
    return;

}

void compute_grad_phi(double *grad_phi,
                      int *node_pos,
                      int num_quad,
                      int num_basis_fcns,
                      double *d_phi_c,
                      double *x,
                      double *J_inv){

    // auxiliary variables
    double tmp_x, tmp_y, tmp_z;
    // set zero
    for(int i = 0; i < num_quad; ++i){
        grad_phi[3*i    ] = 0.0;
        grad_phi[3*i + 1] = 0.0;
        grad_phi[3*i + 2] = 0.0;
    }

    // evaluate the gradients in the reference domain
    for(int i = 0; i < num_quad; ++i){
        for(int j = 0; j < num_basis_fcns; ++j){

            grad_phi[3*i    ] += d_phi_c[3*num_basis_fcns*i + 3*j    ]*x[node_pos[j]];
            grad_phi[3*i + 1] += d_phi_c[3*num_basis_fcns*i + 3*j + 1]*x[node_pos[j]];
            grad_phi[3*i + 2] += d_phi_c[3*num_basis_fcns*i + 3*j + 2]*x[node_pos[j]];

        }

    }

    // transform the gradients in to the global domain
    for(int i = 0; i < num_quad; ++i){

        tmp_x = J_inv[9*i + 0]*grad_phi[3*i] + J_inv[9*i + 3]*grad_phi[3*i + 1] + J_inv[9*i + 6]*grad_phi[3*i + 2];
        tmp_y = J_inv[9*i + 1]*grad_phi[3*i] + J_inv[9*i + 4]*grad_phi[3*i + 1] + J_inv[9*i + 7]*grad_phi[3*i + 2];
        tmp_z = J_inv[9*i + 2]*grad_phi[3*i] + J_inv[9*i + 5]*grad_phi[3*i + 1] + J_inv[9*i + 8]*grad_phi[3*i + 2];

        grad_phi[3*i    ] = tmp_x;
        grad_phi[3*i + 1] = tmp_y;
        grad_phi[3*i + 2] = tmp_z;

    }
}

double kernel_scalar_laplace(double det,
                             double weight,
                             double *J_inv,
                             double *d_phi_1,
                             double *d_phi_2){
    
    double x_1 = J_inv[0]*d_phi_1[0] + J_inv[3]*d_phi_1[1] + J_inv[6]*d_phi_1[2];
    double y_1 = J_inv[1]*d_phi_1[0] + J_inv[4]*d_phi_1[1] + J_inv[7]*d_phi_1[2];    
    double z_1 = J_inv[2]*d_phi_1[0] + J_inv[5]*d_phi_1[1] + J_inv[8]*d_phi_1[2]; 
    
    double x_2 = J_inv[0]*d_phi_2[0] + J_inv[3]*d_phi_2[1] + J_inv[6]*d_phi_2[2];
    double y_2 = J_inv[1]*d_phi_2[0] + J_inv[4]*d_phi_2[1] + J_inv[7]*d_phi_2[2];    
    double z_2 = J_inv[2]*d_phi_2[0] + J_inv[5]*d_phi_2[1] + J_inv[8]*d_phi_2[2];
        
    return det*weight*(x_1*x_2 + y_1*y_2 + z_1*z_2);
    
}

double kernel_curl_curl(double det,
                             double weight,
                             double *J_inv,
                             double *curl_1,
                             double *curl_2){
    
    double curl_1_x = J_inv[0]*curl_1[0] + J_inv[3]*curl_1[1] + J_inv[6]*curl_1[2];
    double curl_1_y = J_inv[1]*curl_1[0] + J_inv[4]*curl_1[1] + J_inv[7]*curl_1[2]; 
    double curl_1_z = J_inv[2]*curl_1[0] + J_inv[5]*curl_1[1] + J_inv[8]*curl_1[2];
    
    double curl_2_x = J_inv[0]*curl_2[0] + J_inv[3]*curl_2[1] + J_inv[6]*curl_2[2];
    double curl_2_y = J_inv[1]*curl_2[0] + J_inv[4]*curl_2[1] + J_inv[7]*curl_2[2]; 
    double curl_2_z = J_inv[2]*curl_2[0] + J_inv[5]*curl_2[1] + J_inv[8]*curl_2[2];
        
    return det*weight*(curl_1_x*curl_2_x + curl_1_y*curl_2_y + curl_1_z*curl_2_z);
    
}

double kernel_div_div(double det,
                             double weight,
                             double *J_inv,
                             double *grad_1,
                             double *grad_2,
                             int row_1,
                             int row_2){

    double div_1 = J_inv[row_1]*grad_1[0] + J_inv[row_1 + 3]*grad_1[1] +  J_inv[row_1 + 6]*grad_1[2];
    double div_2 = J_inv[row_2]*grad_2[0] + J_inv[row_2 + 3]*grad_2[1] +  J_inv[row_2 + 6]*grad_2[2];

    return det*weight*div_1*div_2;

}

void compute_K(triplet_list *triplets,
				mesh *msh,
				quad_3D *quad,
                double *d_phi_c){

	// local variables
	int i, j, k, l; 			// running indices
	triplet_list my_triplets;	// a triplet list for each thread
	int num_triplets;			// the number of triplets to fill the sparse matrix
	int num_elements;			// the number of elements in the mesh
	double* J;					// Jacobians at the integration points 
    double* J_inv;				// inverse of the Jacobians at the integration points
    double* J_det;				// determinants of the Jacobians at the integration points
    double int_val;             // the integration value
    int enable_print = 1;

    // we will use these variables very often
    int nodes_per_el = msh->num_nodes_per_element;
    int num_quad = quad->num_points;

	// get some info about the problem
	num_triplets = triplets->num_triplets;
	num_elements = msh->num_elements;

	printf("=======================\n");
	printf("Mesh info:\n");
	printf("  number of elements = %d\n", msh->num_elements);
	printf("  number of nodes = %d\n", msh->num_nodes);
	printf("  number of nodes per element = %d\n", msh->num_nodes_per_element);
	printf("  number of triplets = %d\n", num_triplets);
	printf("=======================\n");

	#pragma omp parallel
	{

		// make a triplet list for this task
		my_triplets.num_triplets = num_triplets;
		my_triplets.row = (int*) calloc(num_triplets, sizeof(int));
		my_triplets.col = (int*) calloc(num_triplets, sizeof(int));
		my_triplets.vals = (double*) calloc(num_triplets, sizeof(double));
		my_triplets.counter = 0;
        
        // allocate also the space for the Jacobians
        J = (double*) calloc(9*num_quad, sizeof(double));
        J_inv = (double*) calloc(9*num_quad, sizeof(double));
        J_det = (double*) calloc(num_quad, sizeof(double));
                
		#pragma omp for
		for(i = 0; i < num_elements; ++i){
            
            // compute the Jacobian of this element
            compute_jacobian(J, &msh->cells[i*nodes_per_el], msh, num_quad, d_phi_c);
            
            // invert the jacobians and compute the determinants
            invert_jacobians(J_det, J_inv, J, num_quad);
                
            // each kombination of basis functions
            for(k = 0; k < nodes_per_el; ++k){
                for(l = k; l < nodes_per_el; ++l){
                    
                    // reset int val
                    int_val = 0.0;
                    
                    // each integration point
                    for (j = 0; j < num_quad; ++j){
                        

                        int_val += kernel_scalar_laplace(J_det[j], quad->weights[j],
                                                         &J_inv[9*j], &d_phi_c[j*nodes_per_el*3 + 3*k],
                                                         &d_phi_c[j*nodes_per_el*3 + 3*l]);

                        
                    }
                    
                    enable_print += 1;
                    
                    // add to the triplet list
                    my_triplets.row[my_triplets.counter] = msh->cells[i*nodes_per_el + k];
                    my_triplets.col[my_triplets.counter] = msh->cells[i*nodes_per_el + l];
                    my_triplets.vals[my_triplets.counter] = int_val;
                    my_triplets.counter += 1;


                    if (k != l){
                        // apply symmetry
                        my_triplets.row[my_triplets.counter] = msh->cells[i*nodes_per_el + l];
                        my_triplets.col[my_triplets.counter] = msh->cells[i*nodes_per_el + k];
                        my_triplets.vals[my_triplets.counter] = int_val;
                        my_triplets.counter += 1;   
                    }
                    
                }
            }
            
            // zero the Jacobian
            for (j = 0; j < 9*num_quad; ++j){
                J[j] = 0.;
            }
            
		}

    	#pragma omp critical
		{
			// printf("  append %d triplets\n", my_triplets.counter);
            
            for (i = 0; i < my_triplets.counter; ++i){
                triplets->row[triplets->counter] = my_triplets.row[i];
                triplets->col[triplets->counter] = my_triplets.col[i];
                triplets->vals[triplets->counter] = my_triplets.vals[i];
                triplets->counter += 1;
                // printf("( %d , %d , %f )\n", my_triplets.row[i], my_triplets.col[i], my_triplets.vals[i]);
            }
            
		}
                   
        free(my_triplets.row);
        free(my_triplets.col);
        free(my_triplets.vals);
        free(J);
                   
    }
	// for (i = 0; i < 10; ++i){

	// 	printf("Hello %d\n", i);
	// }
	return;

}


void  compute_K_dK(triplet_list *triplets_K, 
                    triplet_list *triplets_dK,
                    mesh *msh,
                    quad_3D *quad,
                    double *d_phi_c,
                    double *x_c,
                    permeability *perm_c){

	// local variables
	int i, j, k, l; 			    // running indices
	triplet_list my_triplets_K;     // triplet lists for stiffness matrix
    triplet_list my_triplets_dK ;	// triplet lists for jacobian
	int num_triplets;			    // the number of triplets to fill the sparse matrix
	int num_elements;			    // the number of elements in the mesh
	double* J;					    // Jacobians at the integration points 
    double* J_inv;				    // inverse of the Jacobians at the integration points
    double* J_det;				    // determinants of the Jacobians at the integration points
    double* grad_phi;               // The gradients of the scalar potential
    double* H_mag;                  // The magnitudes of the magnetic field vectors
    double* mu;                     // The permeability
    double* d_mu;                   // The permeability derivative
    double* grad_u;                 // The gradients of the basis functions in the global coordinates
    double* grad_v;                 // The gradients of the basis functions in the global coordinates
    double int_val_K;               // the integration value for the stiffness matrix
    double int_val_dK;               // the integration value for the jacobian
    int enable_print = 1;

    // we will use these variables very often
    int nodes_per_el = msh->num_nodes_per_element;
    int num_quad = quad->num_points;

	// get some info about the problem
	num_triplets = triplets_K->num_triplets;
	num_elements = msh->num_elements;


	// printf("=======================\n");
	// printf("Mesh info:\n");
	// printf("  number of elements = %d\n", msh->num_elements);
	// printf("  number of nodes = %d\n", msh->num_nodes);
	// printf("  number of nodes per element = %d\n", msh->num_nodes_per_element);
	// printf("  number of triplets = %d\n", num_triplets);
	// printf("=======================\n");

	// #pragma omp parallel
	{

		// make triplet lists for this task
		my_triplets_K.num_triplets = num_triplets;
		my_triplets_K.row = (int*) calloc(num_triplets, sizeof(int));
		my_triplets_K.col = (int*) calloc(num_triplets, sizeof(int));
		my_triplets_K.vals = (double*) calloc(num_triplets, sizeof(double));
		my_triplets_K.counter = 0;
        
		my_triplets_dK.num_triplets = num_triplets;
		my_triplets_dK.row = (int*) calloc(num_triplets, sizeof(int));
		my_triplets_dK.col = (int*) calloc(num_triplets, sizeof(int));
		my_triplets_dK.vals = (double*) calloc(num_triplets, sizeof(double));
		my_triplets_dK.counter = 0;

        // allocate also the space for the Jacobians
        J = (double*) calloc(9*num_quad, sizeof(double));
        J_inv = (double*) calloc(9*num_quad, sizeof(double));
        J_det = (double*) calloc(num_quad, sizeof(double));
        grad_phi = (double*) calloc(3*num_quad, sizeof(double));
        H_mag = (double*) calloc(num_quad, sizeof(double));
        mu = (double*) calloc(num_quad, sizeof(double));
        d_mu = (double*) calloc(num_quad, sizeof(double));

        // the gradients of the basis functions in the global coordinates
        grad_u = (double*) calloc(3, sizeof(double));
        grad_v = (double*) calloc(3, sizeof(double));

		// #pragma omp for
		for(i = 0; i < num_elements; ++i){
            
            // compute the Jacobian of this element
            compute_jacobian(J, &msh->cells[i*nodes_per_el], msh, num_quad, d_phi_c);
            
            // invert the jacobians and compute the determinants
            invert_jacobians(J_det, J_inv, J, num_quad);

            // evaluate the gradient of the scalar potential solution x
            compute_grad_phi(grad_phi, &msh->cells[i*nodes_per_el], num_quad, nodes_per_el, d_phi_c, x_c, J_inv);

            // compute H_mag
            compute_H_mag(H_mag, grad_phi, num_quad);

            // compute mu and d_mu
            compute_mu_and_derivative(mu, d_mu, perm_c, H_mag, num_quad);

            // each kombination of basis functions
            for(k = 0; k < nodes_per_el; ++k){
                for(l = k; l < nodes_per_el; ++l){
                    
                    // reset int val
                    int_val_K = 0.0;
                    int_val_dK = 0.0;
                    
                    // each integration point
                    for (j = 0; j < num_quad; ++j){
                        
                        // map the gradients
                        mat_vec(grad_u, &J_inv[9*j], &d_phi_c[j*nodes_per_el*3 + 3*k], 1);
                        mat_vec(grad_v, &J_inv[9*j], &d_phi_c[j*nodes_per_el*3 + 3*l], 1);
                        
                        // increment kernel K
                        int_val_K += mu[j]*J_det[j]*quad->weights[j]*dot_prod(grad_u, grad_v);


                        if (H_mag[j] >= 1e-14){

                            // increment kernel dK
                            int_val_dK += d_mu[j]*J_det[j]*quad->weights[j]*dot_prod(grad_u, &grad_phi[3*j])*dot_prod(grad_v, &grad_phi[3*j])/H_mag[j];

                        }

                        
                    }
                    
                    enable_print += 1;
                    
                    // add to the triplet list
                    my_triplets_K.row[my_triplets_K.counter] = msh->cells[i*nodes_per_el + k];
                    my_triplets_K.col[my_triplets_K.counter] = msh->cells[i*nodes_per_el + l];
                    my_triplets_K.vals[my_triplets_K.counter] = int_val_K;
                    my_triplets_K.counter += 1;

                    // add to the triplet list
                    my_triplets_dK.row[my_triplets_dK.counter] = msh->cells[i*nodes_per_el + k];
                    my_triplets_dK.col[my_triplets_dK.counter] = msh->cells[i*nodes_per_el + l];
                    my_triplets_dK.vals[my_triplets_dK.counter] = int_val_dK;
                    my_triplets_dK.counter += 1;

                    if (k != l){
                        // apply symmetry
                        my_triplets_K.row[my_triplets_K.counter] = msh->cells[i*nodes_per_el + l];
                        my_triplets_K.col[my_triplets_K.counter] = msh->cells[i*nodes_per_el + k];
                        my_triplets_K.vals[my_triplets_K.counter] = int_val_K;
                        my_triplets_K.counter += 1;   

                        my_triplets_dK.row[my_triplets_dK.counter] = msh->cells[i*nodes_per_el + l];
                        my_triplets_dK.col[my_triplets_dK.counter] = msh->cells[i*nodes_per_el + k];
                        my_triplets_dK.vals[my_triplets_dK.counter] = int_val_dK;
                        my_triplets_dK.counter += 1;   

                    }
                    
                }
            }
            
            // zero the Jacobian
            for (j = 0; j < 9*num_quad; ++j){
                J[j] = 0.;
            }
            
		}

    	// #pragma omp critical
		{
			// printf("  append %d triplets\n", my_triplets.counter);
            
            for (i = 0; i < my_triplets_K.counter; ++i){
                triplets_K->row[triplets_K->counter] = my_triplets_K.row[i];
                triplets_K->col[triplets_K->counter] = my_triplets_K.col[i];
                triplets_K->vals[triplets_K->counter] = my_triplets_K.vals[i];
                triplets_K->counter += 1;

                triplets_dK->row[triplets_dK->counter] = my_triplets_dK.row[i];
                triplets_dK->col[triplets_dK->counter] = my_triplets_dK.col[i];
                triplets_dK->vals[triplets_dK->counter] = my_triplets_dK.vals[i];
                triplets_dK->counter += 1;
                // printf("( %d , %d , %f )\n", my_triplets.row[i], my_triplets.col[i], my_triplets.vals[i]);
            }
            
		}
                   
        free(my_triplets_K.row);
        free(my_triplets_K.col);
        free(my_triplets_K.vals);
        free(my_triplets_dK.row);
        free(my_triplets_dK.col);
        free(my_triplets_dK.vals);
        free(J);
                   
    }

	return;

}


void compute_B_in_element(double *B,
                          double *B_s,
                          int num_points,
                          int num_el_dofs,
                          const int *glob_ids,
                          const double *x,
                          const double *curl_w_hat,
                          const double *J,
                          const double *det_J){


    // the result of a matrix vector product we need to compute below
    double mv[3];


    for(int m = 0; m < num_points; ++m){
        
        // take the source field (for reduced vector potential) otherwise zero
        B[3*m  ] = B_s[3*m  ];
        B[3*m+1] = B_s[3*m+1];
        B[3*m+2] = B_s[3*m+2];
        
        for(int k = 0; k < num_el_dofs; ++k){

            mat_vec(mv,
                    &J[9*m],
                    &curl_w_hat[3*(m*num_el_dofs + k)],
                    0);

            B[3*m  ] += mv[0] * x[glob_ids[k]] / det_J[m];
            B[3*m+1] += mv[1] * x[glob_ids[k]] / det_J[m];
            B[3*m+2] += mv[2] * x[glob_ids[k]] / det_J[m];
            
        }
    }

}

void compute_K_dK_Hcurl(triplet_list *triplets_K, 
                    triplet_list *triplets_dK,
                    int *glob_ids_c,
                    reluctance *rel_c,
                    mesh *msh,
                    quad_3D *quad,
                    double *curls_c,
                    double *d_phi_c,
                    int *orientations_c,
                    double *x_c){

	// local variables
	int i, j, k, l; 			    // running indices
	// triplet_list my_triplets_K;     // triplet lists for stiffness matrix
    // triplet_list my_triplets_dK ;	// triplet lists for jacobian
	int num_triplets;			    // the number of triplets to fill the sparse matrix
	int num_elements;			    // the number of elements in the mesh
	double* J;					    // Jacobians at the integration points 
    double* J_inv;				    // inverse of the Jacobians at the integration points
    double* J_det;				    // determinants of the Jacobians at the integration points
    double* grad_phi;               // The gradients of the scalar potential
    double* B;                      // The components of the magnetic flux density vectors
    double* B_s;                    // The components of the source field here zeros
    double* B_mag;                  // The magnitudes of the magnetic flux density vectors
    double* nu;                     // The reluctance
    double* d_nu;                   // The reluctance derivative
    double* curl_u;                 // The curls of the basis functions in the global coordinates
    double* curl_v;                 // The curls of the basis functions in the global coordinates
    double int_val_K;               // the integration value for the stiffness matrix
    double int_val_dK;              // the integration value for the jacobian
    double proj_u;                  // the projection of B onto the curl of the basis function
    double proj_v;                  // the projection of B onto the curl of the test function
    int offs;                       // offset in the curls array for the given orientation
    int num_el_dofs;                // the number of DoFs per element

    // we will use these variables very often
    int nodes_per_el = msh->num_nodes_per_element;
    int num_quad = quad->num_points;

	// get some info about the problem
	num_triplets = triplets_K->num_triplets;
	num_elements = msh->num_elements;

    // calculate the number of DoFs per finite element
    num_el_dofs = (int) sqrt(num_triplets/num_elements);



	// printf("=======================\n");
	// printf("Mesh info:\n");
	// printf("  number of elements = %d\n", msh->num_elements);
	// printf("  number of nodes = %d\n", msh->num_nodes);
	// printf("  number of nodes per element = %d\n", msh->num_nodes_per_element);
	// printf("  number of triplets = %d\n", num_triplets);
        // printf("  number of DoFs per element = %d\n", num_el_dofs);
	// printf("=======================\n");

	// #pragma omp parallel
	// {

	// make triplet lists for this task (use this for parallel code)
	// my_triplets_K.num_triplets = num_triplets;
	// my_triplets_K.row = (int*) calloc(num_triplets, sizeof(int));
	// my_triplets_K.col = (int*) calloc(num_triplets, sizeof(int));
	// my_triplets_K.vals = (double*) calloc(num_triplets, sizeof(double));
	// my_triplets_K.counter = 0;
        
	// my_triplets_dK.num_triplets = num_triplets;
	// my_triplets_dK.row = (int*) calloc(num_triplets, sizeof(int));
	// my_triplets_dK.col = (int*) calloc(num_triplets, sizeof(int));
	// my_triplets_dK.vals = (double*) calloc(num_triplets, sizeof(double));
	// my_triplets_dK.counter = 0;

    // allocate also the space for the Jacobians
    J = (double*) calloc(9*num_quad, sizeof(double));
    J_inv = (double*) calloc(9*num_quad, sizeof(double));
    J_det = (double*) calloc(num_quad, sizeof(double));
    grad_phi = (double*) calloc(3*num_quad, sizeof(double));

    // allocate space for the B field evaluations
    B = (double*) calloc(3*num_quad, sizeof(double));

    // allocate space for the B field evaluations
    B_s = (double*) calloc(3*num_quad, sizeof(double));

    // allocate space for the B field evaluations
    B_mag = (double*) calloc(3*num_quad, sizeof(double));

    // allocate space for the reluctances
    nu = (double*) calloc(num_quad, sizeof(double));

    // allocate space for the reluctance derivatives
    d_nu = (double*) calloc(num_quad, sizeof(double));

    // the curls of the basis functions in the global coordinates
    curl_u = (double*) calloc(3, sizeof(double));
    curl_v = (double*) calloc(3, sizeof(double));


    // loop over all finite elements
	// 	// #pragma omp for
	for(i = 0; i < num_elements; ++i){
            

        // compute the Jacobian of this element
        compute_jacobian(J, &msh->cells[i*nodes_per_el], msh, num_quad, d_phi_c);
            
        // invert the jacobians and compute the determinants
        invert_jacobians(J_det, J_inv, J, num_quad);
       

        // this is the offset in the curls array
        offs = 3*num_quad*num_el_dofs*orientations_c[i];

        // compute the B field at the quadrature points
        compute_B_in_element(B, B_s, num_quad, num_el_dofs, &glob_ids_c[i*num_el_dofs], x_c, &curls_c[offs], J, J_det);


        // evaluate also the reluctance and its derivative and store the B mag values
        for (j = 0; j < num_quad; ++j){

            B_mag[j] = sqrt(B[3*j]*B[3*j] + B[3*j+1]*B[3*j+1] + B[3*j+2]*B[3*j+2]);

            nu[j] = eval_pchip_interpolator(rel_c->num_intervals,
                                            rel_c->coefficients,
                                            rel_c->knots,
                                            B_mag[j]);

            d_nu[j] = eval_pchip_interpolator_derivative(rel_c->num_intervals,
                                                         rel_c->coefficients,
                                                         rel_c->knots,
                                                         B_mag[j]);

        }

        // sum over all combinations of basis functions (apply symmetry)
        for (j = 0; j < num_el_dofs; ++j){

            for (k = j; k < num_el_dofs; ++k){
                
                // reset the integration values
                int_val_K = 0.0;
                int_val_dK = 0.0;

                // integrate
                for (l = 0; l < num_quad; ++l){
                    
                    // transform the curls to the global frame
                    mat_vec(curl_u, &J[9*l], &curls_c[offs + 3*(l*num_el_dofs + j)], 0);
                    mat_vec(curl_v, &J[9*l], &curls_c[offs + 3*(l*num_el_dofs + k)], 0);

                    // increment the integration value for K
                    int_val_K += nu[l]*(curl_u[0]*curl_v[0] + curl_u[1]*curl_v[1] + curl_u[2]*curl_v[2])*quad->weights[l]/J_det[l];

                    // increment the integration value for dK
                    if (B_mag[l] >= 1e-14){

                        // compute the scalar products curl_u.B and curl_v.B
                        proj_u = curl_u[0]*B[3*l] + curl_u[1]*B[3*l+1] + curl_u[2]*B[3*l+2];
                        proj_v = curl_v[0]*B[3*l] + curl_v[1]*B[3*l+1] + curl_v[2]*B[3*l+2];

                        // increment the integration value for dK
                        int_val_dK += d_nu[l]*proj_u*proj_v*quad->weights[l]/J_det[l]/B_mag[l];
                    }
                }

                // fill the triplet list
                triplets_K->row[triplets_K->counter] = glob_ids_c[i*num_el_dofs + j];
                triplets_K->col[triplets_K->counter] = glob_ids_c[i*num_el_dofs + k];
                triplets_K->vals[triplets_K->counter] = int_val_K;
                triplets_K->counter += 1;

                triplets_dK->row[triplets_dK->counter] = glob_ids_c[i*num_el_dofs + j];
                triplets_dK->col[triplets_dK->counter] = glob_ids_c[i*num_el_dofs + k];
                triplets_dK->vals[triplets_dK->counter] = int_val_dK;
                triplets_dK->counter += 1;

                // apply symmetry
                if(j != k){
                    triplets_K->row[triplets_K->counter] = glob_ids_c[i*num_el_dofs + k];
                    triplets_K->col[triplets_K->counter] = glob_ids_c[i*num_el_dofs + j];
                    triplets_K->vals[triplets_K->counter] = int_val_K;
                    triplets_K->counter += 1;

                    triplets_dK->row[triplets_dK->counter] = glob_ids_c[i*num_el_dofs + k];
                    triplets_dK->col[triplets_dK->counter] = glob_ids_c[i*num_el_dofs + j];
                    triplets_dK->vals[triplets_dK->counter] = int_val_dK;
                    triplets_dK->counter += 1;
                }
            }



        } 

        // zero the Jacobian
        for (j = 0; j < 9*num_quad; ++j){
            J[j] = 0.;
        }
 
	// 	}

    // 	// #pragma omp critical
	// 	{
	// 		// printf("  append %d triplets\n", my_triplets.counter);
            
    //         for (i = 0; i < my_triplets_K.counter; ++i){
    //             triplets_K->row[triplets_K->counter] = my_triplets_K.row[i];
    //             triplets_K->col[triplets_K->counter] = my_triplets_K.col[i];
    //             triplets_K->vals[triplets_K->counter] = my_triplets_K.vals[i];
    //             triplets_K->counter += 1;

    //             triplets_dK->row[triplets_dK->counter] = my_triplets_dK.row[i];
    //             triplets_dK->col[triplets_dK->counter] = my_triplets_dK.col[i];
    //             triplets_dK->vals[triplets_dK->counter] = my_triplets_dK.vals[i];
    //             triplets_dK->counter += 1;
    //             // printf("( %d , %d , %f )\n", my_triplets.row[i], my_triplets.col[i], my_triplets.vals[i]);
    //         }
            
	// 	}
                   
    //     free(my_triplets_K.row);
    //     free(my_triplets_K.col);
    //     free(my_triplets_K.vals);
    //     free(my_triplets_dK.row);
    //     free(my_triplets_dK.col);
    //     free(my_triplets_dK.vals);
    //     free(J);
                   
    }

	return;

}

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
                    double *B_s){

    // local variables
    int i, j, k, l;                 // running indices
    // triplet_list my_triplets_K;     // triplet lists for stiffness matrix
    // triplet_list my_triplets_dK ;   // triplet lists for jacobian
    int num_triplets;               // the number of triplets to fill the sparse matrix
    int num_elements;               // the number of elements in the mesh
    double* J;                      // Jacobians at the integration points 
    double* J_inv;                  // inverse of the Jacobians at the integration points
    double* J_det;                  // determinants of the Jacobians at the integration points
    double* grad_phi;               // The gradients of the scalar potential
    double* B;                      // The components of the magnetic flux density vectors
    double* B_mag;                  // The magnitudes of the magnetic flux density vectors
    double* nu;                     // The reluctance
    double* d_nu;                   // The reluctance derivative
    double* curl_u;                 // The curls of the basis functions in the global coordinates
    double* curl_v;                 // The curls of the basis functions in the global coordinates
    double int_val_K;               // the integration value for the stiffness matrix
    double int_val_dK;              // the integration value for the jacobian
    double proj_u;                  // the projection of B onto the curl of the basis function
    double proj_v;                  // the projection of B onto the curl of the test function
    int offs;                       // offset in the curls array for the given orientation
    int num_el_dofs;                // the number of DoFs per element
    double nu_0 = 1.0/4.0/M_PI*1e7; // The vacuum reluctance

    // we will use these variables very often
    int nodes_per_el = msh->num_nodes_per_element;
    int num_quad = quad->num_points;

    // get some info about the problem
    num_triplets = triplets_K->num_triplets;
    num_elements = msh->num_elements;

    // calculate the number of DoFs per finite element
    num_el_dofs = (int) sqrt(num_triplets/num_elements);


    // allocate also the space for the Jacobians
    J = (double*) calloc(9*num_quad, sizeof(double));
    J_inv = (double*) calloc(9*num_quad, sizeof(double));
    J_det = (double*) calloc(num_quad, sizeof(double));
    grad_phi = (double*) calloc(3*num_quad, sizeof(double));

    // allocate space for the B field evaluations
    B = (double*) calloc(3*num_quad, sizeof(double));

    // allocate space for the B field evaluations
    B_mag = (double*) calloc(3*num_quad, sizeof(double));

    // allocate space for the reluctances
    nu = (double*) calloc(num_quad, sizeof(double));

    // allocate space for the reluctance derivatives
    d_nu = (double*) calloc(num_quad, sizeof(double));

    // the curls of the basis functions in the global coordinates
    curl_u = (double*) calloc(3, sizeof(double));
    curl_v = (double*) calloc(3, sizeof(double));


    // loop over all finite elements
    //  // #pragma omp for
    for(i = 0; i < num_elements; ++i){
            

        // compute the Jacobian of this element
        compute_jacobian(J, &msh->cells[i*nodes_per_el], msh, num_quad, d_phi_c);
            
        // invert the jacobians and compute the determinants
        invert_jacobians(J_det, J_inv, J, num_quad);
       

        // this is the offset in the curls array
        offs = 3*num_quad*num_el_dofs*orientations_c[i];

        // compute the B field at the quadrature points
        compute_B_in_element(B, &B_s[i*3*num_quad], num_quad, num_el_dofs, &glob_ids_c[i*num_el_dofs], x_c, &curls_c[offs], J, J_det);


        // evaluate also the reluctance and its derivative and store the B mag values
        for (j = 0; j < num_quad; ++j){

            B_mag[j] = sqrt(B[3*j]*B[3*j] + B[3*j+1]*B[3*j+1] + B[3*j+2]*B[3*j+2]);

            nu[j] = eval_pchip_interpolator(rel_c->num_intervals,
                                            rel_c->coefficients,
                                            rel_c->knots,
                                            B_mag[j]);

            d_nu[j] = eval_pchip_interpolator_derivative(rel_c->num_intervals,
                                                         rel_c->coefficients,
                                                         rel_c->knots,
                                                         B_mag[j]);

        }

        // sum over all combinations of basis functions (apply symmetry)
        for (j = 0; j < num_el_dofs; ++j){

            for (k = j; k < num_el_dofs; ++k){
                
                // reset the integration values
                int_val_K = 0.0;
                int_val_dK = 0.0;

                // integrate
                for (l = 0; l < num_quad; ++l){
                    
                    // transform the curls to the global frame
                    mat_vec(curl_u, &J[9*l], &curls_c[offs + 3*(l*num_el_dofs + j)], 0);
                    mat_vec(curl_v, &J[9*l], &curls_c[offs + 3*(l*num_el_dofs + k)], 0);

                    // increment the integration value for K
                    int_val_K += nu[l]*(curl_u[0]*curl_v[0] + curl_u[1]*curl_v[1] + curl_u[2]*curl_v[2])*quad->weights[l]/J_det[l];

                    // increment the integration value for dK
                    if (B_mag[l] >= 1e-14){

                        // compute the scalar products curl_u.B and curl_v.B
                        proj_u = curl_u[0]*B[3*l] + curl_u[1]*B[3*l+1] + curl_u[2]*B[3*l+2];
                        proj_v = curl_v[0]*B[3*l] + curl_v[1]*B[3*l+1] + curl_v[2]*B[3*l+2];

                        // increment the integration value for dK
                        int_val_dK += d_nu[l]*proj_u*proj_v*quad->weights[l]/J_det[l]/B_mag[l];
                    }

                    if(j == k){
                        // increment the integration value for the right hand side
                        rhs_c[glob_ids_c[i*num_el_dofs + j]] += (nu_0 - nu[l])*(curl_u[0]*B_s[3*(i*num_quad + l)] + curl_u[1]*B_s[3*(i*num_quad + l) + 1] + curl_u[2]*B_s[3*(i*num_quad + l) + 2])*quad->weights[l];                        
                    }
                    
                }

                // fill the triplet list
                triplets_K->row[triplets_K->counter] = glob_ids_c[i*num_el_dofs + j];
                triplets_K->col[triplets_K->counter] = glob_ids_c[i*num_el_dofs + k];
                triplets_K->vals[triplets_K->counter] = int_val_K;
                triplets_K->counter += 1;

                triplets_dK->row[triplets_dK->counter] = glob_ids_c[i*num_el_dofs + j];
                triplets_dK->col[triplets_dK->counter] = glob_ids_c[i*num_el_dofs + k];
                triplets_dK->vals[triplets_dK->counter] = int_val_dK;
                triplets_dK->counter += 1;

                // apply symmetry
                if(j != k){
                    triplets_K->row[triplets_K->counter] = glob_ids_c[i*num_el_dofs + k];
                    triplets_K->col[triplets_K->counter] = glob_ids_c[i*num_el_dofs + j];
                    triplets_K->vals[triplets_K->counter] = int_val_K;
                    triplets_K->counter += 1;

                    triplets_dK->row[triplets_dK->counter] = glob_ids_c[i*num_el_dofs + k];
                    triplets_dK->col[triplets_dK->counter] = glob_ids_c[i*num_el_dofs + j];
                    triplets_dK->vals[triplets_dK->counter] = int_val_dK;
                    triplets_dK->counter += 1;
                }
            }



        } 

        // zero the Jacobian
        for (j = 0; j < 9*num_quad; ++j){
            J[j] = 0.;
        }
                    
    }

    return;

}

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
                    int num_el_dofs){

    // local variables
    int i, j, k;                    // running indices
    int num_elements;               // the number of elements in the mesh
    double* J;                      // Jacobians at the integration points 
    double* J_inv;                  // inverse of the Jacobians at the integration points
    double* J_det;                  // determinants of the Jacobians at the integration points
    double* curl_u;                 // The curls of the basis functions in the global coordinates
    // double int_val;                 // the integration value
    // double proj_u;                  // the projection of B onto the curl of the basis function
    // double proj_v;                  // the projection of B onto the curl of the test function
    int offs;                       // offset in the curls array for the given orientation
    // int test;
    double* B;                      // The components of the magnetic flux density vectors
    double* B_mag;                  // The magnitudes of the magnetic flux density vectors
    double* nu;                     // The reluctances
    double nu_0 = 1.0/4.0/M_PI*1e7; // The vacuum reluctance

    // we will use these variables very often
    int nodes_per_el = msh->num_nodes_per_element;
    int num_quad = quad->num_points;

    // get some info about the problem
    num_elements = msh->num_elements;

    // allocate also the space for the Jacobians
    J = (double*) calloc(9*num_quad, sizeof(double));
    J_inv = (double*) calloc(9*num_quad, sizeof(double));
    J_det = (double*) calloc(num_quad, sizeof(double));

    // the curls of the basis functions in the global coordinates
    curl_u = (double*) calloc(3, sizeof(double));
    
    // allocate space for the B field evaluations
    B = (double*) calloc(3*num_quad, sizeof(double));

    // allocate space for the B field evaluations
    B_mag = (double*) calloc(3*num_quad, sizeof(double));

    // allocate space for the reluctances
    nu = (double*) calloc(num_quad, sizeof(double));

    // loop over all finite elements
    //  // #pragma omp for
    for(i = 0; i < num_elements; ++i){
            
        // compute the Jacobian of this element
        compute_jacobian(J, &msh->cells[i*nodes_per_el], msh, num_quad, d_phi_c);
            
        // invert the jacobians and compute the determinants
        invert_jacobians(J_det, J_inv, J, num_quad);
       
        // this is the offset in the curls array
        offs = 3*num_quad*num_el_dofs*orientations_c[i];

        // compute the B field at the quadrature points
        compute_B_in_element(B, &B_s[i*3*num_quad], num_quad, num_el_dofs, &glob_ids_c[i*num_el_dofs], x_c, &curls_c[offs], J, J_det);

        // evaluate also the reluctance and its derivative and store the B mag values
        for (j = 0; j < num_quad; ++j){

            B_mag[j] = sqrt(B[3*j]*B[3*j] + B[3*j+1]*B[3*j+1] + B[3*j+2]*B[3*j+2]);

            nu[j] = eval_pchip_interpolator(rel_c->num_intervals,
                                            rel_c->coefficients,
                                            rel_c->knots,
                                            B_mag[j]);

        }

        // sum over all basis functions
        for (j = 0; j < num_el_dofs; ++j){

            
            // integrate
            for (k = 0; k < num_quad; ++k){
                    
                // transform the curls to the global frame
                mat_vec(curl_u, &J[9*k], &curls_c[offs + 3*(k*num_el_dofs + j)], 0);

                // increment the integration value for K
                rhs_c[glob_ids_c[i*num_el_dofs + j]] += (nu_0 - nu[k])*(curl_u[0]*B_s[3*(i*num_quad + k)] + curl_u[1]*B_s[3*(i*num_quad + k) + 1] + curl_u[2]*B_s[3*(i*num_quad + k) + 2])*quad->weights[k];

            }

        } 
                    
    }

    free(J);
    free(J_inv);
    free(J_det);
    free(curl_u);

    return;

}


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
                    double *x_c){

	// local variables
	int i; 			                // running indices
	double* J;					    // Jacobians at the integration points 
    double* J_inv;				    // inverse of the Jacobians at the integration points
    double* J_det;				    // determinants of the Jacobians at the integration points
    double* grad_phi;               // The gradients of the scalar potential
    double* curl_uv;                // The curls of the basis functions in the global coordinates
    double *B_s;
    int offs;                       // offset in the curls array for the given orientation



    // we will use these variables very often
    int nodes_per_el = msh->num_nodes_per_element;
    int num_quad = quad->num_points;


    // allocate also the space for the Jacobians
    J = (double*) calloc(9*num_quad, sizeof(double));
    J_inv = (double*) calloc(9*num_quad, sizeof(double));
    J_det = (double*) calloc(num_quad, sizeof(double));
    grad_phi = (double*) calloc(3*num_quad, sizeof(double));

    // allocate space for the source B field (dummy)
    B_s = (double*) calloc(3*num_quad, sizeof(double));

    // the curls of the basis functions in the global coordinates
    curl_uv = (double*) calloc(3, sizeof(double));

    // loop over all finite elements
	for(i = 0; i < msh->num_elements; ++i){
            
        // compute the Jacobian of this element
        compute_jacobian(J, &msh->cells[i*nodes_per_el], msh, num_quad, d_phi_c);

        // invert the jacobians and compute the determinants
        invert_jacobians(J_det, J_inv, J, num_quad);
        
        // this is the offset in the curls array
        offs = 3*num_quad*num_el_dofs*orientations_c[i];

        // evaluate also the finite element for the position
        evaluate_finite_element(&points[i*3*num_quad], &msh->cells[i*nodes_per_el], num_quad, nodes_per_el, phi_c, msh->nodes);
        
        // compute the B field at the quadrature points
        compute_B_in_element(&B[i*3*num_quad], B_s, num_quad, num_el_dofs, &glob_ids_c[i*num_el_dofs], x_c, &curls_c[offs], J, J_det);

    }
    
	return;

}

void compute_B_line_segs(double* B_ret, const double *src, const double *tar, const double current, const double rad, const int num_src, const int num_tar){
/**
    * Evaluate the B field based on the (5.51) in Field Computation for Accelerator Magnets.
    * by S. Russenschuck
    * Here we consider a collection of line segments as sources.
    * 
    * @param B_ret The return data pointer. You need to have allocated it first to the
                      correct size.
    * @param src The source points in a c vector.
    * @param tar The target points in a c vector.
    * @param current The magnet current.
    * @param rad The radius of the line current.
    * @param num_src The number of sources.
    * @param num_tar The number of targets.
    * @return Nothing.
*/



    int enable_print = 0;   			// flag to set if output is desired

    if(num_tar*num_src > 1e8){
        // print out the details
        printf("********************************\n");
        printf("  Evaluate Biot-Savarts law  \n");
        printf("********************************\n");
        printf("number of sources = %d\n", num_src);
        printf("number of field points = %d\n", num_tar);
        enable_print = 1;
    }
  

    if (enable_print == 1) printf("start computation\n");

    // zero B_ret (cython does not have calloc)
    for (int i = 0; i < num_tar ; ++i){
      B_ret[3*i  ] = 0.0;
      B_ret[3*i+1] = 0.0;
      B_ret[3*i+2] = 0.0;
    }

    #pragma omp parallel
    {

       double *my_B = (double*) calloc(3*num_tar, sizeof(double)); // result B field pointers for each task
       double d1[3];			// difference vector for the start of the segment
       double d2[3];			// difference vector for the end of the segment
       double norm_d1, norm_d2;		// norms of the difference vectors
       double d1_d2;			// the scalar product of d1 and d2
       double factor;			// auxiliary variable
       double n[3];			// the direction vector of the line segment
       double norm_n_sq;		// the norm of n squared
       double t_p;			// the parameter on the line r1 + t*(r2 - r1) that minimizes the distance to the observation point
       double d_min_sq;			// the minimum squared distance to the line r1 + t*(r2 - r1)
       double r_min[3];			// the point on the line r1 + t*(r2 - r1) with minimal distance to the target
       double B_dir[3];			// a container for the direction of B
       double norm_B_dir;		// the norm of the above
       double p[3];			// the difference vector tar - r1 + t_p*(r2 - r1)
       
       
	for(int j = 0; j < num_src ; ++j){
	
	n[0] = src[6*j+3] - src[6*j  ];
	n[1] = src[6*j+4] - src[6*j+1];
	n[2] = src[6*j+5] - src[6*j+2];
	norm_n_sq = n[0]*n[0] + n[1]*n[1] + n[2]*n[2];
	
        #pragma omp single nowait
        {
	    for (int i = 0; i < num_tar ; ++i){            
            
            
                //  difference vectors
                d1[0] = src[6*j  ] - tar[3*i  ];
                d1[1] = src[6*j+1] - tar[3*i+1];
                d1[2] = src[6*j+2] - tar[3*i+2];
                
                d2[0] = src[6*j+3] - tar[3*i  ];
                d2[1] = src[6*j+4] - tar[3*i+1];
                d2[2] = src[6*j+5] - tar[3*i+2];

                //  norms
                norm_d1 = sqrt(d1[0]*d1[0] + d1[1]*d1[1] + d1[2]*d1[2]);
                norm_d2 = sqrt(d2[0]*d2[0] + d2[1]*d2[1] + d2[2]*d2[2]);

		// the parameter on the line r1 + t*(r2 - r1) that minimizes the distance to the target point
		t_p = -1.0*(d1[0]*n[0] + d1[1]*n[1] + d1[2]*n[2])/norm_n_sq;
		

		// point with minimum distance
		r_min[0] = src[6*j  ] + t_p*n[0];
		r_min[1] = src[6*j+1] + t_p*n[1];
		r_min[2] = src[6*j+2] + t_p*n[2];
		
		// the difference vector
		p[0] = tar[3*i  ] - r_min[0];
		p[1] = tar[3*i+1] - r_min[1];
		p[2] = tar[3*i+2] - r_min[2];
		 
		// minimum distance squared
		d_min_sq = p[0]*p[0] + p[1]*p[1] + p[2]*p[2];	        


	        if(((d_min_sq < rad*rad) && (t_p <= 1.0) && (t_p >= 0.0)) || (norm_d1 < rad) || (norm_d2 < rad)){
	        
	           // the point lies inside the cylindrical segment of radius rad
	           // the segment is 'sausaged', so that no singularities can occure at the start and end points.
	           B_dir[0] = n[1]*p[2] - n[2]*p[1];
	           B_dir[1] = n[2]*p[0] - n[0]*p[2];
	           B_dir[2] = n[0]*p[1] - n[1]*p[0];
	           norm_B_dir = sqrt(B_dir[0]*B_dir[0] + B_dir[1]*B_dir[1] + B_dir[2]*B_dir[2]);
	           
		   //  B = mu_0*I*r/2/pi/R**2 (current*1e-7 will be applied later)
		   factor = 2.0*sqrt(d_min_sq)/rad/rad/norm_B_dir;
		   
		   my_B[3*i  ] += B_dir[0]*factor;
		   my_B[3*i+1] += B_dir[1]*factor;
		   my_B[3*i+2] += B_dir[2]*factor;
		   
	        }
	        else{
	        
		   //  scalar product
		   d1_d2 = d1[0]*d2[0] + d1[1]*d2[1] + d1[2]*d2[2];
		        
		   //  factor
		   factor = (norm_d1 + norm_d2)/(norm_d1*norm_d2 + d1_d2)/norm_d1/norm_d2;

		   //  Bx ~ (d1y d2z) - (d1z d2y)
		   my_B[3*i  ] += (d1[1]*d2[2] - d1[2]*d2[1])*factor;

		   //  By ~ (d1z d2x) - (d1x d2z)
		   my_B[3*i+1] += (d1[2]*d2[0] - d1[0]*d2[2])*factor;

		   //  Bz ~ (d1x d2y) - (d1y d2x)
		   my_B[3*i+2] += (d1[0]*d2[1] - d1[1]*d2[0])*factor;

		}


            }
        }
        }
        #pragma omp critical
        {
            for (int i = 0; i < num_tar ; ++i){
            	B_ret[3*i  ] += my_B[3*i  ]*current*1e-7;
            	B_ret[3*i+1] += my_B[3*i+1]*current*1e-7;
            	B_ret[3*i+2] += my_B[3*i+2]*current*1e-7;
            }
            
        }
        
    }
    
    if (enable_print == 1) printf("done\n");

    return;

}
