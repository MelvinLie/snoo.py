#include "ctools.h"
#include <stdio.h>
#include <omp.h>

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
	int i, j, k, l, m; 			// running indices
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


void compute_K_A(triplet_list *triplets,
				mesh *msh,
				quad_3D *quad,
				double *d_phi_c,
				double *curls_c,
                double *grads_c,
                double *v_c, 
                double *vs_c){

	// local variables
	int i, j, k, l, m; 			// running indices
	triplet_list my_triplets;	// a triplet list for each thread
	int num_triplets;			// the number of triplets to fill the sparse matrix
	int num_elements;			// the number of elements in the mesh
	double* J;					// Jacobians at the integration points 
    double* J_inv;				// inverse of the Jacobians at the integration points
    double* J_det;				// determinants of the Jacobians at the integration points
    double int_val_curl, int_val_div;  // the integration values
    double debug;
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

	//#pragma omp parallel
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
                
		// #pragma omp for
		for(i = 0; i < num_elements; ++i){
            
            // compute the Jacobian of this element
            compute_jacobian(J, &msh->cells[i*nodes_per_el], msh, num_quad, d_phi_c);
            
            // invert the jacobians and compute the determinants
            invert_jacobians(J_det, J_inv, J, num_quad);
                
            // each kombination of basis functions
            for(k = 0; k < 3*nodes_per_el; ++k){
                for(l = k; l < 3*nodes_per_el; ++l){
                    
                    // reset int val
                    int_val_curl = 0.0;
                    int_val_div = 0.0;
                    
                    // each integration point
                    for (j = 0; j < num_quad; ++j){
                        
                        int_val_curl += kernel_curl_curl(J_det[j], quad->weights[j],
                                                         &J_inv[9*j], &curls_c[j*nodes_per_el*9 + 3*k],
                                                         &curls_c[j*nodes_per_el*9 + 3*l]);
                        
                        
                        int_val_div += kernel_div_div(J_det[j], quad->weights[j],
                                                         &J_inv[9*j], &grads_c[j*nodes_per_el*9 + 3*k],
                                                         &grads_c[j*nodes_per_el*9 + 3*l], k % 3, l % 3);

                    }
                    
                    enable_print += 1;
                    
                    // add to the triplet list
                    my_triplets.row[my_triplets.counter] = 3*msh->cells[i*nodes_per_el + k/3] + k % 3;
                    my_triplets.col[my_triplets.counter] = 3*msh->cells[i*nodes_per_el + l/3] + l % 3;
                    my_triplets.vals[my_triplets.counter] = v_c[i]*int_val_curl + vs_c[i]*int_val_div;
                    my_triplets.counter += 1;


                    if (k != l){
                        // apply symmetry
                        my_triplets.row[my_triplets.counter] = 3*msh->cells[i*nodes_per_el + l/3] + l % 3;
                        my_triplets.col[my_triplets.counter] = 3*msh->cells[i*nodes_per_el + k/3] + k % 3;
                        my_triplets.vals[my_triplets.counter] = v_c[i]*int_val_curl + vs_c[i]*int_val_div;
                        my_triplets.counter += 1;   
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

void  compute_K_div(triplet_list *triplets, 
                    mesh *msh,
                    quad_3D *quad,
                    double *d_phi_c,
                    double *grads_c){

    // local variables
	int i, j, k, l, m; 			// running indices
	triplet_list my_triplets;	// a triplet list for each thread
	int num_triplets;			// the number of triplets to fill the sparse matrix
	int num_elements;			// the number of elements in the mesh
	double* J;					// Jacobians at the integration points 
    double* J_inv;				// inverse of the Jacobians at the integration points
    double* J_det;				// determinants of the Jacobians at the integration points
    double int_val;             // the integration value
    double debug;
    int enable_print = 1;

    // we will use these variables very often
    int nodes_per_el = msh->num_nodes_per_element;
    int num_quad = quad->num_points;

	// get some info about the problem
	num_triplets = triplets->num_triplets;
	num_elements = msh->num_elements;

	//#pragma omp parallel
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
                
		// #pragma omp for
		for(i = 0; i < num_elements; ++i){
            
            // compute the Jacobian of this element
            compute_jacobian(J, &msh->cells[i*nodes_per_el], msh, num_quad, d_phi_c);
            
            // invert the jacobians and compute the determinants
            invert_jacobians(J_det, J_inv, J, num_quad);
                
            // each kombination of basis functions
            for(k = 0; k < 3*nodes_per_el; ++k){
                for(l = k; l < 3*nodes_per_el; ++l){
                    
                    // reset int val
                    int_val = 0.0;
                    
                    // each integration point
                    for (j = 0; j < num_quad; ++j){                    
                        
                        int_val += kernel_div_div(J_det[j], quad->weights[j],
                                                         &J_inv[9*j], &grads_c[j*nodes_per_el*9 + 3*k],
                                                         &grads_c[j*nodes_per_el*9 + 3*l], k % 3, l % 3);

                    }
                    
                    enable_print += 1;
                    
                    // add to the triplet list
                    my_triplets.row[my_triplets.counter] = 3*msh->cells[i*nodes_per_el + k/3] + k % 3;
                    my_triplets.col[my_triplets.counter] = 3*msh->cells[i*nodes_per_el + l/3] + l % 3;
                    my_triplets.vals[my_triplets.counter] = int_val;
                    my_triplets.counter += 1;


                    if (k != l){
                        // apply symmetry
                        my_triplets.row[my_triplets.counter] = 3*msh->cells[i*nodes_per_el + l/3] + l % 3;
                        my_triplets.col[my_triplets.counter] = 3*msh->cells[i*nodes_per_el + k/3] + k % 3;
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

    	// #pragma omp critical
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

double *compute_rhs_A_c(mesh *msh,
                        quad_3D *quad,
                        double *phi_c,
                        double *d_phi_c,
                        double *j_c){
    
    // local variables
	int i, j, k; 			// running indices

    // we will use these variables very often
    int nodes_per_el = msh->num_nodes_per_element;
    int num_quad = quad->num_points;

	// get some info about the problem
	int num_elements = msh->num_elements;

    // allocate space for the return
    double* rhs = (double*) calloc(3*msh->num_nodes, sizeof(double));

    printf("num_nodes_per_el = %d\n\n", nodes_per_el);

	// #pragma omp parallel
	{

        // allocate also the space for the Jacobians
        double* J = (double*) calloc(9*num_quad, sizeof(double));
        double* J_inv = (double*) calloc(9*num_quad, sizeof(double));
        double* J_det = (double*) calloc(num_quad, sizeof(double));

        double Jv_x, Jv_y, Jv_z;

        // allocate space for the return of this task
        double* my_rhs = (double*) calloc(3*msh->num_nodes, sizeof(double));

        // #pragma omp for
		for(i = 0; i < num_elements; ++i){

            // compute the Jacobian of this element
            compute_jacobian(J, &msh->cells[i*nodes_per_el], msh, num_quad, d_phi_c);
            
            // invert the jacobians and compute the determinants
            invert_jacobians(J_det, J_inv, J, num_quad);

            // loop over all nodes of this element
            for(k = 0; k < nodes_per_el; ++k){

                // loop over the quadrature points
                for (j = 0; j < num_quad; ++j){

                    // compute the scalar products J.v
                    Jv_x = j_c[i*num_quad*3 + j*3]*phi_c[j*nodes_per_el + k];
                    Jv_y = j_c[i*num_quad*3 + j*3 + 1]*phi_c[j*nodes_per_el + k];
                    Jv_z = j_c[i*num_quad*3 + j*3 + 2]*phi_c[j*nodes_per_el + k];

                    my_rhs[3*msh->cells[i*nodes_per_el + k]] += quad->weights[j]*Jv_x*J_det[j];
                    my_rhs[3*msh->cells[i*nodes_per_el + k] + 1] += quad->weights[j]*Jv_y*J_det[j];
                    my_rhs[3*msh->cells[i*nodes_per_el + k] + 2] += quad->weights[j]*Jv_z*J_det[j];

                   
                }
            }
            // zero the Jacobian
            for (j = 0; j < 9*num_quad; ++j){
                J[j] = 0.;
            }
        }



        // #pragma omp critical
		{   
            for (i = 0; i < 3*msh->num_nodes; ++i){ 
                rhs[i] += my_rhs[i];
                // printf("my_rhs[i] =  %f\n", my_rhs[i]);
            }
        }
    }

    return rhs;

}