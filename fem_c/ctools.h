#ifndef CTOOLS_H
#define CTOOLS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


// =======================================
// Compute the FEM stiffness matrix
// 
// Author: Melvin Liebsch
// email: melvin.liebsch@cern.ch
// =======================================

typedef struct mesh {

  int num_nodes;
  int num_elements;
  int num_nodes_per_element;
  double *nodes;
  int *cells;

} mesh;

typedef struct triplet_list {

  int counter;
  int *row;
  int *col;
  double *vals;
  int num_triplets;

} triplet_list;

typedef struct quad_3D{

    int num_points;
    double *points;
    double *weights;

} quad_3D;

typedef struct permeability{

    int type_spec;
    int degree;
    int num_control_points;
    double *knots;
    double *control_points;
    double H_min;
    double mu_H_min;

} permeability;


typedef struct reluctance{

    int type_spec;
    int num_intervals;
    double *coefficients;
    double *knots;

} reluctance;

struct triplet_list* make_triplet_list(int number_of_triplets,
                                       int *row_ptr,
                                       int *col_ptr,
                                       double *vals);

struct mesh* make_mesh(int num_nodes,
                       int num_elements,
                       int num_nodes_per_element,
                       double *nodes,
                       int *cells);

struct quad_3D* make_quadrature_rule(int number_of_points,
                                     double *points,
                                     double *weights);

struct permeability* make_permeability(int type_spec,
                                       int degree,
                                       int num_control_points,
                                       double *knots,
                                       double *control_points,
                                       double H_min,
                                       double mu_H_min);

struct reluctance* make_reluctance(int type_spec,
                                       int num_intervals,
                                       double *coefficients,
                                       double *knots);

double eval_pchip_interpolator(const int num_intervals, const double *c, const double *x_k, const double x);
double eval_pchip_interpolator_derivative(const int num_intervals, const double *c, const double *x_k, const double x);

double eval_cox_de_boor(int i, int k, double *t, double x);
double eval_cox_de_boor_derivative(int i, int k, double *t, double x);

double eval_basis_spline(double *t, double *c, int k, int n, double x);
double eval_basis_spline_derivative(double *t, double *c, int k, int n, double x);

double compute_mu(permeability *perm_c, double H_mag);
double compute_mu_derivative(permeability *perm_c, double H_mag);

void mat_vec(double *result, const double *mat, const double *vec, const int transpose);
double dot_prod(const double *v1, const double *v2);

void compute_B_in_element(double *B,
                          double *B_s,
                          int num_points,
                          int num_el_dofs,
                          const int *glob_ids,
                          const double *x,
                          const double *curl_w_hat,
                          const double *J,
                          const double *det_J);

void evaluate_finite_element(double *phi,
                      int *node_pos,
                      int num_quad,
                      int num_basis_fcns,
                      double *phi_c,
                      double *x);

void compute_grad_phi(double *grad_phi,
                      int *node_pos,
                      int num_quad,
                      int num_basis_fcns,
                      double *d_phi_c,
                      double *x,
                      double *J_inv);

void  compute_K(triplet_list *triplets, 
                mesh *msh,
                quad_3D *quad,
                double *d_phi_c);

void  compute_K_dK(triplet_list *triplets_K, 
                    triplet_list *triplets_dK,
                    mesh *msh,
                    quad_3D *quad,
                    double *d_phi_c,
                    double *x_c,
                    permeability *perm_c);

void  compute_K_dK_Hcurl(triplet_list *triplets_K, 
                    triplet_list *triplets_dK,
                    int *glob_ids_c,
                    reluctance *rel_c,
                    mesh *msh,
                    quad_3D *quad,
                    double *curls_c,
                    double *d_phi_c,
                    int *orientations_c,
                    double *x_c);

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
                    double *B_s);

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
                    int num_el_dofs);

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
                    double *x_c);

void compute_B_line_segs(double* B_ret,
			const double *src_ptr,
			const double *tar_ptr,
			const double current,
			const double rad,
			const int num_src,
			const int num_tar);

#endif
