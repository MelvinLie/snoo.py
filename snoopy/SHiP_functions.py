"""Functions for the automated map generation for certain cases.
   ============================================================

   The functions are:

   o get_vector_field_mag_1
      : used for the mag 1 template (normal conducting magnet) :

   o get_vector_field_mag_2
      : used for the mag 2 template (superconducting magnet) :
   
   o get_vector_field_mag_3
      : used for the mag 1 template (normal conducting magnet) :

   o get_map_ncsc
      : used to simulate the cross talk between hadron absorber and
      superconducting magnet :
"""

import os
import numpy as np
import pandas as pd
import pyvista as pv
import time
import gmsh
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg
from scipy.sparse import diags
from scipy.interpolate import griddata 

import snoopy

def get_vector_field_mag_1(parameters,
                  df_index=0, lc=0.2,
                  geo_th=1e-5, run_gmsh=False, plot_geo=False,
                  plot_result=False, result_directory='none', result_spec='',
                  eval_pos=np.zeros((0, 3)),
                  materials_directory='files/materials'):
   '''Get the vector point cloud for the magnet 1 template.
    
   :params parameters:
      The magnet parameters as pandas dataframe. See the
      SHiP documentation for details.

   :param df_index:
      The row in the pandas parameter dataframe.
   
   :param lc:
      The mesh size parameter. Default = 0.5.

   :param geo_th:
      A threshold for the identification of boundary surfaces.
      Adjust it if You generate very small features.

   :param run_gmsh:
      Set this flag to true if You like to run the gmsh gui after the mesh
      was generated.

   :param plot_geo:
      Set this flag to true if You like to generate a 3D plot of the geometry.

   :param plot_result:
      Set this flag to true if You like to generate a 3D plot of the result.

   :param result_directory:
      The result directory in case You like to store the solution somewhere.

   :param result_spec:
      A specifyer for the result files.

   :param eval_pos:
      Additional positions to be evaluated. Default empty.

   :params materials_directory:
      The directory where the material files are stored. Default files/materials.

   :return:
      The positions and field components in a 3D numpy grid.
   '''

   # launch time measurement
   t_s = time.time()

   # ====================================================
   # read the material data
   reluctance_iron = snoopy.Reluctance(os.path.join(materials_directory,
                                                    parameters["material"][df_index]))
   reluctance_air = snoopy.ConstantReluctance(1e7/4/np.pi)

   # ====================================================
   # mesh generation
   gmsh.initialize()
   gmsh.model.add("make mesh mag 1 template")

   # we get the geometry parameters as variables for convinience
   X_mgap_1 = parameters["Xmgap1(m)"][df_index]
   X_mgap_2 = parameters["Xmgap2(m)"][df_index]

   X_core_1 = parameters["Xcore1(m)"][df_index]
   X_core_2 = parameters["Xcore2(m)"][df_index]

   X_void_1 = parameters["Xvoid1(m)"][df_index]
   X_void_2 = parameters["Xvoid2(m)"][df_index]

   X_yoke_1 = parameters["Xyoke1(m)"][df_index]
   X_yoke_2 = parameters["Xyoke2(m)"][df_index]

   Y_core_1 = parameters["Ycore1(m)"][df_index]
   Y_core_2 = parameters["Ycore2(m)"][df_index]

   Y_void_1 = parameters["Yvoid1(m)"][df_index]
   Y_void_2 = parameters["Yvoid2(m)"][df_index]

   Y_yoke_1 = parameters["Yyoke1(m)"][df_index]
   Y_yoke_2 = parameters["Yyoke2(m)"][df_index]

   Z_len = parameters["Z_len(m)"][df_index]
   Z_pos = parameters["Z_pos(m)"][df_index]

   delta_x = parameters["delta_x(m)"][df_index]
   delta_y = parameters["delta_y(m)"][df_index]
   delta_z = parameters["delta_z(m)"][df_index]
   
   yoke_spacer = parameters["yoke_spacer(m)"][df_index]

   ins = parameters["insulation(m)"][df_index]

   current = parameters["NI(A)"][df_index]

   coil_radius = 0.5*parameters["coil_diam(mm)"][df_index]*1e-3

   field_density = 0.5*parameters["field_density"][df_index]

   # the limits in x, y, and z
   lim_x = max([X_yoke_1, X_yoke_2]) + delta_x
   lim_y = max([Y_yoke_1, Y_yoke_2]) + delta_y
   z_min = Z_pos - delta_z
   z_max = Z_pos + Z_len + delta_z
   
   # the iron domain
   vol_iron = snoopy.add_SHIP_iron_yoke(gmsh.model, X_mgap_1,
                                                    X_core_1,
                                                    X_void_1,
                                                    X_yoke_1,
                                                    X_mgap_2, 
                                                    X_core_2,
                                                    X_void_2,
                                                    X_yoke_2,
                                                    Y_core_1,
                                                    Y_void_1,
                                                    Y_yoke_1,
                                                    Y_core_2,
                                                    Y_void_2,
                                                    Y_yoke_2,
                                                    Z_len, 
                                                    Z_pos=Z_pos,
                                                    lc=lc,
                                                    lc_inner=0.2*lc,
                                                    yoke_type=1)

   # the iron domain
   vol_air = gmsh.model.occ.addBox(0.0, 0.0, z_min, lim_x, lim_y, z_max - z_min)
   gmsh.model.occ.synchronize()

   # fragment perfoms something like a union
   fragments, _ = gmsh.model.occ.fragment([(3, vol_iron)], [(3, vol_air)])
   gmsh.model.occ.synchronize()

   # we get the domains of the fragmentation
   dom_iron = fragments[0][1]
   dom_air = fragments[1][1]

   # and we define physical domains
   gmsh.model.addPhysicalGroup(3, [dom_iron], 1, name = "Iron")
   gmsh.model.occ.synchronize()
   gmsh.model.addPhysicalGroup(3, [dom_air], 2, name = "Air")
   gmsh.model.occ.synchronize()

   # we then generate the mesh
   gmsh.model.mesh.generate(3)
   gmsh.model.occ.synchronize()

   # we now need to collect all Dirichlet boundaries
   boundary_entities = gmsh.model.getEntities(2)

   # this list will store the boundary tags
   dirichlet_boundaries = []

   for i, be in boundary_entities:

      min_uv, max_uv = gmsh.model.getParametrizationBounds(2, be)
      u = 0.5*(max_uv[0] + min_uv[0])
      v = 0.5*(max_uv[1] + min_uv[1])

      coord = gmsh.model.getValue(2, be, [u, v])
      normal = gmsh.model.getNormal(be, [u, v])

      if (abs(coord[0] - 0.0) < geo_th and abs(abs(normal[0]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

      elif (abs(coord[0] - lim_x) < geo_th and abs(abs(normal[0]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

      elif (abs(coord[1] - lim_y) < geo_th and abs(abs(normal[1]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

      elif (abs(coord[2] - z_min) < geo_th and abs(abs(normal[2]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

      elif (abs(coord[2] - z_max) < geo_th and abs(abs(normal[2]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

   # add a physical group for this boundary condition
   gmsh.model.addPhysicalGroup(2, dirichlet_boundaries, 1, name = "Dirichlet Boundary")

   gmsh.model.occ.synchronize()

   if run_gmsh:
      gmsh.fltk.run()


   # ====================================================
   # Make the coil objects

   # this list stores the coil objects
   coil_list = []

   # determine the slot size
   slot_size = 2*min(Y_core_1, Y_core_2)

   # determine the number of conductors
   num_cond = np.int32(slot_size/2/(coil_radius+ins))

   # these are the vertical positions
   y = np.linspace(-0.5*slot_size + coil_radius + ins,
                    0.5*slot_size - coil_radius - ins, num_cond)

   if X_mgap_1 == 0.0 or X_mgap_2 == 0.0:

      # make only a single coil
      kp = np.array([[-X_core_2 - yoke_spacer - ins, Z_pos + Z_len             ],
                     [-X_core_2,          Z_pos + Z_len + yoke_spacer + ins    ],
                     [ X_core_2,          Z_pos + Z_len + yoke_spacer + ins    ],
                     [ X_core_2 + yoke_spacer + ins,   Z_pos + Z_len           ],
                     [ X_core_1 + yoke_spacer + ins,   Z_pos                   ],
                     [ X_core_1,                       Z_pos-yoke_spacer - ins ],
                     [-X_core_1,                       Z_pos-yoke_spacer - ins ],
                     [-X_core_1 - yoke_spacer - ins,   Z_pos                   ]])

      

      coil_list.append(snoopy.RacetrackCoil(kp, y, coil_radius, current/num_cond))

   else:

      # make two coils

      kp_1 = np.array([[ X_mgap_2 - yoke_spacer - ins, Z_pos + Z_len           ],
                     [ X_mgap_2,          Z_pos + Z_len + yoke_spacer + ins    ],
                     [ X_core_2,          Z_pos + Z_len + yoke_spacer + ins    ],
                     [ X_core_2 + yoke_spacer + ins,   Z_pos + Z_len           ],
                     [ X_core_1 + yoke_spacer + ins,   Z_pos                   ],
                     [ X_core_1,                       Z_pos-yoke_spacer - ins ],
                     [ X_mgap_1,                       Z_pos-yoke_spacer - ins ],
                     [ X_mgap_1 - yoke_spacer - ins,   Z_pos                   ]])
      
      kp_2 = kp_1.copy()
      kp_2[:, 0] *= -1.0
      
      coil_list.append(snoopy.RacetrackCoil(kp_1, y, coil_radius, current/num_cond))
      coil_list.append(snoopy.RacetrackCoil(kp_2, y, coil_radius, current/num_cond))

   if plot_geo:

      # ====================================================
      # Plot the magnet
      pl = pv.Plotter(shape=(1, 1), off_screen=False)

      pl.subplot(0, 0)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_iron, reflect_xz=True, reflect_yz=True, show_edges=True, opacity=1.0)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, reflect_xz=True, reflect_yz=True, show_edges=False, opacity=0.0)
      for coil in coil_list:
         coil.plot_pv(pl)
      pl.show_grid()
      pl.add_axes()

      light_1 = pv.Light((1.5, 0.0, -1.2), (0, 0, 0), 'white')
      light_2 = pv.Light((-0.15, 0.2, 0.1), (0, 0, 0), 'white')


      pl.add_light(light_1)
      pl.add_light(light_2)

      pl.camera_position = 'zy'
      pl.camera.elevation = 35
      pl.camera.azimuth = 45
      pl.show()
   

   # ====================================================
   # Solve the problem
   solver = snoopy.RedMVPSolver(gmsh.model, coil_list, 
                                [dom_iron], [dom_air],
                                [reluctance_iron],
                                quad_order=12, max_newton_iterations=150)
   
   x = solver.solve()


   # ====================================================
   # Get the point cloud
   points, B_i = solver.curl_curl_factory.compute_B(x, quad_order=field_density)

   B_coil = 0.0*B_i
   for coil in coil_list:
      B_coil += coil.compute_B(points)

   # stop time measurement
   t_e = time.time()

   print('elapsed time = {:.2f} sec'.format(t_e - t_s))

   if plot_result:
      # ====================================================
      # Plot the solution in 3D
      pl = pv.Plotter(shape=(1, 2))

      pl.subplot(0, 0)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_iron, plot_volume=False)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, plot_volume=False, opacity=0.0)
      snoopy.plot_vector_field(pl, points, B_i, title='B iron in T', mag=0.1, sym_yz=1, sym_xz=2, opacity='linear')
      pl.show_grid()
      pl.camera_position = 'zy'
      pl.camera.elevation = 35
      pl.camera.azimuth = 45
      pl.add_axes()

      pl.subplot(0, 1)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_iron, plot_volume=False)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, plot_volume=False, opacity=0.0)
      snoopy.plot_vector_field(pl, points, B_coil, title='B coil in T', mag=30.0, sym_yz=1, sym_xz=2, opacity='linear')
      pl.show_grid()
      pl.camera_position = 'zy'
      pl.camera.elevation = 35
      pl.camera.azimuth = 45
      pl.add_axes()
      pl.show()

   # ====================================================
   # Store the solution
   if not result_directory == 'none':
      np.save(os.path.join(result_directory, 'x' + result_spec + '.npy'), x)
      
      out_params = parameters.iloc[df_index]
      out_params_df = out_params.to_frame().T
      out_params_df.to_csv(os.path.join(result_directory, 'parameters' + result_spec + '.csv'), index=False)


   ret_vals = [points, B_i + B_coil]

   if eval_pos.shape[0] > 0:
      
      # ====================================================
      # Evaluate at the given positions

      B_iron = snoopy.evaluate_curl_curl_solution(eval_pos, gmsh.model, [dom_iron, dom_air], x, solver.get_global_ids())

      B_coil = 0.0*B_iron
      for coil in coil_list:
         B_coil += coil.compute_B(eval_pos)
      
      B_tot = B_coil + B_iron 

      if not result_directory == 'none':
         output_filename = os.path.join(result_directory, 'B' + result_spec + '.csv')

      else:
         print('Warning, no result directory specified. I store the evaluations in snoopy directory.')
         output_filename = 'B' + result_spec + '.csv'

      out_df = pd.DataFrame(data=np.append(eval_pos, B_tot, axis=1),
                              columns=['x(m)', 'y(m)', 'z(m)', 'Bx(T)', 'By(T)', 'Bz(T)']).to_csv(output_filename, index=False)

   return ret_vals


def get_vector_field_mag_2(parameters, df_index=0, lc=0.4,
                  geo_th=1e-5, run_gmsh=False, plot_geo=False,
                  plot_result=False, result_directory='none', result_spec='',
                  eval_pos=np.zeros((0, 3)),
                  materials_directory='files/materials'):
   '''Get the vector point cloud for the magnet 2 template.
    
   :params parameters:
      The magnet parameters as pandas dataframe. See the
      SHiP documentation for details.

   :param df_index:
      The row in the pandas parameter dataframe.
   
   :param lc:
      The mesh size parameter. Default = 0.5.

   :param geo_th:
      A threshold for the identification of boundary surfaces.
      Adjust it if You generate very small features.

   :param run_gmsh:
      Set this flag to true if You like to run the gmsh gui after the mesh
      was generated.

   :param plot_geo:
      Set this flag to true if You like to generate a 3D plot of the geometry.

   :param plot_result:
      Set this flag to true if You like to generate a 3D plot of the result.

   :param result_directory:
      The result directory in case You like to store the solution somewhere.

   :param result_spec:
      A specifyer for the result files.

   :param eval_pos:
      Additional positions to be evaluated. Default empty.

   :params materials_directory:
      The directory where the material files are stored. Default files/materials.

   :return:
      The positions and field components in a 3D numpy grid.
   '''

   # launch time measurement
   t_s = time.time()

   # ====================================================
   # read the material data
   reluctance_iron = snoopy.Reluctance(os.path.join(materials_directory,
                                                    parameters["material"][df_index]))
   reluctance_air = snoopy.ConstantReluctance(1e7/4/np.pi)

   # ====================================================
   # mesh generation
   gmsh.initialize()
   gmsh.model.add("make mesh mag 2 template")

   # we get the geometry parameters as variables for convinience
   X_core_1 = parameters["Xcore1(m)"][df_index]
   X_core_2 = parameters["Xcore2(m)"][df_index]

   X_void_1 = parameters["Xvoid1(m)"][df_index]
   X_void_2 = parameters["Xvoid2(m)"][df_index]

   X_yoke_1 = parameters["Xyoke1(m)"][df_index]
   X_yoke_2 = parameters["Xyoke2(m)"][df_index]

   Y_core_1 = parameters["Ycore1(m)"][df_index]
   Y_core_2 = parameters["Ycore2(m)"][df_index]

   Y_void_1 = parameters["Yvoid1(m)"][df_index]
   Y_void_2 = parameters["Yvoid2(m)"][df_index]

   Y_yoke_1 = parameters["Yyoke1(m)"][df_index]
   Y_yoke_2 = parameters["Yyoke2(m)"][df_index]

   Z_len = parameters["Z_len(m)"][df_index]
   Z_pos = parameters["Z_pos(m)"][df_index]

   delta_x = parameters["delta_x(m)"][df_index]
   delta_y = parameters["delta_y(m)"][df_index]
   delta_z = parameters["delta_z(m)"][df_index]
   
   yoke_spacer = parameters["yoke_spacer(m)"][df_index]

   ins = parameters["insulation(m)"][df_index]

   current = parameters["NI(A)"][df_index]

   coil_radius = 0.5*parameters["coil_diam(mm)"][df_index]*1e-3

   field_density = 0.5*parameters["field_density"][df_index]

   # the limits in x, y, and z
   lim_x = max([X_yoke_1, X_yoke_2]) + delta_x
   lim_y = max([Y_yoke_1, Y_yoke_2]) + delta_y
   z_min = Z_pos - delta_z
   z_max = Z_pos + Z_len + delta_z
   
   # the core domain
   vol_core = snoopy.add_SHIP_iron_core(gmsh.model, X_core_1,
                                                    X_core_2,
                                                    Y_core_1,
                                                    Y_core_2,
                                                    Z_len,
                                                    Z_pos=Z_pos,
                                                    lc=0.3*lc)

   # the yoke domain
   vol_yoke = snoopy.add_SHIP_iron_yoke(gmsh.model, 0.0, X_core_1,
                                                         X_void_1,
                                                         X_yoke_1,
                                                         0.0, 
                                                         X_core_2,
                                                         X_void_2,
                                                         X_yoke_2,
                                                         Y_core_1,
                                                         Y_void_1,
                                                         Y_yoke_1,
                                                         Y_core_2,
                                                         Y_void_2,
                                                         Y_yoke_2,
                                                         Z_len, 
                                                         Z_pos=Z_pos,
                                                         lc=lc,
                                                         lc_inner=0.3*lc,
                                                         yoke_type=2)

   # the iron domain
   vol_air = gmsh.model.occ.addBox(0.0, 0.0, z_min, lim_x, lim_y, z_max - z_min)
   gmsh.model.occ.synchronize()

   # fragment perfoms something like a union
   fragments, _ = gmsh.model.occ.fragment([(3, vol_core)], [(3, vol_yoke), (3, vol_air)])
   gmsh.model.occ.synchronize()

   # we get the domains of the fragmentation
   dom_core = fragments[0][1]
   dom_yoke = fragments[2][1]
   dom_air = fragments[1][1]

   # and we define physical domains
   gmsh.model.addPhysicalGroup(3, [dom_core, dom_yoke], 1, name = "Iron")
   gmsh.model.occ.synchronize()
   gmsh.model.addPhysicalGroup(3, [dom_air], 2, name = "Air")
   gmsh.model.occ.synchronize()

   # we then generate the mesh
   gmsh.model.mesh.generate(3)
   gmsh.model.occ.synchronize()

   # we now need to collect all Dirichlet boundaries
   boundary_entities = gmsh.model.getEntities(2)

   # this list will store the boundary tags
   dirichlet_boundaries = []

   for i, be in boundary_entities:

      min_uv, max_uv = gmsh.model.getParametrizationBounds(2, be)
      u = 0.5*(max_uv[0] + min_uv[0])
      v = 0.5*(max_uv[1] + min_uv[1])

      coord = gmsh.model.getValue(2, be, [u, v])
      normal = gmsh.model.getNormal(be, [u, v])

      if (abs(coord[0] - 0.0) < geo_th and abs(abs(normal[0]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

      elif (abs(coord[0] - lim_x) < geo_th and abs(abs(normal[0]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

      elif (abs(coord[1] - lim_y) < geo_th and abs(abs(normal[1]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

      elif (abs(coord[2] - z_min) < geo_th and abs(abs(normal[2]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

      elif (abs(coord[2] - z_max) < geo_th and abs(abs(normal[2]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

   # add a physical group for this boundary condition
   gmsh.model.addPhysicalGroup(2, dirichlet_boundaries, 1, name = "Dirichlet Boundary")

   gmsh.model.occ.synchronize()

   if run_gmsh:
      gmsh.fltk.run()


   # ====================================================
   # Make a coil object
   kp = np.array([[-X_core_2 - yoke_spacer - ins, Z_pos + Z_len             ],
                  [-X_core_2,          Z_pos + Z_len + yoke_spacer + ins    ],
                  [ X_core_2,          Z_pos + Z_len + yoke_spacer + ins    ],
                  [ X_core_2 + yoke_spacer + ins,   Z_pos + Z_len           ],
                  [ X_core_1 + yoke_spacer + ins,   Z_pos                   ],
                  [ X_core_1,                       Z_pos-yoke_spacer - ins ],
                  [-X_core_1,                       Z_pos-yoke_spacer - ins ],
                  [-X_core_1 - yoke_spacer - ins,   Z_pos                   ]])
   
   # determine the slot size
   slot_size = 2*min(Y_core_1, Y_core_2)

   # determine the number of conductors
   num_cond = np.int32(slot_size/2/(coil_radius+ins))

   y = np.linspace(-0.5*slot_size + coil_radius + ins,
                    0.5*slot_size - coil_radius - ins, num_cond)

   coil = snoopy.RacetrackCoil(kp, y, coil_radius, current/num_cond)

   if plot_geo:

      # ====================================================
      # Plot the magnet
      pl = pv.Plotter(shape=(1, 1), off_screen=False)

      pl.subplot(0, 0)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_core, reflect_xz=True, reflect_yz=True, show_edges=True, opacity=1.0)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_yoke, reflect_xz=True, reflect_yz=True, show_edges=True, opacity=1.0)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, reflect_xz=True, reflect_yz=True, show_edges=False, opacity=0.0)
      coil.plot_pv(pl)
      pl.show_grid()
      pl.add_axes()

      light_1 = pv.Light((1.5, 0.0, -1.2), (0, 0, 0), 'white')
      light_2 = pv.Light((-0.15, 0.2, 0.1), (0, 0, 0), 'white')


      pl.add_light(light_1)
      pl.add_light(light_2)

      pl.camera_position = 'zy'
      pl.camera.elevation = 35
      pl.camera.azimuth = 45
      pl.show()
   
   # ====================================================
   # Solve the problem
   solver = snoopy.RedMVPSolver(gmsh.model, [coil], 
                                [dom_core, dom_yoke], [dom_air],
                                [reluctance_iron, reluctance_iron],
                                quad_order=10, max_newton_iterations=25)
   
   x = solver.solve()

   # ====================================================
   # Get the point cloud
   points, B_i = solver.curl_curl_factory.compute_B(x, quad_order=field_density)

   B_coil = coil.compute_B(points)

   # stop time measurement
   t_e = time.time()

   print('elapsed time = {:.2f} sec'.format(t_e - t_s))

   if plot_result:
      # ====================================================
      # Plot the solution in 3D
      pl = pv.Plotter(shape=(1, 2))

      pl.subplot(0, 0)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_core, plot_volume=False)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_yoke, plot_volume=False)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, plot_volume=False, opacity=0.0)
      snoopy.plot_vector_field(pl, points, B_i, title='B iron in T', mag=0.05, sym_yz=1, sym_xz=2, opacity='linear')
      pl.show_grid()
      pl.camera_position = 'zy'
      pl.camera.elevation = 35
      pl.camera.azimuth = 45
      pl.add_axes()

      pl.subplot(0, 1)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_core, plot_volume=False)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_yoke, plot_volume=False)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, plot_volume=False, opacity=0.0)
      snoopy.plot_vector_field(pl, points, B_coil, title='B coil in T', mag=0.05, sym_yz=1, sym_xz=2, opacity='linear')
      pl.show_grid()
      pl.camera_position = 'zy'
      pl.camera.elevation = 35
      pl.camera.azimuth = 45
      pl.add_axes()
      pl.show()

   # ====================================================
   # Store the solution
   if not result_directory == 'none':
      np.save(os.path.join(result_directory, 'x' + result_spec + '.npy'), x)
      
      out_params = parameters.iloc[df_index]
      out_params_df = out_params.to_frame().T
      out_params_df.to_csv(os.path.join(result_directory, 'parameters' + result_spec + '.csv'), index=False)


   ret_vals = [points, B_i + B_coil]

   if eval_pos.shape[0] > 0:
      
      # ====================================================
      # Evaluate at the given positions

      B_iron = snoopy.evaluate_curl_curl_solution(eval_pos, gmsh.model, [dom_core, dom_yoke, dom_air], x, solver.get_global_ids())
      B_coil = coil.compute_B(eval_pos)
      
      B_tot = B_coil + B_iron 

      if not result_directory == 'none':
         output_filename = os.path.join(result_directory, 'B' + result_spec + '.csv')

      else:
         print('Warning, no result directory specified. I store the evaluations in snoopy directory.')
         output_filename = 'B' + result_spec + '.csv'

      out_df = pd.DataFrame(data=np.append(eval_pos, B_tot, axis=1),
                              columns=['x(m)', 'y(m)', 'z(m)', 'Bx(T)', 'By(T)', 'Bz(T)']).to_csv(output_filename, index=False)

   return ret_vals


def get_vector_field_mag_3(parameters, df_index=0, lc=0.4,
                  geo_th=1e-5, run_gmsh=False, plot_geo=False,
                  plot_result=False, result_directory='none', result_spec='',
                  eval_pos=np.zeros((0, 3)),
                  materials_directory='files/materials'):
   '''Get the vector point cloud for the magnet 3 template.
    
   :params parameters:
      The magnet parameters as pandas dataframe. See the
      SHiP documentation for details.

   :param df_index:
      The row in the pandas parameter dataframe.
   
   :param lc:
      The mesh size parameter. Default = 0.5.

   :param geo_th:
      A threshold for the identification of boundary surfaces.
      Adjust it if You generate very small features.

   :param run_gmsh:
      Set this flag to true if You like to run the gmsh gui after the mesh
      was generated.

   :param plot_geo:
      Set this flag to true if You like to generate a 3D plot of the geometry.

   :param plot_result:
      Set this flag to true if You like to generate a 3D plot of the result.

   :param result_directory:
      The result directory in case You like to store the solution somewhere.

   :param result_spec:
      A specifyer for the result files.

   :param eval_pos:
      Additional positions to be evaluated. Default empty.

   :params materials_directory:
      The directory where the material files are stored. Default files/materials.

   :return:
      The positions and field components in a 3D numpy grid.
   '''


   # ====================================================
   # read the material data
   reluctance_iron = snoopy.Reluctance(os.path.join(materials_directory,
                                                    parameters["material"][df_index]))
   reluctance_air = snoopy.ConstantReluctance(1e7/4/np.pi)

   # ====================================================
   # mesh generation
   gmsh.initialize()
   gmsh.model.add("make mesh mag 3 template")

   # we get the geometry parameters as variables for convinience
   X_mgap_1 = parameters["Xmgap1(m)"][df_index]
   X_mgap_2 = parameters["Xmgap2(m)"][df_index]

   X_core_1 = parameters["Xcore1(m)"][df_index]
   X_core_2 = parameters["Xcore2(m)"][df_index]

   X_void_1 = parameters["Xvoid1(m)"][df_index]
   X_void_2 = parameters["Xvoid2(m)"][df_index]

   X_yoke_1 = parameters["Xyoke1(m)"][df_index]
   X_yoke_2 = parameters["Xyoke2(m)"][df_index]

   Y_core_1 = parameters["Ycore1(m)"][df_index]
   Y_core_2 = parameters["Ycore2(m)"][df_index]

   Y_void_1 = parameters["Yvoid1(m)"][df_index]
   Y_void_2 = parameters["Yvoid2(m)"][df_index]

   Y_yoke_1 = parameters["Yyoke1(m)"][df_index]
   Y_yoke_2 = parameters["Yyoke2(m)"][df_index]

   Z_len = parameters["Z_len(m)"][df_index]
   Z_pos = parameters["Z_pos(m)"][df_index]

   delta_x = parameters["delta_x(m)"][df_index]
   delta_y = parameters["delta_y(m)"][df_index]
   delta_z = parameters["delta_z(m)"][df_index]
   
   yoke_spacer = parameters["yoke_spacer(m)"][df_index]

   ins = parameters["insulation(m)"][df_index]

   current = parameters["NI(A)"][df_index]

   coil_radius = 0.5*parameters["coil_diam(mm)"][df_index]*1e-3

   field_density = 0.5*parameters["field_density"][df_index]

   # the limits in x, y, and z
   lim_x = max([X_yoke_1, X_yoke_2]) + delta_x
   lim_y = max([Y_yoke_1, Y_yoke_2]) + delta_y
   z_min = Z_pos - delta_z
   z_max = Z_pos + Z_len + delta_z
   
   # the iron domain
   vol_iron = snoopy.add_SHIP_iron_yoke(gmsh.model, X_mgap_1,
                                                    X_core_1,
                                                    X_void_1,
                                                    X_yoke_1,
                                                    X_mgap_2, 
                                                    X_core_2,
                                                    X_void_2,
                                                    X_yoke_2,
                                                    Y_core_1,
                                                    Y_void_1,
                                                    Y_yoke_1,
                                                    Y_core_2,
                                                    Y_void_2,
                                                    Y_yoke_2,
                                                    Z_len, 
                                                    Z_pos=Z_pos,
                                                    lc=lc,
                                                    lc_inner=0.3*lc,
                                                    yoke_type=3)

   # the iron domain
   vol_air = gmsh.model.occ.addBox(0.0, 0.0, z_min, lim_x, lim_y, z_max - z_min)
   gmsh.model.occ.synchronize()

   # fragment perfoms something like a union
   fragments, _ = gmsh.model.occ.fragment([(3, vol_iron)], [(3, vol_air)])
   gmsh.model.occ.synchronize()

   # we get the domains of the fragmentation
   dom_iron = fragments[0][1]
   dom_air = fragments[1][1]

   # and we define physical domains
   gmsh.model.addPhysicalGroup(3, [dom_iron], 1, name = "Iron")
   gmsh.model.occ.synchronize()
   gmsh.model.addPhysicalGroup(3, [dom_air], 2, name = "Air")
   gmsh.model.occ.synchronize()

   # we then generate the mesh
   gmsh.model.mesh.generate(3)
   gmsh.model.occ.synchronize()

   # we now need to collect all Dirichlet boundaries
   boundary_entities = gmsh.model.getEntities(2)

   # this list will store the boundary tags
   dirichlet_boundaries = []

   for i, be in boundary_entities:

      min_uv, max_uv = gmsh.model.getParametrizationBounds(2, be)
      u = 0.5*(max_uv[0] + min_uv[0])
      v = 0.5*(max_uv[1] + min_uv[1])

      coord = gmsh.model.getValue(2, be, [u, v])
      normal = gmsh.model.getNormal(be, [u, v])

      if (abs(coord[0] - 0.0) < geo_th and abs(abs(normal[0]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

      elif (abs(coord[0] - lim_x) < geo_th and abs(abs(normal[0]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

      elif (abs(coord[1] - lim_y) < geo_th and abs(abs(normal[1]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

      elif (abs(coord[2] - z_min) < geo_th and abs(abs(normal[2]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

      elif (abs(coord[2] - z_max) < geo_th and abs(abs(normal[2]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

   # add a physical group for this boundary condition
   gmsh.model.addPhysicalGroup(2, dirichlet_boundaries, 1, name = "Dirichlet Boundary")

   gmsh.model.occ.synchronize()

   if run_gmsh:
      gmsh.fltk.run()


   # ====================================================
   # Make the coil objects

   # this list stores the coil objects
   coil_list = []

   # determine the slot size
   slot_size = 2*min(Y_core_1, Y_core_2)

   # determine the number of conductors
   num_cond = np.int32(slot_size/2/(coil_radius+ins))

   # these are the vertical positions
   y = np.linspace(-0.5*slot_size + coil_radius + ins,
                    0.5*slot_size - coil_radius - ins, num_cond)

   # make two coils

   kp_1 = np.array([[ X_void_2 - yoke_spacer - ins, Z_pos + Z_len           ],
                  [ X_void_2,          Z_pos + Z_len + yoke_spacer + ins    ],
                  [ X_yoke_2,          Z_pos + Z_len + yoke_spacer + ins    ],
                  [ X_yoke_2 + yoke_spacer + ins,   Z_pos + Z_len           ],
                  [ X_yoke_1 + yoke_spacer + ins,   Z_pos                   ],
                  [ X_yoke_1,                       Z_pos-yoke_spacer - ins ],
                  [ X_void_1,                       Z_pos-yoke_spacer - ins ],
                  [ X_void_1 - yoke_spacer - ins,   Z_pos                   ]])
      
   kp_2 = kp_1.copy()
   kp_2[:, 0] *= -1.0
      
   coil_list.append(snoopy.RacetrackCoil(kp_1, y, coil_radius, current/num_cond))
   coil_list.append(snoopy.RacetrackCoil(kp_2, y, coil_radius, current/num_cond))

   if plot_geo:

      # ====================================================
      # Plot the magnet
      pl = pv.Plotter(shape=(1, 1), off_screen=False)

      pl.subplot(0, 0)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_iron, reflect_xz=True, reflect_yz=True, show_edges=True, opacity=1.0)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, reflect_xz=True, reflect_yz=True, show_edges=False, opacity=0.0)
      for coil in coil_list:
         coil.plot_pv(pl)
      pl.show_grid()
      pl.add_axes()

      light_1 = pv.Light((1.5, 0.0, -1.2), (0, 0, 0), 'white')
      light_2 = pv.Light((-0.15, 0.2, 0.1), (0, 0, 0), 'white')


      pl.add_light(light_1)
      pl.add_light(light_2)

      pl.camera_position = 'zy'
      pl.camera.elevation = 35
      pl.camera.azimuth = 45
      pl.show()
   

   # ====================================================
   # Solve the problem
   solver = snoopy.RedMVPSolver(gmsh.model, coil_list, 
                                [dom_iron], [dom_air],
                                [reluctance_iron],
                                quad_order=12, max_newton_iterations=50)
   
   x = solver.solve()


   # ====================================================
   # Get the point cloud
   points, B_i = solver.curl_curl_factory.compute_B(x, quad_order=field_density)

   B_coil = 0.0*B_i
   for coil in coil_list:
      B_coil += coil.compute_B(points)


   if plot_result:
      # ====================================================
      # Plot the solution in 3D
      pl = pv.Plotter(shape=(1, 2))

      pl.subplot(0, 0)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_iron, plot_volume=False)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, plot_volume=False, opacity=0.0)
      snoopy.plot_vector_field(pl, points, B_i, title='B iron in T', mag=0.1, sym_yz=1, sym_xz=2, opacity='linear')
      pl.show_grid()
      pl.camera_position = 'zy'
      pl.camera.elevation = 35
      pl.camera.azimuth = 45
      pl.add_axes()

      pl.subplot(0, 1)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_iron, plot_volume=False)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, plot_volume=False, opacity=0.0)
      snoopy.plot_vector_field(pl, points, B_coil, title='B coil in T', mag=30.0, sym_yz=1, sym_xz=2, opacity='linear')
      pl.show_grid()
      pl.camera_position = 'zy'
      pl.camera.elevation = 35
      pl.camera.azimuth = 45
      pl.add_axes()
      pl.show()

   # ====================================================
   # Store the solution
   if not result_directory == 'none':
      np.save(os.path.join(result_directory, 'x' + result_spec + '.npy'), x)
      
      out_params = parameters.iloc[df_index]
      out_params_df = out_params.to_frame().T
      out_params_df.to_csv(os.path.join(result_directory, 'parameters' + result_spec + '.csv'), index=False)


   ret_vals = [points, B_i + B_coil]

   if eval_pos.shape[0] > 0:
      
      # ====================================================
      # Evaluate at the given positions

      B_iron = snoopy.evaluate_curl_curl_solution(eval_pos, gmsh.model, [dom_iron, dom_air], x, solver.get_global_ids())

      B_coil = 0.0*B_iron
      for coil in coil_list:
         B_coil += coil.compute_B(eval_pos)
      
      B_tot = B_coil + B_iron 

      if not result_directory == 'none':
         output_filename = os.path.join(result_directory, 'B' + result_spec + '.csv')

      else:
         print('Warning, no result directory specified. I store the evaluations in snoopy directory.')
         output_filename = 'B' + result_spec + '.csv'

      out_df = pd.DataFrame(data=np.append(eval_pos, B_tot, axis=1),
                              columns=['x(m)', 'y(m)', 'z(m)', 'Bx(T)', 'By(T)', 'Bz(T)']).to_csv(output_filename, index=False)

   return ret_vals



def get_vector_field_ncsc(parameters, df_index=0, lc=0.4,
                  geo_th=1e-5, run_gmsh=False, plot_geo=False,
                  plot_result=False, result_directory='none', result_spec='',
                  eval_pos=np.zeros((0, 3)),
                  materials_directory='files/materials'):
   '''Get the vector point cloud for the HASC template, i.e. the coupling
   of hardon absorber and superconducting magnet.
    
   :params parameters:
      The magnet parameters as pandas dataframe. See the
      SHiP documentation for details.

   :param df_index:
      The row in the pandas parameter dataframe.
   
   :param lc:
      The mesh size parameter. Default = 0.5.

   :param geo_th:
      A threshold for the identification of boundary surfaces.
      Adjust it if You generate very small features.

   :param run_gmsh:
      Set this flag to true if You like to run the gmsh gui after the mesh
      was generated.

   :param plot_geo:
      Set this flag to true if You like to generate a 3D plot of the geometry.

   :param plot_result:
      Set this flag to true if You like to generate a 3D plot of the result.

   :param result_directory:
      The result directory in case You like to store the solution somewhere.

   :param result_spec:
      A specifyer for the result files.

   :param eval_pos:
      Additional positions to be evaluated. Default empty.

   :params materials_directory:
      The directory where the material files are stored. Default files/materials.

   :return:
      The positions and field components in a 3D numpy grid.
   '''

   # launch time measurement
   t_s = time.time()

   # ====================================================
   # read the material data
   reluctance_iron = snoopy.Reluctance(os.path.join(materials_directory,
                                                    parameters["material"][df_index]))
   reluctance_air = snoopy.ConstantReluctance(1e7/4/np.pi)

   # ====================================================
   # mesh generation
   gmsh.initialize()
   gmsh.model.add("make mesh mag 1 template")

   # Read the geometry parameters for the normal conducting magnet
   X_mgap_1_1 = parameters["Xmgap1(m)"][df_index]
   X_mgap_2_1 = parameters["Xmgap2(m)"][df_index]

   X_core_1_1 = parameters["Xcore1(m)"][df_index]
   X_core_2_1 = parameters["Xcore2(m)"][df_index]

   X_void_1_1 = parameters["Xvoid1(m)"][df_index]
   X_void_2_1 = parameters["Xvoid2(m)"][df_index]

   X_yoke_1_1 = parameters["Xyoke1(m)"][df_index]
   X_yoke_2_1 = parameters["Xyoke2(m)"][df_index]

   Y_core_1_1 = parameters["Ycore1(m)"][df_index]
   Y_core_2_1 = parameters["Ycore2(m)"][df_index]

   Y_void_1_1 = parameters["Yvoid1(m)"][df_index]
   Y_void_2_1 = parameters["Yvoid2(m)"][df_index]

   Y_yoke_1_1 = parameters["Yyoke1(m)"][df_index]
   Y_yoke_2_1 = parameters["Yyoke2(m)"][df_index]

   Z_len_1 = parameters["Z_len(m)"][df_index]
   Z_pos_1 = parameters["Z_pos(m)"][df_index]
   
   yoke_spacer_1 = parameters["yoke_spacer(m)"][df_index]
   ins_1 = parameters["insulation(m)"][df_index]
   current_1 = parameters["NI(A)"][df_index]
   coil_radius_1 = 0.5*parameters["coil_diam(mm)"][df_index]*1e-3

   delta_x_1 = parameters["delta_x(m)"][df_index]
   delta_y_1 = parameters["delta_y(m)"][df_index]
   delta_z_1 = parameters["delta_z(m)"][df_index]

   field_density_1 = 0.5*parameters["field_density"][df_index]

   # Read the geometry parameters for the superconducting magnet
   X_mgap_1_2 = parameters["Xmgap1(m)"][df_index + 1]
   X_mgap_2_2 = parameters["Xmgap2(m)"][df_index + 1]

   X_core_1_2 = parameters["Xcore1(m)"][df_index + 1]
   X_core_2_2 = parameters["Xcore2(m)"][df_index + 1]

   X_void_1_2 = parameters["Xvoid1(m)"][df_index + 1]
   X_void_2_2 = parameters["Xvoid2(m)"][df_index + 1]

   X_yoke_1_2 = parameters["Xyoke1(m)"][df_index + 1]
   X_yoke_2_2 = parameters["Xyoke2(m)"][df_index + 1]

   Y_core_1_2 = parameters["Ycore1(m)"][df_index + 1]
   Y_core_2_2 = parameters["Ycore2(m)"][df_index + 1]

   Y_void_1_2 = parameters["Yvoid1(m)"][df_index + 1]
   Y_void_2_2 = parameters["Yvoid2(m)"][df_index + 1]

   Y_yoke_1_2 = parameters["Yyoke1(m)"][df_index + 1]
   Y_yoke_2_2 = parameters["Yyoke2(m)"][df_index + 1]

   Z_len_2 = parameters["Z_len(m)"][df_index + 1]
   Z_pos_2 = parameters["Z_pos(m)"][df_index + 1]
   
   yoke_spacer_2 = parameters["yoke_spacer(m)"][df_index + 1]
   ins_2 = parameters["insulation(m)"][df_index + 1]
   current_2 = parameters["NI(A)"][df_index + 1]
   coil_radius_2 = 0.5*parameters["coil_diam(mm)"][df_index + 1]*1e-3

   delta_x_2 = parameters["delta_x(m)"][df_index + 1]
   delta_y_2 = parameters["delta_y(m)"][df_index + 1]
   delta_z_2 = parameters["delta_z(m)"][df_index + 1]

   field_density_2 = 0.5*parameters["field_density"][df_index + 1]

   # the limits in x, y, and z
   lim_x = max([max([X_yoke_1_1, X_yoke_2_1]) + delta_x_1, max([X_yoke_1_2, X_yoke_2_2]) + delta_x_2])
   lim_y = max([max([Y_yoke_1_1, Y_yoke_2_1]) + delta_y_1, max([Y_yoke_1_2, Y_yoke_2_2]) + delta_y_2])
   z_min = Z_pos_1 - delta_z_1
   z_max = Z_pos_2 + Z_len_2 + delta_z_2
   
   # the nc magnet iron domain
   vol_nc = snoopy.add_SHIP_iron_yoke(gmsh.model, X_mgap_1_1,
                                                    X_core_1_1,
                                                    X_void_1_1,
                                                    X_yoke_1_1,
                                                    X_mgap_2_1, 
                                                    X_core_2_1,
                                                    X_void_2_1,
                                                    X_yoke_2_1,
                                                    Y_core_1_1,
                                                    Y_void_1_1,
                                                    Y_yoke_1_1,
                                                    Y_core_2_1,
                                                    Y_void_2_1,
                                                    Y_yoke_2_1,
                                                    Z_len_1, 
                                                    Z_pos=Z_pos_1,
                                                    lc=lc,
                                                    lc_inner=lc,
                                                    yoke_type=1)

   # the sc magnet core domain
   vol_sc_core = snoopy.add_SHIP_iron_core(gmsh.model, X_core_1_2,
                                                    X_core_2_2,
                                                    Y_core_1_2,
                                                    Y_core_2_2,
                                                    Z_len_2,
                                                    Z_pos=Z_pos_2,
                                                    lc=lc)

   # the sc magnet yoke domain
   vol_sc_yoke = snoopy.add_SHIP_iron_yoke(gmsh.model, 0.0, X_core_1_2,
                                                         X_void_1_2,
                                                         X_yoke_1_2,
                                                         0.0, 
                                                         X_core_2_2,
                                                         X_void_2_2,
                                                         X_yoke_2_2,
                                                         Y_core_1_2,
                                                         Y_void_1_2,
                                                         Y_yoke_1_2,
                                                         Y_core_2_2,
                                                         Y_void_2_2,
                                                         Y_yoke_2_2,
                                                         Z_len_2, 
                                                         Z_pos=Z_pos_2,
                                                         lc=lc,
                                                         lc_inner=lc,
                                                         yoke_type=2)

   # the iron domain
   vol_air = gmsh.model.occ.addBox(0.0, 0.0, z_min, lim_x, lim_y, z_max - z_min)
   gmsh.model.occ.synchronize()

   # fragment perfoms something like a union
   fragments, _ = gmsh.model.occ.fragment([(3, vol_air)], [(3, vol_nc), (3, vol_sc_core), (3, vol_sc_yoke)])
   gmsh.model.occ.synchronize()

   # we get the domains of the fragmentation
   dom_nc = fragments[0][1]
   dom_sc_core = fragments[1][1]
   dom_air = fragments[2][1]
   dom_sc_yoke = fragments[3][1]

   # and we define physical domains
   gmsh.model.addPhysicalGroup(3, [dom_nc, dom_sc_core, dom_sc_yoke], 1, name = "Iron")
   gmsh.model.occ.synchronize()
   gmsh.model.addPhysicalGroup(3, [dom_air], 2, name = "Air")
   gmsh.model.occ.synchronize()

   gmsh.option.setNumber("Mesh.MeshSizeFactor", lc)

   # we then generate the mesh
   gmsh.model.mesh.generate(3)
   gmsh.model.occ.synchronize()

   # we now need to collect all Dirichlet boundaries
   boundary_entities = gmsh.model.getEntities(2)

   # this list will store the boundary tags
   dirichlet_boundaries = []

   for i, be in boundary_entities:

      min_uv, max_uv = gmsh.model.getParametrizationBounds(2, be)
      u = 0.5*(max_uv[0] + min_uv[0])
      v = 0.5*(max_uv[1] + min_uv[1])

      coord = gmsh.model.getValue(2, be, [u, v])
      normal = gmsh.model.getNormal(be, [u, v])

      if (abs(coord[0] - 0.0) < geo_th and abs(abs(normal[0]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

      elif (abs(coord[0] - lim_x) < geo_th and abs(abs(normal[0]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

      elif (abs(coord[1] - lim_y) < geo_th and abs(abs(normal[1]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

      elif (abs(coord[2] - z_min) < geo_th and abs(abs(normal[2]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

      elif (abs(coord[2] - z_max) < geo_th and abs(abs(normal[2]) - 1.0) < geo_th):
         dirichlet_boundaries.append(be)

   # add a physical group for this boundary condition
   gmsh.model.addPhysicalGroup(2, dirichlet_boundaries, 1, name = "Dirichlet Boundary")

   gmsh.model.occ.synchronize()

   if run_gmsh:
      gmsh.fltk.run()


   # ====================================================
   # Make the coil objects

   # this list stores the coil objects
   coil_list = []

   # determine the slot sizes
   slot_size_1 = 2*min(Y_core_1_1, Y_core_2_1)
   slot_size_2 = 2*min(Y_core_1_2, Y_core_2_2)

   # determine the number of conductors
   num_cond_1 = np.int32(slot_size_1/2/(coil_radius_1 + ins_1))
   num_cond_2 = np.int32(slot_size_2/2/(coil_radius_2 + ins_2))

   # these are the vertical positions
   y_1 = np.linspace(-0.5*slot_size_1 + coil_radius_1 + ins_1,
                    0.5*slot_size_1 - coil_radius_1 - ins_1, num_cond_1)
   y_2 = np.linspace(-0.5*slot_size_2 + coil_radius_2 + ins_2,
                    0.5*slot_size_2 - coil_radius_2 - ins_2, num_cond_2)

   # make only a single coil for the first nc magnet
   kp_1 = np.array([[-X_core_2_1 - yoke_spacer_1 - ins_1, Z_pos_1 + Z_len_1             ],
                     [-X_core_2_1,          Z_pos_1 + Z_len_1 + yoke_spacer_1 + ins_1    ],
                     [ X_core_2_1,          Z_pos_1 + Z_len_1 + yoke_spacer_1 + ins_1    ],
                     [ X_core_2_1 + yoke_spacer_1 + ins_1,   Z_pos_1 + Z_len_1           ],
                     [ X_core_1_1 + yoke_spacer_1 + ins_1,   Z_pos_1                   ],
                     [ X_core_1_1,                       Z_pos_1-yoke_spacer_1 - ins_1 ],
                     [-X_core_1_1,                       Z_pos_1-yoke_spacer_1 - ins_1 ],
                     [-X_core_1_1 - yoke_spacer_1 - ins_1,   Z_pos_1                   ]])

   coil_list.append(snoopy.RacetrackCoil(kp_1, y_1, coil_radius_1, current_1/num_cond_1))

   # make only a single coil for the next sc magnet
   kp_2 = np.array([[-X_core_2_2 - yoke_spacer_2 - ins_2, Z_pos_2 + Z_len_2             ],
                     [-X_core_2_2,          Z_pos_2 + Z_len_2 + yoke_spacer_2 + ins_2    ],
                     [ X_core_2_2,          Z_pos_2 + Z_len_2 + yoke_spacer_2 + ins_2    ],
                     [ X_core_2_2 + yoke_spacer_2 + ins_2,   Z_pos_2 + Z_len_2           ],
                     [ X_core_1_2 + yoke_spacer_2 + ins_2,   Z_pos_2                   ],
                     [ X_core_1_2,                       Z_pos_2-yoke_spacer_2 - ins_2 ],
                     [-X_core_1_2,                       Z_pos_2-yoke_spacer_2 - ins_2 ],
                     [-X_core_1_2 - yoke_spacer_2 - ins_2,   Z_pos_2                   ]]) 

   coil_list.append(snoopy.RacetrackCoil(kp_2, y_2, coil_radius_2, current_2/num_cond_2))


   if plot_geo:

      # ====================================================
      # Plot the magnet
      pl = pv.Plotter(shape=(1, 1), off_screen=False)

      pl.subplot(0, 0)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_nc, reflect_xz=True, reflect_yz=True, show_edges=True, opacity=1.0)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_sc_core, reflect_xz=True, reflect_yz=True, show_edges=True, opacity=1.0)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_sc_yoke, reflect_xz=True, reflect_yz=True, show_edges=True, opacity=1.0)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, reflect_xz=True, reflect_yz=True, show_edges=False, opacity=0.0)
      for coil in coil_list:
         coil.plot_pv(pl)
      pl.show_grid()
      pl.add_axes()

      light_1 = pv.Light((1.5, 0.0, -1.2), (0, 0, 0), 'white')
      light_2 = pv.Light((-0.15, 0.2, 0.1), (0, 0, 0), 'white')


      pl.add_light(light_1)
      pl.add_light(light_2)

      pl.camera_position = 'zy'
      pl.camera.elevation = 35
      pl.camera.azimuth = 45
      pl.show()
   

   # ====================================================
   # Solve the problem
   solver = snoopy.RedMVPSolver(gmsh.model, coil_list, 
                                [dom_nc, dom_sc_core, dom_sc_yoke], [dom_air],
                                [reluctance_iron, reluctance_iron, reluctance_iron],
                                quad_order=12, max_newton_iterations=50)
   
   x = solver.solve()


   # ====================================================
   # Get the point cloud
   
   field_density = max([field_density_1, field_density_2])

   points, B_i = solver.curl_curl_factory.compute_B(x, quad_order=field_density)

   B_coil = 0.0*B_i
   for coil in coil_list:
      B_coil += coil.compute_B(points)

   # stop time measurement
   t_e = time.time()

   print('elapsed time = {:.2f} sec'.format(t_e - t_s))

   if plot_result:
      # ====================================================
      # Plot the solution in 3D
      pl = pv.Plotter(shape=(1, 2))

      pl.subplot(0, 0)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_nc, plot_volume=False)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_sc_core, plot_volume=False)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_sc_yoke, plot_volume=False)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, plot_volume=False, opacity=0.0)
      snoopy.plot_vector_field(pl, points, B_i, title='B iron in T', mag=0.1, sym_yz=1, sym_xz=2, opacity='linear')
      pl.show_grid()
      pl.camera_position = 'zy'
      pl.camera.elevation = 35
      pl.camera.azimuth = 45
      pl.add_axes()

      pl.subplot(0, 1)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_nc, plot_volume=False)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_sc_core, plot_volume=False)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_sc_yoke, plot_volume=False)
      snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, plot_volume=False, opacity=0.0)
      snoopy.plot_vector_field(pl, points, B_coil, title='B coil in T', mag=0.1, sym_yz=1, sym_xz=2, opacity='linear')
      pl.show_grid()
      pl.camera_position = 'zy'
      pl.camera.elevation = 35
      pl.camera.azimuth = 45
      pl.add_axes()
      pl.show()

   # ====================================================
   # Store the solution
   if not result_directory == 'none':
      np.save(os.path.join(result_directory, 'x' + result_spec + '.npy'), x)
      
      out_params = parameters.iloc[df_index]
      out_params_df = out_params.to_frame().T
      out_params_df.to_csv(os.path.join(result_directory, 'parameters' + result_spec + '.csv'), index=False)


   ret_vals = [points, B_i + B_coil]

   if eval_pos.shape[0] > 0:
      
      # ====================================================
      # Evaluate at the given positions

      B_iron = snoopy.evaluate_curl_curl_solution(eval_pos, gmsh.model, [dom_nc, dom_sc_core, dom_sc_yoke, dom_air], x, solver.get_global_ids())

      B_coil = 0.0*B_iron
      for coil in coil_list:
         B_coil += coil.compute_B(eval_pos)
      
      B_tot = B_coil + B_iron 

      if not result_directory == 'none':
         output_filename = os.path.join(result_directory, 'B' + result_spec + '.csv')

      else:
         print('Warning, no result directory specified. I store the evaluations in snoopy directory.')
         output_filename = 'B' + result_spec + '.csv'

      out_df = pd.DataFrame(data=np.append(eval_pos, B_tot, axis=1),
                              columns=['x(m)', 'y(m)', 'z(m)', 'Bx(T)', 'By(T)', 'Bz(T)']).to_csv(output_filename, index=False)

   return ret_vals

def plot_geometry_mag_1(pl, parameters, df_index=0, lc=1.0, opacity=0.0, show_edges=False, plot_feature_edges=True):
   '''Plot the geometry for the magnet 1 template.
    
   :params pl:
      The plotter object.

   :params parameters:
      The magnet parameters as pandas dataframe. See the
      SHiP documentation for details.

   :param df_index:
      The row in the pandas parameter dataframe.
   
   :param lc:
      The mesh size parameter. Default = 0.5.

   :param opacity:
      The opacity of the iron domain. Default=False.

   :param show_edges:
      Set this flag to enable plotting the edges. Default=True.

   :return:
      None
   '''

   # ====================================================
   # mesh generation
   gmsh.initialize()
   gmsh.model.add("make mesh mag 1 template")

   # we get the geometry parameters as variables for convinience
   X_mgap_1 = parameters["Xmgap1(m)"][df_index]
   X_mgap_2 = parameters["Xmgap2(m)"][df_index]

   X_core_1 = parameters["Xcore1(m)"][df_index]
   X_core_2 = parameters["Xcore2(m)"][df_index]

   X_void_1 = parameters["Xvoid1(m)"][df_index]
   X_void_2 = parameters["Xvoid2(m)"][df_index]

   X_yoke_1 = parameters["Xyoke1(m)"][df_index]
   X_yoke_2 = parameters["Xyoke2(m)"][df_index]

   Y_core_1 = parameters["Ycore1(m)"][df_index]
   Y_core_2 = parameters["Ycore2(m)"][df_index]

   Y_void_1 = parameters["Yvoid1(m)"][df_index]
   Y_void_2 = parameters["Yvoid2(m)"][df_index]

   Y_yoke_1 = parameters["Yyoke1(m)"][df_index]
   Y_yoke_2 = parameters["Yyoke2(m)"][df_index]

   Z_len = parameters["Z_len(m)"][df_index]
   Z_pos = parameters["Z_pos(m)"][df_index]

   delta_x = parameters["delta_x(m)"][df_index]
   delta_y = parameters["delta_y(m)"][df_index]
   delta_z = parameters["delta_z(m)"][df_index]
   
   yoke_spacer = parameters["yoke_spacer(m)"][df_index]

   ins = parameters["insulation(m)"][df_index]

   current = parameters["NI(A)"][df_index]

   coil_radius = 0.5*parameters["coil_diam(mm)"][df_index]*1e-3

   field_density = 0.5*parameters["field_density"][df_index]

   # the limits in x, y, and z
   lim_x = max([X_yoke_1, X_yoke_2]) + delta_x
   lim_y = max([Y_yoke_1, Y_yoke_2]) + delta_y
   z_min = Z_pos - delta_z
   z_max = Z_pos + Z_len + delta_z
   
   # the iron domain
   vol_iron = snoopy.add_SHIP_iron_yoke(gmsh.model, X_mgap_1,
                                                    X_core_1,
                                                    X_void_1,
                                                    X_yoke_1,
                                                    X_mgap_2, 
                                                    X_core_2,
                                                    X_void_2,
                                                    X_yoke_2,
                                                    Y_core_1,
                                                    Y_void_1,
                                                    Y_yoke_1,
                                                    Y_core_2,
                                                    Y_void_2,
                                                    Y_yoke_2,
                                                    Z_len, 
                                                    Z_pos=Z_pos,
                                                    lc=lc,
                                                    lc_inner=0.2*lc,
                                                    yoke_type=1)

   # the iron domain
   vol_air = gmsh.model.occ.addBox(0.0, 0.0, z_min, lim_x, lim_y, z_max - z_min)
   gmsh.model.occ.synchronize()

   # fragment perfoms something like a union
   fragments, _ = gmsh.model.occ.fragment([(3, vol_iron)], [(3, vol_air)])
   gmsh.model.occ.synchronize()

   # we get the domains of the fragmentation
   dom_iron = fragments[0][1]
   dom_air = fragments[1][1]

   # and we define physical domains
   gmsh.model.addPhysicalGroup(3, [dom_iron], 1, name = "Iron")
   gmsh.model.occ.synchronize()
   gmsh.model.addPhysicalGroup(3, [dom_air], 2, name = "Air")
   gmsh.model.occ.synchronize()

   # we then generate the mesh
   gmsh.model.mesh.generate(3)
   gmsh.model.occ.synchronize()

   gmsh.model.occ.synchronize()

   # ====================================================
   # Make the coil objects

   # this list stores the coil objects
   coil_list = []

   # determine the slot size
   slot_size = 2*min(Y_core_1, Y_core_2)

   # determine the number of conductors
   num_cond = np.int32(slot_size/2/(coil_radius+ins))

   # these are the vertical positions
   y = np.linspace(-0.5*slot_size + coil_radius + ins,
                    0.5*slot_size - coil_radius - ins, num_cond)

   if X_mgap_1 == 0.0 or X_mgap_2 == 0.0:

      # make only a single coil
      kp = np.array([[-X_core_2 - yoke_spacer - ins, Z_pos + Z_len             ],
                     [-X_core_2,          Z_pos + Z_len + yoke_spacer + ins    ],
                     [ X_core_2,          Z_pos + Z_len + yoke_spacer + ins    ],
                     [ X_core_2 + yoke_spacer + ins,   Z_pos + Z_len           ],
                     [ X_core_1 + yoke_spacer + ins,   Z_pos                   ],
                     [ X_core_1,                       Z_pos-yoke_spacer - ins ],
                     [-X_core_1,                       Z_pos-yoke_spacer - ins ],
                     [-X_core_1 - yoke_spacer - ins,   Z_pos                   ]])

      

      coil_list.append(snoopy.RacetrackCoil(kp, y, coil_radius, current/num_cond))

   else:

      # make two coils

      kp_1 = np.array([[ X_mgap_2 - yoke_spacer - ins, Z_pos + Z_len           ],
                     [ X_mgap_2,          Z_pos + Z_len + yoke_spacer + ins    ],
                     [ X_core_2,          Z_pos + Z_len + yoke_spacer + ins    ],
                     [ X_core_2 + yoke_spacer + ins,   Z_pos + Z_len           ],
                     [ X_core_1 + yoke_spacer + ins,   Z_pos                   ],
                     [ X_core_1,                       Z_pos-yoke_spacer - ins ],
                     [ X_mgap_1,                       Z_pos-yoke_spacer - ins ],
                     [ X_mgap_1 - yoke_spacer - ins,   Z_pos                   ]])
      
      kp_2 = kp_1.copy()
      kp_2[:, 0] *= -1.0
      
      coil_list.append(snoopy.RacetrackCoil(kp_1, y, coil_radius, current/num_cond))
      coil_list.append(snoopy.RacetrackCoil(kp_2, y, coil_radius, current/num_cond))

   snoopy.plot_domain(pl, gmsh.model.mesh, dom_iron, reflect_xz=True, reflect_yz=True, show_edges=show_edges, opacity=opacity, plot_feature_edges=plot_feature_edges)
   # snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, reflect_xz=True, reflect_yz=True, show_edges=False, opacity=0.0)
   for coil in coil_list:
      coil.plot_pv(pl)

   return

def plot_geometry_mag_2(pl, parameters, df_index=0, lc=1.0, opacity=0.0, show_edges=False, plot_feature_edges=True):
   '''Plot the geometry for the magnet 2 template.
    
   :params pl:
      The plotter object.

   :params parameters:
      The magnet parameters as pandas dataframe. See the
      SHiP documentation for details.

   :param df_index:
      The row in the pandas parameter dataframe.
   
   :param lc:
      The mesh size parameter. Default = 0.5.

   :param opacity:
      The opacity of the iron domain. Default=False.

   :param show_edges:
      Set this flag to enable plotting the edges. Default=True.

   :return:
      None
   '''


   # ====================================================
   # mesh generation
   gmsh.initialize()
   gmsh.model.add("make mesh mag 2 template")

   # we get the geometry parameters as variables for convinience
   X_core_1 = parameters["Xcore1(m)"][df_index]
   X_core_2 = parameters["Xcore2(m)"][df_index]

   X_void_1 = parameters["Xvoid1(m)"][df_index]
   X_void_2 = parameters["Xvoid2(m)"][df_index]

   X_yoke_1 = parameters["Xyoke1(m)"][df_index]
   X_yoke_2 = parameters["Xyoke2(m)"][df_index]

   Y_core_1 = parameters["Ycore1(m)"][df_index]
   Y_core_2 = parameters["Ycore2(m)"][df_index]

   Y_void_1 = parameters["Yvoid1(m)"][df_index]
   Y_void_2 = parameters["Yvoid2(m)"][df_index]

   Y_yoke_1 = parameters["Yyoke1(m)"][df_index]
   Y_yoke_2 = parameters["Yyoke2(m)"][df_index]

   Z_len = parameters["Z_len(m)"][df_index]
   Z_pos = parameters["Z_pos(m)"][df_index]

   delta_x = parameters["delta_x(m)"][df_index]
   delta_y = parameters["delta_y(m)"][df_index]
   delta_z = parameters["delta_z(m)"][df_index]
   
   yoke_spacer = parameters["yoke_spacer(m)"][df_index]

   ins = parameters["insulation(m)"][df_index]

   current = parameters["NI(A)"][df_index]

   coil_radius = 0.5*parameters["coil_diam(mm)"][df_index]*1e-3

   field_density = 0.5*parameters["field_density"][df_index]

   # the limits in x, y, and z
   lim_x = max([X_yoke_1, X_yoke_2]) + delta_x
   lim_y = max([Y_yoke_1, Y_yoke_2]) + delta_y
   z_min = Z_pos - delta_z
   z_max = Z_pos + Z_len + delta_z
   
   # the core domain
   vol_core = snoopy.add_SHIP_iron_core(gmsh.model, X_core_1,
                                                    X_core_2,
                                                    Y_core_1,
                                                    Y_core_2,
                                                    Z_len,
                                                    Z_pos=Z_pos,
                                                    lc=0.3*lc)

   # the yoke domain
   vol_yoke = snoopy.add_SHIP_iron_yoke(gmsh.model, 0.0, X_core_1,
                                                         X_void_1,
                                                         X_yoke_1,
                                                         0.0, 
                                                         X_core_2,
                                                         X_void_2,
                                                         X_yoke_2,
                                                         Y_core_1,
                                                         Y_void_1,
                                                         Y_yoke_1,
                                                         Y_core_2,
                                                         Y_void_2,
                                                         Y_yoke_2,
                                                         Z_len, 
                                                         Z_pos=Z_pos,
                                                         lc=lc,
                                                         lc_inner=0.3*lc,
                                                         yoke_type=2)

   # the iron domain
   vol_air = gmsh.model.occ.addBox(0.0, 0.0, z_min, lim_x, lim_y, z_max - z_min)
   gmsh.model.occ.synchronize()

   # fragment perfoms something like a union
   fragments, _ = gmsh.model.occ.fragment([(3, vol_core)], [(3, vol_yoke), (3, vol_air)])
   gmsh.model.occ.synchronize()

   # we get the domains of the fragmentation
   dom_core = fragments[0][1]
   dom_yoke = fragments[2][1]
   dom_air = fragments[1][1]

   # and we define physical domains
   gmsh.model.addPhysicalGroup(3, [dom_core, dom_yoke], 1, name = "Iron")
   gmsh.model.occ.synchronize()
   gmsh.model.addPhysicalGroup(3, [dom_air], 2, name = "Air")
   gmsh.model.occ.synchronize()

   # we then generate the mesh
   gmsh.model.mesh.generate(3)
   gmsh.model.occ.synchronize()

   gmsh.model.occ.synchronize()


   # ====================================================
   # Make a coil object
   kp = np.array([[-X_core_2 - yoke_spacer - ins, Z_pos + Z_len             ],
                  [-X_core_2,          Z_pos + Z_len + yoke_spacer + ins    ],
                  [ X_core_2,          Z_pos + Z_len + yoke_spacer + ins    ],
                  [ X_core_2 + yoke_spacer + ins,   Z_pos + Z_len           ],
                  [ X_core_1 + yoke_spacer + ins,   Z_pos                   ],
                  [ X_core_1,                       Z_pos-yoke_spacer - ins ],
                  [-X_core_1,                       Z_pos-yoke_spacer - ins ],
                  [-X_core_1 - yoke_spacer - ins,   Z_pos                   ]])
   
   # determine the slot size
   slot_size = 2*min(Y_core_1, Y_core_2)

   # determine the number of conductors
   num_cond = np.int32(slot_size/2/(coil_radius+ins))

   y = np.linspace(-0.5*slot_size + coil_radius + ins,
                    0.5*slot_size - coil_radius - ins, num_cond)

   coil = snoopy.RacetrackCoil(kp, y, coil_radius, current/num_cond)

   snoopy.plot_domain(pl, gmsh.model.mesh, dom_core, reflect_xz=True, reflect_yz=True, show_edges=show_edges, opacity=opacity, plot_feature_edges=plot_feature_edges)
   snoopy.plot_domain(pl, gmsh.model.mesh, dom_yoke, reflect_xz=True, reflect_yz=True, show_edges=show_edges, opacity=opacity, plot_feature_edges=plot_feature_edges)
   # snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, reflect_xz=True, reflect_yz=True, show_edges=False, opacity=0.0)
   coil.plot_pv(pl)

   return None

def plot_geometry_mag_3(pl, parameters, df_index=0, lc=1.0, opacity=0.0, show_edges=False, plot_feature_edges=True):
   '''Plot the geometry for the magnet 2 template.
    
   :params pl:
      The plotter object.

   :params parameters:
      The magnet parameters as pandas dataframe. See the
      SHiP documentation for details.

   :param df_index:
      The row in the pandas parameter dataframe.
   
   :param lc:
      The mesh size parameter. Default = 0.5.

   :param opacity:
      The opacity of the iron domain. Default=False.

   :param show_edges:
      Set this flag to enable plotting the edges. Default=True.

   :return:
      None
   '''
   # ====================================================
   # mesh generation
   gmsh.initialize()
   gmsh.model.add("make mesh mag 3 template")

   # we get the geometry parameters as variables for convinience
   X_mgap_1 = parameters["Xmgap1(m)"][df_index]
   X_mgap_2 = parameters["Xmgap2(m)"][df_index]

   X_core_1 = parameters["Xcore1(m)"][df_index]
   X_core_2 = parameters["Xcore2(m)"][df_index]

   X_void_1 = parameters["Xvoid1(m)"][df_index]
   X_void_2 = parameters["Xvoid2(m)"][df_index]

   X_yoke_1 = parameters["Xyoke1(m)"][df_index]
   X_yoke_2 = parameters["Xyoke2(m)"][df_index]

   Y_core_1 = parameters["Ycore1(m)"][df_index]
   Y_core_2 = parameters["Ycore2(m)"][df_index]

   Y_void_1 = parameters["Yvoid1(m)"][df_index]
   Y_void_2 = parameters["Yvoid2(m)"][df_index]

   Y_yoke_1 = parameters["Yyoke1(m)"][df_index]
   Y_yoke_2 = parameters["Yyoke2(m)"][df_index]

   Z_len = parameters["Z_len(m)"][df_index]
   Z_pos = parameters["Z_pos(m)"][df_index]

   delta_x = parameters["delta_x(m)"][df_index]
   delta_y = parameters["delta_y(m)"][df_index]
   delta_z = parameters["delta_z(m)"][df_index]
   
   yoke_spacer = parameters["yoke_spacer(m)"][df_index]

   ins = parameters["insulation(m)"][df_index]

   current = parameters["NI(A)"][df_index]

   coil_radius = 0.5*parameters["coil_diam(mm)"][df_index]*1e-3

   field_density = 0.5*parameters["field_density"][df_index]

   # the limits in x, y, and z
   lim_x = max([X_yoke_1, X_yoke_2]) + delta_x
   lim_y = max([Y_yoke_1, Y_yoke_2]) + delta_y
   z_min = Z_pos - delta_z
   z_max = Z_pos + Z_len + delta_z
   
   # the iron domain
   vol_iron = snoopy.add_SHIP_iron_yoke(gmsh.model, X_mgap_1,
                                                    X_core_1,
                                                    X_void_1,
                                                    X_yoke_1,
                                                    X_mgap_2, 
                                                    X_core_2,
                                                    X_void_2,
                                                    X_yoke_2,
                                                    Y_core_1,
                                                    Y_void_1,
                                                    Y_yoke_1,
                                                    Y_core_2,
                                                    Y_void_2,
                                                    Y_yoke_2,
                                                    Z_len, 
                                                    Z_pos=Z_pos,
                                                    lc=lc,
                                                    lc_inner=0.3*lc,
                                                    yoke_type=3)

   # the iron domain
   vol_air = gmsh.model.occ.addBox(0.0, 0.0, z_min, lim_x, lim_y, z_max - z_min)
   gmsh.model.occ.synchronize()

   # fragment perfoms something like a union
   fragments, _ = gmsh.model.occ.fragment([(3, vol_iron)], [(3, vol_air)])
   gmsh.model.occ.synchronize()

   # we get the domains of the fragmentation
   dom_iron = fragments[0][1]
   dom_air = fragments[1][1]

   # and we define physical domains
   gmsh.model.addPhysicalGroup(3, [dom_iron], 1, name = "Iron")
   gmsh.model.occ.synchronize()
   gmsh.model.addPhysicalGroup(3, [dom_air], 2, name = "Air")
   gmsh.model.occ.synchronize()

   # we then generate the mesh
   gmsh.model.mesh.generate(3)
   gmsh.model.occ.synchronize()

   gmsh.model.occ.synchronize()

   # ====================================================
   # Make the coil objects

   # this list stores the coil objects
   coil_list = []

   # determine the slot size
   slot_size = 2*min(Y_core_1, Y_core_2)

   # determine the number of conductors
   num_cond = np.int32(slot_size/2/(coil_radius+ins))

   # these are the vertical positions
   y = np.linspace(-0.5*slot_size + coil_radius + ins,
                    0.5*slot_size - coil_radius - ins, num_cond)

   # make two coils

   kp_1 = np.array([[ X_void_2 - yoke_spacer - ins, Z_pos + Z_len           ],
                  [ X_void_2,          Z_pos + Z_len + yoke_spacer + ins    ],
                  [ X_yoke_2,          Z_pos + Z_len + yoke_spacer + ins    ],
                  [ X_yoke_2 + yoke_spacer + ins,   Z_pos + Z_len           ],
                  [ X_yoke_1 + yoke_spacer + ins,   Z_pos                   ],
                  [ X_yoke_1,                       Z_pos-yoke_spacer - ins ],
                  [ X_void_1,                       Z_pos-yoke_spacer - ins ],
                  [ X_void_1 - yoke_spacer - ins,   Z_pos                   ]])
      
   kp_2 = kp_1.copy()
   kp_2[:, 0] *= -1.0
      
   coil_list.append(snoopy.RacetrackCoil(kp_1, y, coil_radius, current/num_cond))
   coil_list.append(snoopy.RacetrackCoil(kp_2, y, coil_radius, current/num_cond))


   snoopy.plot_domain(pl, gmsh.model.mesh, dom_iron, reflect_xz=True, reflect_yz=True, show_edges=show_edges, opacity=opacity, plot_feature_edges=plot_feature_edges)
   # snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, reflect_xz=True, reflect_yz=True, show_edges=False, opacity=0.0)
   for coil in coil_list:
      coil.plot_pv(pl)