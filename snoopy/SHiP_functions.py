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
import json

import snoopy.red_mvp_solver_diluted
from .geometry_tools import compute_area_polygon
from .materials import Reluctance
from snoopy.evaluators import evaluate_fem_solution_node_interpolation

import snoopy

def get_NI(B_goal, parameters_df, df_index, materials_directory='files/materials'):
    """ Get the ampere turns for a SHiP type magnet geometry.

    :param B_goal:
        The desired B field in the goal region (depends on magnet type).

    :param parameters_df:
        The magnet parameters dataframe.

    :param df_index:
        The index of the magnet in the dataframe.

    :param materials_directory:
        The directory where the materials are found. Default='files/materials'

    :return:
        The value for the ampere turns. We also return a flag that indicates if the magnet
        is valid for air cooling.
    """

    # get the reluctance
    reluctance = Reluctance(os.path.join(materials_directory,
                                         parameters_df["material"][df_index]))

    # to simplify the area calculations we define keypoints
    kp_1 = np.zeros((5, 2))
    kp_2 = np.zeros((5, 2))
    kp_3 = np.zeros((5, 2))

    # get the parameters for convinience
    X_mgap_1 = parameters_df["Xmgap1(m)"][df_index]
    X_core_1 = parameters_df["Xcore1(m)"][df_index]
    X_void_1 = parameters_df["Xvoid1(m)"][df_index]
    X_yoke_1 = parameters_df["Xyoke1(m)"][df_index]

    X_mgap_2 = parameters_df["Xmgap2(m)"][df_index]
    X_core_2 = parameters_df["Xcore2(m)"][df_index]
    X_void_2 = parameters_df["Xvoid2(m)"][df_index]
    X_yoke_2 = parameters_df["Xyoke2(m)"][df_index]

    Y_core_1 = parameters_df["Ycore1(m)"][df_index]
    Y_void_1 = parameters_df["Yvoid1(m)"][df_index]
    Y_yoke_1 = parameters_df["Yyoke1(m)"][df_index]

    Y_core_2 = parameters_df["Ycore2(m)"][df_index]
    Y_void_2 = parameters_df["Yvoid2(m)"][df_index]
    Y_yoke_2 = parameters_df["Yyoke2(m)"][df_index]

    Z_len = parameters_df["Z_len(m)"][df_index]

    # the left horizontal cut
    kp_1[0, 0] = X_mgap_1
    kp_1[0, 0] = 0.0

    kp_1[1, 0] = X_core_1
    kp_1[1, 1] = 0.0

    kp_1[2, 0] = X_core_2
    kp_1[2, 1] = Z_len

    kp_1[3, 0] = X_mgap_2
    kp_1[3, 1] = Z_len

    kp_1[4, 0] = X_mgap_1
    kp_1[4, 1] = 0.0

    # the vertical cut
    kp_2[0, 0] = Y_void_1
    kp_2[0, 0] = 0.0

    kp_2[1, 0] = Y_yoke_1
    kp_2[1, 1] = 0.0

    kp_2[2, 0] = Y_yoke_2
    kp_2[2, 1] = Z_len

    kp_2[3, 0] = Y_void_2
    kp_2[3, 1] = Z_len

    kp_2[4, 0] = Y_void_1
    kp_2[4, 1] = 0.0


    # the right horizontal cut
    kp_3[0, 0] = X_void_1
    kp_3[0, 0] = 0.0

    kp_3[1, 0] = X_yoke_1
    kp_3[1, 1] = 0.0

    kp_3[2, 0] = X_yoke_2
    kp_3[2, 1] = Z_len

    kp_3[3, 0] = X_void_2
    kp_3[3, 1] = Z_len

    kp_3[4, 0] = X_void_1
    kp_3[4, 1] = 0.0

    # compute the areas
    A1 = compute_area_polygon(kp_1)
    A2 = compute_area_polygon(kp_2)
    A3 = compute_area_polygon(kp_3)

    valid_geometry = True


    print(f"area 1 = {A1:.2f} m2")
    print(f"area 2 = {A2:.2f} m2")
    print(f"area 3 = {A3:.2f} m2")

    if parameters_df["yoke_type"][df_index] == 'Mag1' or parameters_df["yoke_type"][df_index] == 'Mag2':

        # the type 1 or 2 magnet
        # the coil is wound around the core

        # compute the path lengths
        l1 = 0.25*(Y_core_1 + Y_core_2 + Y_yoke_1 + Y_yoke_2)
        l2 = (   0.5*(X_yoke_1 + X_yoke_2 ) + 0.5*(X_void_1 + X_void_2)
               - 0.5*(X_core_1 + X_core_2 ) - 0.5*(X_mgap_1 + X_mgap_2)  )
        l3 = l1

        # compute the Flux
        Phi = abs(B_goal)*A1

        # compute the magnetic fields
        H1 = reluctance.evaluate_nu(B_goal)*B_goal
        H2 = reluctance.evaluate_nu(Phi/A2)*Phi/A2
        H3 = reluctance.evaluate_nu(Phi/A3)*Phi/A3

        NI = H1*l1 + H2*l2 + H3*l3

        if B_goal < 0.0:
            NI *= -1.0


        # we return this flag to inducate the validity
        valid_geometry = A1 < A3

    elif parameters_df["yoke_type"][df_index] == 'Mag3':

        # the type 3 magnet
        # the coil is wound around the leg

        # compute the path lengths
        l1 = 0.25*(Y_core_1 + Y_core_2 + Y_yoke_1 + Y_yoke_2)
        l2 = (   0.5*(X_yoke_1 + X_yoke_2 ) + 0.5*(X_void_1 + X_void_2)
               - 0.5*(X_core_1 + X_core_2 ) - 0.5*(X_mgap_1 + X_mgap_2)  )
        l3 = l1

        # compute the Flux
        Phi = abs(B_goal)*A3

        # compute the magnetic fields
        H1 = reluctance.evaluate_nu(Phi/A1)*Phi/A1
        H2 = reluctance.evaluate_nu(Phi/A2)*Phi/A2
        H3 = reluctance.evaluate_nu(Phi/A3)*Phi/A3

        NI = H1*l1 + H2*l2 + H3*l3

        if B_goal < 0.0:
            NI *= -1.0

        # we return this flag to inducate the validity
        valid_geometry = A1 > A3

    return NI, valid_geometry

def compute_prices(magnet_parameters, df_index, M_iron,
                   M_coil, Q, electricity_costs=5.0/72000.0, runtime=72000,
                   materials_directory='files/materials',
                   price_peanuts=0.5):
    '''Compute the costs of a magnet given the parameters iron mass,
    coil mass and power consumption.
    
    :param magnet parameters:
        The magnet parameters. These are needed to know the material data.

    :param df_index:
        The row index in the pandas dataframe for the magnet parameters.

    :param M_iron:
        The mass of the iron domain in g.

    :param M_coil:
        The mass of the coil in g.
    
    :param Q:
        The power consumption in W.

    :param electricity costs:
        The cost for the electricity in CHF/Wh. Default 5 CHF/Wh.

    :param runtime:
        The runtime of the experiment in hours. Default 72000 h, means 15 years
        with 4800 h. 

    :param materials_directory:
        The directory where to find the materials.

    :param price_peanuts:
        The price of a peanut in peanuts/CHF. Default is 0.5.
    
    :return:
        The costs for iron, coil and electricity.
    '''

    # read the material data
    with open(os.path.join(materials_directory, magnet_parameters["material"][df_index])) as f:
        iron_material_data = json.load(f)

    with open(os.path.join(materials_directory, magnet_parameters["coil_material"][df_index])) as f:
        coil_material_data = json.load(f)

    C_iron = 1e-3*M_iron*(iron_material_data["material_cost(CHF/kg)"]
                     +  iron_material_data["manufacturing_cost(CHF/kg)"])*price_peanuts

    C_coil = 1e-3*M_coil*(coil_material_data["material_cost(CHF/kg)"]
                     +  coil_material_data["manufacturing_cost(CHF/kg)"])*price_peanuts

    C_edf = Q*electricity_costs*runtime*price_peanuts

    return C_iron, C_coil, C_edf

def compute_J_opt(conductor_material_data, kappa_elec):
    '''Compute the optimal current density based on the material propoerties and
    the electricity costs.

    :param conductor_material_data:
        The conductor material data.

    :param kappa_elec:
        The electricity costs.

    :return:
        The optimal current density in A/mm2.
    '''
    
    kappa_cu = conductor_material_data["material_cost(CHF/kg)"]
    kappa_cu += conductor_material_data["manufacturing_cost(CHF/kg)"]
    dens = conductor_material_data["density(g/m3)"]
    rho = conductor_material_data["resistivity(Ohm.m)"]

    return np.sqrt(kappa_cu*dens*1e-3/kappa_elec/rho)*1e-6


def get_vector_field_mag_1(parameters,
                  df_index=0, lc=0.2,
                  geo_th=1e-5, run_gmsh=False, plot_geo=False,
                  plot_result=False, result_directory='none', result_spec='',
                  eval_pos=np.zeros((0, 3)),
                  materials_directory='files/materials',
                  quad_order=8,
                  max_coil_size=0.05,
                  use_diluted_steel=False,
                  kappa_elec=5.0):
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

    :param quad_order:
       The quadrature order. Default 8.

    :param max_coil_size:
       The maximum coil size in m2.

    :param use_diluted_steel:
       Set this flag to true if You like to use a diluted steel in the
       return yoke.
    
    :param kappa_elec:
        The costs for a W of power in CHF.
    
    :return:
       The positions and field components in a 3D numpy grid.
    '''

    # launch time measurement
    t_s = time.time()

    # ====================================================
    # read the material data
    reluctance_iron = snoopy.Reluctance(os.path.join(materials_directory,
                                                     parameters["material"][df_index]))

    with open(os.path.join(materials_directory, parameters["material"][df_index])) as f:
        iron_material_data = json.load(f)

    with open(os.path.join(materials_directory, parameters["coil_material"][df_index])) as f:
        conductor_material_data = json.load(f)

    # get the target current density
    if parameters["J_tar(A/mm2)"][df_index] < 0:
        J_tar = compute_J_opt(conductor_material_data, kappa_elec)*1e6
    else:
        J_tar = parameters["J_tar(A/mm2)"][df_index]*1e6

    # ====================================================
    # mesh generation
    gmsh.initialize()
    gmsh.model.add("make mesh mag 1 template")
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")

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
   
    yoke_spacer = parameters["yoke_spacer(mm)"][df_index]*1e-3

    ins = parameters["insulation(mm)"][df_index]*1e-3

    current = parameters["NI(A)"][df_index]

    coil_radius = 0.5*parameters["coil_diam(mm)"][df_index]*1e-3

    field_density = 0.5*parameters["field_density"][df_index]

    if use_diluted_steel:
        # setup the dilution
        dilution_wing = snoopy.Dilution(X_mgap_1, X_core_1, X_void_1, X_yoke_1,
                                        X_mgap_2, X_core_2, X_void_2, X_yoke_2,
                                        Z_pos, Z_len)

    # the limits in x, y, and z
    lim_x = max([X_yoke_1, X_yoke_2]) + delta_x
    lim_y = max([Y_yoke_1, Y_yoke_2]) + delta_y
    z_min = Z_pos - delta_z
    z_max = Z_pos + Z_len + delta_z

    # the maximum number of turns
    max_turns = np.int64(parameters["max_turns"][df_index])

    if use_diluted_steel:
        # the iron domain
        vol_iron_1, vol_iron_2 = snoopy.add_SHIP_iron_yoke_diluted_wing(gmsh.model, X_mgap_1,
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
                                                                     lc_inner=0.2*lc)

        # the iron domain
        vol_air = gmsh.model.occ.addBox(0.0, 0.0, z_min, lim_x, lim_y, z_max - z_min)
        gmsh.model.occ.synchronize()

        # fragment perfoms something like a union
        fragments, _ = gmsh.model.occ.fragment([(3, vol_air)], [(3, vol_iron_1),(3, vol_iron_2)])
        gmsh.model.occ.synchronize()

        # we get the domains of the fragmentation
        dom_iron = fragments[0][1]
        dom_wing = fragments[1][1]
        dom_air = fragments[2][1]

        # and we define physical domains
        gmsh.model.addPhysicalGroup(3, [dom_iron, dom_wing], 1, name = "Iron")
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [dom_air], 2, name = "Air")
        gmsh.model.occ.synchronize()

        # get the volume of the iron domain
        M_iron = 4*gmsh.model.occ.getMass(3, dom_iron)*iron_material_data["density(g/m3)"]
        M_iron += 4*gmsh.model.occ.getMass(3, dom_wing)*iron_material_data["density(g/m3)"]

    else:
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

        # get the volume of the iron domain
        M_iron = 4*gmsh.model.occ.getMass(3, dom_iron)*iron_material_data["density(g/m3)"]

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

    # do not allow more conductors than a certain amount
    if num_cond > max_turns:
        num_cond = max_turns

    # these are the vertical positions
    y = np.linspace(-0.5*slot_size + coil_radius + ins,
                     0.5*slot_size - coil_radius - ins, num_cond)

    # the horizontal slot size
    slot_size_horz = min(X_void_1 - X_core_1, X_void_2 - X_core_2)

    # this is the space we have available for the coils
    A_geo = slot_size_horz*slot_size

    # this is the copper cross section using the target current density
    A_cu = abs(current)/J_tar

    # this is the coil cross section using the target current density (including the filling factor)
    A_coil = abs(current)/J_tar/conductor_material_data["filling_factor"]

    # we compute the current density (for monitoring reasons)
    if current == 0.0:
        current_density = 0.0
    else:
        current_density = abs(current)/min([A_geo, A_coil])/conductor_material_data["filling_factor"]

    # the turn perimeter
    turn_perimeter = 0.0

    if X_mgap_1 == 0.0 or X_mgap_2 == 0.0:

        # make only a single coil
        kp = np.array([[-X_core_2 - yoke_spacer - ins - coil_radius, Z_pos + Z_len             ],
                       [-X_core_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                       [ X_core_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                       [ X_core_2 + yoke_spacer + ins + coil_radius,   Z_pos + Z_len           ],
                       [ X_core_1 + yoke_spacer + ins + coil_radius,   Z_pos                   ],
                       [ X_core_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                       [-X_core_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                       [-X_core_1 - yoke_spacer - ins - coil_radius,   Z_pos                   ]])

        coil_list.append(snoopy.RacetrackCoil(kp, y, coil_radius, current/num_cond))

        turn_perimeter = coil_list[-1].get_length()

    else:

        # make two coils

        kp_1 = np.array([[ X_mgap_2 - yoke_spacer - ins - coil_radius, Z_pos + Z_len + coil_radius ],
                       [ X_mgap_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius        ],
                       [ X_core_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius        ],
                       [ X_core_2 + yoke_spacer + ins + coil_radius,   Z_pos + Z_len               ],
                       [ X_core_1 + yoke_spacer + ins + coil_radius,   Z_pos                       ],
                       [ X_core_1,                       Z_pos-yoke_spacer - ins - coil_radius     ],
                       [ X_mgap_1,                       Z_pos-yoke_spacer - ins - coil_radius     ],
                       [ X_mgap_1 - yoke_spacer - ins - coil_radius,   Z_pos                       ]])


        kp_2 = kp_1.copy()
        kp_2[:, 0] *= -1.0

        coil_list.append(snoopy.RacetrackCoil(kp_1, y, coil_radius, current/num_cond))
        coil_list.append(snoopy.RacetrackCoil(kp_2, y, coil_radius, current/num_cond))

        turn_perimeter = coil_list[0].get_length() +  coil_list[1].get_length()

    # this is the available coil volume
    M_coil = A_cu*turn_perimeter*conductor_material_data['density(g/m3)']

    # ====================================================
    # The power consumption
    if current == 0:
        Q = 0.0
    else:
        Q = abs(current*current)*turn_perimeter*conductor_material_data['resistivity(Ohm.m)']/A_cu

    if plot_geo:

        # ====================================================
        # Plot the magnet
        pl = pv.Plotter(shape=(1, 1), off_screen=False)

        pl.subplot(0, 0)
        if use_diluted_steel:
            snoopy.plot_domain(pl, gmsh.model.mesh, dom_iron, reflect_xz=True, reflect_yz=True, show_edges=True, opacity=1.0)
            snoopy.plot_domain(pl, gmsh.model.mesh, dom_wing, reflect_xz=True, reflect_yz=True, show_edges=True, opacity=1.0)
        else:
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
    if use_diluted_steel:
        solver = snoopy.RedMVPSolverDiluted(gmsh.model, coil_list,
                                      [dom_iron, dom_wing], [dom_air],
                                      [reluctance_iron, reluctance_iron],
                                      [None, dilution_wing],
                                      quad_order=quad_order, max_newton_iterations=25)
      
    else:
        solver = snoopy.RedMVPSolver(gmsh.model, coil_list,
                                      [dom_iron], [dom_air],
                                      [reluctance_iron],
                                      quad_order=quad_order, max_newton_iterations=25)
   

    # ====================================================
    # If current is zero, don't solve
    if current == 0.0:
        x = np.zeros((solver.num_dofs_mvp, ))
    else:
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


    ret_vals = [points, B_i + B_coil, M_iron, M_coil, Q, current_density]

    if eval_pos.shape[0] > 0:
      
        # ====================================================
        # Evaluate at the given positions

        if use_diluted_steel:
            B_tot = evaluate_fem_solution_node_interpolation(gmsh.model,
                                                             eval_pos,
                                                             solver,
                                                             coil_list,
                                                             x,
                                                             [dom_iron, dom_wing, dom_air])
            
        else:
            B_tot = evaluate_fem_solution_node_interpolation(gmsh.model,
                                                             eval_pos,
                                                             solver,
                                                             coil_list,
                                                             x,
                                                             [dom_iron, dom_air])
            
        if not result_directory == 'none':
           output_filename = os.path.join(result_directory, 'B' + result_spec + '.csv')

        else:
           print('Warning, no result directory specified. I store the evaluations in snoopy directory.')
           output_filename = 'B' + result_spec + '.csv'

        out_df = pd.DataFrame(data=np.append(eval_pos, B_tot, axis=1),
                               columns=['x(m)', 'y(m)', 'z(m)', 'Bx(T)', 'By(T)', 'Bz(T)']).to_csv(output_filename, index=False)

    # print some feedback
    print('**********************************************')
    print('The iron mass is = {:.2f} t'.format(M_iron*1e-6))
    print('The coil mass is {:.2f} kg'.format(M_coil*1e-3))
    print('The coil surface is = {:.2f} m2'.format(A_coil))
    print('The coil perimeter is = {:.2f} m'.format(turn_perimeter))
    print('The current density is = {:.2f} A/mm2'.format(current_density*1e-6))
    print('The power consumption = {:.2f} W'.format(Q))
    print('**********************************************')

    return ret_vals


def get_vector_field_mag_2(parameters, df_index=0, lc=0.4,
                   geo_th=1e-5, run_gmsh=False, plot_geo=False,
                   plot_result=False, result_directory='none', result_spec='',
                   eval_pos=np.zeros((0, 3)),
                   materials_directory='files/materials',
                   quad_order=8,
                   max_coil_size=0.05,
                   kappa_elec=5.0):
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

    :param quad_order:
        The quadrature order. Default 8.

    :param max_coil_size:
        The maximum coil size in m2.

    :param kappa_elec:
        The costs for a W of power in CHF.

    :return:
        The positions and field components in a 3D numpy grid.
    '''

    # launch time measurement
    t_s = time.time()

    # ====================================================
    # read the material data
    reluctance_iron = snoopy.Reluctance(os.path.join(materials_directory,
                                                     parameters["material"][df_index]))

    with open(os.path.join(materials_directory, parameters["material"][df_index])) as f:
        iron_material_data = json.load(f)

    with open(os.path.join(materials_directory, parameters["coil_material"][df_index])) as f:
        conductor_material_data = json.load(f)

    # get the target current density
    if parameters["J_tar(A/mm2)"][df_index] < 0:
        J_tar = compute_J_opt(conductor_material_data, kappa_elec)*1e6
    else:
        J_tar = parameters["J_tar(A/mm2)"][df_index]*1e6


    # ====================================================
    # mesh generation
    gmsh.initialize()
    gmsh.model.add("make mesh mag 2 template")
    gmsh.option.setString("Geometry.OCCTargetUnit", "M") 

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

    yoke_spacer = parameters["yoke_spacer(mm)"][df_index]*1e-3

    ins = parameters["insulation(mm)"][df_index]*1e-3

    current = parameters["NI(A)"][df_index]

    coil_radius = 0.5*parameters["coil_diam(mm)"][df_index]*1e-3

    field_density = 0.5*parameters["field_density"][df_index]

    # the maximum number of turns
    max_turns = np.int64(parameters["max_turns"][df_index])

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

    # get the volume of the iron domain
    M_iron = 4*(gmsh.model.occ.getMass(3, dom_core)
               + gmsh.model.occ.getMass(3, dom_yoke))*iron_material_data["density(g/m3)"]

    if run_gmsh:
        gmsh.fltk.run()

    # ====================================================
    # Make a coil object
    kp = np.array([[-X_core_2 - yoke_spacer - ins - coil_radius, Z_pos + Z_len             ],
                  [-X_core_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                  [ X_core_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                  [ X_core_2 + yoke_spacer + ins + coil_radius,   Z_pos + Z_len           ],
                  [ X_core_1 + yoke_spacer + ins + coil_radius,   Z_pos                   ],
                  [ X_core_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                  [-X_core_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                  [-X_core_1 - yoke_spacer - ins - coil_radius,   Z_pos                   ]])

    # determine the slot size
    slot_size = 2*min(Y_core_1, Y_core_2)

    # determine the number of conductors
    num_cond = np.int32(slot_size/2/(coil_radius+ins))

    # do not allow more conductors than a certain amount
    if num_cond > max_turns:
        num_cond = max_turns

    y = np.linspace(-0.5*slot_size + coil_radius + ins,
                    0.5*slot_size - coil_radius - ins, num_cond)

    coil = snoopy.RacetrackCoil(kp, y, coil_radius, current/num_cond)

    # the horizontal slot size
    slot_size_horz = min(X_void_1 - X_core_1, X_void_2 - X_core_2)

    # this is the space we have available for the coils
    A_geo = slot_size_horz*slot_size

    # this is the cupper cross section using the target current density
    A_cu = abs(current)/J_tar

    # this is the coil cross section using the target current density (including the filling factor)
    A_coil = abs(current)/J_tar/conductor_material_data["filling_factor"]

    # we compute the current density (for monitoring reasons)
    if current == 0.0:
        current_density = 0.0
    else:
        current_density = abs(current)/min([A_geo, A_coil])/conductor_material_data["filling_factor"]

    # the turn perimeter
    turn_perimeter = coil.get_length()

    # this is the available coil volume
    M_coil = A_cu*turn_perimeter*conductor_material_data["density(g/m3)"]

    # ====================================================
    # The power consumption
    if current == 0.0:
        Q = 0.0
    else:
        Q = abs(current*current)*turn_perimeter*conductor_material_data['resistivity(Ohm.m)']/A_cu

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
                                 quad_order=quad_order, max_newton_iterations=25)
   
    if current == 0.0:
        x = np.zeros((solver.num_dofs_mvp, ))
    else:
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


    ret_vals = [points, B_i + B_coil, M_iron, M_coil, Q, current_density]

    if eval_pos.shape[0] > 0:
      
        # ====================================================
        # Evaluate at the given positions

        B_tot = evaluate_fem_solution_node_interpolation(gmsh.model,
                                                             eval_pos,
                                                             solver,
                                                             [coil],
                                                             x,
                                                             [dom_core, dom_yoke, dom_air])
            
        if not result_directory == 'none':
            output_filename = os.path.join(result_directory, 'B' + result_spec + '.csv')

        else:
            print('Warning, no result directory specified. I store the evaluations in snoopy directory.')
            output_filename = 'B' + result_spec + '.csv'

        out_df = pd.DataFrame(data=np.append(eval_pos, B_tot, axis=1),
                              columns=['x(m)', 'y(m)', 'z(m)', 'Bx(T)', 'By(T)', 'Bz(T)']).to_csv(output_filename, index=False)

    # print some feedback
    print('**********************************************')
    print('The iron mass is = {:.2f} t'.format(M_iron*1e-6))
    print('The coil mass is {:.2f} kg'.format(M_coil*1e-3))
    print('The coil surface is = {:.2f} m2'.format(A_coil))
    print('The coil perimeter is = {:.2f} m'.format(turn_perimeter))
    print('The current density is = {} A/mm2'.format(current_density*1e-6))
    print('The power consumption = {:.2f} W'.format(Q))
    print('**********************************************')

    return ret_vals


def get_vector_field_mag_3(parameters, df_index=0, lc=0.2,
                  geo_th=1e-5, run_gmsh=False, plot_geo=False,
                  plot_result=False, result_directory='none', result_spec='',
                  eval_pos=np.zeros((0, 3)),
                  materials_directory='files/materials',
                  quad_order=8,
                  max_coil_size=0.05,
                  use_diluted_steel=False,
                  kappa_elec=5.0):
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

    :param quad_order:
        The quadrature order. Default 8.
   
    :param max_coil_size:
        The maximum coil size in m2.

    :param use_diluted_steel:
        Set this flag if You like to use diluted steel in the core.

    :param kappa_elec:
        The costs for a W of power in CHF.

    :return:
        The positions and field components in a 3D numpy grid.
    '''


    # ====================================================
    # read the material data
    reluctance_iron = snoopy.Reluctance(os.path.join(materials_directory,
                                                     parameters["material"][df_index]))
   
    with open(os.path.join(materials_directory, parameters["material"][df_index])) as f:
        iron_material_data = json.load(f)


    with open(os.path.join(materials_directory, parameters["coil_material"][df_index])) as f:
        conductor_material_data = json.load(f)

    # get the target current density
    if parameters["J_tar(A/mm2)"][df_index] < 0:
        J_tar = compute_J_opt(conductor_material_data, kappa_elec)*1e6
    else:
        J_tar = parameters["J_tar(A/mm2)"][df_index]*1e6


    # ====================================================
    # mesh generation
    gmsh.initialize()
    gmsh.model.add("make mesh mag 3 template")
    gmsh.option.setString("Geometry.OCCTargetUnit", "M") 

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
   
    yoke_spacer = parameters["yoke_spacer(mm)"][df_index]*1e-3

    ins = parameters["insulation(mm)"][df_index]*1e-3

    current = parameters["NI(A)"][df_index]

    coil_radius = 0.5*parameters["coil_diam(mm)"][df_index]*1e-3

    field_density = 0.5*parameters["field_density"][df_index]

    # the maximum number of turns
    max_turns = np.int64(parameters["max_turns"][df_index])

    # the following is used in case of diluted iron core
    if use_diluted_steel:
        # setup the dilution
        dilution_wing = snoopy.Dilution(X_mgap_1, X_core_1, X_void_1, X_yoke_1,
                                        X_mgap_2, X_core_2, X_void_2, X_yoke_2,
                                        Z_pos, Z_len, mag_type=1)
       

    # the limits in x, y, and z
    lim_x = max([X_yoke_1, X_yoke_2]) + delta_x
    lim_y = max([Y_yoke_1, Y_yoke_2]) + delta_y
    z_min = Z_pos - delta_z
    z_max = Z_pos + Z_len + delta_z
   
    if use_diluted_steel:
        # the iron domain
        vol_iron_1, vol_iron_2 = snoopy.add_SHIP_iron_yoke_diluted_wing(gmsh.model, X_mgap_1,
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
                                                                     lc_inner=0.2*lc)

        # the iron domain
        vol_air = gmsh.model.occ.addBox(0.0, 0.0, z_min, lim_x, lim_y, z_max - z_min)
        gmsh.model.occ.synchronize()

        # fragment perfoms something like a union
        fragments, _ = gmsh.model.occ.fragment([(3, vol_air)], [(3, vol_iron_1),(3, vol_iron_2)])
        gmsh.model.occ.synchronize()

        # we get the domains of the fragmentation
        dom_iron = fragments[0][1]
        dom_wing = fragments[1][1]
        dom_air = fragments[2][1]

        # and we define physical domains
        gmsh.model.addPhysicalGroup(3, [dom_iron, dom_wing], 1, name = "Iron")
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [dom_air], 2, name = "Air")
        gmsh.model.occ.synchronize()

        # get the volume of the iron domain
        M_iron = 4*gmsh.model.occ.getMass(3, dom_iron)*iron_material_data["density(g/m3)"]
        M_iron += 4*gmsh.model.occ.getMass(3, dom_wing)*iron_material_data["density(g/m3)"]
    
    else:
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

        # get the volume of the iron domain
        M_iron = 4*gmsh.model.occ.getMass(3, dom_iron)*iron_material_data["density(g/m3)"]


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

    # do not allow more conductors than a certain amount
    if num_cond > max_turns:
        num_cond = max_turns

    # these are the vertical positions
    y = np.linspace(-0.5*slot_size + coil_radius + ins,
                     0.5*slot_size - coil_radius - ins, num_cond)

    # make two coils
    kp_1 = np.array([[ X_void_2 - yoke_spacer - ins - coil_radius, Z_pos + Z_len           ],
                   [ X_void_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                   [ X_yoke_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                   [ X_yoke_2 + yoke_spacer + ins + coil_radius,   Z_pos + Z_len           ],
                   [ X_yoke_1 + yoke_spacer + ins + coil_radius,   Z_pos                   ],
                   [ X_yoke_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                   [ X_void_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                   [ X_void_1 - yoke_spacer - ins - coil_radius,   Z_pos                   ]])

    kp_2 = kp_1.copy()
    kp_2[:, 0] *= -1.0
      
    coil_list.append(snoopy.RacetrackCoil(kp_1, y, coil_radius, current/num_cond))
    coil_list.append(snoopy.RacetrackCoil(kp_2, y, coil_radius, current/num_cond))

    # the horizontal slot size
    slot_size_horz = min(X_void_1 - X_core_1, X_void_2 - X_core_2)   

    # this is the space we have available for the coils
    A_geo = slot_size_horz*slot_size

    # this is the cupper cross section using the target current density
    A_cu = abs(current)/J_tar

    # this is the coil cross section using the target current density (including the filling factor)
    A_coil = abs(current)/J_tar/conductor_material_data["filling_factor"]

    # we compute the current density (for monitoring reasons)
    if current == 0.0:
        current_density = 0.0
    else:
        current_density = abs(current)/min([A_geo, A_coil])/conductor_material_data["filling_factor"]


    # the turn perimeter
    turn_perimeter = coil_list[0].get_length() + coil_list[1].get_length()

    # this is the available coil volume
    M_coil = A_cu*turn_perimeter*conductor_material_data["density(g/m3)"]

    # ====================================================
    # The power consumption

    if current == 0.0:
        Q = 0.0
    else:
        Q = abs(current*current)*turn_perimeter*conductor_material_data['resistivity(Ohm.m)']/A_cu


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
    if use_diluted_steel:
      solver = snoopy.RedMVPSolverDiluted(gmsh.model, coil_list, 
                                    [dom_iron, dom_wing], [dom_air],
                                    [reluctance_iron, reluctance_iron],
                                    [None, dilution_wing],
                                    quad_order=quad_order, max_newton_iterations=25)
      
    else:
      solver = snoopy.RedMVPSolver(gmsh.model, coil_list, 
                                    [dom_iron], [dom_air],
                                    [reluctance_iron],
                                    quad_order=quad_order, max_newton_iterations=25)
   
    if current == 0.0:
        x = np.zeros((solver.num_dofs_mvp, ))
    else:
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
        if use_diluted_steel:
            snoopy.plot_domain(pl, gmsh.model.mesh, dom_wing, plot_volume=False)
        snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, plot_volume=False, opacity=0.0)
        snoopy.plot_vector_field(pl, points, B_i, title='B iron in T', mag=0.1, sym_yz=1, sym_xz=2, opacity='linear')
        pl.show_grid()
        pl.camera_position = 'zy'
        pl.camera.elevation = 35
        pl.camera.azimuth = 45
        pl.add_axes()

        pl.subplot(0, 1)
        snoopy.plot_domain(pl, gmsh.model.mesh, dom_iron, plot_volume=False)
        if use_diluted_steel:
            snoopy.plot_domain(pl, gmsh.model.mesh, dom_wing, plot_volume=False)
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


    ret_vals = [points, B_i + B_coil, M_iron, M_coil, Q, current_density]

    if eval_pos.shape[0] > 0:
      
        # ====================================================
        # Evaluate at the given positions
        if use_diluted_steel:
            B_tot = evaluate_fem_solution_node_interpolation(gmsh.model,
                                                             eval_pos,
                                                             solver,
                                                             coil_list,
                                                             x,
                                                             [dom_iron, dom_wing, dom_air])
            
        else:
            B_tot = evaluate_fem_solution_node_interpolation(gmsh.model,
                                                             eval_pos,
                                                             solver,
                                                             coil_list,
                                                             x,
                                                             [dom_iron, dom_air])
            
        if not result_directory == 'none':
            output_filename = os.path.join(result_directory, 'B' + result_spec + '.csv')

        else:
            print('Warning, no result directory specified. I store the evaluations in snoopy directory.')
            output_filename = 'B' + result_spec + '.csv'

        out_df = pd.DataFrame(data=np.append(eval_pos, B_tot, axis=1),
                              columns=['x(m)', 'y(m)', 'z(m)', 'Bx(T)', 'By(T)', 'Bz(T)']).to_csv(output_filename, index=False)

    # print some feedback
    print('**********************************************')
    print('The iron mass is = {:.2f} t'.format(M_iron*1e-6))
    print('The coil mass is {:.2f} kg'.format(M_coil*1e-3))
    print('The coil surface is = {:.2f} m2'.format(A_coil))
    print('The coil perimeter is = {:.2f} m'.format(turn_perimeter))
    print('The current density is = {} A/mm2'.format(current_density*1e-6))
    print('The power consumption = {:.2f} W'.format(Q))
    print('**********************************************')

    return ret_vals


def get_vector_field_mag_4(parameters, df_index=0, lc=0.2,
                  geo_th=1e-5, run_gmsh=False, plot_geo=False,
                  plot_result=False, result_directory='none', result_spec='',
                  eval_pos=np.zeros((0, 3)),
                  materials_directory='files/materials',
                  quad_order=8,
                  max_coil_size=0.05,
                  use_diluted_steel=False,
                  kappa_elec=5.0):
    '''Get the vector point cloud for the magnet 4 template.
    
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

    :param quad_order:
        The quadrature order. Default 8.
   
    :param max_coil_size:
        The maximum coil size in m2.

    :param use_diluted_steel:
        Set this flag if You like to use diluted steel in the core.

    :param kappa_elec:
        The costs for a W of power in CHF.

    :return:
        The positions and field components in a 3D numpy grid.
    '''


    # ====================================================
    # read the material data
    reluctance_iron = snoopy.Reluctance(os.path.join(materials_directory,
                                                     parameters["material"][df_index]))
   
    with open(os.path.join(materials_directory, parameters["material"][df_index])) as f:
        iron_material_data = json.load(f)


    with open(os.path.join(materials_directory, parameters["coil_material"][df_index])) as f:
        conductor_material_data = json.load(f)

    # get the target current density
    if parameters["J_tar(A/mm2)"][df_index] < 0:
        J_tar = compute_J_opt(conductor_material_data, kappa_elec)*1e6
    else:
        J_tar = parameters["J_tar(A/mm2)"][df_index]*1e6


    # ====================================================
    # mesh generation
    gmsh.initialize()
    gmsh.model.add("make mesh mag 4 template")
    gmsh.option.setString("Geometry.OCCTargetUnit", "M") 

    # we get the geometry parameters as variables for convinience
    X_A_1 = parameters["Xmgap1(m)"][df_index]
    X_A_2 = parameters["Xmgap2(m)"][df_index]

    X_B_1 = parameters["Xcore1(m)"][df_index]
    X_B_2 = parameters["Xcore2(m)"][df_index]

    X_C_1 = parameters["Xvoid1(m)"][df_index]
    X_C_2 = parameters["Xvoid2(m)"][df_index]

    X_D_1 = parameters["Xcore1(m)"][df_index+1]
    X_D_2 = parameters["Xcore2(m)"][df_index+1]

    X_E_1 = parameters["Xvoid1(m)"][df_index+1]
    X_E_2 = parameters["Xvoid2(m)"][df_index+1]

    X_F_1 = parameters["Xyoke1(m)"][df_index+1]
    X_F_2 = parameters["Xyoke2(m)"][df_index+1]

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
   
    yoke_spacer = parameters["yoke_spacer(mm)"][df_index]*1e-3

    ins = parameters["insulation(mm)"][df_index]*1e-3

    current = parameters["NI(A)"][df_index]

    coil_radius = 0.5*parameters["coil_diam(mm)"][df_index]*1e-3

    field_density = 0.5*parameters["field_density"][df_index]

    # the maximum number of turns
    max_turns = np.int64(parameters["max_turns"][df_index])

    # the following is used in case of diluted iron core
    if use_diluted_steel:
        # setup the dilution
        dilution_iron = snoopy.Dilution(X_A_1 + X_E_1, X_B_1 + X_F_1, X_C_1, X_D_1,
                                       X_A_2 + X_E_2, X_B_2 + X_F_2, X_C_2, X_D_2,
                                       Z_pos, Z_len)
       
    # the limits in x, y, and z
    lim_x = max([X_F_1, X_F_2]) + delta_x
    lim_y = max([Y_yoke_1, Y_yoke_2]) + delta_y
    z_min = Z_pos - delta_z
    z_max = Z_pos + Z_len + delta_z

    # the iron domain
    vol_core, vol_yoke = snoopy.add_SHIP_iron_yoke_mag_4(gmsh.model,
                                                         X_A_1, X_B_1, X_C_1, X_D_1, X_E_1, X_F_1,
                                                         X_A_2, X_B_2, X_C_2, X_D_2, X_E_2, X_F_2,
                                                         Y_core_1, Y_yoke_1,
                                                         Y_core_2, Y_yoke_2,
                                                         Z_len,
                                                         Z_pos=Z_pos,
                                                         lc=lc,
                                                         lc_inner=lc)
    # the iron domain
    vol_air = gmsh.model.occ.addBox(0.0, 0.0, z_min, lim_x, lim_y, z_max - z_min)
    gmsh.model.occ.synchronize()


    # fragment perfoms something like a union
    fragments, _ = gmsh.model.occ.fragment([(3, vol_air)], [(3, vol_core),(3, vol_yoke)])
    gmsh.model.occ.synchronize()

    # we get the domains of the fragmentation
    dom_iron = fragments[0][1]
    dom_wing = fragments[1][1]
    dom_air = fragments[2][1]

    # and we define physical domains
    gmsh.model.addPhysicalGroup(3, [dom_iron, dom_wing], 1, name = "Iron")
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [dom_air], 2, name = "Air")
    gmsh.model.occ.synchronize()

    # get the volume of the iron domain
    M_iron = 4*gmsh.model.occ.getMass(3, dom_iron)*iron_material_data["density(g/m3)"]
    M_iron += 4*gmsh.model.occ.getMass(3, dom_wing)*iron_material_data["density(g/m3)"]
    
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

    # do not allow more conductors than a certain amount
    if num_cond > max_turns:
        num_cond = max_turns

    # these are the vertical positions
    y = np.linspace(-0.5*slot_size + coil_radius + ins,
                     0.5*slot_size - coil_radius - ins, num_cond)

    # make two coils
    kp_1 = np.array([[ X_C_2 - yoke_spacer - ins - coil_radius, Z_pos + Z_len           ],
                   [ X_C_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                   [ X_D_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                   [ X_D_2 + yoke_spacer + ins + coil_radius,   Z_pos + Z_len           ],
                   [ X_D_1 + yoke_spacer + ins + coil_radius,   Z_pos                   ],
                   [ X_D_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                   [ X_C_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                   [ X_C_1 - yoke_spacer - ins - coil_radius,   Z_pos                   ]])

    kp_2 = kp_1.copy()
    kp_2[:, 0] *= -1.0
      
    coil_list.append(snoopy.RacetrackCoil(kp_1, y, coil_radius, current/num_cond))
    coil_list.append(snoopy.RacetrackCoil(kp_2, y, coil_radius, current/num_cond))

    # the horizontal slot size
    slot_size_horz = min([X_C_1 - X_B_1, X_C_2 - X_B_2,
                          X_E_1 - X_D_1, X_E_2 - X_D_2,])   

    # this is the space we have available for the coils
    A_geo = slot_size_horz*slot_size

    # this is the cupper cross section using the target current density
    A_cu = abs(current)/J_tar

    # this is the coil cross section using the target current density (including the filling factor)
    A_coil = abs(current)/J_tar/conductor_material_data["filling_factor"]

    # we compute the current density (for monitoring reasons)
    current_density = abs(current)/min([A_geo, A_coil])/conductor_material_data["filling_factor"]


    # the turn perimeter
    turn_perimeter = coil_list[0].get_length() + coil_list[1].get_length()

    # this is the available coil volume
    M_coil = A_cu*turn_perimeter*conductor_material_data["density(g/m3)"]

    print('The coil mass is {:.2f} kg'.format(M_coil*1e-3))

    # ====================================================
    # The power consumption
    print('The current density is = {} A/mm2'.format(current_density*1e-6))

    Q = abs(current*current)*turn_perimeter*conductor_material_data['resistivity(Ohm.m)']/A_cu

    print('The power consumption = {:.2f} W'.format(Q))

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
    if use_diluted_steel:
      solver = snoopy.RedMVPSolverDiluted(gmsh.model, coil_list, 
                                    [dom_iron, dom_wing], [dom_air],
                                    [reluctance_iron, reluctance_iron],
                                    [dilution_iron, None],
                                    quad_order=quad_order, max_newton_iterations=150)
      # solver = snoopy.RedMVPSolver(gmsh.model, coil_list, 
      #                               [dom_iron, dom_wing], [dom_air],
      #                               [reluctance_iron, reluctance_wing],
      #                               quad_order=quad_order, max_newton_iterations=150)
      
    else:
      solver = snoopy.RedMVPSolver(gmsh.model, coil_list, 
                                    [dom_iron, dom_wing], [dom_air],
                                    [reluctance_iron, reluctance_iron],
                                    quad_order=quad_order, max_newton_iterations=150)
   
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
        snoopy.plot_domain(pl, gmsh.model.mesh, dom_wing, plot_volume=False)
        snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, plot_volume=False, opacity=0.0)
        snoopy.plot_vector_field(pl, points, B_i, title='B iron in T', mag=0.1, sym_yz=1, sym_xz=2, opacity='linear')
        pl.show_grid()
        pl.camera_position = 'zy'
        pl.camera.elevation = 35
        pl.camera.azimuth = 45
        pl.add_axes()

        pl.subplot(0, 1)
        snoopy.plot_domain(pl, gmsh.model.mesh, dom_iron, plot_volume=False)
        snoopy.plot_domain(pl, gmsh.model.mesh, dom_wing, plot_volume=False)
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


    ret_vals = [points, B_i + B_coil, M_iron, M_coil, Q, current_density]

    if eval_pos.shape[0] > 0:
      
        # ====================================================
        # Evaluate at the given positions
        B_tot = evaluate_fem_solution_node_interpolation(gmsh.model,
                                                             eval_pos,
                                                             solver,
                                                             coil_list,
                                                             x,
                                                             [dom_iron, dom_wing, dom_air])

        if not result_directory == 'none':
            output_filename = os.path.join(result_directory, 'B' + result_spec + '.csv')

        else:
            print('Warning, no result directory specified. I store the evaluations in snoopy directory.')
            output_filename = 'B' + result_spec + '.csv'

        out_df = pd.DataFrame(data=np.append(eval_pos, B_tot, axis=1),
                              columns=['x(m)', 'y(m)', 'z(m)', 'Bx(T)', 'By(T)', 'Bz(T)']).to_csv(output_filename, index=False)


    # print some feedback
    print('**********************************************')
    print('The iron mass is = {:.2f} t'.format(M_iron*1e-6))
    print('The coil mass is {:.2f} kg'.format(M_coil*1e-3))
    print('The coil surface is = {:.2f} m2'.format(A_coil))
    print('The coil perimeter is = {:.2f} m'.format(turn_perimeter))
    print('The current density is = {} A/mm2'.format(current_density*1e-6))
    print('The power consumption = {:.2f} W'.format(Q))
    print('**********************************************')

    return ret_vals


def get_vector_field_ncsc(parameters, df_index=0, lc=0.4,
                  geo_th=1e-5, run_gmsh=False, plot_geo=False,
                  plot_result=False, result_directory='none', result_spec='',
                  eval_pos=np.zeros((0, 3)),
                  materials_directory='files/materials',
                  quad_order=8):
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

    :param quad_order:
        The quadrature order. Default 8.

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
    gmsh.option.setString("Geometry.OCCTargetUnit", "M") 
   
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
   
    yoke_spacer_1 = parameters["yoke_spacer(mm)"][df_index]*1e-3
    ins_1 = parameters["insulation(mm)"][df_index]*1e-3
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
   
    yoke_spacer_2 = parameters["yoke_spacer(mm)"][df_index + 1]*1e-3
    ins_2 = parameters["insulation(mm)"][df_index + 1]*1e-3
    current_2 = parameters["NI(A)"][df_index + 1]
    coil_radius_2 = 0.5*parameters["coil_diam(mm)"][df_index + 1]*1e-3

    delta_x_2 = parameters["delta_x(m)"][df_index + 1]
    delta_y_2 = parameters["delta_y(m)"][df_index + 1]
    delta_z_2 = parameters["delta_z(m)"][df_index + 1]

    field_density_2 = 0.5*parameters["field_density"][df_index + 1]

    # the maximum number of turns
    max_turns_1 = np.int64(parameters["max_turns"][df_index])
    max_turns_2 = np.int64(parameters["max_turns"][df_index + 1])

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
    vol_air = vol_air = snoopy.add_SHIP_box(gmsh.model, 0.0, 0.0, z_min, lim_x, lim_y, z_max - z_min, lc=5*lc)
    # gmsh.model.occ.addBox(0.0, 0.0, z_min, lim_x, lim_y, z_max - z_min)
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

    # do not allow more conductors than a certain amount
    if num_cond_1 > max_turns_1:
        num_cond_1 = max_turns_1

    num_cond_2 = np.int32(slot_size_2/2/(coil_radius_2 + ins_2))

    # do not allow more conductors than a certain amount
    if num_cond_2 > max_turns_2:
        num_cond_2 = max_turns_2

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
                                quad_order=quad_order, max_newton_iterations=50)
   
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
        snoopy.plot_vector_field(pl, points, B_i, title='B iron in T', mag=0.1, sym_yz=1, sym_xz=2)
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
        snoopy.plot_vector_field(pl, points, B_coil, title='B coil in T', mag=0.1, sym_yz=1, sym_xz=2)
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
      
        # print([dom_nc, dom_sc_core, dom_sc_yoke, dom_air])
        # ====================================================
        # Evaluate at the given positions
        B_tot = evaluate_fem_solution_node_interpolation(gmsh.model,
                                                             eval_pos,
                                                             solver,
                                                             coil_list,
                                                             x,
                                                             [dom_nc, dom_sc_core, dom_air, dom_sc_yoke])
        # [dom_nc, dom_sc_core, dom_sc_yoke, dom_air]
        
        # B_iron = snoopy.evaluate_curl_curl_solution(eval_pos, gmsh.model, [dom_nc, dom_sc_core, dom_sc_yoke, dom_air], x, solver.get_global_ids())

        # B_coil = 0.0*B_iron
        # for coil in coil_list:
        #     B_coil += coil.compute_B(eval_pos)
      
        # B_tot = B_coil + B_iron 

        if not result_directory == 'none':
            output_filename = os.path.join(result_directory, 'B' + result_spec + '.csv')

        else:
            print('Warning, no result directory specified. I store the evaluations in snoopy directory.')
            output_filename = 'B' + result_spec + '.csv'

        out_df = pd.DataFrame(data=np.append(eval_pos, B_tot, axis=1),
                              columns=['x(m)', 'y(m)', 'z(m)', 'Bx(T)', 'By(T)', 'Bz(T)']).to_csv(output_filename, index=False)

    return ret_vals

def plot_geometry_mag_1(pl, parameters, df_index=0, lc=1.0, opacity=0.0, show_edges=False, plot_feature_edges=True, step_filename=''):
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
   gmsh.option.setString("Geometry.OCCTargetUnit", "M") 
   
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
   
   yoke_spacer = parameters["yoke_spacer(mm)"][df_index]*1e-3

   ins = parameters["insulation(mm)"][df_index]*1e-3

   current = parameters["NI(A)"][df_index]

   coil_radius = 0.5*parameters["coil_diam(mm)"][df_index]*1e-3

   field_density = 0.5*parameters["field_density"][df_index]

   max_turns = np.int64(parameters["max_turns"][df_index])

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

   if not len(step_filename) == 0:
      gmsh.write(step_filename)

   # ====================================================
   # Make the coil objects

   # this list stores the coil objects
   coil_list = []

   # determine the slot size
   slot_size = 2*min(Y_core_1, Y_core_2)

   # determine the number of conductors
   num_cond = np.int32(slot_size/2/(coil_radius+ins))

   # do not allow more conductors than a certain amount
   if num_cond > max_turns:
      num_cond = max_turns

   # these are the vertical positions
   y = np.linspace(-0.5*slot_size + coil_radius + ins,
                    0.5*slot_size - coil_radius - ins, num_cond)

   if X_mgap_1 == 0.0 or X_mgap_2 == 0.0:

      # make only a single coil
      kp = np.array([[-X_core_2 - yoke_spacer - ins - coil_radius, Z_pos + Z_len             ],
                     [-X_core_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                     [ X_core_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                     [ X_core_2 + yoke_spacer + ins + coil_radius,   Z_pos + Z_len           ],
                     [ X_core_1 + yoke_spacer + ins + coil_radius,   Z_pos                   ],
                     [ X_core_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                     [-X_core_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                     [-X_core_1 - yoke_spacer - ins - coil_radius,   Z_pos                   ]])

      

      coil_list.append(snoopy.RacetrackCoil(kp, y, coil_radius, current/num_cond))

   else:

      # make two coils

      kp_1 = np.array([[ X_mgap_2 - yoke_spacer - ins - coil_radius, Z_pos + Z_len + coil_radius ],
                     [ X_mgap_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius        ],
                     [ X_core_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius        ],
                     [ X_core_2 + yoke_spacer + ins + coil_radius,   Z_pos + Z_len               ],
                     [ X_core_1 + yoke_spacer + ins + coil_radius,   Z_pos                       ],
                     [ X_core_1,                       Z_pos-yoke_spacer - ins - coil_radius     ],
                     [ X_mgap_1,                       Z_pos-yoke_spacer - ins - coil_radius     ],
                     [ X_mgap_1 - yoke_spacer - ins - coil_radius,   Z_pos                       ]])
      
      kp_2 = kp_1.copy()
      kp_2[:, 0] *= -1.0
      
      coil_list.append(snoopy.RacetrackCoil(kp_1, y, coil_radius, current/num_cond))
      coil_list.append(snoopy.RacetrackCoil(kp_2, y, coil_radius, current/num_cond))

   snoopy.plot_domain(pl, gmsh.model.mesh, dom_iron, reflect_xz=True, reflect_yz=True, show_edges=show_edges, opacity=opacity, plot_feature_edges=plot_feature_edges)
   # snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, reflect_xz=True, reflect_yz=True, show_edges=False, opacity=0.0)
   for coil in coil_list:
      coil.plot_pv(pl)

   return

def plot_geometry_mag_2(pl, parameters, df_index=0, lc=1.0, opacity=0.0, show_edges=False, plot_feature_edges=True, step_filename=''):
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
   gmsh.option.setString("Geometry.OCCTargetUnit", "M") 
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
   
   yoke_spacer = parameters["yoke_spacer(mm)"][df_index]*1e-3

   ins = parameters["insulation(mm)"][df_index]*1e-3

   current = parameters["NI(A)"][df_index]

   coil_radius = 0.5*parameters["coil_diam(mm)"][df_index]*1e-3

   field_density = 0.5*parameters["field_density"][df_index]

   max_turns = np.int64(parameters["max_turns"][df_index])

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

   if not len(step_filename) == 0:
      gmsh.write(step_filename)


   # ====================================================
   # Make a coil object   
   kp = np.array([[-X_core_2 - yoke_spacer - ins - coil_radius, Z_pos + Z_len             ],
                  [-X_core_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                  [ X_core_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                  [ X_core_2 + yoke_spacer + ins + coil_radius,   Z_pos + Z_len           ],
                  [ X_core_1 + yoke_spacer + ins + coil_radius,   Z_pos                   ],
                  [ X_core_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                  [-X_core_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                  [-X_core_1 - yoke_spacer - ins - coil_radius,   Z_pos                   ]])

   # determine the slot size
   slot_size = 2*min(Y_core_1, Y_core_2)

   # determine the number of conductors
   num_cond = np.int32(slot_size/2/(coil_radius+ins))

   # do not allow more conductors than a certain amount
   if num_cond > max_turns:
      num_cond = max_turns

   y = np.linspace(-0.5*slot_size + coil_radius + ins,
                    0.5*slot_size - coil_radius - ins, num_cond)

   coil = snoopy.RacetrackCoil(kp, y, coil_radius, current/num_cond)

   snoopy.plot_domain(pl, gmsh.model.mesh, dom_core, reflect_xz=True, reflect_yz=True, show_edges=show_edges, opacity=opacity, plot_feature_edges=plot_feature_edges)
   snoopy.plot_domain(pl, gmsh.model.mesh, dom_yoke, reflect_xz=True, reflect_yz=True, show_edges=show_edges, opacity=opacity, plot_feature_edges=plot_feature_edges)
   # snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, reflect_xz=True, reflect_yz=True, show_edges=False, opacity=0.0)
   coil.plot_pv(pl)

   return None

def plot_geometry_mag_3(pl, parameters, df_index=0, lc=1.0, opacity=0.0, show_edges=False, plot_feature_edges=True, step_filename=''):
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
   gmsh.option.setString("Geometry.OCCTargetUnit", "M") 

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
   
   yoke_spacer = parameters["yoke_spacer(mm)"][df_index]*1e-3

   ins = parameters["insulation(mm)"][df_index]*1e-3

   current = parameters["NI(A)"][df_index]

   coil_radius = 0.5*parameters["coil_diam(mm)"][df_index]*1e-3

   field_density = 0.5*parameters["field_density"][df_index]

   max_turns = np.int64(parameters["max_turns"][df_index])

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

   if not len(step_filename) == 0:
      gmsh.write(step_filename)

   # ====================================================
   # Make the coil objects

   # this list stores the coil objects
   coil_list = []

   # determine the slot size
   slot_size = 2*min(Y_core_1, Y_core_2)

   # determine the number of conductors
   num_cond = np.int32(slot_size/2/(coil_radius+ins))

   # do not allow more conductors than a certain amount
   if num_cond > max_turns:
      num_cond = max_turns

   # these are the vertical positions
   y = np.linspace(-0.5*slot_size + coil_radius + ins,
                    0.5*slot_size - coil_radius - ins, num_cond)

   # make two coils
   kp_1 = np.array([[ X_void_2 - yoke_spacer - ins - coil_radius, Z_pos + Z_len           ],
                  [ X_void_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                  [ X_yoke_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                  [ X_yoke_2 + yoke_spacer + ins + coil_radius,   Z_pos + Z_len           ],
                  [ X_yoke_1 + yoke_spacer + ins + coil_radius,   Z_pos                   ],
                  [ X_yoke_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                  [ X_void_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                  [ X_void_1 - yoke_spacer - ins - coil_radius,   Z_pos                   ]])

   kp_2 = kp_1.copy()
   kp_2[:, 0] *= -1.0
      
   coil_list.append(snoopy.RacetrackCoil(kp_1, y, coil_radius, current/num_cond))
   coil_list.append(snoopy.RacetrackCoil(kp_2, y, coil_radius, current/num_cond))


   snoopy.plot_domain(pl, gmsh.model.mesh, dom_iron, reflect_xz=True, reflect_yz=True, show_edges=show_edges, opacity=opacity, plot_feature_edges=plot_feature_edges)
   # snoopy.plot_domain(pl, gmsh.model.mesh, dom_air, reflect_xz=True, reflect_yz=True, show_edges=False, opacity=0.0)
   for coil in coil_list:
      coil.plot_pv(pl)

def plot_geometry_mag_4(pl, parameters, df_index=0, lc=1.0, opacity=0.0, show_edges=False, plot_feature_edges=True, step_filename=''):
    '''Plot the geometry for the magnet 4 template.
     
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
    gmsh.option.setString("Geometry.OCCTargetUnit", "M") 

    # we get the geometry parameters as variables for convinience
    X_A_1 = parameters["Xmgap1(m)"][df_index]
    X_A_2 = parameters["Xmgap2(m)"][df_index]

    X_B_1 = parameters["Xcore1(m)"][df_index]
    X_B_2 = parameters["Xcore2(m)"][df_index]

    X_C_1 = parameters["Xvoid1(m)"][df_index]
    X_C_2 = parameters["Xvoid2(m)"][df_index]

    X_D_1 = parameters["Xcore1(m)"][df_index+1]
    X_D_2 = parameters["Xcore2(m)"][df_index+1]

    X_E_1 = parameters["Xvoid1(m)"][df_index+1]
    X_E_2 = parameters["Xvoid2(m)"][df_index+1]

    X_F_1 = parameters["Xyoke1(m)"][df_index+1]
    X_F_2 = parameters["Xyoke2(m)"][df_index+1]

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
   
    yoke_spacer = parameters["yoke_spacer(mm)"][df_index]*1e-3

    ins = parameters["insulation(mm)"][df_index]*1e-3

    current = parameters["NI(A)"][df_index]

    coil_radius = 0.5*parameters["coil_diam(mm)"][df_index]*1e-3

    field_density = 0.5*parameters["field_density"][df_index]

    max_turns = np.int64(parameters["max_turns"][df_index])

    # the limits in x, y, and z
    lim_x = max([X_F_1, X_F_2]) + delta_x
    lim_y = max([Y_yoke_1, Y_yoke_2]) + delta_y
    z_min = Z_pos - delta_z
    z_max = Z_pos + Z_len + delta_z
    
    # the iron domain
    vol_core, vol_yoke = snoopy.add_SHIP_iron_yoke_mag_4(gmsh.model, 
                                                         X_A_1, X_B_1, X_C_1, X_D_1, X_E_1, X_F_1,
                                                         X_A_2, X_B_2, X_C_2, X_D_2, X_E_2, X_F_2,
                                                         Y_core_1, Y_yoke_1,
                                                         Y_core_2, Y_yoke_2,
                                                         Z_len, 
                                                         Z_pos=Z_pos,
                                                         lc=lc,
                                                         lc_inner=0.3*lc)
    # the iron domain
    vol_air = gmsh.model.occ.addBox(0.0, 0.0, z_min, lim_x, lim_y, z_max - z_min)
    gmsh.model.occ.synchronize()


    # fragment perfoms something like a union
    fragments, _ = gmsh.model.occ.fragment([(3, vol_air)], [(3, vol_core),(3, vol_yoke)])
    gmsh.model.occ.synchronize()

    # we get the domains of the fragmentation
    dom_iron = fragments[0][1]
    dom_wing = fragments[1][1]
    dom_air = fragments[2][1]

    # and we define physical domains
    gmsh.model.addPhysicalGroup(3, [dom_iron, dom_wing], 1, name = "Iron")
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [dom_air], 2, name = "Air")
    gmsh.model.occ.synchronize()

    
    # we then generate the mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.occ.synchronize()

    if not len(step_filename) == 0:
       gmsh.write(step_filename)

    # ====================================================
    # Make the coil objects

    # this list stores the coil objects
    coil_list = []

    # determine the slot size
    slot_size = 2*min(Y_core_1, Y_core_2)

    # determine the number of conductors
    num_cond = np.int32(slot_size/2/(coil_radius+ins))

    # do not allow more conductors than a certain amount
    if num_cond > max_turns:
        num_cond = max_turns

    # these are the vertical positions
    y = np.linspace(-0.5*slot_size + coil_radius + ins,
                     0.5*slot_size - coil_radius - ins, num_cond)

    # make two coils
    kp_1 = np.array([[ X_C_2 - yoke_spacer - ins - coil_radius, Z_pos + Z_len           ],
                   [ X_C_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                   [ X_D_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                   [ X_D_2 + yoke_spacer + ins + coil_radius,   Z_pos + Z_len           ],
                   [ X_D_1 + yoke_spacer + ins + coil_radius,   Z_pos                   ],
                   [ X_D_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                   [ X_C_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                   [ X_C_1 - yoke_spacer - ins - coil_radius,   Z_pos                   ]])

    kp_2 = kp_1.copy()
    kp_2[:, 0] *= -1.0
      
    coil_list.append(snoopy.RacetrackCoil(kp_1, y, coil_radius, current/num_cond))
    coil_list.append(snoopy.RacetrackCoil(kp_2, y, coil_radius, current/num_cond))

    snoopy.plot_domain(pl, gmsh.model.mesh, dom_iron, reflect_xz=True, reflect_yz=True, show_edges=show_edges, opacity=opacity, plot_feature_edges=plot_feature_edges)
    snoopy.plot_domain(pl, gmsh.model.mesh, dom_wing, reflect_xz=True, reflect_yz=True, show_edges=show_edges, opacity=opacity, plot_feature_edges=plot_feature_edges)

    for coil in coil_list:
       coil.plot_pv(pl)