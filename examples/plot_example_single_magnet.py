"""Example single magnet designer.
   ==========

   This script allows You to model a single superconducting magnet.
"""

import os
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
import json
import gmsh

import snoopy


# this is the current woring directory
cwd = os.getcwd()

# %%
# Script parameters
# =================

# get the files directoy. We need the following statements to make the auto gallery running
if cwd.split('\\')[-1] == 'examples':
    directory = os.path.join('..', 'files', 'parameters')
    materials_directory = os.path.join('..', 'files', 'materials')
    result_directory = os.path.join('..', 'files', 'results', 'single_magnet_option')
else:
    directory = os.path.join('files', 'parameters')
    materials_directory = os.path.join('files', 'materials')
    result_directory = os.path.join('files', 'results', 'single_magnet_option')

# read magnet parameters
param_file = open(os.path.join(directory, "single_magnet.json"))

params = json.load(param_file)

# read the core geometry
params_core = pd.read_csv(os.path.join(directory, params["core_geometry_filename"]))
w_core_1 = params_core["width_entrance(m)"][0]
w_core_2 = params_core["width_exit(m)"][0]
h_core_1 = params_core["height_entrance(m)"][0]
h_core_2 = params_core["height_exit(m)"][0]

# read the keypoints
kp_yoke = pd.read_csv(os.path.join(directory, params["yoke_geometry_filename"]))

# read the coil geometry
coil_geom = pd.read_csv(os.path.join(directory, params["coil_geometry_filename"]))

# the coil winding radius
rad_coil = params["winding_radius"]

# get the mesh size parameters
lc_core = params["mesh_size_core"]
lc_yoke = params["mesh_size_yoke"]
lc_air = params["bounding_box_mesh_size_parameter"]

# the magnet length
L = params["magnet_length(m)"]

# the extension of the bounding box
dx = params["bounding_box_extension"][0]
dy = params["bounding_box_extension"][1]
dz = params["bounding_box_extension"][2]

# get the maximum dimensions in the cross sections
max_x = max(kp_yoke.values[:, [0, 2]].flatten())
max_y = max(kp_yoke.values[:, [1, 3]].flatten())

# read the materials
reluctances = []
reluctances.append(snoopy.Reluctance(os.path.join(materials_directory,
                                                     params["core_material_filename"])))
reluctances.append(snoopy.Reluctance(os.path.join(materials_directory,
                                                     params["yoke_material_filename"])))

# A geometry threshold to identify the dirichlet boundary.
GEO_TH = 1e-6

# the density to create the point cloud
FIELD_DENS = 5

# %%
# Plot the cross sections
# =======================

fig = plt.figure()
ax = fig.add_subplot(121)
ax.fill([0.0, w_core_1, w_core_1, 0.0],
         [0.0, 0.0, h_core_1, h_core_1], color='gray')
ax.plot([0.0, w_core_1, w_core_1, 0.0, 0.0],
        [0.0, 0.0, h_core_1, h_core_1, 0.0], color='k', linewidth=0.5)
ax.fill(kp_yoke.values[:, 0], kp_yoke.values[:, 1], color='gray')
ax.plot(np.append(kp_yoke.values[:, 0], kp_yoke.values[0, 0]), 
        np.append(kp_yoke.values[:, 1], kp_yoke.values[0, 1]), color='k', linewidth=0.5)
ax.set_xlabel('$x$ in m')
ax.set_ylabel('$y$ in m')
ax.set_xlim((0.0, max_x + dx))
ax.set_ylim((0.0, max_y + dy))
ax.set_title('entrance')
ax.set_aspect('equal')
ax = fig.add_subplot(122)
ax.fill([0.0, w_core_2, w_core_2, 0.0],
         [0.0, 0.0, h_core_2, h_core_2], color='gray')
ax.plot([0.0, w_core_2, w_core_2, 0.0, 0.0],
        [0.0, 0.0, h_core_2, h_core_2, 0.0], color='k', linewidth=0.5)
ax.fill(kp_yoke.values[:, 2], kp_yoke.values[:, 3], color='gray')
ax.plot(np.append(kp_yoke.values[:, 2], kp_yoke.values[0, 2]), 
            np.append(kp_yoke.values[:, 3], kp_yoke.values[0, 3]), color='k', linewidth=0.5)
ax.set_xlim((0.0, max_x + dx))
ax.set_ylim((0.0, max_y + dy))
ax.set_title('exit')
ax.set_xlabel('$x$ in m')
ax.set_ylabel('$y$ in m')
ax.set_aspect('equal')
plt.tight_layout()
plt.show()

# %%
# Make the mesh
# =============
gmsh.initialize()
gmsh.model.add("make mesh")

# the labels for the iron volumes
vol_iron = []

# the core geometry
kp_core_1 = [gmsh.model.occ.addPoint(0.0, 0.0, 0.0, lc_core),
             gmsh.model.occ.addPoint(0.0, 0.0, L, lc_core),
             gmsh.model.occ.addPoint(w_core_2 - rad_coil, 0.0, L, lc_core),
             gmsh.model.occ.addPoint(w_core_2, 0.0, L - rad_coil, lc_core),
             gmsh.model.occ.addPoint(w_core_1, 0.0, rad_coil, lc_core),
             gmsh.model.occ.addPoint(w_core_1 - rad_coil, 0.0, 0.0, lc_core)]

kp_core_2 = [gmsh.model.occ.addPoint(0.0, h_core_1, 0.0, lc_core),
             gmsh.model.occ.addPoint(0.0, h_core_2, L, lc_core),
             gmsh.model.occ.addPoint(w_core_2 - rad_coil, h_core_2, L, lc_core),
             gmsh.model.occ.addPoint(w_core_2, h_core_2, L - rad_coil, lc_core),
             gmsh.model.occ.addPoint(w_core_1, h_core_1, rad_coil, lc_core),
             gmsh.model.occ.addPoint(w_core_1 - rad_coil, h_core_1, 0.0, lc_core)]


vol_iron = [snoopy.add_SHiP_volume(gmsh.model, kp_core_1, kp_core_2)]

# the yoke geometry
kp_1 = [gmsh.model.occ.addPoint(kp_yoke.values[i, 0],
                                       kp_yoke.values[i, 1],
                                       0.0, lc_yoke) for i in range(kp_yoke.shape[0])]

kp_2 = [gmsh.model.occ.addPoint(kp_yoke.values[i, 2],
                                       kp_yoke.values[i, 3],
                                       L, lc_yoke) for i in range(kp_yoke.shape[0])]
    
vol_iron.append(snoopy.add_SHiP_volume(gmsh.model, kp_1, kp_2))

# the air volumes
vol_air = snoopy.add_SHIP_box(gmsh.model, 0.0, 0.0, -dz, max_x + dx, max_y + dy, L + 2*dz, lc=lc_air)

# iron dimtags
iron_dimtags = [(3, t) for t in vol_iron]

# fragment perfoms something like a union
fragments, _ = gmsh.model.occ.fragment([(3, vol_air)], iron_dimtags)

gmsh.model.occ.synchronize()

# the iron domain indices
dom_i = [fragments[i][1] for i in range(len(vol_iron))]
# the air domain index
dom_a = fragments[-1][1]

# define physical domains
gmsh.model.addPhysicalGroup(3, dom_i, 1, name = "Iron")
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(3, [dom_a], 2, name = "Air")
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

    if (abs(coord[0] - 0.0) < GEO_TH and abs(abs(normal[0]) - 1.0) < GEO_TH):
        dirichlet_boundaries.append(be)

    elif (abs(coord[0] - max_x - dx) < GEO_TH and abs(abs(normal[0]) - 1.0) < GEO_TH):
        dirichlet_boundaries.append(be)

    elif (abs(coord[1] - max_y - dy) < GEO_TH and abs(abs(normal[1]) - 1.0) < GEO_TH):
        dirichlet_boundaries.append(be)

    elif (abs(coord[2] + dz) < GEO_TH and abs(abs(normal[2]) - 1.0) < GEO_TH):
        dirichlet_boundaries.append(be)

    elif (abs(coord[2] - L - dz) < GEO_TH and abs(abs(normal[2]) - 1.0) < GEO_TH):
        dirichlet_boundaries.append(be)

# add a physical group for this boundary condition
gmsh.model.addPhysicalGroup(2, dirichlet_boundaries, 1, name = "Dirichlet Boundary")

gmsh.model.occ.synchronize()

# %%
# Make the coil
# =============

# the number of coils
num_coils = coil_geom.values.shape[0]

# the list of coils
coil_list = []

for i in range(num_coils):

    # some auxiliary variables
    x1 = coil_geom["X_1(m)"][i]
    x2 = coil_geom["X_2(m)"][i]
    y = coil_geom["Y(m)"][i]
    rad = 0.5*coil_geom["coil_diam(mm)"][i]*1e-3
    current = coil_geom["NI(A)"][i]

    # make the keypoints for this turn
    kp = np.array([[ -x2 - rad , L - rad_coil ],
                   [ -x2 - rad + rad_coil , L + rad ],
                   [ x2 + rad- rad_coil , L + rad ],
                   [ x2 + rad, L - rad_coil],
                   [ x1 + rad, rad_coil ],
                   [ x1 + rad - rad_coil, -rad ],
                   [-x1 - rad + rad_coil, -rad ],
                   [-x1 - rad , rad_coil ]])

    coil_list.append(snoopy.RacetrackCoil(kp, [y], rad, current))

# %%
# Plot the whole thing
# ====================

pl = pv.Plotter(shape=(1, 1), off_screen=False)
for di in dom_i:
    snoopy.plot_domain(pl, gmsh.model.mesh, di, reflect_xz=True, reflect_yz=True, show_edges=True, opacity=1.0)
snoopy.plot_domain(pl, gmsh.model.mesh, dom_a, reflect_xz=True, reflect_yz=True, show_edges=False, opacity=0.0)
for coil in coil_list:
    coil.plot_pv(pl)
pl.show()

# %%
# Solve the problem
# =================
solver = snoopy.RedMVPSolver(gmsh.model, coil_list, dom_i, [dom_a], reluctances)

x = solver.solve()

# %%
# Compute the point cloud
# =======================
points, B_i = solver.curl_curl_factory.compute_B(x, quad_order=FIELD_DENS)

B_coil = 0.0*B_i
for coil in coil_list:
    B_coil += coil.compute_B(points)

B_tot = B_i + B_coil

pl = pv.Plotter(shape=(1, 2))

pl.subplot(0, 0)
snoopy.plot_domain(pl, gmsh.model.mesh, dom_i[0], plot_volume=False)
snoopy.plot_domain(pl, gmsh.model.mesh, dom_i[1], plot_volume=False)
snoopy.plot_domain(pl, gmsh.model.mesh, dom_a, plot_volume=False, opacity=0.0)
snoopy.plot_vector_field(pl, points, B_i, title='B iron in T', mag=0.1, sym_yz=1, sym_xz=2)
pl.show_grid()
pl.camera_position = 'zy'
pl.camera.elevation = 35
pl.camera.azimuth = 45
pl.add_axes()

pl.subplot(0, 1)
snoopy.plot_domain(pl, gmsh.model.mesh, dom_i[0], plot_volume=False)
snoopy.plot_domain(pl, gmsh.model.mesh, dom_i[1], plot_volume=False)
snoopy.plot_domain(pl, gmsh.model.mesh, dom_a, plot_volume=False, opacity=0.0)
snoopy.plot_vector_field(pl, points, B_i + B_coil, title='B in T', mag=0.1, sym_yz=1, sym_xz=2)
pl.show_grid()
pl.camera_position = 'zy'
pl.camera.elevation = 35
pl.camera.azimuth = 45
pl.add_axes()
pl.show()

# %%
# Compute the field along a line in x
# ===================================
z_cut = 0.75*L
y_line = 0.0

RESOL_X = 200

eval_pos = np.zeros((RESOL_X, 3))
eval_pos[:, 0] = np.linspace(0.0, max_x+dx, RESOL_X)
eval_pos[:, 1] = y_line
eval_pos[:, 2] = z_cut

B_x_l = snoopy.evaluate_fem_solution_node_interpolation(gmsh.model, eval_pos, solver,
                                                        coil_list, x, dom_i + [dom_a])

B_x_coil = 0.0*B_x_l
for coil in coil_list:
    B_x_coil += coil.compute_B(eval_pos)

B_x_i = B_x_l - B_x_coil

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(eval_pos[:, 0], B_x_l[:, 1], label='total field')
ax.plot(eval_pos[:, 0], B_x_coil[:, 1], label='coil field')
ax.plot(eval_pos[:, 0], B_x_i[:, 1], label='iron field')
ax.set_xlabel("$x$ in m")
ax.legend()
ax.set_title(f"$B_y(x)$ over $x$ at y = {y_line:.2f}, z = {z_cut:.2f} in T")
plt.show()

# %%
# Compute the field in the xy plane
# =================================
z_cut = 0.75*L

RESOL_X = 70
RESOL_Y = 70

X, Y = np.meshgrid(np.linspace(0.0, max_x+dx, RESOL_X),
                   np.linspace(0.0, max_y+dy, RESOL_Y))

eval_pos = np.zeros((RESOL_X*RESOL_Y, 3))
eval_pos[:, 0] = X.flatten()
eval_pos[:, 1] = Y.flatten()
eval_pos[:, 2] = z_cut

B_xy = snoopy.evaluate_fem_solution_node_interpolation(gmsh.model, eval_pos, solver,
                                                        coil_list, x, dom_i + [dom_a])

Bx = B_xy[:, 0]
By = B_xy[:, 1]
Bx.shape = X.shape
By.shape = X.shape
B_norm = np.sqrt(Bx*Bx + By*By)

B_xy_coil = 0.0*B_xy
for coil in coil_list:
    B_xy_coil += coil.compute_B(eval_pos)

Bx_c = B_xy_coil[:, 0]
By_c = B_xy_coil[:, 1]
Bx_c.shape = X.shape
By_c.shape = X.shape
B_c_norm = np.sqrt(Bx_c*Bx_c + By_c*By_c)


fig = plt.figure()
ax = fig.add_subplot(121)
cntrf=ax.contourf(X, Y, B_norm, cmap="rainbow", levels=20)
ax.streamplot(X, Y, Bx, By, density=0.5, color='k')
ax.set_ylabel("$x$ in m")
ax.set_xlabel("$y$ in m")
cbar=plt.colorbar(cntrf)
cbar.set_label("$|B|$ in T")
ax.set_title(f"Total field in the $z = ${z_cut:.2f} m plane")

ax = fig.add_subplot(122)
cntrf=ax.contourf(X, Y, B_c_norm, cmap="rainbow", levels=20)
ax.streamplot(X, Y, Bx_c, By_c, density=0.5, color='k')
ax.set_ylabel("$x$ in m")
ax.set_xlabel("$y$ in m")
cbar=plt.colorbar(cntrf)
cbar.set_label("$|B|$ in T")
ax.set_title(f"Coil field in the $z = ${z_cut:.2f} m plane")
plt.show()



# %%
# Compute the field in the xz plane
# =================================

RESOL_X = 30
RESOL_Z = 200

X, Z = np.meshgrid(np.linspace(0.0, max_x+dx, RESOL_X),
                   np.linspace(-dz, L+dz, RESOL_Z))

eval_pos = np.zeros((RESOL_X*RESOL_Z, 3))
eval_pos[:, 0] = X.flatten()
eval_pos[:, 2] = Z.flatten()

B_xz = snoopy.evaluate_fem_solution_node_interpolation(gmsh.model, eval_pos, solver,
                                                coil_list, x, dom_i + [dom_a])

By = B_xz[:, 1]
By.shape = X.shape

fig = plt.figure()
ax = fig.add_subplot(111)
cntrf=ax.contourf(Z.T, X.T, By.T, cmap="rainbow", levels=100)
ax.set_ylabel("$x$ in m")
ax.set_xlabel("$z$ in m")
cbar=plt.colorbar(cntrf)
cbar.set_label("$|B|$ in T")
plt.show()


