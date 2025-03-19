"""Example: Run the standard analysis.
   ==========

   We read a parameters file and generate the following (magnet center at 000):

   - field profile in z (x=y=0)
   - field profile in x (z=y=0)
   - field map in xy plane (z=0)
   - field map in xz plane (y=0)
   - field map in (yz) plane (x=0)

"""

import os
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy import interpolate
from scipy import optimize
import copy

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
    result_directory = os.path.join('..', 'files', 'results')
else:
    directory = os.path.join('files', 'parameters')
    materials_directory = os.path.join('files', 'materials')
    result_directory = os.path.join('files', 'results')

# enable this flag if You like to plot the configuration
PLOT_CONFIG = True

# the parameters file
params_df = pd.read_csv(os.path.join(directory, 'baseline_1.csv'))

# the magnets in the parameters file to analyse
magnet_numbers = [0]

# the number of magnets
num_mag = len(magnet_numbers)

# the parameters used for the cost estimation
cost_parameters = np.zeros((num_mag, 7))

# number of evaluation positions in z
num_eval_z = 200

# number of evaluation positions in x
num_eval_x = 100

# the resolutions for the field maps
resol_x = 40
resol_y = 40
resol_z = 100

# the total number of field evaluations
num_eval = num_eval_z + num_eval_x + resol_x*resol_y + resol_x*resol_z + resol_y*resol_z

# the sizes of the arrays
array_sizes = [num_eval_z, num_eval_x, resol_x*resol_y, resol_x*resol_z, resol_y*resol_z]

# the indices in the result array
indx_from_to = [0]
for a_size in array_sizes:
    indx_from_to.append(indx_from_to[-1]+a_size)

if PLOT_CONFIG:
    # In case plot config is enabled, we open a pyvista plotter
    pl = pv.Plotter()

# %%
# Loop over all magnets
# =====================

# We loop over all magnets
for j, i in enumerate(magnet_numbers):

    # the domain size
    z_min = params_df['Z_pos(m)'][i] - params_df['delta_z(m)'][i]
    z_max = params_df['Z_pos(m)'][i] + params_df['Z_len(m)'][i] + params_df['delta_z(m)'][i]
    x_max = max([params_df['Xyoke2(m)'][i], params_df['Xyoke1(m)'][i]]) + params_df['delta_x(m)'][i]
    y_max = max([params_df['Yyoke2(m)'][i], params_df['Yyoke1(m)'][i]]) + params_df['delta_y(m)'][i]


    # get the z center position
    x_ctr = 0.25*(params_df['Xcore1(m)'][i] + params_df['Xcore2(m)'][i] 
                    + params_df['Xmgap1(m)'][i] + params_df['Xmgap2(m)'][i])
    z_ctr = params_df['Z_pos(m)'][i] + 0.5*params_df['Z_len(m)'][i]

    # the central x position of the return yoke in entrance and exit
    x_ret_ctr_2 = 0.5*(params_df['Xyoke2(m)'][i] + params_df['Xvoid2(m)'][i])
    x_ret_ctr_1 = 0.5*(params_df['Xyoke1(m)'][i] + params_df['Xvoid1(m)'][i])

    # the slope in the return yoke
    m_ret = (x_ret_ctr_2 - x_ret_ctr_1)/params_df['Z_len(m)'][i]

    # the intercept of the line representing the center of the return yoke
    b_ret = x_ret_ctr_2 - m_ret*params_df['Z_len(m)'][i]

    # get the horizontal points at the beginning and end of the line crossing the
    # return yoke
    x_ret_line_1 = -m_ret*params_df['delta_z(m)'][i] + b_ret
    x_ret_line_2 = m_ret*(params_df['delta_z(m)'][i] + params_df['Z_len(m)'][i]) + b_ret

    # make the lines for the evaluation
    z_line = np.linspace(z_min, z_max, num_eval_z)
    x_line = np.linspace(0.0, x_max, num_eval_x)
    x_line_ret = np.linspace(x_ret_line_1, x_ret_line_2, num_eval_z)

    # make the lines for the grid evaluation
    x_grid = np.linspace(0.0, x_max, resol_x)
    y_grid = np.linspace(0.0, y_max, resol_y)
    z_grid = np.linspace(z_min, z_max, resol_z)

    # make the mesh grids
    X_xy, Y_xy = np.meshgrid(x_grid, y_grid)
    X_xz, Z_xz = np.meshgrid(x_grid, z_grid)
    Y_yz, Z_yz = np.meshgrid(y_grid, z_grid)

    # make the evaluation positions
    eval_pos = np.zeros((num_eval, 3))

    if params_df['yoke_type'][i] == 'Mag3':
        # the line  through the return yoke
        eval_pos[indx_from_to[0]:indx_from_to[1], 0] = x_line_ret
        eval_pos[indx_from_to[0]:indx_from_to[1], 2] = z_line
    else:
        # the line x=y=0
        eval_pos[indx_from_to[0]:indx_from_to[1], 0] = 0.0
        eval_pos[indx_from_to[0]:indx_from_to[1], 2] = z_line

    # the line y=0, z=z_ctr
    eval_pos[indx_from_to[1]:indx_from_to[2], 0] = x_line
    eval_pos[indx_from_to[1]:indx_from_to[2], 2] = z_ctr
    
    # the plane z = z_ctr
    eval_pos[indx_from_to[2]:indx_from_to[3], 0] = X_xy.flatten()
    eval_pos[indx_from_to[2]:indx_from_to[3], 1] = Y_xy.flatten()
    eval_pos[indx_from_to[2]:indx_from_to[3], 2] = z_ctr
    
    # the plane y = 0
    eval_pos[indx_from_to[3]:indx_from_to[4], 0] = X_xz.flatten()
    eval_pos[indx_from_to[3]:indx_from_to[4], 2] = Z_xz.flatten()

    if params_df['yoke_type'][i] == 'Mag3':
        # the plane through the return yoke
        eval_pos[indx_from_to[4]:indx_from_to[5], 0] = m_ret*(Z_yz.flatten() - params_df['Z_pos(m)'][i]) + b_ret
        eval_pos[indx_from_to[4]:indx_from_to[5], 1] = Y_yz.flatten()
        eval_pos[indx_from_to[4]:indx_from_to[5], 2] = Z_yz.flatten()
    else:
        # the plane x = x_ctr
        eval_pos[indx_from_to[4]:indx_from_to[5], 0] = 0.0
        eval_pos[indx_from_to[4]:indx_from_to[5], 1] = Y_yz.flatten()
        eval_pos[indx_from_to[4]:indx_from_to[5], 2] = Z_yz.flatten()

    if params_df['yoke_type'][i] == 'Mag1':

        points, B, M_i, M_c, Q, J = snoopy.get_vector_field_mag_1(params_df, i,
                                    materials_directory=materials_directory,
                                    eval_pos=eval_pos, result_directory=result_directory,
                                    result_spec=f"_mag{i}")

        if PLOT_CONFIG:
            snoopy.plot_geometry_mag_1(pl, params_df, i)
            snoopy.plot_vector_field(pl, points, B, title='B in T',
                                    mag=0.05, sym_yz=1, sym_xz=2)
            
    if params_df['yoke_type'][i] == 'Mag3':

        points, B, M_i, M_c, Q, J = snoopy.get_vector_field_mag_3(params_df, i,
                                    materials_directory=materials_directory,
                                    eval_pos=eval_pos, result_directory=result_directory,
                                    result_spec=f"_mag{i}")
        
        if PLOT_CONFIG:
            snoopy.plot_geometry_mag_3(pl, params_df, i)
            snoopy.plot_vector_field(pl, points, B, title='B in T',
                                    mag=0.05, sym_yz=1, sym_xz=2)
            
    # compute the prices
    C_i, C_c, C_edf = snoopy.compute_prices(params_df, i, M_i, M_c, Q)
    cost_parameters[j, :] = np.array([M_i, C_i, M_c, C_c, Q, C_edf, J])

    # load the B field result
    B_res = pd.read_csv(os.path.join(result_directory, f"B_mag{i}.csv"))

    # compute the central field integral
    Bdl = np.trapezoid(B_res['By(T)'][:indx_from_to[1]], x=eval_pos[:indx_from_to[1], 2])

    print(f"The field integral is {Bdl:.2f} Tm")

    # plot the field
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(231)
    ax.plot(eval_pos[:indx_from_to[1], 2], B_res['By(T)'][:indx_from_to[1]])
    ax.set_xlabel('$z$ in m')
    ax.set_title("$B_y(y=0)$ along line 1 in T")
    ax = fig.add_subplot(232)
    ax.plot(eval_pos[indx_from_to[1]:indx_from_to[2], 0], B_res['By(T)'][indx_from_to[1]:indx_from_to[2]])
    ax.set_xlabel('$x$ in m')
    ax.set_title('$B_y(x,y=z=0)$ in T')
    ax = fig.add_subplot(233)
    Bx_xy = B_res['Bx(T)'].values[indx_from_to[2]:indx_from_to[3]]
    By_xy = B_res['By(T)'].values[indx_from_to[2]:indx_from_to[3]]
    Bz_xy = B_res['Bz(T)'].values[indx_from_to[2]:indx_from_to[3]]
    Bx_xy.shape = X_xy.shape
    By_xy.shape = X_xy.shape
    Bz_xy.shape = X_xy.shape
    Bmag_xy = np.sqrt(Bx_xy**2 + By_xy**2 + Bz_xy**2)
    cntrf = ax.contourf(X_xy, Y_xy, Bmag_xy, cmap='rainbow', levels=100)
    ax.quiver(X_xy, Y_xy, Bx_xy, By_xy, scale=100, angles='xy')
    plt.colorbar(cntrf)
    ax.set_xlabel('$x$ in m')
    ax.set_ylabel('$y$ in m')
    ax.set_title(f"$z={z_ctr:.2f}$ m plane")
    ax = fig.add_subplot(234)
    Bx_xz = B_res['Bx(T)'].values[indx_from_to[3]:indx_from_to[4]]
    By_xz = B_res['By(T)'].values[indx_from_to[3]:indx_from_to[4]]
    Bz_xz = B_res['Bz(T)'].values[indx_from_to[3]:indx_from_to[4]]
    Bx_xz.shape = X_xz.shape
    By_xz.shape = X_xz.shape
    Bz_xz.shape = X_xz.shape
    Bmag_xz = np.sqrt(Bx_xz**2 + By_xz**2 + Bz_xz**2)
    cntrf = ax.contourf(Z_xz.T, X_xz.T, Bmag_xz.T, cmap='rainbow', levels=100)
    ax.quiver(Z_xz, X_xz, Bz_xz, Bx_xz, scale=100, angles='xy')
    if params_df['yoke_type'][i] == 'Mag3':
        ax.plot(z_line, x_line_ret, '--', color='k', label='line 1')
    else:
        ax.plot(z_line, 0.0*x_line_ret, '--', color='k', label='line 1')

    cbar = plt.colorbar(cntrf)
    ax.set_xlabel('$z$ in m')
    ax.set_ylabel('$x$ in m')
    cbar.set_label('$|B|$ in T')
    ax.set_title('$y=0$ plane')
    ax.legend()
    ax = fig.add_subplot(235)
    Bx_yz = B_res['Bx(T)'].values[indx_from_to[4]:indx_from_to[5]]
    By_yz = B_res['By(T)'].values[indx_from_to[4]:indx_from_to[5]]
    Bz_yz = B_res['Bz(T)'].values[indx_from_to[4]:indx_from_to[5]]
    Bx_yz.shape = Y_yz.shape
    By_yz.shape = Y_yz.shape
    Bz_yz.shape = Y_yz.shape
    Bmag_yz = np.sqrt(Bx_yz**2 + By_yz**2 + Bz_yz**2)
    cntrf = ax.contourf(Z_yz.T, Y_yz.T, Bmag_yz.T, cmap='rainbow', levels=100)
    ax.quiver(Z_yz, Y_yz, Bz_yz, By_yz, scale=100, angles='xy')
    cbar = plt.colorbar(cntrf)
    ax.set_xlabel('$z$ in m')
    ax.set_ylabel('$y$ in m')
    cbar.set_label('$|B|$ in T')
    ax.set_title("(line 1, $y$) plane")
    plt.tight_layout()
    fig.savefig(os.path.join(result_directory, f"field_mag_{i}.pdf"))


# plot the cost parameters
fig = plt.figure()
ax = fig.add_subplot(221)
ax.bar(np.linspace(1, num_mag, num_mag), cost_parameters[:, 0]*1e-6, color='C0')
ax.set_xlabel('magnet number')
ax.set_title('iron weight in t')
ax = fig.add_subplot(222)
ax.bar(np.linspace(1, num_mag, num_mag), cost_parameters[:, 2]*1e-6, color='C1')
ax.set_xlabel('magnet number')
ax.set_title('coil weight in t')
ax = fig.add_subplot(223)
ax.bar(np.linspace(1, num_mag, num_mag), cost_parameters[:, 4]*1e-3, color='C2')
ax.set_xlabel('magnet number')
ax.set_title('power consumption in kW')
ax = fig.add_subplot(224)
ax.bar(np.linspace(1, num_mag, num_mag), cost_parameters[:, 6]*1e-6, color='C3')
ax.set_xlabel('magnet number')
ax.set_title('current density in A/mm2')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(np.linspace(1, num_mag, num_mag)-0.25, cost_parameters[:, 1]*1e-3, color='C0', width=0.2, label='Iron')
ax.bar(np.linspace(1, num_mag, num_mag), cost_parameters[:, 3]*1e-3, color='C1', width=0.2, label='Coil')
ax.bar(np.linspace(1, num_mag, num_mag)+0.25, cost_parameters[:, 5]*1e-3, color='C2', width=0.2, label='Electricity')
ax.legend()
ax.set_title('Costs in peanuts')
ax.set_xlabel('magnet number')

total_cost = (np.sum(cost_parameters[:, 1]) + np.sum(cost_parameters[:, 3]) + np.sum(cost_parameters[:, 5]))*1e-3

print(f"the total cost is {total_cost:.0f} peanuts")

plt.show()

# show also the pyvista plot
if PLOT_CONFIG:
    pl.show_grid()
    pl.camera_position = 'zy'
    pl.camera.elevation = 35
    pl.camera.azimuth = 45
    pl.add_axes()
    pl.show()
