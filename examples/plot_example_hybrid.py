"""Example: The script for the optimization of the hybrid muon shield.
   ==========

   Here the input parameters do not include the magnetized hadron stopper.
"""

import os
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt

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
else:
    directory = os.path.join('files', 'parameters')
    materials_directory = os.path.join('files', 'materials')


# the parameters file
params_df = pd.read_csv(os.path.join(directory, '250403_supernut_1.csv'))

# the number of magnets
num_mag = params_df.values.shape[0]

# field maps list
maps = []

# set this flag to plot the configuration
PLOT_CONFIG = True

# the parameters used for the cost estimation
cost_parameters = np.zeros((num_mag-1, 7))

# %%
# Loop over all magnets
# =====================

if PLOT_CONFIG:
    # In case plot config is enabled, we open a pyvista plotter
    pl = pv.Plotter()


# the first two lines always encode the magnetized hadron stopper and the superconducting magnet
points, B, M_i, M_c, Q, J = snoopy.get_vector_field_ncsc(params_df, df_index=0)

maps.append((points, B))

C_i, C_c, C_edf = snoopy.compute_prices(params_df, 0, M_i, M_c, Q)
cost_parameters[0, :] = np.array([M_i, C_i, M_c, C_c, Q, C_edf, J])

if PLOT_CONFIG:
    snoopy.plot_geometry_ncsc(pl, params_df, 0)
    snoopy.plot_vector_field(pl, maps[0][0], maps[0][1], title='B in T',
                                    mag=0.05, sym_yz=1, sym_xz=2)

# the next magnets encode the normal conducting magnets
for i in range(2, num_mag):

    if params_df['yoke_type'][i] == 'Mag1':

        points, B, M_i, M_c, Q, J = snoopy.get_vector_field_mag_1(params_df, i,
                                  materials_directory=materials_directory)
        maps.append((points, B))

        C_i, C_c, C_edf = snoopy.compute_prices(params_df, i, M_i, M_c, Q)

        cost_parameters[i-1, :] = np.array([M_i, C_i, M_c, C_c, Q, C_edf, J])

        if PLOT_CONFIG:
            snoopy.plot_geometry_mag_1(pl, params_df, i)
            snoopy.plot_vector_field(pl, maps[i-1][0], maps[i-1][1], title='B in T',
                                    mag=0.05, sym_yz=1, sym_xz=2)

    if params_df['yoke_type'][i] == 'Mag3':

        points, B, M_i, M_c, Q, J = snoopy.get_vector_field_mag_3(params_df, i,
                                       materials_directory=materials_directory)
        maps.append((points, B))

        C_i, C_c, C_edf = snoopy.compute_prices(params_df, i, M_i, M_c, Q)

        cost_parameters[i-1, :] = np.array([M_i, C_i, M_c, C_c, Q, C_edf, J])

        if PLOT_CONFIG:
            snoopy.plot_geometry_mag_3(pl, params_df, i)
            snoopy.plot_vector_field(pl, maps[i-1][0], maps[i-1][1], title='B in T',
                                    mag=0.05, sym_yz=1, sym_xz=2)


# plot the cost parameters
fig = plt.figure()
ax = fig.add_subplot(221)
ax.bar(np.linspace(1, num_mag-1, num_mag-1), cost_parameters[:, 0]*1e-6, color='C0')
ax.set_xlabel('magnet number')
ax.set_title('iron weight in t')
ax = fig.add_subplot(222)
ax.bar(np.linspace(1, num_mag-1, num_mag-1), cost_parameters[:, 2]*1e-6, color='C1')
ax.set_xlabel('magnet number')
ax.set_title('coil weight in t')
ax = fig.add_subplot(223)
ax.bar(np.linspace(1, num_mag-1, num_mag-1), cost_parameters[:, 4]*1e-3, color='C2')
ax.set_xlabel('magnet number')
ax.set_title('power consumption in kW')
ax = fig.add_subplot(224)
ax.bar(np.linspace(1, num_mag-1, num_mag-1), cost_parameters[:, 6]*1e-6, color='C3')
ax.set_xlabel('magnet number')
ax.set_title('current density in A/mm2')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(np.linspace(1, num_mag-1, num_mag-1)-0.25, cost_parameters[:, 1]*1e-3, color='C0', width=0.2, label='Iron')
ax.bar(np.linspace(1, num_mag-1, num_mag-1), cost_parameters[:, 3]*1e-3, color='C1', width=0.2, label='Coil')
ax.bar(np.linspace(1, num_mag-1, num_mag-1)+0.25, cost_parameters[:, 5]*1e-3, color='C2', width=0.2, label='Electricity')
ax.legend()
ax.set_title('Costs in peanuts')
ax.set_xlabel('magnet number')
plt.show()

if PLOT_CONFIG:
    # pl.show_grid()
    pl.camera_position = 'zy'
    pl.camera.elevation = 35
    pl.camera.azimuth = 45
    pl.add_axes()
    pl.show()

total_cost = np.sum(cost_parameters[:, 1] + cost_parameters[:, 3] + cost_parameters[:, 5])
print(f"The total cost is {total_cost*1e-3:.1f} mega peanuts")