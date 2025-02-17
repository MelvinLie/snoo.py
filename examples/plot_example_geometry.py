"""Example: Plot the baseline configuration.
   ==========

   Here we plot a baseline configuration for the hybrid version.
"""

import os
import numpy as np
import pandas as pd
import pyvista as pv
import time
import matplotlib.pyplot as plt

import snoopy as snoopy


# %%
# Script parameters
# =================

# the parameters file
params_df = pd.read_csv(os.path.join('files', 'parameters', 'baseline_1_mod_2.csv'))

# the number of magnets
num_mag = params_df.values.shape[0]

# field maps list
maps = []

# set this flag to plot the configuration
plot_config = True


# %%
# Loop over all magnets
# =====================

if plot_config:
   # In case plot config is enabled, we open a pyvista plotter
   pl = pv.Plotter()


# We loop over all magnets
for i in range(num_mag):

   if params_df['yoke_type'][i] == 'Mag1':

      # points, B = snoopy.get_vector_field_mag_1(params_df, i)
      # maps.append((points, B))

      if plot_config:
         snoopy.plot_geometry_mag_1(pl, params_df, i, opacity=1, show_edges=True)
         # snoopy.plot_vector_field(pl, maps[i][0], maps[i][1], title='B_1 in T', mag=0.05, sym_yz=1, sym_xz=2)

   if params_df['yoke_type'][i] == 'Mag2':

      # points, B = snoopy.get_vector_field_mag_2(params_df, i)
      # maps.append((points, B))

      if plot_config:
         snoopy.plot_geometry_mag_2(pl, params_df, i, opacity=1, show_edges=True)
         # snoopy.plot_vector_field(pl, maps[i][0], maps[i][1], title='B_2 in T', mag=0.01, sym_yz=1, sym_xz=2)

   if params_df['yoke_type'][i] == 'Mag3':

      # points, B = snoopy.get_vector_field_mag_3(params_df, i)
      # maps.append((points, B))

      if plot_config:
         snoopy.plot_geometry_mag_3(pl, params_df, i, opacity=1, show_edges=True)
         # snoopy.plot_vector_field(pl, maps[i][0], maps[i][1], title='B_3 in T', mag=0.05, sym_yz=1, sym_xz=2)



if plot_config:
   pl.show_grid()
   pl.camera_position = 'zy'
   pl.camera.elevation = 35
   pl.camera.azimuth = 45
   pl.add_axes()
   pl.show()