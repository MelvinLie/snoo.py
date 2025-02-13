"""Example: Materials example.
   ==========

   Plot the materials available in snoo.py
"""

import os
import numpy as np
import pandas as pd
import pyvista as pv
import time
import matplotlib.pyplot as plt

import snoopy as snoopy


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

# %%
# Load all materials we have
# ==========================
aisi1010 = snoopy.Reluctance(os.path.join(materials_directory, 'aisi1010.csv'))
roxie_iron = snoopy.Reluctance(os.path.join(materials_directory, 'bhiron_1.csv'))
comsol_pure_iron = snoopy.Reluctance(os.path.join(materials_directory, 'pure_iron_comsol.csv'))

# %%
# Plot the permeabilities
# =======================
B_max = 5.0

fig = plt.figure()
ax = fig.add_subplot(111)
aisi1010.plot_permeability(ax, B_max, label='AISI 1010', resol=1000)
roxie_iron.plot_permeability(ax, B_max, label='ROXIE iron', resol=1000)
comsol_pure_iron.plot_permeability(ax, B_max, label='COMSOL pure iron', resol=1000)
ax.set_yscale('log')
ax.set_xlabel('$B$ in T')
ax.set_title(r'relative permeability $\mu_{\mathrm{0}}$')
ax.grid('minor')
ax.legend()
plt.show()