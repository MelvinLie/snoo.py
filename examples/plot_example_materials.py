"""Example: Materials example.
   ==========

   Plot the materials available in snoo.py
"""

import os
import matplotlib.pyplot as plt
import glob

import snoopy as snoopy


# this is the current woring directory
cwd = os.getcwd()

# %%
# Script parameters
# =================

B_max = 5.0

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
material_files = ['files\\materials\\aisi1010.json',
                  'files\\materials\\bhiron_1.json',
                  'files\\materials\\pure_iron_comsol.json',
                  'files\\materials\\iron_pw.json']

# %%
# Open a figure
# ==========================
fig = plt.figure()
ax = fig.add_subplot(111)


# %%
# Plot the permeabilities
# =======================
for i, fn in enumerate(material_files):

    # open the material
    material = snoopy.Reluctance(fn)

    # get the name
    material_name = fn.split('\\')[-1]

    # plot it
    material.plot_permeability(ax, B_max, label=material_name, resol=1000)

ax.set_yscale('log')
ax.set_xlabel('$B$ in T')
ax.set_title(r'relative permeability $\mu_{\mathrm{r}}$')
ax.grid('minor')
ax.legend()
plt.show()