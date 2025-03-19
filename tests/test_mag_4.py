"""Test the Mag 4 template.
   ==========

   This is a unit test for the Mag 4 template.
"""

import os

# enforce open mp to run on single thread
os.environ["OMP_NUM_THREADS"] = "1"
# enforce blas to run on single thread
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import snoopy


# this is the current woring directory
cwd = os.getcwd()

# %%
# Test parameters
# ===============

# get the files directoy. We need the following statements to make the auto gallery running
if cwd.split('\\')[-1] == 'examples':
    test_directory = os.path.join('..', 'files', 'tests', 'test_mag_4')
    materials_directory = os.path.join('..', 'files', 'materials')
    test_directory = os.path.join('..', 'tests')
else:
    test_directory = os.path.join('files', 'tests', 'test_mag_4')
    materials_directory = os.path.join('files', 'materials')

# the parameters file
params_df = pd.read_csv(os.path.join(test_directory, 'parameters.csv'))

# set this flag if You like to generate a plot
plot_result = False

# %%
# Read Piets Field map
# ====================
num_z = 136
num_y = 23

z_cut = 21.5

map_piet = pd.read_csv(os.path.join('files', 'FieldMapPiWaFEA35eQS.txt'), sep='\t', skipinitialspace=True, header=None).values

num_x = np.int64(map_piet.shape[0]/num_z/num_y)

mask_xy = map_piet[:, 2] == z_cut

X_p = map_piet[mask_xy, 0]
Y_p = map_piet[mask_xy, 1]
Bx_p = map_piet[mask_xy, 3]
By_p = map_piet[mask_xy, 4]


X_p.shape = (num_x, num_y)
Y_p.shape = (num_x, num_y)
Bx_p.shape = (num_x, num_y)
By_p.shape = (num_x, num_y)
Bn_p = np.sqrt(Bx_p**2 + By_p**2)

# %%
# Make the evaluation positions
# =============================
resol_x = 50
resol_y = 50

X, Y = np.meshgrid(np.linspace(0.0, 1.85, resol_x),
                   np.linspace(0.0, 1.1, resol_y))

# the additional evaluation positions
eval_pos = np.zeros((resol_x*resol_y, 3))
eval_pos[:, 0] = X.flatten()
eval_pos[:, 1] = Y.flatten()

# %%
# Compute the map
# ===============
points, B, M_i, M_c, Q, J = snoopy.get_vector_field_mag_4(params_df, 0,
                                   materials_directory=materials_directory,
                                   eval_pos=eval_pos, result_directory=test_directory,
                                   result_spec="_test_mag_4",
                                   use_diluted_steel=True)

B_eval = pd.read_csv(os.path.join(test_directory, 'B_test_mag_4.csv')).values

# map_out = np.append(points, B, axis=1)
# np.save(os.path.join(test_directory, 'map.npy'), map_out)

B_norm = np.linalg.norm(B_eval[:, 3:], axis=1)
B_norm.shape = (resol_x, resol_y)

Bx = B_eval[:, 3]
Bx.shape = (resol_x, resol_y)

By = B_eval[:, 4]
By.shape = (resol_x, resol_y)

if plot_result:
    fig = plt.figure()
    ax = fig.add_subplot(121)
    cntrf=ax.contourf(X, Y, B_norm, levels=100, cmap='rainbow')
    ax.streamplot(X, Y, Bx, By, density=0.4, color='k')
    cbar=fig.colorbar(cntrf)
    cbar.set_label('magnetic flux density $B$ in T')
    ax.set_xlabel('$x$ in m')
    ax.set_ylabel('$y$ in m')
    ax.set_title(r'sn$\infty$.py')
    ax = fig.add_subplot(122)
    cntrf=ax.contourf(X_p, Y_p, Bn_p, levels=100, cmap='rainbow')
    ax.streamplot(X_p.T, Y_p.T, Bx_p.T, By_p.T, density=0.4, color='k')
    cbar=fig.colorbar(cntrf)
    cbar.set_label('magnetic flux density $B$ in T')
    ax.set_xlabel('$x$ in m')
    ax.set_ylabel('$y$ in m')
    ax.set_title('Piet')
    plt.show()

# %%
# Read the reference data
# =======================
map_ref = np.load(os.path.join(test_directory, 'map.npy'))
B_eval_ref = pd.read_csv(os.path.join(test_directory, 'B_eval_ref.csv')).values

# %%
# run the checks
# ==============
np.testing.assert_allclose(map_ref[:, :3], points)
np.testing.assert_allclose(map_ref[:, 3:], B)
np.testing.assert_allclose(B_eval_ref, B_eval)
assert M_i == 265423620.0
assert M_c == 231182.49808090203
assert Q == 12483.854896368712
assert J == 5228345.170351609
