"""Test the Mag 1 template.
   ==========

   This is a unit test for the Mag 1 template.
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
# =================

# get the files directoy. We need the following statements to make the auto gallery running
if cwd.split('\\')[-1] == 'examples':
    test_directory = os.path.join('..', 'files', 'tests', 'test_mag_1')
    materials_directory = os.path.join('..', 'files', 'materials')
    test_directory = os.path.join('..', 'tests')
else:
    test_directory = os.path.join('files', 'tests', 'test_mag_1')
    materials_directory = os.path.join('files', 'materials')

# the parameters file
params_df = pd.read_csv(os.path.join(test_directory, 'parameters.csv'))


# make the evaluatio positions
X, Y = np.meshgrid(np.linspace(0.0, 2.0, 20),
                   np.linspace(0.0, 2.0, 20))

# the additional evaluation positions
eval_pos = np.zeros((20*20, 3))
eval_pos[:, 0] = X.flatten()
eval_pos[:, 1] = Y.flatten()

# %%
# Compute the map
# =================
points, B, M_i, M_c, Q, J = snoopy.get_vector_field_mag_1(params_df, 0,
                                   materials_directory=materials_directory,
                                   eval_pos=eval_pos, result_directory=test_directory,
                                   result_spec="_test_mag_1")

B_eval = pd.read_csv(os.path.join(test_directory, 'B_test_mag_1.csv')).values

# map_out = np.append(points, B, axis=1)
# np.save(os.path.join(test_directory, 'map.npy'), map_out)


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
assert M_i == 241892319.99999994
assert M_c == 447739.94824795314
assert Q == 88448.62816951754
assert J == 10000000.0
