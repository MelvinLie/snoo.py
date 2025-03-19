"""Test the Mag 3 template.
   ==========

   This is a unit test for the Mag 3 template.
"""

import os

# enforce open mp to run on single thread
os.environ["OMP_NUM_THREADS"] = "1"
# enforce blas to run on single thread
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

import snoopy

# this is the current woring directory
cwd = os.getcwd()

# %%
# Test parameters
# =================

# get the files directoy. We need the following statements to make the auto gallery running
if cwd.split('\\')[-1] == 'examples':
    test_directory = os.path.join('..', 'files', 'tests', 'test_mag_3')
    materials_directory = os.path.join('..', 'files', 'materials')
    test_directory = os.path.join('..', 'tests')
else:
    test_directory = os.path.join('files', 'tests', 'test_mag_3')
    materials_directory = os.path.join('files', 'materials')

# the parameters file
params_df = pd.read_csv(os.path.join(test_directory, 'parameters.csv'))

# the additional evaluation positions
eval_pos = np.zeros((100, 3))
eval_pos[:, 2] = np.linspace(-1.7, 1.7, 100)

# %%
# Compute the map
# =================
points, B, M_i, M_c, Q, J = snoopy.get_vector_field_mag_3(params_df, 0,
                                   materials_directory=materials_directory,
                                   eval_pos=eval_pos, result_directory=test_directory,
                                   result_spec="_test_mag_3")

B_eval = pd.read_csv(os.path.join(test_directory, 'B_test_mag_3.csv')).values


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

assert M_i == 89058843.98683298
assert M_c == 260537.6988856567
assert Q == 51467.82667718888
assert J == 10000000.0
