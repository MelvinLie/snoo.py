import os
import numpy
import platform
import shutil
from distutils.core import Distribution, Extension

from Cython.Build import build_ext, cythonize

# ========================================
# Build file written by M. Liebsch for the
# fem_c module.
#
# ========================================

# compiler arguments are system dependent
if platform.system() == 'Windows':
    compile_args = ['/O2', '-openmp']
    extra_link_args = []
elif platform.system() == 'Linux':
    compile_args = ['-O3', '-fopenmp']
    extra_link_args = ['-fopenmp']

fem_c_dir = "fem_c"

extensions = [Extension(
    "fem_c_mod",
    [
        os.path.join(fem_c_dir, "interface_c.pyx"),
    ],
    extra_compile_args=compile_args,
    include_dirs=[numpy.get_include()],
    extra_link_args=extra_link_args,
    ),
]

ext_modules = cythonize(extensions, include_path=[fem_c_dir])

dist = Distribution({"ext_modules": ext_modules})
cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()

for output in cmd.get_outputs():
    relative_extension = os.path.relpath(output, cmd.build_lib)
    shutil.copyfile(output, os.path.join("snoopy", relative_extension))