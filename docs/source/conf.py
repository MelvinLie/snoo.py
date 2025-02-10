# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

topdir = "../../"

project = 'snoopy'
copyright = '2024, Melvin Liebsch'
author = 'Melvin Liebsch'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              "sphinx.ext.autosummary",
              "sphinxcontrib.apidoc",
              'sphinx.ext.duration',
              'sphinx.ext.doctest',
              'sphinx_gallery.gen_gallery',
              'myst_parser']

sys.path.insert(0, os.path.abspath(topdir))

apidoc_module_dir = topdir + project
apidoc_output_dir = "gen"
apidoc_excluded_paths = ["tests"]
apidoc_separate_modules = True

source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = []

sphinx_gallery_conf = {
     	'examples_dirs': '../../examples',   # path to your example scripts
     	'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
        'backreferences_dir': 'gen_modules/backreferences',
        'doc_module': ('sphinx_gallery', 'numpy'),
        'reference_url': {
            'sphinx_gallery': None,
        },
        'image_scrapers': ('matplotlib', 'pyvista'),
        'compress_images': ('images', 'thumbnails'),
        'show_memory': True,
        'promote_jupyter_magic': False,
        'junit': os.path.join('sphinx-gallery', 'junit-results.xml'),
        # capture raw HTML or, if not present, __repr__ of last expression in
        # each code block
        'capture_repr': ('_repr_html_', '__repr__'),
        'matplotlib_animations': True,
        'image_srcset': ["4x"],
        'nested_sections': False,
        # Modules for which function level galleries are created.  In
        "doc_module": "pyvista",
        "first_notebook_cell": "%matplotlib inline\n"
        "from pyvista import set_plot_theme\n"
        'set_plot_theme("document")\n',
	}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']


from plotly.io._sg_scraper import plotly_sg_scraper

image_scrapers = ("matplotlib", plotly_sg_scraper, "pyvista")

# -- pyvista configuration ---------------------------------------------------
import pyvista

# Manage errors
pyvista.set_error_output_file("errors.txt")
# Ensure that offscreen rendering is used for docs generation
pyvista.OFF_SCREEN = True  # Not necessary - simply an insurance policy
# Preferred plotting style for documentation
pyvista.set_plot_theme("document")
pyvista.global_theme.window_size = [1024, 768]
pyvista.global_theme.font.size = 22
pyvista.global_theme.font.label_size = 22
pyvista.global_theme.font.title_size = 22
pyvista.global_theme.return_cpos = False
pyvista.set_jupyter_backend(None)
# Save figures in specified directory
pyvista.FIGURE_PATH = os.path.join(os.path.abspath("./images/"), "auto-generated/")
if not os.path.exists(pyvista.FIGURE_PATH):
    os.makedirs(pyvista.FIGURE_PATH)

pyvista.BUILDING_GALLERY = True
os.environ["PYVISTA_BUILDING_GALLERY"] = "true"
