# The snoo.py package

<img src="logo_with_name.pdf" alt="The snoo.py package" width="700"/>

The snoo.py package is a leightweight but efficient finite element method based on gmsh and scipy.
It has been developed to provide field maps for the optimization of a muon shield at CERN.

# Installation

The easiest way to use snoo.py is to install it in pip. Simply run

pip install snoopy-SHiP

The second option is by using poetry. Just clone this repository and run

poetry build
poetry install

Thats it!

# Requirements

It is important that You have python dev installed. In Ubuntu You get it like this:

sudo apt-get install python3-dev

In CentOS You get it like this:

sudo yum install python3-devel

This package is based on python3.11.

# Getting started.

There is an examples folder in the root directory. Also, You can generate autodocs
by running:

poetry run make html

in the docs directory.
This will also generate the examples gallery.
