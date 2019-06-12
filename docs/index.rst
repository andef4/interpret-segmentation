.. interpret-segmentation documentation master file, created by
   sphinx-quickstart on Tue Jun 11 09:15:06 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to interpret-segmentation's documentation!
==================================================

interpret-segmentation is a one-stop shop for the interpretability of image segmentation models.
This code was extracted from the code of my bachelor thesis: https://github.com/andef4/thesis-code.

The PDF of the thesis is available here: https://github.com/andef4/thesis-doc/releases/download/release/thesis.pdf.
It contains detailed explanations of the methods used here.

Installation
------------

| For pip based environments:
| ``pip install interpret_segmentation``

| For anaconda users:
| ``conda install interpret_segmentation``

All dependencies except pytorch and torchvisison are installed automatically.
Please install pytorch and torchvision manually as described on https://pytorch.org/get-started/locally/.

Methods
-------
.. toctree::
   :maxdepth: 2

   rise
   hdm

Examples
--------
.. toctree::
   :maxdepth: 2

   examples
   testnet
