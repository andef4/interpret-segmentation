.. interpret-segmentation documentation master file, created by
   sphinx-quickstart on Tue Jun 11 09:15:06 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to interpret-segmentation's documentation!
==================================================

A one-stop shop for the interpretability of image segmentation models.
This code was extracted from my bachelors thesis: https://github.com/andef4/thesis-code

The following methods are currently implemented:

.. toctree::
   :maxdepth: 2

   rise
   hdm

Installation
------------
All dependencies except pytorch and torchvisison are installed automatically.
Please install pytorch and torchvision manually as described on https://pytorch.org/get-started/locally/.

Examples
--------
Examples how to use the two algorithms are provided in the examples/ subdirectory in the git repository.
The examples use the testnet dataset, which was specifically built as a showcase for these algorithms.

.. toctree::
   :maxdepth: 2

   examples
   testnet

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
