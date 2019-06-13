========
Overview
========

The GitHub repository has an ``examples/`` folder which contains two Python scripts to show how to apply the methods
in this library on a PyTorch model. The used dataset and neural network is "testnet", a simple generated segmentation
dataset using the U-Net architecture. See :ref:`testnet` for more information.

Run the examples
----------------------
- Install interpret-segmentation into a pip virtualenv or anaconda environment
- Clone GitHub repository: ``git clone https://github.com/andef4/interpret-segmentation``
- Install additional dependencies: ``pip install scikit-image requests`` / ``conda install scikit-image requests``

The example uses the "testnet" dataset, you can download the dataset and a pretrained model by running the ``examples/testnet/download.py`` script.
Alternatively, you can generate and train the dataset yourself with the ``examples/testnet/generate.py`` and ``examples/testnet/train.py`` scripts.

The run one of the example scripts:

* ``python3 examples/hdm.py``
* ``python3 examples/rise.py``

Both scripts generate PNG visualizations in the ``examples`` directory.
The runtime of the scripts are around 30-60 seconds on a current generation high-end graphics card (GeForce 1080 Ti/RTX 2080).
