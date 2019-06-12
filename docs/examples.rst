========
Examples
========

- Install interpret-segmentation
- Clone GitHub repository: `git clone https://github.com/andef4/interpret-segmentation`
- Install additional dependencies: `pip install scikit-image requests`

The example uses the "testnet" dataset, a dataset built to show how interpret-segmentation works.
You can download the dataset and a pretrained model by running the `examples/testnet/download.py` script,
or generate and train the dataset yourself with the `examples/testnet/generate.py` and `examples/testnet/train.py` scripts.

The scripts `examples/hdm.py` and `examples/rise.py` show how to use this library. The generated visualizations are saved in the `examples/` directory.
