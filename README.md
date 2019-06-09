# Interpret Segmentation

A one-stop shop for the interpretability of image segmentation models.
This code was extracted from my bachelors thesis: https://github.com/andef4/thesis-code

Available algorithms

- Hausdorff Distance Masks
- RISE

## Installation

All dependencies except pytorch and torchvisison are installed automatically.
Please install pytorch and torchvision manually as described on https://pytorch.org/get-started/locally/.

# Howto run example

- Install interpret-segmentation
- Clone GitHub repository: `git clone https://github.com/andef4/interpret-segmentation`

The example uses the "testnet" dataset, a dataset built to show how interpret-segmentation works.
You can download the dataset and a pretrained model by running the `examples/testnet/download.py` script,
or generate and train the dataset yourself with the `examples/testnet/generate.py` and `examples/testnet/train.py` scripts.

The scripts `examples/hdm.py` and `examples/rise.py` show how to use this library. The generated visualizations are saved in the `examples/` directory.

# Development

To hack on interpret-segmentation, do the following:

- Clone the git repository
- Create a new virtualenv and activate it, e.g. `python3 -m venv venv/; source venv/bin/activate`
- Inside the git repository, install the library in development mode including development dependencies: `pip install -e .[dev]`
- Install the flake8 pre-commit hook with `pre-commit`
