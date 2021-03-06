# Interpret Segmentation

A one-stop shop for the interpretability of image segmentation models.
This code was extracted from my bachelors thesis: https://github.com/andef4/thesis-code

Available algorithms:

- Hausdorff Distance Masks
- RISE

## Documentation
The documentation is available on Read the Docs: https://interpret-segmentation.readthedocs.io/en/latest/.

# Development

To hack on interpret-segmentation, do the following:

- Clone the git repository
- Create a new virtualenv and activate it, e.g. `python3 -m venv venv/; source venv/bin/activate`
- Inside the git repository, install the library in development mode including development dependencies: `pip install -e .[dev]`
- Install the flake8 pre-commit hook with `pre-commit`
