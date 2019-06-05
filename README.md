# Interpret Segmentation
A one-stop shop for the interpretability of image segmentation models.
This code was extracted from my bachelors thesis: https://github.com/andef4/thesis-code

Modified algorithms for use in image segmentation:
* RISE
* Grad-CAM

New algorithm developed specifically for the interpretability of image segmentations:
* Hausdorff Distance Masks

# Howto run example

- git clone https://github.com/andef4/interpret-segmentation
- make sure you have at least python 3.4 installer
- cd examples
- python3 -m venv venv/
- source venv/bin/activate
- pip install -r requirements.txt
- python3 download.py # or python generate.py; python train.py
- jupyter notebook
- Open browser at http://localhost
- Open notebook rise.ipynb or hdm.ipynb

