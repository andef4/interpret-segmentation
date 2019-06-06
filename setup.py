import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="interpret_segmentation",
    version="0.0.1",
    author="Fabio Anderegg",
    author_email="andef4@bfh.ch",
    description="Interpreing image segmentation models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=['interpret_segmentation'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
