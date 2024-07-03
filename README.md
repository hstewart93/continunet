# ContinUNet
[![Pytest](https://github.com/hstewart93/continunet/actions/workflows/pytest.yml/badge.svg)](https://github.com/hstewart93/continunet/actions/workflows/pytest.yml)

Source finding package for radio continuum data powered by U-Net segmentation algorithm.

- [Paper](https://academic.oup.com/rasti/article/3/1/315/7685538?utm_source=advanceaccess&utm_campaign=rasti&utm_medium=email#supplementary-data)
- [Installation](#installation)
- [Developer Installation](#developer-installation)
- [Quickstart](#quickstart)
- [Example Notebook](https://github.com/hstewart93/continunet/tree/finder/continunet/user_example.ipynb)
- [Training Dataset](https://www.kaggle.com/datasets/harrietstewart/continunet)
- [Next Release](#development)

## Installation
The project is available on [PyPI](https://pypi.org/project/continunet/), to install latest stable release use:

```bash
pip install continunet
```

To install version in development, use:

```bash
pip install git+https://github.com/hstewart93/continunet
```

**ContinUNet has a `Python 3.9` minimum requirement.**

## Developer Installation
If you want to contribute to the repository, install as follows:

Once you have cloned down this repository using `git clone`, cd into the app directory:

```bash
git clone git@github.com:hstewart93/continunet.git
cd continunet
```

Create a virtual environment for development, if you are using bash:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .[dev,ci]
```

To exit the virtual environment use `deactivate`.

This project used the black auto formatter which can be run on git commit along with flake8 if you install pre-commit. To do this run the following in your terminal from within your virtual environment.

```bash
pre-commit install
```

Now pre-commit hooks should run on `git commit`.

To run the test suite use `pytest`.

## Quickstart
The package currently support `.FITS` type images. To perform source finding you can import the `finder` module:

```python
from continunet.finder import Finder
```

Load your image file:

```python
finder = Finder("<filepath>")
```

To produce a source catalogue and populate the `Finder` instance:

```python
sources = finder.find()
```

If you want to calculate the model map and residuals image as part of source finding, use the `Finder.find()` method with `generate_maps=True`:

```python
sources = finder.find(generate_maps=True)
model_map = finder.model_map
residuals = finder.residuals
```

Alternatively, manually calculate model map and residual images using:

```python
model_map = finder.get_model_map()
residuals = finder.get_residuals()
```

Useful available attributes of the `Finder` object are:
```python
finder.sources # cleaned source catalogue
finder.reconstructed_image # predicted image reconstructed by unet module
finder.segmentation_map # predicted segmentation map
finder.model_map # model map of cleaned predicted sources
finder.residuals # residual image as numpy array
finder.raw_sources # sources from labelled segmentation map before cleaning
```

Export source catalogue using `finder.export_sources` as `.csv` by default or `.FITS` by setting `export_fits=True`:

```python
finder.export_sources("<filepath>", export_fits=<Boolean>)
```

Source parameters extracted are:

| **Parameter**              | **Description**                                                                                        |
|----------------------------|--------------------------------------------------------------------------------------------------------|
| `x_location_original`      | x coordinate of the source from the cutout used for inference                                          |
| `y_location_original`      | y coordinate of the source from the cutout used for inference                                          |
| `orientation`              | orientation of source ellipse in radians                                                               |
| `major_axis`               | major axis of source ellipse                                                                           |
| `minor_axis`               | minor axis of source ellipse                                                                           |
| `flux_density_uncorrected` | total intensity of the segmented source region before beam corrections applied                         |
| `label`                    | class label in predicted segmentation map                                                              |
| `x_location`               | x coordinate of the source in the original input image dimensions                                      |
| `y_location`               | y coordinate of the source in the original input image dimensions                                      |
| `right_ascension`          | RA coordinate of the source in the original input image dimensions                                     |
| `declination`              | Dec coordinate of the source in the original input image dimensions                                    |
| `area`                     | area of source ellipse                                                                                 |
| `position_angle`           | position angle of source ellipse in degrees                                                            |
| `correction_factor`        | correction factor applied to flux density measurement to account for undersampling of synthesised beam |
| `flux_density`             | corrected flux density                                                                                 |

## Development
ContinUNet is subject to ongoing development. To see the backlog of features and bug fixes please go to the [project board](https://github.com/users/hstewart93/projects/4/views/1). Please raise any feature requests or bugs as [issues](https://github.com/hstewart93/continunet/issues).

The following features will be added in the next release:

1. Exporting processed images to `.npy` and `.FTIS` [(#33)](https://github.com/hstewart93/continunet/issues/33)
2. Inference for non-square images [(#27)](https://github.com/hstewart93/continunet/issues/27)
3. Taking cutout of `ImageSquare` object before inference [(#28)](https://github.com/hstewart93/continunet/issues/28)