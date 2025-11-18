# ContinUNet
[![Pytest](https://github.com/hstewart93/continunet/actions/workflows/pytest.yml/badge.svg)](https://github.com/hstewart93/continunet/actions/workflows/pytest.yml)

Source finding package for radio continuum data powered by U-Net segmentation algorithm.

- [Paper](https://academic.oup.com/rasti/article/3/1/315/7685538)
- [Documentation](https://hstewart93.github.io/continunet/index.html)
- [Installation](#installation)
- [Developer Installation](#developer-installation)
- [Example Notebook](https://github.com/hstewart93/continunet/blob/main/continunet/user_example.ipynb)
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

**ContinUNet requires `>=Python3.9, <Python3.12`.**

## Developer Setup
If you want to setup a developer environment, install as follows:

Once you have cloned down this repository using `git clone`, cd into the app directory:

```bash
git clone git@github.com:hstewart93/continunet.git
cd continunet
```

Create a virtual environment for development, if you are using zsh:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev,ci]"
```

For bash, run `pip install -e .[dev,ci]` instead. To exit the virtual environment use `deactivate`.

This project used the black auto formatter which can be run on git commit along with flake8 if you install pre-commit. To do this run the following in your terminal from within your virtual environment.

```bash
pre-commit install
```

Now pre-commit hooks should run on `git commit`.

To run the test suite use `pytest`.

## Development
ContinUNet is subject to ongoing development. To see the backlog of features and bug fixes please go to the [project board](https://github.com/users/hstewart93/projects/4/views/1). Please raise any feature requests or bugs as [issues](https://github.com/hstewart93/continunet/issues).

The following features will be added in the next release:

1. Exporting processed images to `.npy` and `.FTIS` [(#33)](https://github.com/hstewart93/continunet/issues/33)
2. Inference for non-square images [(#27)](https://github.com/hstewart93/continunet/issues/27)
3. Taking cutout of `ImageSquare` object before inference [(#28)](https://github.com/hstewart93/continunet/issues/28)
