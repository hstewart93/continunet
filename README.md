# ContinUNet
Source finding package for radio continuum data powered by U-Net segmentation algorithm.

## Installation
The project is available on [PyPI](https://pypi.org/project/continunet/), to install latest stable release use:

```pip install continunet```

To install version in development, use:

```pip install git+https://github.com/hstewart93/continunet```

## Developer Installation
If you want to contribute to the repository, install as follows:

Once you have cloned down this repository using `git clone` cd into the app directory eg.

```
cd continunet
```

Create a virtual environment for development, if you are using bash:

```
python3 -m venv venv
source venv/bin/activate
pip install -e .[dev,ci]
```

To exit the virtual environment use `deactivate`.

This project used the black auto formatter which can be run on git commit along with flake8 if you install pre-commit. To do this run the following in your terminal from within your virtual environment.

```
pre-commit install
```

Now pre-commit hooks should run on `git commit`.

The run the test suite use `pytest`.