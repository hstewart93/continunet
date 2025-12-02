# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))  # or your package folder

project = "ContinUNet"
copyright = "2025, Hattie Stewart"
author = "Hattie Stewart"
release = "v0.10"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # supports Google & NumPy-style docstrings
    "sphinx.ext.viewcode",  # adds source links
    "sphinx.ext.autosummary",
]
autosummary_generate = True

html_theme = "sphinx_rtd_theme"

master_doc = "index"
