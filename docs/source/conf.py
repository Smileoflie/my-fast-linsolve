"""Sphinx configuration for linsolve."""
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "linsolve"
copyright = "2026, linsolve contributors"
author = "linsolve contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
