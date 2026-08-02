"""Sphinx configuration for the scFair documentation (built on Read the Docs)."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

# -- Project information ------------------------------------------------------
# Footer (sphinx-book-theme): "By Zhao Li (李钊)" and
# "© Copyright YYYY, Zhao Li (李钊)." — same pattern as scATrans.
project = "scFair"
author = "Zhao Li (李钊)"
copyright = f"{datetime.now():%Y}, {author}"
repository_url = "https://github.com/leelieber2025/scFair"
default_branch = "main"

try:
    from scfair._version import __version__ as release
except Exception:
    try:
        from importlib.metadata import version as _pkg_version

        release = _pkg_version("scfair")
    except Exception:
        release = "0.0.0+unknown"
version = release

html_context = {
    "display_github": True,
    "github_user": "leelieber2025",
    "github_repo": "scFair",
    "github_version": default_branch,
    "conf_py_path": "/docs/",
}

# -- Extensions ----------------------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

# -- Autodoc / autosummary / napoleon -------------------------------------------
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True

# -- MyST / myst-nb --------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "substitution",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# Tutorial notebooks are executed ahead of time when figures are needed;
# render stored outputs on RTD instead of re-running heavy data loads.
nb_execution_mode = "off"
nb_merge_streams = True

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Local research notes — not part of the public site
    "PAPER_*",
    "DEVELOPMENT_LOG.md",
    "PACKAGE_VALUE.md",
    "gold.md",
    "review.md",
    "AUTO_N_*",
    "CELLBRF_*",
    "DATA_SOURCES.md",
]
needs_sphinx = "5.0"
nitpicky = False

# -- intersphinx ------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
}

# -- HTML / sphinx_book_theme ----------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = project

html_theme_options = {
    "repository_url": repository_url,
    "repository_branch": default_branch,
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "home_page_in_toc": True,
    "show_navbar_depth": 1,
    "navigation_with_keys": True,
}

pygments_style = "tango"
pygments_dark_style = "monokai"

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_show_sphinx = False
html_show_copyright = True

if os.environ.get("READTHEDOCS"):
    html_baseurl = "https://scfair.readthedocs.io/"
