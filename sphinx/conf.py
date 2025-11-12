# Configuration file for Sphinx Documentation

import importlib
from datetime import datetime
from datetime import timezone
from pathlib import Path

# -----------------------------------------------------------------------------
# Disable Sphinx's epub3 builder *before* it imports anything else
# -----------------------------------------------------------------------------
spec = importlib.util.find_spec("sphinx.application")
if spec is not None:
    import sphinx.application

    be = list(getattr(sphinx.application, "builtin_extensions", []))
    if "sphinx.builders.epub3" in be:
        be.remove("sphinx.builders.epub3")
    sphinx.application.builtin_extensions = tuple(be)

# -----------------------------------------------------------------------------
# Path setup — ensure Sphinx imports from the local `src` tree, not site-packages
# -----------------------------------------------------------------------------
here = Path(__file__).resolve().parent

"""
# Uncomment this if using auto-import of local modules:
src_path = here.parent / "src"
if not src_path.exists():
    raise RuntimeError(f"Cannot find local source directory at {src_path}")

# sys.path.insert(0, str(src_path))

# Remove any installed copy of 'laser' from sys.path so autodoc uses local code
for p in list(sys.path):
    if "site-packages" in p and (Path(p) / "laser").is_dir():
        sys.path.remove(p)
"""

# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------
project = "LASER-GENERIC"
author = "Institute for Disease Modeling"
copyright = f"{datetime.now(timezone.utc).year}, {author}"
release = "0.4.1"

# -----------------------------------------------------------------------------
# Optional: disable epub output to avoid loading the epub3 builder
# -----------------------------------------------------------------------------
epub_show_urls = "none"

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # Render docstrings
    "sphinx.ext.autosummary",  # Generate summary tables
    "sphinx.ext.napoleon",  # Parse Google/Numpy-style docstrings
    "sphinx.ext.viewcode",  # Add [source] links
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",  # Include type hints
    "myst_parser",  # Markdown support
    "sphinx_rtd_theme",
    "autoapi.extension",
]

autoapi_type = "python"
autoapi_dirs = ["../src/laser/generic"]
autoapi_add_toctree_entry = True
autoapi_keep_files = True
autoapi_imported_members = False

autosummary_generate = True
autoclass_content = "both"
add_module_names = False

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
html_static_path = ["_static"]

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"

# -----------------------------------------------------------------------------
# LaTeX / PDF output
# -----------------------------------------------------------------------------
latex_engine = "xelatex"
latex_documents = [
    ("index", "LASER-GENERIC.tex", "LASER-GENERIC Documentation", author, "manual"),
]
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "preamble": r"""
        \usepackage{titlesec}
        \titleformat{\chapter}[display]
            {\normalfont\huge\bfseries}{\chaptername\ \thechapter}{20pt}{\Huge}
        \setcounter{secnumdepth}{3}
        \setcounter{tocdepth}{3}
    """,
}

# -----------------------------------------------------------------------------
# MyST Markdown options
# -----------------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "linkify"]


# -----------------------------------------------------------------------------
# Workaround: prevent Sphinx from loading the epub3 builder
# -----------------------------------------------------------------------------
def setup(app):
    app.registry.builders.pop("epub", None)
    app.registry.builders.pop("epub3", None)
