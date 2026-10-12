# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Giulia'
copyright = '2024, David López Pérez'
author = 'David López Pérez'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_tabs.tabs',
    # Parses the Markdown pages under docs/ that the .rst wrappers ``include``
    # with ``:parser: myst_parser.sphinx_``. Without it those includes fail.
    'myst_parser',
    # For generating PDF files
    'sphinx_simplepdf',
]

# Generate anchor slugs for headings down to level 4, so the cross-page fragment
# links the Markdown pages use (e.g. ``pairing.md#3-how-cells-get-paired``) resolve
# to the same slugs GitHub produces.
myst_heading_anchors = 4

# Base URL used to rewrite links that point at repository files (see
# ``_rewrite_markdown_links`` below). Override it if you fork the project.
REPO_BLOB_URL = os.environ.get(
    'GIULIA_REPO_BLOB_URL',
    'https://github.com/david-lopez-perez/Giulia/blob/main',
)

# The Markdown pages in docs/ are dual-target: Sphinx renders them through the .rst
# wrappers in this folder, and they are also read directly on the Git host. So they
# link to their targets with paths relative to docs/ — for example
# ``[Models](models.md)``, ``[main.py](../main.py)`` or
# ``[catalog](output-saveable/saveable-catalog.md)``.
#
# MyST cannot resolve those (they are file paths, not Sphinx document names) and
# leaves a dead ``href="#models.md"`` in the HTML. The hooks below resolve them:
#
#   * a page that IS wrapped by one of the .rst files here becomes a normal internal
#     link to the corresponding HTML page;
#   * anything else (source files, config files, the Markdown-only doc sets under
#     energy-efficiency/ and output-saveable/) becomes an absolute link into the
#     repository, so it still works from the published site.
_SOURCE_ROOT = 'docs'  # every wrapped Markdown page lives directly in docs/

# Filled at builder-inited: {'running-giulia.md': 'running-giulia', ...}
_WRAPPED_MARKDOWN = {}


def _collect_wrapped_markdown(app):
    """Map each included Markdown page to the docname of the .rst that wraps it."""
    import glob
    import re

    pattern = re.compile(r'^\s*\.\.\s+include::\s*\.\./(\S+\.md)\s*$', re.MULTILINE)
    for rst in glob.glob(os.path.join(app.srcdir, '*.rst')):
        with open(rst, encoding='utf-8') as handle:
            for match in pattern.finditer(handle.read()):
                _WRAPPED_MARKDOWN[match.group(1)] = os.path.splitext(os.path.basename(rst))[0]


def _rewrite_markdown_links(app, doctree, docname):
    import posixpath

    from docutils import nodes

    for node in doctree.findall(nodes.reference):
        # MyST leaves an unresolved file target as a refid with no refuri.
        target = node.get('refid')
        if not target or node.get('refuri'):
            continue
        path, _, fragment = target.partition('#')
        # A genuine in-page anchor has no path separator and no file extension.
        if '/' not in path and '.' not in path:
            continue

        wrapper = _WRAPPED_MARKDOWN.get(path)
        if wrapper is not None:
            uri = app.builder.get_relative_uri(docname, wrapper)
            if fragment:
                uri += f'#{fragment}'
        else:
            resolved = posixpath.normpath(posixpath.join(_SOURCE_ROOT, path))
            if resolved.startswith('..'):
                continue  # escapes the repository: leave it alone
            uri = f'{REPO_BLOB_URL}/{resolved}'
            if fragment:
                uri += f'#{fragment}'

        del node['refid']
        node['refuri'] = uri


def setup(app):
    app.connect('builder-inited', _collect_wrapped_markdown)
    # Priority 900 so this runs after MyST's own reference post-transforms.
    app.connect('doctree-resolved', _rewrite_markdown_links, priority=900)


# The links handled by _rewrite_markdown_links above are reported as missing
# cross-references by MyST before we get a chance to resolve them. Every other
# warning class still surfaces normally.
suppress_warnings = ['myst.xref_missing']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
