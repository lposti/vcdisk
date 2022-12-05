# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information

project = 'vcdisk'
copyright = '2022, Lorenzo Posti'
author = 'Lorenzo Posti'

release = '0.3'
version = '0.3.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    # 'sphinxjp.themes.basicstrap',
    'nbsphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

# html_theme = 'sphinx_rtd_theme'
# html_theme = 'basicstrap'
html_theme = 'classic'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# autodoc options
autodoc_default_options = {'members': True, 'special-members': '__call__',}
