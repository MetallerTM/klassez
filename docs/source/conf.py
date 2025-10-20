r"""
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
"""

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -- Project information -----------------------------------------------------

project = 'klassez'
copyright = '2025, Francesco Bruno and Letizia Fiorucci'
author = 'Francesco Bruno and Letizia Fiorucci'
release = '0.4a.11'

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",        # Genera la documentazione dalle docstring
    "sphinx.ext.napoleon",       # Supporta NumPy e Google style docstrings
    "sphinx.ext.viewcode",       # Aggiunge link al sorgente
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    }

mathjax3_config = {
    'tex': {
        'equationNumbers': {'autoNumber': 'AMS'},
    },
    'options': {
        'processHtmlClass': 'math|output_area',
    },
    'chtml': {
        'displayAlign': 'left',  # Allinea le equazioni a sinistra
        'displayIndent': '0em'   # Rimuove l'indentazione
    }
}

# Formato docstring: NumPy
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True  # Usa la sezione Parameters per i parametri
napoleon_use_rtype = True  # Usa la sezione Returns per i tipi di ritorno
napoleon_preprocess_types = True
napoleon_custom_sections = [('Parameters', 'params_style')]  # Forza stile params


# -- Options for HTML output -------------------------------------------------

#html_theme = "sphinx_rtd_theme"
html_theme = "sphinx_rtd_theme"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ['custom.css']

html_logo = '_static/klassez_logo.png'
html_favicon = '_static/klassez_logo.png'
html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 5,
    'includehidden': True,
}

# Configurazione per autodoc
autoclass_content = 'class'  # Include sia la docstring della classe che dell'__init__
autodoc_member_order = 'alphabetical'  # Ordina i membri secondo il sorgente (per moduli)
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__,__call__',
    'member-order': 'alphabetical',  # Ordina le funzioni alfabeticamente all'interno dei moduli
}



