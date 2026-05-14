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
suppress_warnings = ['autosummary.*not found']

import os
import sys
import inspect
from pathlib import Path
basedir_kz = Path(__file__).parent.parent.parent
sys.path.insert(0, str(basedir_kz))

# -- Project information -----------------------------------------------------

project = 'klassez'
copyright = '2025, Francesco Bruno and Letizia Fiorucci'
author = 'Francesco Bruno and Letizia Fiorucci'
release = "0.2.1.dev0"

if 'dev' in release:
    branch = 'dev'
else:
    branch = 'main'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.linkcode',
    'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.graphviz',
    'sphinx.ext.todo',
    'sphinx.ext.ifconfig',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.mathjax',
    'sphinx.ext.imgconverter',
]

autosummary_generate = True
numpydoc_class_members_toctree = False
#numpydoc_show_class_members = False

numfig = True
numfig_format = {
    'figure': 'Figure %s',
    'table':  'Table %s',
    'code-block': 'Listing %s',
}

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
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

html_theme = 'sphinx_rtd_theme'
templates_path = ['_templates']
html_static_path = ['_static']
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

html_context = {
    'github_url': 'https://github.com/MetallerTM/klassez',
    'github_version': branch,
    'conf_py_path': '/docs/',
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

# -- Options for LaTeX output ------------------------------------------------

# Options for LaTeX output
# ------------------------

# The paper size ('letter' or 'a4').
latex_paper_size = 'letter'

# Grouping the document tree into LaTeX files.
# List of tuples:
#   (source start file, target name, title, author,
#    document class [howto/manual])

latex_documents = [
    ('index', 'klassez.tex', 'klassez Documentation',
     'Francesco Bruno and Letizia Fiorucci', 'manual'),
]


# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = None

# Use Unicode aware LaTeX engine
latex_engine = 'pdflatex'  # or 'lualatex'

latex_elements = {}

# Keep babel usage also with xelatex (Sphinx default is polyglossia)
# If this key is removed or changed, latex build directory must be cleaned
latex_elements['babel'] = r'\usepackage{babel}'

# Font configuration
# Fix fontspec converting " into right curly quotes in PDF
# cf https://github.com/sphinx-doc/sphinx/pull/6888/

# Fix fancyhdr complaining about \headheight being too small
latex_elements['passoptionstopackages'] = r"""
    \PassOptionsToPackage{headheight=14pt}{geometry}
"""

# Additional stuff for the LaTeX preamble.
latex_elements['preamble'] = r"""
   % Show Parts and Chapters in Table of Contents
   \setcounter{tocdepth}{0}
   % One line per author on title page
   \DeclareRobustCommand{\and}%
     {\end{tabular}\kern-\tabcolsep\\\begin{tabular}[t]{c}}%
   \usepackage{etoolbox}
   \let\latexdescription=\description
   \def\description{\latexdescription{}{} \breaklabel}
   % But expdlist old LaTeX package requires fixes:
   % 1) remove extra space
   \makeatletter
   \patchcmd\@item{{\@breaklabel} }{{\@breaklabel}}{}{}
   \makeatother
   % 2) fix bug in expdlist's way of breaking the line after long item label
   \makeatletter
   \def\breaklabel{%
       \def\@breaklabel{%
           \leavevmode\par
           % now a hack because Sphinx inserts \leavevmode after term node
           \def\leavevmode{\def\leavevmode{\unhbox\voidb@x}}%
      }%
   }
   \newcommand{\ui}{\mathrm{i}}
   \makeatother
"""
# Sphinx 1.5 provides this to avoid "too deeply nested" LaTeX error
# and usage of "enumitem" LaTeX package is unneeded.
# Value can be increased but do not set it to something such as 2048
# which needlessly would trigger creation of thousands of TeX macros
latex_elements['maxlistdepth'] = '10'
latex_elements['pointsize'] = '11pt'

# Better looking general index in PDF
latex_elements['printindex'] = r'\footnotesize\raggedright\printindex'

# Documents to append as an appendix to all manuals.
latex_appendices = []

# If false, no module index is generated.
latex_use_modindex = True

latex_toplevel_sectioning = 'chapter'

# Including additional LaTeX files if needed
latex_additional_files = ['_static/klassez_logo.png']

def linkcode_resolve(domain, info):
    """ Returns the URL for a given function or class """
    if domain != 'py' or not info['module']:
        return None

    try:
        module = __import__(info['module'], fromlist=[''])
        obj = module
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
    except Exception:
        return None

    if isinstance(obj, property):
        obj = obj.fget
    # If it is a descriptior (classmethod, staticmethod, etc.)
    elif hasattr(obj, '__func__'):
        obj = obj.__func__

    # Special cases that are not inspectable
    if isinstance(obj, (dict, list, tuple, set, str, int, float, bool, type(None))):
        # Return only the module type
        filename = info['module'].replace('.', '/')
        return f'https://github.com/MetallerTM/klassez/blob/{branch}/{filename}.py'

    source_file = inspect.getsourcefile(obj)
    if source_file is None:
        return None

    # Convert absolute path in relative with respect to the root
    rel_path = Path(source_file).relative_to(Path(__file__).parent.parent.parent)

    _, lineno = inspect.getsourcelines(obj)

    lineno_str = f'#L{lineno}' if lineno else ''

    return f'https://github.com/MetallerTM/klassez/blob/{branch}/{rel_path}{lineno_str}'
