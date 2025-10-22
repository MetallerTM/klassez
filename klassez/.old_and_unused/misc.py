#! /usr/bin/env python3

def procpar(txt):
    """
    Takes as input the path of a file containing a "key" in the first column and a "value" in the second column.
    Returns the correspondant dictionary.


    Parameters
    ----------
    txt : str
        Path to a file that contains "key" in first column and "value" in the second

    Returns
    -------
    procpars : dict
        Dictionary of shape ``key : value``

    """
    fyle = open(txt).readlines()
    procpars = {}
    for line in fyle:
        if line[0] == '#':
            continue    # Skip comments
        string = line.split('\t')
        procpars[string[0]] = float(string[1].strip())
    return procpars


def write_help(request, file=None):
    """
    Gets the documentation of request, and tries to save it in a text file.

    Parameters
    ----------
    request : function or class or package
        Whatever you need documentation of
    file : str or None or False
        Name of the output documentation file. If it is None, a default name is given. If it is False, the output is printed on screen.
    """
    import pydoc
    if file is None:
        file = request.__name__+'.hlp'
    hlp_text = pydoc.render_doc(request, renderer=pydoc.plaintext)
    if bool(file):
        with open(file, 'w') as F:
            F.write(hlp_text)
    else:
        print(hlp_text)
