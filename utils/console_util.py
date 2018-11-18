import numpy as np

from utils.misc_util import zipsame


def cell(x, width):
    """Format a tabular cell to the specified width"""
    if isinstance(x, np.ndarray):
        assert x.ndim == 0
        x = x.item()
    rep = "{:G}".format(x) if isinstance(x, float) else str(x)
    return rep + (' ' * (width - len(rep)))


def columnize(names, tuples, widths, indent=2):
    """Generate and return the content of table
    (w/o logging or printing anything)

    Args:
        width (int): Width of each cell in the table
        indent (int): Indentation spacing prepended to every row in the table
    """
    indent_space = indent * ' '
    # Add row containing the names
    table = indent_space + " | ".join(cell(name, width) for name, width in zipsame(names, widths))
    table_width = len(table)
    # Add header hline
    table += '\n' + indent_space + ('-' * table_width)
    for tuple_ in tuples:
        # Add a new row
        table += '\n' + indent_space
        table += " | ".join(cell(value, width) for value, width in zipsame(tuple_, widths))
    # Add closing hline
    table += '\n' + indent_space + ('-' * table_width)
    return table
