def in_jupyter():
    # Assert we're in a jupyter notebook environment. Not the cleanest, but gets the job done.
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter

if not in_jupyter():
    raise RuntimeError("Jupyter environment not found. SchedulerPlotter can only be used with Jupyter.")

from ._plotter import Plotter
