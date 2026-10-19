"""optiforge: a modular optimisation platform with an autonomous operator/algorithm development layer."""
__version__ = "0.1.0"

from .core import *  # noqa: F401,F403
from .solvers import SOLVERS, get_solver  # noqa: F401
from .api import plan, solve  # noqa: F401,E402
