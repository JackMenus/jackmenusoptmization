import sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
from optiforge.core.space import Space, Integer, Real, Binary, Categorical, Permutation
from optiforge.core.problem import Problem
from optiforge.core.constraints import ineq, eq, LinearStructure
from optiforge.core.budget import Budget
from optiforge.solvers import get_solver
def show(r):
    print(r.summary())
    print("  x=", r.x, "certified_optimal=", r.certified_optimal)
    for m in r.messages: print("  msg:", m)
