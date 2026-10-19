import numpy as np
import pytest

from optiforge import Problem, Space, Real, Integer, Binary, Categorical, Permutation
from optiforge.isolation.sandbox import probe


def sphere_problem(dim=3, lo=-5.0, hi=5.0, **kw):
    sp = Space([Real(f"x{i}", lo, hi) for i in range(dim)])
    return Problem(sp, lambda x: float(np.sum(np.asarray(x) ** 2)), name="sphere-test",
                   gradient=lambda x: 2 * np.asarray(x), **kw)


@pytest.fixture
def sphere():
    return sphere_problem()


@pytest.fixture(scope="session")
def isolation():
    return probe()


@pytest.fixture
def needs_isolation(isolation):
    if not isolation.available:
        pytest.skip(f"OS isolation unavailable: {isolation.reason}")
    return isolation
