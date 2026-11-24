"""Shared pytest fixtures for the pgmuvi test suite."""

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def seed_rngs():
    """Seed global random number generators and restore the torch dtype.

    Seeds ``random``, ``numpy.random`` and ``torch`` with 0 before every
    test, so results do not depend on test order. After the test, the torch
    default dtype is reset to its value before the test, so a test that
    changes it cannot leak into later tests.
    """
    dtype = torch.get_default_dtype()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(dtype)
