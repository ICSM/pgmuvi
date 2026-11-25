"""Shared pytest fixtures for the pgmuvi test suite."""

import json
import math
import os
import random
from pathlib import Path

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


# --- golden snapshots (roadmap 1.4) -------------------------------------------

GOLDEN_DIR = Path(__file__).parent / "golden"


def _jsonable(value):
    """Convert tensors, arrays, numpy scalars and tuples to JSON-safe values."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _assert_close(expected, actual, rtol, atol, path):
    """Recursively compare two JSON-like values; numbers use rtol/atol."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected a dict"
        assert sorted(expected) == sorted(actual), (
            f"{path}: keys differ\n expected {sorted(expected)}\n"
            f" actual   {sorted(actual)}"
        )
        for key in expected:
            _assert_close(expected[key], actual[key], rtol, atol, f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), (
            f"{path}: length {len(actual)} != {len(expected)}"
        )
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            _assert_close(e, a, rtol, atol, f"{path}[{i}]")
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert expected == actual, f"{path}: {actual!r} != {expected!r}"
    else:
        assert isinstance(actual, (int, float)), f"{path}: {actual!r} is not a number"
        if math.isnan(expected):
            assert math.isnan(actual), f"{path}: {actual!r} != nan"
        else:
            assert math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), (
                f"{path}: {actual!r} != {expected!r} (rtol={rtol}, atol={atol})"
            )


@pytest.fixture
def golden():
    """Compare a value with ``tests/golden/<name>.json``.

    Returns a callable ``check(name, value, rtol=1e-5, atol=1e-8)``. Set
    ``PGMUVI_UPDATE_GOLDEN=1`` to (re)write the files instead of comparing;
    a missing file is written and the test fails once, so that an update
    is always a visible diff.
    """

    def check(name, value, rtol=1e-5, atol=1e-8):
        path = GOLDEN_DIR / f"{name}.json"
        value = _jsonable(value)
        update = os.environ.get("PGMUVI_UPDATE_GOLDEN") == "1"
        if update or not path.exists():
            GOLDEN_DIR.mkdir(exist_ok=True)
            path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
            if not update:
                pytest.fail(
                    f"golden file {path.name} was missing and has been written; re-run"
                )
            return
        _assert_close(json.loads(path.read_text()), value, rtol, atol, name)

    return check
