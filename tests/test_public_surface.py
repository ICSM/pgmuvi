"""Snapshot of the public API, so phase-2 moves must leave aliases (roadmap 1.4).

For each module, the sorted public names (no leading underscore) are recorded
with ``own`` if the object's ``__module__`` is that module and ``re-export``
otherwise (``var`` for objects without ``__module__``, such as constants).
Names that move are expected to stay importable from the old module as shims,
so the names must not disappear; an intentional addition or removal is a
visible diff in ``tests/golden/public_surface.json``.

The ``pgmuvi.models`` deprecation shim is covered by
``tests/test_package_hygiene.py::test_models_shim_warns_and_reexports``.
"""

import importlib
import inspect

import pytest

import pgmuvi
from pgmuvi.lightcurve import Lightcurve

MODULES = sorted(
    {
        *(f"pgmuvi.{m}" for m in pgmuvi.__all__ if m != "Lightcurve"),
        "pgmuvi.lightcurve",
        "pgmuvi.gps",
        "pgmuvi.kernels",
        "pgmuvi.priors",
        "pgmuvi.constraints",
        "pgmuvi.trainers",
        "pgmuvi.synthetic",
        "pgmuvi.initialization",
        "pgmuvi.multiband_ls_significance",
        "pgmuvi.preprocess",
        "pgmuvi.preprocess.quality",
        "pgmuvi.preprocess.variability",
        "pgmuvi.datasets",
    }
)


def _kind(modname, obj):
    owner = getattr(obj, "__module__", None)
    if owner is None:
        return "var"
    return "own" if owner == modname else "re-export"


def _module_surface(modname):
    mod = importlib.import_module(modname)
    return {
        name: _kind(modname, obj)
        for name, obj in sorted(vars(mod).items())
        if not name.startswith("_") and not inspect.ismodule(obj)
    }


def _lightcurve_members():
    return sorted(
        name
        for name, _ in inspect.getmembers(Lightcurve)
        if not name.startswith("_")
    )


def test_public_surface(golden):
    snapshot = {name: _module_surface(name) for name in MODULES}
    snapshot["Lightcurve members"] = _lightcurve_members()
    golden("public_surface", snapshot)


@pytest.mark.parametrize("modname", MODULES)
def test_module_importable(modname):
    importlib.import_module(modname)
