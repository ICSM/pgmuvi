"""Golden snapshot of every model name ``Lightcurve.set_model`` accepts.

For each name the model class and the ``state_dict`` keys and shapes of the
model and likelihood are recorded in ``tests/golden/registry_state_dicts.json``
(roadmap 1.4). A name that raises on construction is recorded by exception
type, so a later change in either direction is visible in the diff.

The dispatch dicts live inside ``Lightcurve.set_model`` and are not
importable, so the names are listed here; ``test_unknown_name_raises`` and the
snapshot keep the list honest, and
``test_model_names_match_set_model`` compares it with the source.
"""

import ast
import inspect
import textwrap
import warnings

import pytest

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.synthetic import make_chromatic_sinusoid_2d, make_simple_sinusoid_1d

MODEL_NAMES = [
    "1D",
    "2D",
    "1DLinear",
    "2DLinear",
    "2DPowerLaw",
    "2DDust",
    "1DSKI",
    "2DSKI",
    "1DLinearSKI",
    "2DLinearSKI",
    "2DPowerLawSKI",
    "2DDustSKI",
    "1DQuasiPeriodic",
    "1DMatern",
    "1DPeriodicStochastic",
    "1DLinearQuasiPeriodic",
    "2DSeparable",
    "2DAchromatic",
    "2DWavelengthDependent",
    "2DDustMean",
    "2DPowerLawMean",
]


def _shapes(module):
    return [[k, list(v.shape)] for k, v in sorted(module.state_dict().items())]


def _snapshot(name):
    if name.startswith("1D"):
        lc = make_simple_sinusoid_1d(n_obs=30, seed=0)
    else:
        lc = make_chromatic_sinusoid_2d(n_per_band=15, seed=0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lc.set_model(name, num_mixtures=2)
    except Exception as exc:  # recorded, not skipped: a change must be visible
        return {"raises": type(exc).__name__}
    return {
        "model_class": type(lc.model).__name__,
        "model_keys": _shapes(lc.model),
        "likelihood_class": type(lc.likelihood).__name__,
        "likelihood_keys": _shapes(lc.likelihood),
    }


def test_registry_state_dicts(golden):
    snapshot = {name: _snapshot(name) for name in MODEL_NAMES}
    golden("registry_state_dicts", snapshot)


def test_model_names_match_set_model():
    """The typed ``MODEL_NAMES`` equals the keys of the ``model_dic_*`` dicts."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(Lightcurve.set_model)))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(t.startswith("model_dic") for t in targets):
                found.update(
                    k.value
                    for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                )
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.value, ast.Dict)
            and isinstance(node.target, ast.Name)
            and node.target.id.startswith("model_dic")
        ):
            found.update(
                k.value for k in node.value.keys if isinstance(k, ast.Constant)
            )
    assert found == set(MODEL_NAMES), (
        f"AST names {sorted(found)}\n typed names {sorted(MODEL_NAMES)}"
    )


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="1DLinearSKI cannot be constructed: num_dims is missing for the SKI "
    "grid (roadmap 1.12)",
)
def test_1dlinearski_constructs():
    lc = make_simple_sinusoid_1d(n_obs=30, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lc.set_model("1DLinearSKI", num_mixtures=2)


def test_unknown_name_raises():
    lc = make_simple_sinusoid_1d(n_obs=20, seed=0)
    with pytest.raises(ValueError):
        lc.set_model("NoSuchModel", num_mixtures=2)
