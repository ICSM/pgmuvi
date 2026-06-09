import numpy as np
import pytest

from pgmuvi.parameter_estimates import (
    ParameterEstimate,
    ParameterEstimateCollection,
)
from pgmuvi.parameter_specs import (
    ParameterDomain,
    ParameterRole,
    ParameterScale,
    ParameterSpec,
)


def test_parameter_estimate_stores_value_and_constraint():
    spec = ParameterSpec(
        name="mean_module.log_amplitude",
        role=ParameterRole.AMPLITUDE,
        domain=ParameterDomain.FLUX,
        scale=ParameterScale.LOG,
    )

    estimate = ParameterEstimate(
        spec=spec,
        value=100.0,
        constraint=(10.0, 1000.0),
        value_source="robust_flux_range",
        constraint_source="robust_flux_range",
    )

    estimate.validate()

    assert estimate.name == "mean_module.log_amplitude"
    assert estimate.value == 100.0
    assert estimate.constraint == (10.0, 1000.0)
    assert estimate.value_source == "robust_flux_range"


def test_parameter_estimate_rejects_invalid_constraint():
    spec = ParameterSpec(
        name="bad_parameter",
        role=ParameterRole.OTHER,
        domain=ParameterDomain.OTHER,
    )

    estimate = ParameterEstimate(
        spec=spec,
        value=1.0,
        constraint=(2.0, 1.0),
    )

    with pytest.raises(ValueError, match="lower bound"):
        estimate.validate()


def test_parameter_estimate_rejects_value_outside_constraint():
    spec = ParameterSpec(
        name="mean_module.offset",
        role=ParameterRole.OFFSET,
        domain=ParameterDomain.FLUX,
    )

    estimate = ParameterEstimate(
        spec=spec,
        value=100.0,
        constraint=(-10.0, 10.0),
    )

    with pytest.raises(ValueError, match="outside its constraint"):
        estimate.validate()


def test_parameter_estimate_validates_shape_from_spec():
    spec = ParameterSpec(
        name="covar_module.mixture_means",
        role=ParameterRole.FREQUENCY,
        domain=ParameterDomain.FREQUENCY,
        shape=(3,),
    )

    estimate = ParameterEstimate(
        spec=spec,
        value=np.array([0.01, 0.02, 0.03]),
        constraint=(
            np.array([0.001, 0.001, 0.001]),
            np.array([0.1, 0.1, 0.1]),
        ),
    )

    estimate.validate()


def test_parameter_estimate_rejects_wrong_shape():
    spec = ParameterSpec(
        name="covar_module.mixture_means",
        role=ParameterRole.FREQUENCY,
        domain=ParameterDomain.FREQUENCY,
        shape=(3,),
    )

    estimate = ParameterEstimate(
        spec=spec,
        value=np.array([0.01, 0.02]),
    )

    with pytest.raises(ValueError, match="Shape mismatch"):
        estimate.validate()


def test_parameter_estimate_collection_rejects_duplicate_names():
    spec = ParameterSpec(
        name="same",
        role=ParameterRole.OTHER,
        domain=ParameterDomain.OTHER,
    )

    estimate = ParameterEstimate(spec=spec)

    with pytest.raises(ValueError, match="duplicate"):
        ParameterEstimateCollection([estimate, estimate])


def test_parameter_estimate_collection_lookup_and_add():
    spec = ParameterSpec(
        name="mean_module.offset",
        role=ParameterRole.OFFSET,
        domain=ParameterDomain.FLUX,
    )

    estimate = ParameterEstimate(
        spec=spec,
        value=100.0,
        constraint=(-1000.0, 1000.0),
    )

    collection = ParameterEstimateCollection()
    collection.add(estimate)

    assert "mean_module.offset" in collection
    assert collection["mean_module.offset"] is estimate
    assert collection.names() == ["mean_module.offset"]
    assert collection.as_dict()["mean_module.offset"] is estimate