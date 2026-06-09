import numpy as np
import pytest

from pgmuvi.parameter_specs import (
    ParameterDomain,
    ParameterRole,
    ParameterScale,
    ParameterSpec,
    ParameterSpecCollection,
    GuessStrategy,
    ConstraintStrategy,
)


def test_parameter_spec_accepts_physical_space_values():
    spec = ParameterSpec(
        name="mean_module.log_tau",
        role=ParameterRole.TIMESCALE,
        domain=ParameterDomain.TIME,
        scale=ParameterScale.LOG,
        initial_value=250.0,
        constraint=(50.0, 2000.0),
        units="days",
        description="Characteristic variability timescale in physical time units.",
    )

    spec.validate()

    assert spec.initial_value == 250.0
    assert spec.constraint == (50.0, 2000.0)
    assert spec.scale is ParameterScale.LOG
    assert spec.description is not None


def test_parameter_spec_rejects_initial_value_outside_constraint():
    spec = ParameterSpec(
        name="mean_module.log_amplitude",
        role=ParameterRole.AMPLITUDE,
        domain=ParameterDomain.FLUX,
        scale=ParameterScale.LOG,
        initial_value=7.3,
        constraint=(10.0, 100.0),
    )

    with pytest.raises(ValueError, match="outside its constraint"):
        spec.validate()


def test_parameter_spec_rejects_invalid_constraint():
    spec = ParameterSpec(
        name="bad_parameter",
        role=ParameterRole.OTHER,
        domain=ParameterDomain.OTHER,
        initial_value=1.0,
        constraint=(2.0, 1.0),
    )

    with pytest.raises(ValueError, match="lower bound"):
        spec.validate()


def test_parameter_spec_validates_array_shape():
    spec = ParameterSpec(
        name="covar_module.mixture_means",
        role=ParameterRole.FREQUENCY,
        domain=ParameterDomain.FREQUENCY,
        initial_value=np.array([0.01, 0.02, 0.03]),
        constraint=(np.array([0.005, 0.005, 0.005]), np.array([0.05, 0.05, 0.05])),
        shape=(3,),
    )

    spec.validate()


def test_parameter_spec_rejects_wrong_array_shape():
    spec = ParameterSpec(
        name="covar_module.mixture_means",
        role=ParameterRole.FREQUENCY,
        domain=ParameterDomain.FREQUENCY,
        initial_value=np.array([0.01, 0.02]),
        shape=(3,),
    )

    with pytest.raises(ValueError, match="Shape mismatch"):
        spec.validate()


def test_parameter_spec_collection_rejects_duplicate_names():
    spec = ParameterSpec(
        name="same",
        role=ParameterRole.OTHER,
        domain=ParameterDomain.OTHER,
    )

    with pytest.raises(ValueError, match="duplicate"):
        ParameterSpecCollection([spec, spec])


def test_parameter_spec_collection_lookup_and_add():
    collection = ParameterSpecCollection()

    spec = ParameterSpec(
        name="mean_module.offset",
        role=ParameterRole.OFFSET,
        domain=ParameterDomain.FLUX,
        initial_value=100.0,
        constraint=(-1000.0, 1000.0),
    )

    collection.add(spec)

    assert "mean_module.offset" in collection
    assert collection["mean_module.offset"] is spec
    assert collection.names() == ["mean_module.offset"]


def test_parameter_spec_accepts_strategy_metadata():
    spec = ParameterSpec(
        name="mean_module.log_tau",
        role=ParameterRole.TIMESCALE,
        domain=ParameterDomain.TIME,
        scale=ParameterScale.LOG,
        guess_strategy=GuessStrategy.VARIABILITY_TIMESCALE,
        constraint_strategy=ConstraintStrategy.VARIABILITY_TIMESCALE,
    )

    assert spec.guess_strategy is GuessStrategy.VARIABILITY_TIMESCALE
    assert (
        spec.constraint_strategy
        is ConstraintStrategy.VARIABILITY_TIMESCALE
    )
