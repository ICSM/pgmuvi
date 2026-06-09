from pgmuvi.gps import PowerLawMean
from pgmuvi.parameter_specs import ParameterDomain, ParameterRole, ParameterScale


def test_power_law_mean_parameter_schema_names():
    schema = PowerLawMean().parameter_schema()

    assert schema.names() == [
        "mean_module.offset",
        "mean_module.weight",
        "mean_module.exponent",
    ]


def test_power_law_mean_parameter_schema_semantics():
    schema = PowerLawMean().parameter_schema()

    offset = schema["mean_module.offset"]
    weight = schema["mean_module.weight"]
    exponent = schema["mean_module.exponent"]

    assert offset.role is ParameterRole.OFFSET
    assert offset.domain is ParameterDomain.FLUX
    assert offset.scale is ParameterScale.LINEAR

    assert weight.role is ParameterRole.AMPLITUDE
    assert weight.domain is ParameterDomain.FLUX
    assert weight.scale is ParameterScale.LINEAR

    assert exponent.role is ParameterRole.SHAPE
    assert exponent.domain is ParameterDomain.DIMENSIONLESS
    assert exponent.scale is ParameterScale.LINEAR


def test_power_law_mean_parameter_schema_accepts_empty_prefix():
    schema = PowerLawMean().parameter_schema(prefix="")

    assert schema.names() == [
        "offset",
        "weight",
        "exponent",
    ]