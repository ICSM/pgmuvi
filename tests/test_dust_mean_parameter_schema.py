from pgmuvi.gps import DustMean
from pgmuvi.parameter_specs import ParameterDomain, ParameterRole, ParameterScale


def test_dust_mean_parameter_schema_names():
    schema = DustMean().parameter_schema()

    assert schema.names() == [
        "mean_module.offset",
        "mean_module.log_amplitude",
        "mean_module.log_tau",
        "mean_module.log_alpha",
    ]


def test_dust_mean_parameter_schema_semantics():
    schema = DustMean().parameter_schema()

    offset = schema["mean_module.offset"]
    amplitude = schema["mean_module.log_amplitude"]
    tau = schema["mean_module.log_tau"]
    alpha = schema["mean_module.log_alpha"]

    assert offset.role is ParameterRole.OFFSET
    assert offset.domain is ParameterDomain.FLUX
    assert offset.scale is ParameterScale.LINEAR

    assert amplitude.role is ParameterRole.AMPLITUDE
    assert amplitude.domain is ParameterDomain.FLUX
    assert amplitude.scale is ParameterScale.LOG

    assert tau.role is ParameterRole.SHAPE
    assert tau.domain is ParameterDomain.DIMENSIONLESS
    assert tau.scale is ParameterScale.LOG

    assert alpha.role is ParameterRole.SHAPE
    assert alpha.domain is ParameterDomain.DIMENSIONLESS
    assert alpha.scale is ParameterScale.LOG


def test_dust_mean_parameter_schema_accepts_empty_prefix():
    schema = DustMean().parameter_schema(prefix="")

    assert schema.names() == [
        "offset",
        "log_amplitude",
        "log_tau",
        "log_alpha",
    ]
