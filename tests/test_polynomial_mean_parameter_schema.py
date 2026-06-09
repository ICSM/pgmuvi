from pgmuvi.gps import CustomLinearConstantMean, CustomQuadConstantMean
from pgmuvi.parameter_specs import ParameterDomain, ParameterRole, ParameterScale


def test_custom_linear_constant_mean_schema_names():
    schema = CustomLinearConstantMean().parameter_schema()

    assert schema.names() == [
        "mean_module.wavelength_slope",
        "mean_module.bias",
    ]


def test_custom_linear_constant_mean_schema_semantics():
    schema = CustomLinearConstantMean().parameter_schema()

    slope = schema["mean_module.wavelength_slope"]
    bias = schema["mean_module.bias"]

    assert slope.role is ParameterRole.WAVELENGTH_SCALE
    assert slope.domain is ParameterDomain.FLUX
    assert slope.scale is ParameterScale.LINEAR

    assert bias.role is ParameterRole.OFFSET
    assert bias.domain is ParameterDomain.FLUX
    assert bias.scale is ParameterScale.LINEAR


def test_custom_quad_constant_mean_schema_names():
    schema = CustomQuadConstantMean().parameter_schema()

    assert schema.names() == [
        "mean_module.weights",
        "mean_module.bias",
    ]


def test_custom_quad_constant_mean_schema_semantics():
    schema = CustomQuadConstantMean().parameter_schema()

    weights = schema["mean_module.weights"]
    bias = schema["mean_module.bias"]

    assert weights.role is ParameterRole.WAVELENGTH_SCALE
    assert weights.domain is ParameterDomain.FLUX
    assert weights.scale is ParameterScale.LINEAR
    assert weights.shape == (2,)

    assert bias.role is ParameterRole.OFFSET
    assert bias.domain is ParameterDomain.FLUX
    assert bias.scale is ParameterScale.LINEAR


def test_polynomial_mean_schemas_accept_empty_prefix():
    linear = CustomLinearConstantMean().parameter_schema(prefix="")
    quadratic = CustomQuadConstantMean().parameter_schema(prefix="")

    assert linear.names() == ["wavelength_slope", "bias"]
    assert quadratic.names() == ["weights", "bias"]