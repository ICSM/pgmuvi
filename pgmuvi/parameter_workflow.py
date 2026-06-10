"""High-level parameter-estimation workflow helpers."""

from __future__ import annotations

from pgmuvi.parameter_application import ParameterEstimateApplicator
from pgmuvi.parameter_builders import ParameterEstimateBuilder


def get_parameter_schema(model):
    """Return a model parameter schema if one is available."""
    if not hasattr(model, "parameter_schema"):
        return None

    return model.parameter_schema()


def build_parameter_estimates(model, context, builder=None):
    """Build parameter estimates for a model if a schema is available."""
    schema = get_parameter_schema(model)

    if schema is None:
        return None

    if builder is None:
        builder = ParameterEstimateBuilder()

    return builder.build(
        schema=schema,
        context=context,
    )


def apply_parameter_estimates(model, estimates, applicator=None):
    """Apply parameter estimates to a model if estimates are available."""
    if estimates is None:
        return None

    if applicator is None:
        applicator = ParameterEstimateApplicator()

    return applicator.apply(
        model=model,
        estimates=estimates,
    )