"""High-level parameter-estimation workflow helpers."""

from __future__ import annotations

from typing import Any

from pgmuvi.parameter_application import ParameterEstimateApplicator
from pgmuvi.parameter_builders import ParameterEstimateBuilder
from pgmuvi.parameter_context import ParameterEstimationContext
from pgmuvi.parameter_estimates import ParameterEstimateCollection
from pgmuvi.parameter_specs import ParameterSpecCollection


def get_parameter_schema(model: Any) -> ParameterSpecCollection | None:
    """Return a model parameter schema if one is available."""
    schema_attr = getattr(model, "parameter_schema", None)
    if schema_attr is None:
        return None

    return schema_attr()

    return model.parameter_schema()


def build_parameter_estimates(
        model: Any,
        context: ParameterEstimationContext,
        builder: ParameterEstimateBuilder | None = None,
        ) -> ParameterEstimateCollection | None:
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


def apply_parameter_estimates(
        model: Any,
        estimates: ParameterEstimateCollection | None,
        applicator: ParameterEstimateApplicator | None = None,
        ) -> dict[str, dict[str, bool]] | None:
    """Apply parameter estimates to a model if estimates are available."""
    if estimates is None:
        return None

    if applicator is None:
        applicator = ParameterEstimateApplicator()

    return applicator.apply(
        model=model,
        estimates=estimates,
    )
