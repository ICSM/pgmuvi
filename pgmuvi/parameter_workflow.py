"""High-level parameter-estimation workflow helpers."""

from __future__ import annotations


def get_parameter_schema(model):
    """Return a model parameter schema if one is available."""
    if not hasattr(model, "parameter_schema"):
        return None

    return model.parameter_schema()