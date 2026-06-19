"""
Feature encoder registry.

Register encoders with @register_encoder and instantiate via get_encoder(name, **params).
YAML key: features.encoder (default: rna_g4_enriched).
"""

from __future__ import annotations

from typing import Callable, Type

from features.base import FeatureEncoder

_REGISTRY: dict[str, Type[FeatureEncoder]] = {}


def register_encoder(name: str) -> Callable[[Type[FeatureEncoder]], Type[FeatureEncoder]]:
    """Decorator to register a FeatureEncoder subclass under a string name."""

    def decorator(cls: Type[FeatureEncoder]) -> Type[FeatureEncoder]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_encoder(name: str, **params) -> FeatureEncoder:
    """
    Instantiate a registered feature encoder.

    Args:
        name: Registry key (e.g. rna_g4_enriched).
        **params: Constructor kwargs passed to the encoder.

    Raises:
        KeyError: If name is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown feature encoder {name!r}. Available: {available}")
    return _REGISTRY[name](**params)


def list_encoders() -> list[str]:
    """Return registered encoder names."""
    return sorted(_REGISTRY.keys())
