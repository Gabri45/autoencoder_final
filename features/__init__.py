"""Feature encoding plugin system for molecular trajectories."""

from features.base import FeatureEncoder, FeatureMeta
from features.registry import get_encoder, register_encoder

__all__ = ["FeatureEncoder", "FeatureMeta", "get_encoder", "register_encoder"]
