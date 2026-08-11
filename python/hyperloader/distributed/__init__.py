"""Distributed topology and map-placement contracts."""

from .map import MapPlacement, build_map_placement, validate_elastic_restore

__all__ = ["MapPlacement", "build_map_placement", "validate_elastic_restore"]
