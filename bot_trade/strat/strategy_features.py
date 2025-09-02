from __future__ import annotations

"""Strategy feature registry stub.

Plugins can register additional feature builders by updating
``FEATURE_REGISTRY`` at import time. Training code will later select a
builder via :func:`get_feature_builder`.
"""

from typing import Any, Callable, Dict


def build_features(df_like, cfg) -> Dict[str, Any]:
    """Baseline feature builder returning an empty feature set."""
    return {}

FEATURE_REGISTRY: Dict[str, Callable[[Any, Dict[str, Any]], Dict[str, Any]]] = {
    "baseline": build_features,
}


def get_feature_builder(name: str) -> Callable[[Any, Dict[str, Any]], Dict[str, Any]]:
    """Select a feature builder by name from ``FEATURE_REGISTRY``."""
    return FEATURE_REGISTRY.get(name, build_features)
