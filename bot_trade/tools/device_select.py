from __future__ import annotations
"""Device enumeration helpers."""

from typing import List, Dict

from bot_trade.config.device import list_devices


def get_devices() -> List[Dict[str, str]]:
    """Compatibility wrapper around :func:`config.device.list_devices`."""
    devices = []
    for d in list_devices():
        devices.append({"label": d["label"], "value": d["id"]})
    return devices


__all__ = ["get_devices"]
