from __future__ import annotations
"""Device enumeration helpers."""

from typing import List, Dict


def get_devices() -> List[Dict[str, str]]:
    devices = [{"label": "CPU", "value": "cpu"}]
    try:  # torch optional
        import torch

        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                vram_gb = props.total_memory / (1024 ** 3)
                label = f"GPU {idx} (VRAM {vram_gb:.1f} GB)"
                devices.append({"label": label, "value": f"cuda:{idx}"})
    except Exception:  # pragma: no cover - torch may be missing
        pass
    return devices


__all__ = ["get_devices"]
