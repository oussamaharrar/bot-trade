import types
import sys

from bot_trade.config import device


def test_device_selector(monkeypatch):
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_name=lambda idx: f"GPU{idx}",
        get_device_properties=lambda idx: types.SimpleNamespace(total_memory=2 * 1024 ** 3),
        set_device=lambda idx: None,
        mem_get_info=lambda: (1024 ** 3, 2 * 1024 ** 3),
    )
    torch_mod = types.SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    devs = device.list_devices()
    labels = [d["label"] for d in devs]
    assert labels[0] == "CPU"
    assert any("CUDA:0" in l for l in labels)
