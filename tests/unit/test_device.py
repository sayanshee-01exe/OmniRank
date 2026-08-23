"""Device resolution. The one rule: nothing may implicitly assume CUDA."""

from __future__ import annotations

import pytest

from omnirank.core.device import DeviceType, resolve_device
from omnirank.core.exceptions import UnsupportedDeviceError


class TestResolution:
    def test_auto_returns_a_concrete_device(self):
        assert resolve_device(DeviceType.AUTO) in {"cpu", "mps"}

    def test_auto_never_selects_cuda(self, monkeypatch):
        """Even on a CUDA host, `auto` must not pick it."""
        monkeypatch.setattr("omnirank.core.device._torch_backend_available", lambda device: True)
        assert resolve_device(DeviceType.AUTO) == "mps"

    def test_auto_falls_back_to_cpu_without_mps(self, monkeypatch):
        monkeypatch.setattr("omnirank.core.device._torch_backend_available", lambda device: False)
        assert resolve_device(DeviceType.AUTO) == "cpu"

    def test_cpu_always_resolves(self):
        assert resolve_device(DeviceType.CPU) == "cpu"

    def test_string_input_is_accepted(self):
        assert resolve_device("cpu") == "cpu"

    def test_unknown_device_name_is_rejected(self):
        with pytest.raises(ValueError):
            resolve_device("tpu")


class TestCuda:
    def test_explicit_cuda_is_refused_unless_allowed(self):
        with pytest.raises(UnsupportedDeviceError) as exc:
            resolve_device(DeviceType.CUDA)
        assert "allow_cuda" in str(exc.value)

    def test_allowed_cuda_still_requires_a_visible_device(self, monkeypatch):
        monkeypatch.setattr("omnirank.core.device._torch_backend_available", lambda device: False)
        with pytest.raises(UnsupportedDeviceError) as exc:
            resolve_device(DeviceType.CUDA, allow_cuda=True)
        assert "no CUDA device" in str(exc.value)

    def test_allowed_and_present_cuda_resolves(self, monkeypatch):
        monkeypatch.setattr("omnirank.core.device._torch_backend_available", lambda device: True)
        assert resolve_device(DeviceType.CUDA, allow_cuda=True) == "cuda"


class TestMps:
    def test_explicit_mps_without_support_gives_actionable_advice(self, monkeypatch):
        monkeypatch.setattr("omnirank.core.device._torch_backend_available", lambda device: False)
        with pytest.raises(UnsupportedDeviceError) as exc:
            resolve_device(DeviceType.MPS)
        assert "device.preferred" in str(exc.value)


class TestWithoutTorch:
    def test_probe_returns_false_when_torch_is_absent(self):
        """Phase 1 installs no torch; the probe must not explode."""
        from omnirank.core.device import _torch_backend_available

        assert isinstance(_torch_backend_available(DeviceType.MPS), bool)

    def test_config_resolution_works_without_torch(self, config):
        assert config.device.resolve() in {"cpu", "mps"}
