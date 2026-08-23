"""Compute device resolution.

OmniRank's primary development target is Apple Silicon: CPU and MPS, no CUDA.
Every module that needs a device asks this module rather than calling
``torch.cuda.is_available()`` directly, so that no code path can hard-assume a
GPU that is not there.

Design notes:

* ``torch`` is imported lazily, inside the availability probe. Phase 1 does not
  depend on torch at all, and importing this module must not require it.
* ``auto`` never resolves to ``cuda``. A CUDA host is opt-in via
  ``device.allow_cuda`` plus an explicit ``preferred: cuda``, so that a config
  copied from a cloud GPU box cannot silently change behaviour locally.
"""

from __future__ import annotations

from enum import StrEnum

from omnirank.core.exceptions import UnsupportedDeviceError


class DeviceType(StrEnum):
    """Supported compute targets."""

    AUTO = "auto"
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"


def _torch_backend_available(device: DeviceType) -> bool:
    """Probe a torch backend without making torch a hard dependency."""
    try:
        import torch
    except ImportError:
        # No torch installed: only CPU work is possible, which needs no probe.
        return False

    if device is DeviceType.MPS:
        return bool(torch.backends.mps.is_available())
    if device is DeviceType.CUDA:
        return bool(torch.cuda.is_available())
    return True


def resolve_device(
    preferred: DeviceType | str = DeviceType.AUTO, *, allow_cuda: bool = False
) -> str:
    """Resolve a configured preference to a concrete torch device string.

    Args:
        preferred: Requested device, or ``auto`` to detect.
        allow_cuda: Whether an explicit ``cuda`` request may be honoured. Has no
            effect on ``auto``, which never selects CUDA.

    Returns:
        A concrete device string: ``"cpu"``, ``"mps"``, or ``"cuda"``.

    Raises:
        UnsupportedDeviceError: An explicit request cannot be satisfied here.
    """
    device = DeviceType(preferred)

    if device is DeviceType.AUTO:
        # Prefer MPS when the wheel and hardware support it; otherwise CPU.
        if _torch_backend_available(DeviceType.MPS):
            return DeviceType.MPS.value
        return DeviceType.CPU.value

    if device is DeviceType.CPU:
        return DeviceType.CPU.value

    if device is DeviceType.MPS:
        if not _torch_backend_available(DeviceType.MPS):
            raise UnsupportedDeviceError(
                "MPS was requested but is not available. Install torch with MPS "
                "support on Apple Silicon, or set device.preferred to 'cpu'.",
                requested="mps",
            )
        return DeviceType.MPS.value

    # device is CUDA
    if not allow_cuda:
        raise UnsupportedDeviceError(
            "CUDA was requested but device.allow_cuda is false. Enable it "
            "explicitly on a GPU host; it stays off on the local Apple Silicon "
            "target so configs cannot silently assume a GPU.",
            requested="cuda",
        )
    if not _torch_backend_available(DeviceType.CUDA):
        raise UnsupportedDeviceError(
            "CUDA was requested and permitted, but no CUDA device is visible.",
            requested="cuda",
        )
    return DeviceType.CUDA.value


__all__ = ["DeviceType", "resolve_device"]
