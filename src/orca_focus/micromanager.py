"""Micro-Manager integration for orca-focus.

Reads frames from Micro-Manager's live acquisition buffer (owned by the
Micro-Manager GUI) and optionally controls the Z stage through MMCore.

Requires either:
  - pycromanager: pip install pycromanager
  - MMCorePy (bundled with Micro-Manager)

Typical usage:
    # Connect to a running Micro-Manager instance
    source = create_micromanager_frame_source()
    camera = HamamatsuOrcaCamera(frame_source=source, control_source_lifecycle=False)

    # Or use the MMCore stage adapter instead of direct MCL control
    stage = MicroManagerStage(core=source.core)
"""

from __future__ import annotations

import time
from typing import Any

from .dcam import _to_image_2d
from .interfaces import Image2D, StageInterface


class MicroManagerFrameSource:
    """Frame source that reads from Micro-Manager's live circular buffer.

    This does NOT control the camera â€” Micro-Manager handles acquisition,
    exposure, ROI cropping, etc. We just grab the latest frame.
    """

    def __init__(self, core: Any) -> None:
        self._core = core
        self._last_image: Image2D | None = None
        self._last_ts: float = 0.0

    @property
    def core(self) -> Any:
        return self._core

    def start(self) -> None:
        # No-op: Micro-Manager controls acquisition
        pass

    def stop(self) -> None:
        # No-op: Micro-Manager controls acquisition
        pass

    def __call__(self) -> tuple[Image2D, float]:
        core = self._core

        # Try circular buffer first (live mode), fall back to snap.
        if _is_live(core):
            # If the buffer hasn't advanced since our last read, return the
            # previous frame with the same timestamp so the controller's
            # duplicate-frame guard can skip it.
            if self._last_image is not None and not _buffer_has_new_frame(core):
                return self._last_image, self._last_ts
            frame = _get_last_image(core)
        else:
            core.snapImage()
            frame = core.getImage()

        image = _to_image_2d(frame)
        self._last_image = image
        self._last_ts = time.time()
        return image, self._last_ts


class MicroManagerStage(StageInterface):
    """Z stage adapter that goes through Micro-Manager's device layer.

    Use this if your stage is configured as a Micro-Manager device rather
    than being controlled directly via MCL DLL/wrapper.
    """

    def __init__(
        self,
        core: Any,
        z_stage_name: str | None = None,
    ) -> None:
        self._core = core
        # Use the default focus device if no name given
        self._z_name = z_stage_name or core.getFocusDevice()

    def get_z_um(self) -> float:
        return float(self._core.getPosition(self._z_name))

    def move_z_um(self, target_z_um: float) -> None:
        self._core.setPosition(self._z_name, target_z_um)
        self._core.waitForDevice(self._z_name)


def _is_live(core: Any) -> bool:
    """Check if Micro-Manager is currently in live/continuous acquisition."""
    try:
        return bool(core.isSequenceRunning())
    except Exception:
        return False


def _buffer_has_new_frame(core: Any) -> bool:
    """Check if the circular buffer has received a new frame since last read.

    Falls back to True (assume new) if the API is unavailable, so the caller
    always gets a frame rather than starving.
    """
    try:
        return int(core.getRemainingImageCount()) > 0
    except Exception:
        return True


def _get_last_image(core: Any) -> Any:
    """Grab the most recent frame from the circular buffer."""
    try:
        return core.getLastImage()
    except Exception:
        # Buffer might be empty momentarily â€” snap as fallback
        core.snapImage()
        return core.getImage()


def _try_create_pycromanager_core(host: str, port: int) -> Any | None:
    """Return a working pycromanager Core, trying remote then local bridge."""
    try:
        from pycromanager import Core
    except Exception:
        return None

    # 1) explicit host/port bridge (remote or local TCP)
    try:
        core = Core(host=host, port=port)
        core.getVersionInfo()
        return core
    except Exception:
        pass

    # 2) default local bridge parameters (common desktop setup)
    try:
        core = Core()
        core.getVersionInfo()
        return core
    except Exception:
        return None


def create_micromanager_frame_source(
    *,
    host: str = "localhost",
    port: int = 4827,
    core: Any | None = None,
) -> MicroManagerFrameSource:
    """Connect to a running Micro-Manager instance.

    Tries connection methods in order:
    1. Existing core object passed directly
    2. pycromanager Core(host, port)
    3. pycromanager Core() with default bridge settings
    4. MMCorePy (direct, if running in the same process)
    """

    if core is not None:
        return MicroManagerFrameSource(core=core)

    mm_core = _try_create_pycromanager_core(host=host, port=port)
    if mm_core is not None:
        return MicroManagerFrameSource(core=mm_core)

    # Fall back to MMCorePy (in-process)
    try:
        import MMCorePy

        mm_core = MMCorePy.CMMCore()
        return MicroManagerFrameSource(core=mm_core)
    except Exception:
        pass

    raise RuntimeError(
        "Could not connect to Micro-Manager. Make sure one of these is true:\n"
        "  1. Micro-Manager is running with pycromanager bridge enabled\n"
        "     (Tools -> Options -> Run server on port), or\n"
        "  2. pycromanager can attach via its default local bridge (Core()), or\n"
        "  3. MMCorePy is available in the Python environment.\n"
        f"  Tried pycromanager at {host}:{port} and default Core()"
    )
