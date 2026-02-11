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

from typing import Any

from .dcam import _to_image_2d
from .interfaces import Image2D, StageInterface


class MicroManagerFrameSource:
    """Frame source that reads from Micro-Manager's live circular buffer.

    This does NOT control the camera — Micro-Manager handles acquisition,
    exposure, ROI cropping, etc. We just grab the latest frame.
    """

    def __init__(self, core: Any, *, allow_snap_fallback: bool = False) -> None:
        self._core = core
        self._allow_snap_fallback = allow_snap_fallback
        self._last_image: Image2D | None = None
        self._last_ts: float = 0.0
        self._last_frame_token: int | float | str | None = None

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

        if _is_live(core):
            token = _get_frame_token(core)
            if self._last_image is not None and token is not None and token == self._last_frame_token:
                return self._last_image, self._last_ts

            frame = _get_last_image(core)
            timestamp_s = _extract_frame_timestamp_s(core, frame)
            image = _to_image_2d(_extract_image_payload(frame))

            self._last_frame_token = token
            self._last_image = image
            self._last_ts = timestamp_s
            return image, timestamp_s

        if not self._allow_snap_fallback:
            raise RuntimeError(
                "Micro-Manager sequence is not running. Enable Live mode in Micro-Manager "
                "or construct the frame source with allow_snap_fallback=True."
            )

        core.snapImage()
        frame = core.getImage()
        image = _to_image_2d(frame)
        timestamp_s = self._last_ts + 1e-6 if self._last_ts > 0 else 1e-6
        self._last_image = image
        self._last_ts = timestamp_s
        self._last_frame_token = None
        return image, timestamp_s


class MicroManagerStage(StageInterface):
    """Z stage adapter that goes through Micro-Manager's device layer.

    Use this if your stage is configured as a Micro-Manager device rather
    than being controlled directly via MCL DLL/wrapper.
    """

    def __init__(
        self,
        core: Any,
        z_stage_name: str | None = None,
        wait_for_device: bool = True,
    ) -> None:
        self._core = core
        self._wait_for_device = wait_for_device
        # Use the default focus device if no name given
        self._z_name = z_stage_name or core.getFocusDevice()

    def get_z_um(self) -> float:
        return float(self._core.getPosition(self._z_name))

    def move_z_um(self, target_z_um: float) -> None:
        self._core.setPosition(self._z_name, target_z_um)
        if self._wait_for_device:
            self._core.waitForDevice(self._z_name)


def _is_live(core: Any) -> bool:
    """Check if Micro-Manager is currently in live/continuous acquisition."""
    try:
        return bool(core.isSequenceRunning())
    except Exception:
        return False


def _get_frame_token(core: Any) -> int | float | str | None:
    """Best-effort monotonically changing frame token from MMCore.

    Preferred order:
    1) getLastImageTimeStamp() when available (acquisition-tied)
    2) getImageCount() / getRemainingImageCount() as weaker fallbacks
    """
    for attr in ("getLastImageTimeStamp", "getImageCount"):
        fn = getattr(core, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass

    fn = getattr(core, "getRemainingImageCount", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            pass

    return None


def _extract_image_payload(frame: Any) -> Any:
    """Extract image pixel payload from tagged-image style containers."""
    pix = getattr(frame, "pix", None)
    if pix is not None:
        return pix
    if isinstance(frame, dict) and "pix" in frame:
        return frame["pix"]
    return frame


def _extract_frame_timestamp_s(core: Any, frame: Any) -> float:
    """Extract acquisition timestamp (seconds) from metadata if available."""
    tags = getattr(frame, "tags", None)
    if isinstance(tags, dict):
        for key in ("ElapsedTime-ms", "ElapsedTimeMs", "Timestamp-ms"):
            if key in tags:
                try:
                    return float(tags[key]) / 1000.0
                except Exception:
                    pass

    metadata = getattr(frame, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("ElapsedTime-ms", "ElapsedTimeMs", "Timestamp-ms"):
            if key in metadata:
                try:
                    return float(metadata[key]) / 1000.0
                except Exception:
                    pass

    fn = getattr(core, "getLastImageTimeStamp", None)
    if callable(fn):
        try:
            return float(fn()) / 1000.0
        except Exception:
            pass

    return 0.0


def _get_last_image(core: Any) -> Any:
    """Grab the most recent frame from the circular buffer."""
    for method in ("getLastTaggedImage", "getLastImage"):
        fn = getattr(core, method, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                continue
    raise RuntimeError("Unable to fetch latest Micro-Manager frame from live buffer")


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
    allow_snap_fallback: bool = False,
) -> MicroManagerFrameSource:
    """Connect to a running Micro-Manager instance.

    Tries connection methods in order:
    1. Existing core object passed directly
    2. pycromanager Core(host, port)
    3. pycromanager Core() with default bridge settings
    4. MMCorePy (direct, if running in the same process)
    """

    if core is not None:
        return MicroManagerFrameSource(core=core, allow_snap_fallback=allow_snap_fallback)

    mm_core = _try_create_pycromanager_core(host=host, port=port)
    if mm_core is not None:
        return MicroManagerFrameSource(core=mm_core, allow_snap_fallback=allow_snap_fallback)

    # Fall back to MMCorePy (in-process)
    try:
        import MMCorePy

        mm_core = MMCorePy.CMMCore()
        return MicroManagerFrameSource(core=mm_core, allow_snap_fallback=allow_snap_fallback)
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
