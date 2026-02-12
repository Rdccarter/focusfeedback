"""Micro-Manager integration for orca-focus.

Reads frames from Micro-Manager's live acquisition buffer (owned by the
Micro-Manager GUI) and optionally controls the Z stage through MMCore.
"""

from __future__ import annotations

import sys
import threading
import time
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
        self._last_frame_identity: int | float | str | None = None
        self._lock = threading.Lock()

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
        with self._lock:
            core = self._core
            token = _get_frame_token(core)

            # Fast path: if token is unchanged, verify identity before deciding stale.
            if self._last_image is not None and token is not None and token == self._last_frame_token:
                probe_frame = _try_get_last_image(core)
                if probe_frame is None:
                    return self._last_image, self._last_ts
                probe_identity = _frame_identity(probe_frame)
                if probe_identity is None or probe_identity == self._last_frame_identity:
                    return self._last_image, self._last_ts
                frame = probe_frame
            else:
                frame = _try_get_last_image(core)

            if frame is None:
                if self._last_image is not None:
                    return self._last_image, self._last_ts
                if not self._allow_snap_fallback:
                    raise RuntimeError(
                        "Micro-Manager sequence/live mode is not running. Enable Live mode in "
                        "Micro-Manager or construct the frame source with allow_snap_fallback=True."
                    )
                core.snapImage()
                frame = core.getImage()
                ts_for_sample = time.monotonic()
                token = None
                frame_identity = None
            else:
                ts_for_sample = _extract_frame_timestamp_s(frame)
                if ts_for_sample is None:
                    ts_for_sample = time.monotonic()
                frame_identity = _frame_identity(frame)

        image = _to_image_2d(_extract_image_payload(frame))

        with self._lock:
            self._last_image = image
            self._last_ts = ts_for_sample
            self._last_frame_token = token
            self._last_frame_identity = frame_identity
            return image, ts_for_sample


class MicroManagerStage(StageInterface):
    """Z stage adapter that goes through Micro-Manager's device layer."""

    def __init__(
        self,
        core: Any,
        z_stage_name: str | None = None,
        wait_for_device: bool = True,
    ) -> None:
        self._core = core
        self._wait_for_device = wait_for_device
        self._z_name = z_stage_name or core.getFocusDevice()

    def get_z_um(self) -> float:
        return float(self._core.getPosition(self._z_name))

    def move_z_um(self, target_z_um: float) -> None:
        self._core.setPosition(self._z_name, target_z_um)
        if self._wait_for_device:
            self._core.waitForDevice(self._z_name)


def _get_frame_token(core: Any) -> int | float | str | None:
    """Return acquisition token for duplicate detection in live mode.

    `getLastImageTimeStamp` is preferred because it is acquisition-coupled in
    most MM backends. If unavailable, token-based duplicate detection is
    disabled and callers always process newest frame.
    """
    fn = getattr(core, "getLastImageTimeStamp", None)
    if callable(fn):
        try:
            return fn()
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


def _frame_metadata_dict(frame: Any) -> dict[str, Any] | None:
    tags = getattr(frame, "tags", None)
    if tags is None and isinstance(frame, dict):
        tags = frame.get("tags")
    if isinstance(tags, dict):
        return tags

    metadata = getattr(frame, "metadata", None)
    if metadata is None and isinstance(frame, dict):
        metadata = frame.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return None


def _extract_frame_timestamp_s(frame: Any) -> float | None:
    """Extract acquisition timestamp in seconds when metadata is explicit.

    Only metadata with explicit millisecond units is used to avoid unit-guessing
    across MM backends. If no explicit timestamp exists, caller should provide a
    local monotonic fallback.
    """
    md = _frame_metadata_dict(frame)
    if not md:
        return None

    for key in ("ElapsedTime-ms", "ElapsedTimeMs", "Timestamp-ms"):
        if key in md:
            try:
                return float(md[key]) / 1000.0
            except Exception:
                pass
    return None


def _frame_identity(frame: Any) -> int | float | str | None:
    """Best-effort per-frame identity to supplement stale token detection."""
    md = _frame_metadata_dict(frame)
    if md:
        for key in ("ImageNumber", "FrameIndex", "Frame", "Index"):
            if key in md:
                return str(md[key])

    pix = _extract_image_payload(frame)
    return id(pix)


def _try_get_last_image(core: Any) -> Any | None:
    """Grab most recent frame from the circular buffer, or None if unavailable."""
    for method in ("getLastTaggedImage", "getLastImage"):
        fn = getattr(core, method, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                continue
    return None


def _try_create_pycromanager_core(host: str, port: int) -> Any | None:
    try:
        from pycromanager import Core
    except Exception:
        return None

    try:
        core = Core(host=host, port=port)
        core.getVersionInfo()
        return core
    except Exception:
        pass

    try:
        core = Core()
        core.getVersionInfo()
        return core
    except Exception:
        return None


def _try_create_pymmcore_core() -> Any | None:
    """Best-effort local MMCore fallback without pycromanager bridge.

    Supports both modern `pymmcore` and legacy `MMCorePy` module names.
    """
    for module_name in ("pymmcore", "MMCorePy"):
        try:
            module = __import__(module_name)
        except Exception:
            continue

        core_ctor = getattr(module, "CMMCore", None)
        if not callable(core_ctor):
            continue
        try:
            return core_ctor()
        except Exception:
            continue
    return None


def create_micromanager_frame_source(
    *,
    host: str = "localhost",
    port: int = 4827,
    core: Any | None = None,
    allow_snap_fallback: bool = False,
    allow_standalone_core: bool = False,
) -> MicroManagerFrameSource:
    """Connect to a running Micro-Manager instance."""
    if core is not None:
        return MicroManagerFrameSource(core=core, allow_snap_fallback=allow_snap_fallback)

    mm_core = _try_create_pycromanager_core(host=host, port=port)
    if mm_core is not None:
        return MicroManagerFrameSource(core=mm_core, allow_snap_fallback=allow_snap_fallback)

    if allow_standalone_core:
        mm_core = _try_create_pymmcore_core()
        if mm_core is not None:
            print(
                "Warning: using bare MMCore CMMCore(); this core is not attached to a running "
                "Micro-Manager GUI session and may require loading a hardware config.",
                file=sys.stderr,
            )
            return MicroManagerFrameSource(core=mm_core, allow_snap_fallback=allow_snap_fallback)

    raise RuntimeError(
        "Could not connect to Micro-Manager. Make sure one of these is true:\n"
        "  1. Micro-Manager is running with pycromanager bridge enabled\n"
        "     (Tools -> Options -> Run server on port), or\n"
        "  2. pycromanager can attach via its default local bridge (Core()), or\n"
        "  3. (optional) use --mm-allow-standalone-core to allow standalone "
        "pymmcore/MMCorePy fallback when you are not attaching to MM GUI.\n"
        f"  Tried pycromanager at {host}:{port} and default Core()"
    )
