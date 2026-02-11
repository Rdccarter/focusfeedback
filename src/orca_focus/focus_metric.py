from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .interfaces import Image2D


@dataclass(slots=True)
class Roi:
    """Axis-aligned region of interest in image coordinates."""

    x: int
    y: int
    width: int
    height: int

    def clamp(self, image_shape: tuple[int, int]) -> "Roi":
        h, w = image_shape
        x = min(max(0, self.x), max(0, w - 1))
        y = min(max(0, self.y), max(0, h - 1))
        width = min(self.width, w - x)
        height = min(self.height, h - y)
        if width <= 0 or height <= 0:
            raise ValueError("ROI does not intersect image")
        return Roi(x=x, y=y, width=width, height=height)


def _coerce_image_2d(image: Any) -> Image2D:
    if hasattr(image, "tolist") and callable(image.tolist):
        image = image.tolist()
    if isinstance(image, tuple):
        image = list(image)
    if not isinstance(image, list):
        raise TypeError("Image must be a 2D list/tuple or expose tolist()")

    out: Image2D = []
    width: int | None = None
    for row in image:
        if isinstance(row, tuple):
            row = list(row)
        if not isinstance(row, list):
            raise TypeError("Image rows must be list/tuple")
        if width is None:
            width = len(row)
            if width == 0:
                raise ValueError("Empty image")
        elif len(row) != width:
            raise ValueError("Image rows must have equal length")
        out.append([float(v) for v in row])

    if not out:
        raise ValueError("Empty image")
    return out


def _image_shape(image: Image2D) -> tuple[int, int]:
    if not image or not image[0]:
        raise ValueError("Empty image")
    return len(image), len(image[0])


def extract_roi(image: Image2D, roi: Roi) -> Image2D:
    safe_image = _coerce_image_2d(image)
    h, w = _image_shape(safe_image)
    safe_roi = roi.clamp((h, w))
    return [
        row[safe_roi.x : safe_roi.x + safe_roi.width]
        for row in safe_image[safe_roi.y : safe_roi.y + safe_roi.height]
    ]


def _astigmatic_error_signal_numpy(patch: Image2D) -> float:
    try:
        import numpy as np
    except Exception:
        return _astigmatic_error_signal_python(patch)

    arr = np.asarray(patch, dtype=float)
    total = float(arr.sum())
    if total <= 0:
        return 0.0

    y_idx, x_idx = np.indices(arr.shape, dtype=float)
    cx = float((x_idx * arr).sum() / total)
    cy = float((y_idx * arr).sum() / total)

    var_x = float((((x_idx - cx) ** 2) * arr).sum() / total)
    var_y = float((((y_idx - cy) ** 2) * arr).sum() / total)

    denom = var_x + var_y
    if denom == 0:
        return 0.0
    return (var_x - var_y) / denom


def _astigmatic_error_signal_python(patch: Image2D) -> float:
    total = sum(sum(row) for row in patch)
    if total <= 0:
        return 0.0

    sum_x = 0.0
    sum_y = 0.0
    for y, row in enumerate(patch):
        for x, val in enumerate(row):
            sum_x += x * val
            sum_y += y * val

    cx = sum_x / total
    cy = sum_y / total

    var_x = 0.0
    var_y = 0.0
    for y, row in enumerate(patch):
        for x, val in enumerate(row):
            var_x += ((x - cx) ** 2) * val
            var_y += ((y - cy) ** 2) * val

    var_x /= total
    var_y /= total

    denom = var_x + var_y
    if denom == 0:
        return 0.0
    return (var_x - var_y) / denom


def _centroid_python(patch: Image2D) -> tuple[float, float]:
    """Return (cx, cy) intensity-weighted centroid of the patch."""
    total = sum(sum(row) for row in patch)
    if total <= 0:
        h = len(patch)
        w = len(patch[0]) if patch else 0
        return (w - 1) / 2.0, (h - 1) / 2.0
    sum_x = 0.0
    sum_y = 0.0
    for y, row in enumerate(patch):
        for x, val in enumerate(row):
            sum_x += x * val
            sum_y += y * val
    return sum_x / total, sum_y / total


def centroid_near_edge(image: Image2D, roi: Roi, margin_px: float) -> bool:
    """Return True if the intensity centroid is within *margin_px* of the ROI boundary.

    This is a guard against truncated PSFs: when the bead drifts near the ROI
    edge, the second-moment error signal becomes biased and can drive runaway
    corrections.
    """
    if margin_px <= 0:
        return False
    patch = extract_roi(image, roi)
    h = len(patch)
    w = len(patch[0]) if patch else 0
    if h == 0 or w == 0:
        return True
    cx, cy = _centroid_python(patch)
    if cx < margin_px or cx > (w - 1) - margin_px:
        return True
    if cy < margin_px or cy > (h - 1) - margin_px:
        return True
    return False


def roi_total_intensity(image: Image2D, roi: Roi) -> float:
    patch = extract_roi(image, roi)
    return float(sum(sum(row) for row in patch))


def astigmatic_error_signal(image: Image2D, roi: Roi) -> float:
    """Return focus error based on anisotropic second moments.

    Uses a NumPy-accelerated path when NumPy is available; otherwise falls back
    to a pure-Python implementation.
    """

    patch = extract_roi(image, roi)
    return _astigmatic_error_signal_numpy(patch)
