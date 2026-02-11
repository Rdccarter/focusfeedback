import pytest

from orca_focus.focus_metric import Roi, astigmatic_error_signal, centroid_near_edge
from orca_focus.hardware import SimulatedScene


def test_astigmatic_error_changes_sign_across_focus() -> None:
    scene = SimulatedScene(focal_plane_um=0.0, alpha_px_per_um=0.2)
    roi = Roi(x=16, y=16, width=32, height=32)

    error_below = astigmatic_error_signal(scene.render_dot(z_um=-1.0), roi)
    error_above = astigmatic_error_signal(scene.render_dot(z_um=1.0), roi)

    assert error_below < 0
    assert error_above > 0


def test_astigmatic_error_rejects_empty_image() -> None:
    with pytest.raises(ValueError, match="Empty image"):
        astigmatic_error_signal([], Roi(x=0, y=0, width=1, height=1))


def test_astigmatic_error_rejects_empty_image_rows() -> None:
    with pytest.raises(ValueError, match="Empty image"):
        astigmatic_error_signal([[]], Roi(x=0, y=0, width=1, height=1))


class _ArrayLike:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


def test_astigmatic_error_accepts_array_like_images() -> None:
    img = _ArrayLike([[0, 1], [2, 3]])
    value = astigmatic_error_signal(img, Roi(x=0, y=0, width=2, height=2))
    assert isinstance(value, float)


def test_astigmatic_error_rejects_ragged_rows() -> None:
    with pytest.raises(ValueError, match="equal length"):
        astigmatic_error_signal([[1.0], [1.0, 2.0]], Roi(x=0, y=0, width=1, height=1))


def test_centroid_near_edge_centered_dot() -> None:
    """A centered bright pixel in a 5x5 ROI should not be near the edge."""
    image = [[0.0] * 5 for _ in range(5)]
    image[2][2] = 100.0
    assert centroid_near_edge(image, Roi(x=0, y=0, width=5, height=5), margin_px=1.0) is False


def test_centroid_near_edge_corner_dot() -> None:
    """A bright pixel in the corner should be flagged as near the edge."""
    image = [[0.0] * 5 for _ in range(5)]
    image[0][0] = 100.0
    assert centroid_near_edge(image, Roi(x=0, y=0, width=5, height=5), margin_px=1.0) is True


def test_centroid_near_edge_zero_margin_never_triggers() -> None:
    """With margin_px=0, centroid_near_edge should always return False."""
    image = [[0.0] * 5 for _ in range(5)]
    image[0][0] = 100.0
    assert centroid_near_edge(image, Roi(x=0, y=0, width=5, height=5), margin_px=0) is False
