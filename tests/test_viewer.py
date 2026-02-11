import pytest

from orca_focus.focus_metric import Roi
from orca_focus.viewer import RoiSelector


def test_roi_selector_finalize_requires_selection() -> None:
    selector = RoiSelector()

    with pytest.raises(ValueError, match="No ROI selected"):
        selector.finalize([[1.0]])


def test_roi_selector_finalize_clamps_to_image() -> None:
    selector = RoiSelector()
    selector.begin(2, 3)
    selector.update(9, 12)

    roi = selector.finalize([[0.0] * 6 for _ in range(8)])

    assert roi == Roi(x=2, y=3, width=4, height=5)
