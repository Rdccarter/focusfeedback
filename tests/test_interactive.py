import os
import sys
import types

import pytest

from orca_focus import interactive
from orca_focus.interactive import launch_matplotlib_viewer
from orca_focus.interfaces import CameraFrame


class _FakeCamera:
    def get_frame(self) -> CameraFrame:
        return CameraFrame(image=[[1.0, 2.0], [3.0, 4.0]], timestamp_s=0.0)


class _FakeAxes:
    def imshow(self, image, cmap=None):
        return _FakeImage(image, cmap)

    def set_title(self, _title):
        return None


class _FakeImage:
    def __init__(self, image, cmap):
        self.image = image
        self.cmap = cmap

    def set_data(self, image):
        self.image = image


class _FakeFigure:
    pass

class _FakePyplotModule(types.ModuleType):
    def __init__(self):
        super().__init__("matplotlib.pyplot")
        self.show_called = False

    def subplots(self):
        return _FakeFigure(), _FakeAxes()

    def show(self):
        self.show_called = True


class _FakeAnimationModule(types.ModuleType):
    def __init__(self):
        super().__init__("matplotlib.animation")

    class FuncAnimation:
        def __init__(self, _fig, update, interval, blit):
            assert interval == 40
            assert blit is True
            update(0)


def test_launch_matplotlib_viewer_raises_without_matplotlib(monkeypatch) -> None:
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("matplotlib"):
            raise ImportError("missing matplotlib")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match="matplotlib is required"):
        launch_matplotlib_viewer(_FakeCamera())


def test_launch_matplotlib_viewer_uses_animation(monkeypatch) -> None:
    fake_pyplot = _FakePyplotModule()
    fake_animation = _FakeAnimationModule()
    monkeypatch.setitem(sys.modules, "matplotlib", types.ModuleType("matplotlib"))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)
    monkeypatch.setitem(sys.modules, "matplotlib.animation", fake_animation)

    launch_matplotlib_viewer(_FakeCamera())

    assert fake_pyplot.show_called is True


def test_prepare_napari_environment_sets_disable_plugins(monkeypatch):
    monkeypatch.delenv("NAPARI_DISABLE_PLUGINS", raising=False)

    interactive._prepare_napari_environment()

    assert os.environ.get("NAPARI_DISABLE_PLUGINS") == "1"
    assert os.environ.get("NAPARI_DISABLE_PLUGIN_ENTRY_POINTS") == "1"


def test_prepare_napari_environment_respects_existing_value(monkeypatch):
    monkeypatch.setenv("NAPARI_DISABLE_PLUGINS", "0")

    interactive._prepare_napari_environment()

    assert os.environ.get("NAPARI_DISABLE_PLUGINS") == "0"


def test_prepare_napari_environment_sets_legacy_entrypoints_flag(monkeypatch):
    monkeypatch.delenv("NAPARI_DISABLE_PLUGIN_ENTRYPOINTS", raising=False)

    interactive._prepare_napari_environment()

    assert os.environ.get("NAPARI_DISABLE_PLUGIN_ENTRYPOINTS") == "1"


def test_calibration_plan_from_nm_computes_expected_bounds_and_steps() -> None:
    z_min, z_max, steps, effective_step_nm = interactive._calibration_plan_from_nm(
        center_z_um=10.0,
        half_range_nm=500.0,
        step_nm=100.0,
    )

    assert z_min == pytest.approx(9.5)
    assert z_max == pytest.approx(10.5)
    assert steps == 11
    assert effective_step_nm == pytest.approx(100.0)


def test_build_runtime_calibration_for_roi_anchors_error_offset(monkeypatch):
    from orca_focus.calibration import FocusCalibration
    from orca_focus.focus_metric import Roi

    camera = _FakeCamera()
    roi = Roi(x=0, y=0, width=2, height=2)
    base = FocusCalibration(error_at_focus=0.0, error_to_um=3.0)

    monkeypatch.setattr(interactive, "astigmatic_error_signal", lambda _img, _roi: 0.125)

    cal = interactive._build_runtime_calibration_for_roi(camera, roi, base)

    assert cal.error_to_um == pytest.approx(3.0)
    assert cal.error_at_focus == pytest.approx(0.125)


def test_build_runtime_calibration_for_roi_falls_back_on_error(monkeypatch):
    from orca_focus.calibration import FocusCalibration
    from orca_focus.focus_metric import Roi

    camera = _FakeCamera()
    roi = Roi(x=0, y=0, width=2, height=2)
    base = FocusCalibration(error_at_focus=0.05, error_to_um=2.0)

    def _boom(_img, _roi):
        raise RuntimeError("metric failed")

    monkeypatch.setattr(interactive, "astigmatic_error_signal", _boom)

    cal = interactive._build_runtime_calibration_for_roi(camera, roi, base)

    assert cal is base
