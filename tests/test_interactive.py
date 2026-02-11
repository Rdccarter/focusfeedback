import sys
import types

import pytest

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
