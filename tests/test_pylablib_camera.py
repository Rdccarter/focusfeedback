import sys
import types

from orca_focus.pylablib_camera import PylablibFrameSource, _default_read_frame, create_pylablib_frame_source


class _ArrayLike:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class _CameraWithNewest:
    def read_newest_image(self):
        return _ArrayLike([[5, 6], [7, 8]])


class _FakeOrcaCam:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def read_newest_image(self):
        return _ArrayLike([[1, 2], [3, 4]])


class _FakeAndorCam:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def read_newest_image(self):
        return _ArrayLike([[9, 10], [11, 12]])


def test_default_read_frame_uses_newest_method() -> None:
    frame = _default_read_frame(_CameraWithNewest())
    assert frame.tolist() == [[5, 6], [7, 8]]


def test_pylablib_frame_source_callable() -> None:
    source = PylablibFrameSource(camera=_FakeOrcaCam(), read_frame=_default_read_frame)
    image, ts = source()
    assert image == [[1.0, 2.0], [3.0, 4.0]]
    assert isinstance(ts, float)


def test_create_pylablib_frame_source_with_injected_module(monkeypatch) -> None:
    hamamatsu_mod = types.SimpleNamespace(DCAMCamera=_FakeOrcaCam)
    andor_mod = types.SimpleNamespace(AndorSDK2Camera=_FakeAndorCam)
    monkeypatch.setitem(sys.modules, "pylablib", types.ModuleType("pylablib"))
    monkeypatch.setitem(sys.modules, "pylablib.devices", types.ModuleType("pylablib.devices"))
    monkeypatch.setitem(sys.modules, "pylablib.devices.Hamamatsu", hamamatsu_mod)
    monkeypatch.setitem(sys.modules, "pylablib.devices.Andor", andor_mod)

    source_orca = create_pylablib_frame_source("orca", idx=0)
    source_andor = create_pylablib_frame_source("andor", idx=1)

    image_orca, _ = source_orca()
    image_andor, _ = source_andor()

    assert image_orca == [[1.0, 2.0], [3.0, 4.0]]
    assert image_andor == [[9.0, 10.0], [11.0, 12.0]]
