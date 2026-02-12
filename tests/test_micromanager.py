import sys
from types import ModuleType, SimpleNamespace

import pytest

from orca_focus.micromanager import MicroManagerFrameSource, MicroManagerStage, create_micromanager_frame_source


class _FakeCore:
    def getVersionInfo(self):
        return "ok"


class _RemoteFailLocalOkCoreFactory:
    def __call__(self, *args, **kwargs):
        if "host" in kwargs or "port" in kwargs:
            raise RuntimeError("remote fail")
        return _FakeCore()


class _AlwaysFailCoreFactory:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("all fail")


def test_micromanager_source_uses_pycromanager_default_if_host_port_fails(monkeypatch):
    fake_module = ModuleType("pycromanager")
    fake_module.Core = _RemoteFailLocalOkCoreFactory()
    monkeypatch.setitem(sys.modules, "pycromanager", fake_module)

    source = create_micromanager_frame_source(host="localhost", port=4827)

    assert isinstance(source.core, _FakeCore)


def test_micromanager_source_falls_back_to_mmcorepy(monkeypatch):
    fake_py = ModuleType("pycromanager")
    fake_py.Core = _AlwaysFailCoreFactory()
    monkeypatch.setitem(sys.modules, "pycromanager", fake_py)

    # Force pymmcore import to fail so legacy MMCorePy path is exercised.
    monkeypatch.delitem(sys.modules, "pymmcore", raising=False)

    fake_mmcorepy = ModuleType("MMCorePy")
    fake_mmcorepy.CMMCore = lambda: SimpleNamespace(source="mmcorepy")
    monkeypatch.setitem(sys.modules, "MMCorePy", fake_mmcorepy)

    source = create_micromanager_frame_source(
        host="localhost",
        port=4827,
        allow_standalone_core=True,
    )

    assert source.core.source == "mmcorepy"


def test_micromanager_source_falls_back_to_pymmcore(monkeypatch):
    fake_py = ModuleType("pycromanager")
    fake_py.Core = _AlwaysFailCoreFactory()
    monkeypatch.setitem(sys.modules, "pycromanager", fake_py)

    fake_pymmcore = ModuleType("pymmcore")
    fake_pymmcore.CMMCore = lambda: SimpleNamespace(source="pymmcore")
    monkeypatch.setitem(sys.modules, "pymmcore", fake_pymmcore)

    source = create_micromanager_frame_source(
        host="localhost",
        port=4827,
        allow_standalone_core=True,
    )

    assert source.core.source == "pymmcore"


def test_micromanager_source_errors_with_clear_message(monkeypatch):
    fake_py = ModuleType("pycromanager")
    fake_py.Core = _AlwaysFailCoreFactory()
    monkeypatch.setitem(sys.modules, "pycromanager", fake_py)
    fake_mmcorepy = ModuleType("MMCorePy")
    monkeypatch.setitem(sys.modules, "MMCorePy", fake_mmcorepy)

    with pytest.raises(RuntimeError, match=r"--mm-allow-standalone-core"):
        create_micromanager_frame_source(host="localhost", port=4827)




def test_micromanager_source_does_not_fallback_to_standalone_core_without_opt_in(monkeypatch):
    fake_py = ModuleType("pycromanager")
    fake_py.Core = _AlwaysFailCoreFactory()
    monkeypatch.setitem(sys.modules, "pycromanager", fake_py)

    fake_pymmcore = ModuleType("pymmcore")
    fake_pymmcore.CMMCore = lambda: SimpleNamespace(source="pymmcore")
    monkeypatch.setitem(sys.modules, "pymmcore", fake_pymmcore)

    with pytest.raises(RuntimeError, match=r"--mm-allow-standalone-core"):
        create_micromanager_frame_source(host="localhost", port=4827)
def test_micromanager_source_uses_metadata_timestamp_and_detects_duplicate_by_token():
    class _TaggedImage:
        def __init__(self, pix, elapsed_ms):
            self.pix = pix
            self.tags = {"ElapsedTime-ms": elapsed_ms, "ImageNumber": 1}

    class _LiveCoreWithToken:
        def __init__(self):
            self.token = 1

        def getLastImageTimeStamp(self):
            return self.token

        def getLastTaggedImage(self):
            if self.token == 1:
                return _TaggedImage([[10, 20], [30, 40]], 1250.0)
            tagged = _TaggedImage([[11, 22], [33, 44]], 1500.0)
            tagged.tags["ImageNumber"] = 2
            return tagged

    core = _LiveCoreWithToken()
    source = MicroManagerFrameSource(core)

    image1, ts1 = source()
    assert image1 == [[10.0, 20.0], [30.0, 40.0]]
    assert ts1 == pytest.approx(1.25)

    # Same token + same frame identity => duplicate.
    image2, ts2 = source()
    assert ts2 == ts1
    assert image2 == image1

    # Advance token => new frame.
    core.token = 2
    image3, ts3 = source()
    assert image3 == [[11.0, 22.0], [33.0, 44.0]]
    assert ts3 == pytest.approx(1.5)


def test_micromanager_source_uses_identity_when_token_stuck():
    class _TaggedImage:
        def __init__(self, i):
            self.pix = [[i]]
            self.tags = {"ElapsedTime-ms": 1000 + i, "ImageNumber": i}

    class _Core:
        def __init__(self):
            self.i = 0

        def getLastImageTimeStamp(self):
            return 1  # stuck token

        def getLastTaggedImage(self):
            self.i += 1
            return _TaggedImage(self.i)

    source = MicroManagerFrameSource(_Core())
    image1, ts1 = source()
    image2, ts2 = source()
    assert image1 != image2
    assert ts2 > ts1


def test_micromanager_source_requires_live_mode_by_default():
    class _CoreNotLive:
        pass

    source = MicroManagerFrameSource(_CoreNotLive())
    with pytest.raises(RuntimeError, match="not running"):
        source()


def test_micromanager_source_snap_fallback_is_opt_in():
    class _CoreNotLive:
        def snapImage(self):
            return None

        def getImage(self):
            return [[1, 2], [3, 4]]

    source = MicroManagerFrameSource(_CoreNotLive(), allow_snap_fallback=True)
    image, ts = source()
    assert image == [[1.0, 2.0], [3.0, 4.0]]
    assert ts > 0


def test_micromanager_source_without_explicit_timestamp_uses_monotonic():
    class _Core:
        def getLastImage(self):
            return [[1, 1], [1, 1]]

    source = MicroManagerFrameSource(_Core())
    _, ts = source()
    assert ts > 0


def test_micromanager_stage_wait_for_device_is_optional():
    class _Core:
        def __init__(self):
            self.wait_calls = 0
            self.position = 0.0

        def getFocusDevice(self):
            return "Z"

        def getPosition(self, _name):
            return self.position

        def setPosition(self, _name, z):
            self.position = z

        def waitForDevice(self, _name):
            self.wait_calls += 1

    core = _Core()
    stage = MicroManagerStage(core=core, wait_for_device=False)
    stage.move_z_um(2.5)
    assert stage.get_z_um() == pytest.approx(2.5)
    assert core.wait_calls == 0
