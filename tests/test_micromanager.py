import sys
from types import ModuleType, SimpleNamespace

import pytest

from orca_focus.micromanager import create_micromanager_frame_source


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

    fake_mmcorepy = ModuleType("MMCorePy")
    fake_mmcorepy.CMMCore = lambda: SimpleNamespace(source="mmcorepy")
    monkeypatch.setitem(sys.modules, "MMCorePy", fake_mmcorepy)

    source = create_micromanager_frame_source(host="localhost", port=4827)

    assert source.core.source == "mmcorepy"


def test_micromanager_source_errors_with_clear_message(monkeypatch):
    fake_py = ModuleType("pycromanager")
    fake_py.Core = _AlwaysFailCoreFactory()
    monkeypatch.setitem(sys.modules, "pycromanager", fake_py)
    fake_mmcorepy = ModuleType("MMCorePy")
    monkeypatch.setitem(sys.modules, "MMCorePy", fake_mmcorepy)

    with pytest.raises(RuntimeError, match=r"default Core\(\)"):
        create_micromanager_frame_source(host="localhost", port=4827)


def test_micromanager_source_returns_stale_timestamp_when_buffer_empty():
    """When the circular buffer has no new frames, the source should return
    the previous timestamp so the controller can detect a duplicate."""
    from orca_focus.micromanager import MicroManagerFrameSource

    class _LiveCoreWithBuffer:
        def __init__(self):
            self.remaining = 1  # starts with one frame available

        def isSequenceRunning(self):
            return True

        def getRemainingImageCount(self):
            return self.remaining

        def getLastImage(self):
            return [[10, 20], [30, 40]]

    core = _LiveCoreWithBuffer()
    source = MicroManagerFrameSource(core)

    # First call: buffer has a frame → fresh timestamp
    image1, ts1 = source()
    assert ts1 > 0

    # Drain the buffer
    core.remaining = 0

    # Second call: buffer empty → same timestamp returned
    image2, ts2 = source()
    assert ts2 == ts1
    assert image2 == image1

    # New frame arrives
    core.remaining = 1
    image3, ts3 = source()
    assert ts3 > ts1
