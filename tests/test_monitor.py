import threading
import time

from orca_focus.interfaces import CameraFrame
from orca_focus.monitor import run_live_monitor


class _FakeCamera:
    def __init__(self) -> None:
        self.calls = 0

    def get_frame(self) -> CameraFrame:
        self.calls += 1
        return CameraFrame(image=[[1.0, 2.0], [3.0, 4.0]], timestamp_s=time.time())


def test_run_live_monitor_dispatches_frames_until_stopped() -> None:
    stop_event = threading.Event()
    camera = _FakeCamera()
    received = []

    def _on_frame(image):
        received.append(image)
        stop_event.set()

    run_live_monitor(camera=camera, on_frame=_on_frame, stop_event=stop_event, loop_hz=100.0)

    assert camera.calls >= 1
    assert received == [[[1.0, 2.0], [3.0, 4.0]]]
