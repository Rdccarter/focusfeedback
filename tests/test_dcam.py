from orca_focus.dcam import DcamFrameSource, _to_image_2d
from orca_focus.hardware import HamamatsuOrcaCamera


class _ArrayLike:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class _FakeDcamCamera:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def get_latest_frame(self):
        return _ArrayLike([[1, 2], [3, 4]])


def test_to_image_2d_supports_array_like() -> None:
    image = _to_image_2d(_ArrayLike([[1, 2], [3, 4]]))
    assert image == [[1.0, 2.0], [3.0, 4.0]]


def test_to_image_2d_supports_tuple_rows() -> None:
    image = _to_image_2d(((1, 2), (3, 4)))
    assert image == [[1.0, 2.0], [3.0, 4.0]]


def test_hamamatsu_camera_default_is_passive_relay() -> None:
    backend = _FakeDcamCamera()
    source = DcamFrameSource(backend)
    camera = HamamatsuOrcaCamera(frame_source=source)

    camera.start()
    frame = camera.get_frame()
    camera.stop()

    assert backend.started is False
    assert backend.stopped is False
    assert frame.image == [[1.0, 2.0], [3.0, 4.0]]


def test_hamamatsu_camera_can_optionally_control_source_lifecycle() -> None:
    backend = _FakeDcamCamera()
    source = DcamFrameSource(backend)
    camera = HamamatsuOrcaCamera(frame_source=source, control_source_lifecycle=True)

    camera.start()
    camera.stop()

    assert backend.started is True
    assert backend.stopped is True
