from pathlib import Path
from unittest.mock import patch

import pytest

from orca_focus.cli import _load_startup_calibration, build_parser, main


def test_build_parser_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.duration == 2.0
    assert args.loop_hz == 30.0
    assert args.camera == "simulate"
    assert args.show_live is False
    assert args.calibration_csv == "calibration_sweep.csv"


def test_load_startup_calibration_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "calibration_sweep.csv"
    csv_path.write_text("z_um,error,weight\n-1.0,-0.5,1\n0.0,0.0,1\n1.0,0.5,1\n", encoding="utf-8")

    calibration = _load_startup_calibration(str(csv_path))

    assert calibration.error_to_um == pytest.approx(2.0)
    assert calibration.error_at_focus == pytest.approx(0.0)


def test_main_simulate_smoke(tmp_path: Path) -> None:
    csv_path = tmp_path / "calibration_sweep.csv"
    csv_path.write_text("z_um,error,weight\n-1.0,-0.5,1\n0.0,0.0,1\n1.0,0.5,1\n", encoding="utf-8")

    with patch(
        "sys.argv",
        [
            "orca-focus",
            "--duration",
            "0.01",
            "--loop-hz",
            "100",
            "--calibration-csv",
            str(csv_path),
        ],
    ):
        assert main() == 0


def test_build_camera_and_stage_respects_simulate_stage_override(tmp_path: Path) -> None:
    from orca_focus.cli import _build_camera_and_stage

    args = build_parser().parse_args(["--camera", "orca", "--stage", "simulate"])

    with patch("orca_focus.cli.create_pylablib_frame_source", return_value=lambda: ([[0.0]], 0.0)):
        camera, stage = _build_camera_and_stage(args)

    # Simulate stage backend should stay in-memory even with hardware camera mode.
    assert stage._dll is None  # noqa: SLF001
    assert stage._wrapper is None  # noqa: SLF001
    camera.start()
    camera.stop()


def test_main_stops_camera_when_viewer_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "calibration_sweep.csv"
    csv_path.write_text("z_um,error,weight\n-1.0,-0.5,1\n0.0,0.0,1\n1.0,0.5,1\n", encoding="utf-8")

    class _DummyCamera:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

    dummy_camera = _DummyCamera()

    with patch("sys.argv", ["orca-focus", "--show-live", "--calibration-csv", str(csv_path)]), \
        patch("orca_focus.cli._build_camera_and_stage", return_value=(dummy_camera, object())), \
        patch("orca_focus.cli.launch_autofocus_viewer", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            main()

    assert dummy_camera.started is True
    assert dummy_camera.stopped is True
