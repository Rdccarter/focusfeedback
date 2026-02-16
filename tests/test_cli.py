from pathlib import Path
from unittest.mock import patch

import pytest

from orca_focus.cli import _load_startup_calibration, build_parser, main


def test_build_parser_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.duration == 2.0
    assert args.loop_hz == 30.0
    assert args.max_dt_s == 0.2
    assert args.camera == "simulate"
    assert args.show_live is False
    assert args.calibration_csv == "calibration_sweep.csv"
    assert args.mm_allow_standalone_core is False
    assert args.stage_min_um is None
    assert args.stage_max_um is None
    assert args.af_max_excursion_um == 5.0
    assert args.command_deadband_um == 0.02


def test_build_parser_stage_accepts_micromanager() -> None:
    args = build_parser().parse_args(["--stage", "micromanager"])
    assert args.stage == "micromanager"


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




def test_build_camera_and_stage_forwards_mm_standalone_flag() -> None:
    from orca_focus.cli import _build_camera_and_stage

    args = build_parser().parse_args(["--camera", "micromanager", "--mm-allow-standalone-core"])

    with patch("orca_focus.micromanager.create_micromanager_frame_source") as create_mm:
        mm_source = lambda: ([[0.0]], 0.0)
        mm_source.core = object()
        create_mm.return_value = mm_source

        with patch("orca_focus.micromanager.MicroManagerStage") as mm_stage_cls:
            mm_stage_cls.return_value = object()
            _build_camera_and_stage(args)

    assert create_mm.call_args.kwargs["allow_standalone_core"] is True
def test_build_camera_and_stage_defaults_to_mcl_stage_for_mm_camera() -> None:
    from orca_focus.cli import _build_camera_and_stage

    args = build_parser().parse_args(["--camera", "micromanager", "--stage-dll", "C:/path/Madlib.dll"])

    with patch("orca_focus.micromanager.create_micromanager_frame_source") as create_mm,         patch("orca_focus.cli.MclNanoZStage") as mcl_stage_cls:
        core = object()
        create_mm.return_value = lambda: ([[0.0]], 0.0)
        create_mm.return_value.core = core
        mcl_stage = object()
        mcl_stage_cls.return_value = mcl_stage

        camera, stage = _build_camera_and_stage(args)

    assert stage is mcl_stage
    mcl_stage_cls.assert_called_once_with(dll_path="C:/path/Madlib.dll", wrapper_module=None)
    assert camera is not None




def test_build_camera_and_stage_supports_explicit_micromanager_stage_for_mm_camera() -> None:
    from orca_focus.cli import _build_camera_and_stage

    args = build_parser().parse_args(["--camera", "micromanager", "--stage", "micromanager"])

    with patch("orca_focus.micromanager.create_micromanager_frame_source") as create_mm:
        core = object()
        create_mm.return_value = lambda: ([[0.0]], 0.0)
        create_mm.return_value.core = core

        with patch("orca_focus.micromanager.MicroManagerStage") as mm_stage_cls:
            mm_stage = object()
            mm_stage_cls.return_value = mm_stage

            _camera, stage = _build_camera_and_stage(args)

    assert stage is mm_stage
    mm_stage_cls.assert_called_once_with(core=core)

def test_build_stage_mcl_failure_has_actionable_message() -> None:
    from orca_focus.cli import _build_stage
    from orca_focus.hardware import NotConnectedError

    args = build_parser().parse_args(["--camera", "orca", "--stage", "mcl", "--stage-dll", "C:/bad/path/Madlib.dll"])

    with patch("orca_focus.cli.MclNanoZStage", side_effect=NotConnectedError("Failed to initialize MCL handle")):
        with pytest.raises(RuntimeError, match="--stage micromanager"):
            _build_stage(args)


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


def test_main_starts_micromanager_camera_wrapper_for_get_frame(tmp_path: Path) -> None:
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

    with patch("sys.argv", ["orca-focus", "--camera", "micromanager", "--show-live", "--calibration-csv", str(csv_path)]), \
        patch("orca_focus.cli._build_camera_and_stage", return_value=(dummy_camera, object())), \
        patch("orca_focus.cli.launch_autofocus_viewer", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            main()

    assert dummy_camera.started is True
    assert dummy_camera.stopped is True


def test_load_startup_calibration_rejects_low_signal_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "calibration_sweep.csv"
    csv_path.write_text(
        "z_um,error,weight\n0.0,0.0100,1\n1.0,0.0105,1\n2.0,0.0110,1\n3.0,0.0102,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="quality checks"):
        _load_startup_calibration(str(csv_path))




def test_main_maps_negative_af_max_excursion_to_none(tmp_path: Path) -> None:
    csv_path = tmp_path / "calibration_sweep.csv"
    csv_path.write_text("z_um,error,weight\n-1.0,-0.5,1\n0.0,0.0,1\n1.0,0.5,1\n", encoding="utf-8")

    with patch(
        "sys.argv",
        [
            "orca-focus",
            "--duration",
            "0.01",
            "--af-max-excursion-um",
            "-1",
            "--calibration-csv",
            str(csv_path),
        ],
    ), patch("orca_focus.cli.AstigmaticAutofocusController") as ctrl_cls:
        ctrl_cls.return_value.run.return_value = []
        assert main() == 0

    config = ctrl_cls.call_args.kwargs["config"]
    assert config.max_abs_excursion_um is None
    assert config.command_deadband_um == 0.02


def test_main_forwards_command_deadband(tmp_path: Path) -> None:
    csv_path = tmp_path / "calibration_sweep.csv"
    csv_path.write_text("z_um,error,weight\n-1.0,-0.5,1\n0.0,0.0,1\n1.0,0.5,1\n", encoding="utf-8")

    with patch(
        "sys.argv",
        [
            "orca-focus",
            "--duration",
            "0.01",
            "--command-deadband-um",
            "0.05",
            "--calibration-csv",
            str(csv_path),
        ],
    ), patch("orca_focus.cli.AstigmaticAutofocusController") as ctrl_cls:
        ctrl_cls.return_value.run.return_value = []
        assert main() == 0

    config = ctrl_cls.call_args.kwargs["config"]
    assert config.command_deadband_um == 0.05


def test_load_startup_calibration_uses_zero_control_error_offset(tmp_path: Path) -> None:
    csv_path = tmp_path / "calibration_sweep.csv"
    csv_path.write_text(
        "z_um,error,weight\n0.0,0.10,1\n1.0,0.20,1\n2.0,0.30,1\n",
        encoding="utf-8",
    )

    calibration = _load_startup_calibration(str(csv_path))

    assert calibration.error_to_um == pytest.approx(10.0)
    assert calibration.error_at_focus == pytest.approx(0.0)


def test_main_forwards_max_dt_s(tmp_path: Path) -> None:
    csv_path = tmp_path / "calibration_sweep.csv"
    csv_path.write_text("z_um,error,weight\n-1.0,-0.5,1\n0.0,0.0,1\n1.0,0.5,1\n", encoding="utf-8")

    with patch(
        "sys.argv",
        [
            "orca-focus",
            "--duration",
            "0.01",
            "--max-dt-s",
            "0.05",
            "--calibration-csv",
            str(csv_path),
        ],
    ), patch("orca_focus.cli.AstigmaticAutofocusController") as ctrl_cls:
        ctrl_cls.return_value.run.return_value = []
        assert main() == 0

    config = ctrl_cls.call_args.kwargs["config"]
    assert config.max_dt_s == 0.05
