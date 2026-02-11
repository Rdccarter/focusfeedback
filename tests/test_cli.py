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
