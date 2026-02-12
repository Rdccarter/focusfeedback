import pytest

from orca_focus.calibration import (
    CalibrationSample,
    auto_calibrate,
    fit_linear_calibration,
    load_calibration_samples_csv,
    save_calibration_samples_csv,
)
from orca_focus.focus_metric import Roi
from orca_focus.interfaces import CameraFrame


def test_fit_linear_calibration_recovers_mapping() -> None:
    samples = [
        CalibrationSample(z_um=-1.0, error=-0.5),
        CalibrationSample(z_um=0.0, error=0.0),
        CalibrationSample(z_um=1.0, error=0.5),
    ]
    cal = fit_linear_calibration(samples)

    assert abs(cal.error_at_focus) < 1e-9
    assert abs(cal.error_to_um - 2.0) < 1e-9


def test_fit_linear_calibration_requires_at_least_two_samples() -> None:
    with pytest.raises(ValueError, match="Need at least two calibration samples"):
        fit_linear_calibration([CalibrationSample(z_um=0.0, error=0.0)])


def test_fit_linear_calibration_rejects_degenerate_samples() -> None:
    samples = [
        CalibrationSample(z_um=-1.0, error=0.0),
        CalibrationSample(z_um=0.0, error=0.0),
        CalibrationSample(z_um=1.0, error=0.0),
    ]

    with pytest.raises(ValueError, match="Calibration samples are degenerate"):
        fit_linear_calibration(samples)


def test_calibration_samples_csv_round_trip(tmp_path) -> None:
    samples = [
        CalibrationSample(z_um=-0.2, error=-0.1, weight=10.0),
        CalibrationSample(z_um=0.0, error=0.0, weight=20.0),
        CalibrationSample(z_um=0.2, error=0.1, weight=30.0),
    ]
    csv_path = tmp_path / "calibration_sweep.csv"

    save_calibration_samples_csv(csv_path, samples)
    loaded = load_calibration_samples_csv(csv_path)

    assert loaded == samples


def test_auto_calibrate_skips_failed_stage_moves() -> None:
    class _Stage:
        def __init__(self):
            self.z = 0.0

        def move_z_um(self, target_z_um: float) -> None:
            if target_z_um > 0.2:
                raise RuntimeError("MCL_SingleWriteN failed with status 6")
            self.z = target_z_um

        def get_z_um(self) -> float:
            return self.z

    class _Camera:
        def get_frame(self) -> CameraFrame:
            return CameraFrame(image=[[0.0] * 64 for _ in range(64)], timestamp_s=0.0)

    samples = auto_calibrate(
        camera=_Camera(),
        stage=_Stage(),
        roi=Roi(x=20, y=20, width=24, height=24),
        z_min_um=-0.2,
        z_max_um=0.4,
        n_steps=4,
    )

    # Two points should succeed: -0.2 and 0.0
    assert len(samples) == 2


def test_auto_calibrate_raises_clear_error_if_too_many_moves_fail() -> None:
    class _Stage:
        def move_z_um(self, target_z_um: float) -> None:
            raise RuntimeError("MCL_SingleWriteN failed with status 6")

        def get_z_um(self) -> float:
            return 0.0

    class _Camera:
        def get_frame(self) -> CameraFrame:
            return CameraFrame(image=[[0.0] * 64 for _ in range(64)], timestamp_s=0.0)

    with pytest.raises(RuntimeError, match="could not collect enough valid points"):
        auto_calibrate(
            camera=_Camera(),
            stage=_Stage(),
            roi=Roi(x=20, y=20, width=24, height=24),
            z_min_um=-0.2,
            z_max_um=0.2,
            n_steps=3,
        )


def test_auto_calibrate_can_be_cancelled_via_should_stop() -> None:
    class _Stage:
        def move_z_um(self, target_z_um: float) -> None:
            return None

        def get_z_um(self) -> float:
            return 0.0

    class _Camera:
        def get_frame(self) -> CameraFrame:
            return CameraFrame(image=[[0.0] * 64 for _ in range(64)], timestamp_s=0.0)

    with pytest.raises(RuntimeError, match="cancelled"):
        auto_calibrate(
            camera=_Camera(),
            stage=_Stage(),
            roi=Roi(x=20, y=20, width=24, height=24),
            z_min_um=-0.2,
            z_max_um=0.2,
            n_steps=3,
            should_stop=lambda: True,
        )


def test_auto_calibrate_reports_step_progress() -> None:
    class _Stage:
        def __init__(self):
            self.z = 0.0

        def move_z_um(self, target_z_um: float) -> None:
            self.z = target_z_um

        def get_z_um(self) -> float:
            return self.z

    class _Camera:
        def get_frame(self) -> CameraFrame:
            return CameraFrame(image=[[0.0] * 64 for _ in range(64)], timestamp_s=0.0)

    events: list[tuple[int, int, float, float | None, bool]] = []
    auto_calibrate(
        camera=_Camera(),
        stage=_Stage(),
        roi=Roi(x=20, y=20, width=24, height=24),
        z_min_um=-0.2,
        z_max_um=0.2,
        n_steps=3,
        on_step=lambda i, total, target, measured, ok: events.append(
            (i, total, target, measured, ok)
        ),
    )

    assert len(events) == 3
    assert [event[0] for event in events] == [1, 2, 3]
    assert all(event[1] == 3 for event in events)
    assert all(event[4] is True for event in events)
