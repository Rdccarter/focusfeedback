import pytest

from orca_focus.calibration import (
    CalibrationSample,
    auto_calibrate,
    calibration_quality_issues,
    fit_linear_calibration,
    fit_linear_calibration_with_report,
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

    # Forward+reverse sweep repeats successful points in both directions.
    assert len(samples) == 4


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

    assert len(events) == 6
    assert [event[0] for event in events] == [1, 2, 3, 4, 5, 6]
    assert all(event[1] == 6 for event in events)
    assert all(event[4] is True for event in events)


def test_calibration_quality_issues_flags_non_monotonic_sweeps() -> None:
    samples = [
        CalibrationSample(z_um=0.0, error=0.10),
        CalibrationSample(z_um=1.0, error=0.05),
        CalibrationSample(z_um=2.0, error=0.09),
        CalibrationSample(z_um=3.0, error=0.14),
    ]
    report = fit_linear_calibration_with_report(samples, robust=True)

    issues = calibration_quality_issues(samples, report, min_abs_corr=0.9)

    assert any("weakly correlated" in issue for issue in issues)


def test_calibration_quality_issues_accepts_well_behaved_sweep() -> None:
    samples = [
        CalibrationSample(z_um=-1.0, error=-0.5),
        CalibrationSample(z_um=0.0, error=0.0),
        CalibrationSample(z_um=1.0, error=0.5),
    ]
    report = fit_linear_calibration_with_report(samples, robust=True)

    assert calibration_quality_issues(samples, report) == []


def test_calibration_quality_issues_flags_bidirectional_hysteresis() -> None:
    samples = [
        CalibrationSample(z_um=-0.5, error=-0.20),
        CalibrationSample(z_um=0.0, error=0.00),
        CalibrationSample(z_um=0.5, error=0.20),
        CalibrationSample(z_um=0.5, error=0.26),
        CalibrationSample(z_um=0.0, error=0.04),
        CalibrationSample(z_um=-0.5, error=-0.16),
    ]
    report = fit_linear_calibration_with_report(samples, robust=True)

    issues = calibration_quality_issues(samples, report)

    assert any("up/down sweep mismatch" in issue for issue in issues)


def test_calibration_quality_issues_tolerates_small_focus_extrapolation() -> None:
    samples = [
        CalibrationSample(z_um=-0.5, error=-0.030),
        CalibrationSample(z_um=0.0, error=-0.010),
        CalibrationSample(z_um=0.5, error=0.010),
        CalibrationSample(z_um=1.0, error=0.030),
    ]
    report = fit_linear_calibration_with_report(samples, robust=True)

    # Nudge fitted center just outside sampled range to emulate small noisy extrapolation.
    report.calibration.error_at_focus = 0.035

    issues = calibration_quality_issues(samples, report)

    assert not any("outside sampled error range" in issue for issue in issues)
