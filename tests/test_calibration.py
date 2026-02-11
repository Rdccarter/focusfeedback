import pytest

from orca_focus.calibration import (
    CalibrationSample,
    fit_linear_calibration,
    load_calibration_samples_csv,
    save_calibration_samples_csv,
)


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
