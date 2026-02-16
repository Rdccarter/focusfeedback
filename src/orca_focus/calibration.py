from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .focus_metric import Roi, astigmatic_error_signal, roi_total_intensity
from .interfaces import CameraInterface, StageInterface


@dataclass(slots=True)
class FocusCalibration:
    """Maps astigmatic error to physical Z offset.

    The mapping follows a local linear approximation around focus:
    z_offset_um ~= error_to_um * (error - error_at_focus)

    Note: `error_at_focus` is derived from calibration samples and is interpreted
    in the local sweep frame used by the fitter. With symmetric sweeps centered
    near focus this approximates true best-focus error well. Strongly asymmetric
    sweeps can bias this estimate; prefer centered bidirectional sweeps.
    """

    error_at_focus: float
    error_to_um: float

    def error_to_z_offset_um(self, error: float) -> float:
        return (error - self.error_at_focus) * self.error_to_um


@dataclass(slots=True)
class CalibrationSample:
    z_um: float
    error: float
    weight: float = 1.0


@dataclass(slots=True)
class CalibrationFitReport:
    calibration: FocusCalibration
    intercept_um: float
    r2: float
    rmse_um: float
    n_samples: int
    n_inliers: int
    robust: bool


def _weighted_linear_fit(samples: list[CalibrationSample]) -> tuple[float, float]:
    if len(samples) < 2:
        raise ValueError("Need at least two calibration samples")

    sum_w = sum(max(0.0, s.weight) for s in samples)
    if sum_w <= 0:
        raise ValueError("Calibration sample weights must contain positive mass")

    sum_e = sum(max(0.0, s.weight) * s.error for s in samples)
    sum_z = sum(max(0.0, s.weight) * s.z_um for s in samples)
    sum_ee = sum(max(0.0, s.weight) * s.error * s.error for s in samples)
    sum_ez = sum(max(0.0, s.weight) * s.error * s.z_um for s in samples)

    denom = sum_w * sum_ee - sum_e * sum_e
    if denom == 0.0:
        raise ValueError("Calibration samples are degenerate")

    slope = (sum_w * sum_ez - sum_e * sum_z) / denom
    intercept = (sum_z - slope * sum_e) / sum_w
    if slope == 0.0:
        raise ValueError("Calibration slope is zero")
    return slope, intercept




def _weighted_z_reference(samples: list[CalibrationSample]) -> float:
    sum_w = sum(max(0.0, s.weight) for s in samples)
    if sum_w <= 0:
        raise ValueError("Calibration sample weights must contain positive mass")
    return sum(max(0.0, s.weight) * s.z_um for s in samples) / sum_w


def _center_samples_on_reference(
    samples: list[CalibrationSample],
    z_reference_um: float,
) -> list[CalibrationSample]:
    return [
        CalibrationSample(
            z_um=s.z_um - z_reference_um,
            error=s.error,
            weight=s.weight,
        )
        for s in samples
    ]
def _robust_seed_fit(samples: list[CalibrationSample]) -> tuple[float, float]:
    if len(samples) < 2:
        raise ValueError("Need at least two calibration samples")

    best: tuple[float, float] | None = None
    best_med = float("inf")
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            e0 = samples[i].error
            e1 = samples[j].error
            if e1 == e0:
                continue
            slope = (samples[j].z_um - samples[i].z_um) / (e1 - e0)
            if slope == 0.0:
                continue
            intercept = samples[i].z_um - slope * e0
            residuals = [abs(s.z_um - (slope * s.error + intercept)) for s in samples]
            residuals.sort()
            med = residuals[len(residuals) // 2]
            if med < best_med:
                best_med = med
                best = (slope, intercept)
    if best is None:
        return _weighted_linear_fit(samples)
    return best


def _fit_report(
    samples: list[CalibrationSample],
    slope: float,
    intercept: float,
    *,
    robust: bool,
    n_inliers: int,
    metric_samples: list[CalibrationSample] | None = None,
) -> CalibrationFitReport:
    error_at_focus = -intercept / slope
    cal = FocusCalibration(error_at_focus=error_at_focus, error_to_um=slope)

    metric_samples = samples if metric_samples is None else metric_samples
    weights = [max(0.0, s.weight) for s in metric_samples]
    w_sum = sum(weights)
    if w_sum <= 0:
        raise ValueError("Calibration sample weights must contain positive mass")

    z_mean = sum(w * s.z_um for w, s in zip(weights, metric_samples)) / w_sum
    ss_res = 0.0
    ss_tot = 0.0
    for w, s in zip(weights, metric_samples):
        pred = slope * s.error + intercept
        ss_res += w * ((s.z_um - pred) ** 2)
        ss_tot += w * ((s.z_um - z_mean) ** 2)
    r2 = 1.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    rmse = (ss_res / w_sum) ** 0.5

    return CalibrationFitReport(
        calibration=cal,
        intercept_um=intercept,
        r2=r2,
        rmse_um=rmse,
        n_samples=len(samples),
        n_inliers=n_inliers,
        robust=robust,
    )


def fit_linear_calibration_with_report(
    samples: list[CalibrationSample],
    *,
    robust: bool = False,
    outlier_threshold_um: float = 0.2,
) -> CalibrationFitReport:
    """Fit z = slope*error + intercept and return quality metrics."""

    # Fit in a local Z frame to avoid large absolute-stage offsets skewing
    # intercept-derived error_at_focus estimates. This keeps slope unchanged.
    # Caveat: if the sweep is strongly asymmetric around true focus, the local
    # reference can introduce small bias in error_at_focus (symmetric sweeps are
    # recommended and are the GUI default).
    z_reference_um = _weighted_z_reference(samples)
    centered_samples = _center_samples_on_reference(samples, z_reference_um)

    slope, intercept = _weighted_linear_fit(centered_samples)
    n_inliers = len(centered_samples)

    if robust:
        seed_slope, seed_intercept = _robust_seed_fit(centered_samples)
        inliers: list[CalibrationSample] = []
        for s in centered_samples:
            pred = seed_slope * s.error + seed_intercept
            if abs(s.z_um - pred) <= outlier_threshold_um:
                inliers.append(s)
        if len(inliers) >= 2:
            slope, intercept = _weighted_linear_fit(inliers)
            n_inliers = len(inliers)

    metric_samples = centered_samples
    if robust and n_inliers < len(centered_samples):
        seed_slope, seed_intercept = _robust_seed_fit(centered_samples)
        metric_samples = [
            s
            for s in centered_samples
            if abs(s.z_um - (seed_slope * s.error + seed_intercept)) <= outlier_threshold_um
        ]
        if len(metric_samples) < 2:
            metric_samples = centered_samples

    return _fit_report(
        centered_samples,
        slope,
        intercept,
        robust=robust,
        n_inliers=n_inliers,
        metric_samples=metric_samples,
    )


def fit_linear_calibration(
    samples: list[CalibrationSample],
    *,
    robust: bool = False,
    outlier_threshold_um: float = 0.2,
) -> FocusCalibration:
    """Fit z = slope*error + intercept via weighted ordinary least squares."""

    report = fit_linear_calibration_with_report(
        samples,
        robust=robust,
        outlier_threshold_um=outlier_threshold_um,
    )
    return report.calibration


def auto_calibrate(
    camera: CameraInterface,
    stage: StageInterface,
    roi: Roi,
    *,
    z_min_um: float,
    z_max_um: float,
    n_steps: int,
    bidirectional: bool = True,
    should_stop: Callable[[], bool] | None = None,
    on_step: Callable[[int, int, float, float | None, bool], None] | None = None,
) -> list[CalibrationSample]:
    """Collect calibration samples from a deterministic stage sweep."""

    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if z_max_um <= z_min_um:
        raise ValueError("z_max_um must be greater than z_min_um")

    step = (z_max_um - z_min_um) / float(n_steps - 1)
    forward_targets = [z_min_um + i * step for i in range(n_steps)]
    targets = forward_targets
    if bidirectional:
        targets = forward_targets + list(reversed(forward_targets))

    out: list[CalibrationSample] = []
    failed_moves: list[tuple[float, Exception]] = []
    total_steps = len(targets)
    for i, target_z in enumerate(targets):
        if should_stop is not None and should_stop():
            raise RuntimeError("Calibration cancelled by user")

        step_index = i + 1
        try:
            stage.move_z_um(target_z)
        except Exception as exc:
            failed_moves.append((target_z, exc))
            if on_step is not None:
                on_step(step_index, total_steps, target_z, None, False)
            continue

        frame = camera.get_frame()
        err = astigmatic_error_signal(frame.image, roi)
        weight = roi_total_intensity(frame.image, roi)

        # Record where the stage actually ended up (important if hardware clamps).
        measured_z = target_z
        try:
            measured_z = float(stage.get_z_um())
        except Exception:
            pass

        if on_step is not None:
            on_step(step_index, total_steps, target_z, measured_z, True)

        out.append(CalibrationSample(z_um=measured_z, error=err, weight=max(0.0, weight)))

    if len(out) < 2:
        if failed_moves:
            first_z, first_exc = failed_moves[0]
            raise RuntimeError(
                "Calibration sweep could not collect enough valid points: "
                f"{len(out)} succeeded, {len(failed_moves)} failed. "
                f"First failed move at z={first_z:+0.3f} um: {first_exc}"
            ) from first_exc
        raise RuntimeError(
            "Calibration sweep could not collect enough valid points; "
            "need at least 2 successful stage positions."
        )

    return out


def save_calibration_samples_csv(path: str | Path, samples: list[CalibrationSample]) -> None:
    """Write calibration sweep samples for later GUI/model reuse."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["z_um", "error", "weight"])
        writer.writeheader()
        for s in samples:
            writer.writerow({"z_um": s.z_um, "error": s.error, "weight": s.weight})


def load_calibration_samples_csv(path: str | Path) -> list[CalibrationSample]:
    """Read calibration sweep samples previously exported from the GUI."""

    in_path = Path(path)
    with in_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out: list[CalibrationSample] = []
        for row in reader:
            out.append(
                CalibrationSample(
                    z_um=float(row["z_um"]),
                    error=float(row["error"]),
                    weight=float(row.get("weight", "1.0")),
                )
            )
    return out


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    var_x = sum((x - x_mean) ** 2 for x in xs)
    var_y = sum((y - y_mean) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return 0.0
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return cov / ((var_x * var_y) ** 0.5)


def calibration_quality_issues(
    samples: list[CalibrationSample],
    report: CalibrationFitReport,
    *,
    min_abs_corr: float = 0.2,
    min_error_span: float = 0.01,
    focus_margin_fraction: float = 0.1,
    max_bidirectional_hysteresis: float = 0.02,
) -> list[str]:
    """Return human-readable issues when a sweep is not safely usable for control."""

    if len(samples) < 2:
        return ["need at least 2 samples"]

    errors = [s.error for s in samples]
    z_vals = [s.z_um for s in samples]
    min_err = min(errors)
    max_err = max(errors)
    err_span = max_err - min_err

    issues: list[str] = []

    if err_span < min_error_span:
        issues.append(
            f"error span too small ({err_span:0.4f}); increase Z range or improve ROI SNR"
        )

    abs_corr = abs(_pearson_corr(z_vals, errors))
    if abs_corr < min_abs_corr:
        # Astigmatic curves are often locally non-linear around lobe transitions.
        # Keep this as advisory text while relying on fit+range checks for gating.
        issues.append(
            f"error-vs-Z is weakly correlated (|corr|={abs_corr:0.3f}); keep ROI centered and reduce sweep range around focus"
        )

    # For bidirectional sweeps (up/down), the same Z is sampled twice. Ensure
    # the error signal is reasonably consistent to catch backlash/hysteresis.
    z_to_errors: dict[float, list[float]] = {}
    for s in samples:
        key = round(float(s.z_um), 3)
        z_to_errors.setdefault(key, []).append(float(s.error))
    hysteresis_deltas = [max(v) - min(v) for v in z_to_errors.values() if len(v) > 1]
    if hysteresis_deltas and (max(hysteresis_deltas) > max_bidirectional_hysteresis):
        issues.append(
            "up/down sweep mismatch is high (possible backlash or stage settling issue); "
            "reduce step size, slow sweep, or tighten stage settling"
        )

    err0 = report.calibration.error_at_focus
    nearest_err_dist = min(abs(err - err0) for err in errors)
    # Be tolerant to slightly out-of-range fitted centers: astigmatic curves can
    # be asymmetric/noisy near the lobe crossover even when focus is bracketed.
    margin = max(0.02, err_span * focus_margin_fraction)
    near_focus_tolerance = max(0.02, err_span * 0.25)
    if (err0 < (min_err - margin) or err0 > (max_err + margin)) and (
        nearest_err_dist > near_focus_tolerance
    ):
        issues.append(
            "fitted focus lies outside sampled error range; sweep likely does not bracket focus"
        )

    return issues


def validate_calibration_sign(
    calibration: FocusCalibration,
    *,
    expected_positive_slope: bool = True,
) -> None:
    """Raise if fitted slope sign does not match the expected setup convention."""

    if expected_positive_slope and calibration.error_to_um <= 0:
        raise ValueError("Calibration slope sign is inverted for expected-positive setup")
    if (not expected_positive_slope) and calibration.error_to_um >= 0:
        raise ValueError("Calibration slope sign is inverted for expected-negative setup")
