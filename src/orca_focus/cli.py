from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .autofocus import AstigmaticAutofocusController, AutofocusConfig
from .calibration import (
    FocusCalibration,
    calibration_quality_issues,
    fit_linear_calibration_with_report,
    load_calibration_samples_csv,
    validate_calibration_sign,
)
from .focus_metric import Roi
from .hardware import HamamatsuOrcaCamera, MclNanoZStage, NotConnectedError, SimulatedCamera
from .interfaces import StageInterface
from .interactive import launch_autofocus_viewer
from .pylablib_camera import create_pylablib_frame_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run astigmatic autofocus loop")
    parser.add_argument("--duration", type=float, default=2.0, help="Loop runtime in seconds")
    parser.add_argument("--loop-hz", type=float, default=30.0, help="Control loop frequency")
    parser.add_argument(
        "--show-live",
        action="store_true",
        help="Open interactive live image viewer with ROI selection",
    )
    parser.add_argument(
        "--camera",
        choices=["simulate", "orca", "andor", "micromanager"],
        default="simulate",
        help="Camera backend selection",
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, help="Camera index for pylablib backend"
    )
    parser.add_argument("--mm-host", default="localhost", help="Micro-Manager pycromanager host")
    parser.add_argument("--mm-port", type=int, default=4827, help="Micro-Manager pycromanager port")
    parser.add_argument(
        "--mm-allow-standalone-core",
        action="store_true",
        help=(
            "Allow fallback to standalone pymmcore/MMCorePy CMMCore if bridge attach fails. "
            "Use only when intentionally running without attaching to MM GUI."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=["mcl", "simulate", "micromanager"],
        default=None,
        help=(
            "Stage backend. Defaults: simulate camera -> simulate stage, "
            "all hardware cameras (including micromanager) -> mcl stage."
        ),
    )
    parser.add_argument("--stage-dll", default=None, help="Path to MCL stage DLL")
    parser.add_argument(
        "--stage-wrapper", default=None, help="Python module path for MCL wrapper"
    )
    parser.add_argument("--kp", type=float, default=0.8, help="Proportional gain")
    parser.add_argument("--ki", type=float, default=0.2, help="Integral gain")
    parser.add_argument("--max-step", type=float, default=0.2, help="Max correction step in µm")
    parser.add_argument("--stage-min-um", type=float, default=None, help="Lower clamp for commanded stage Z (µm)")
    parser.add_argument("--stage-max-um", type=float, default=None, help="Upper clamp for commanded stage Z (µm)")
    parser.add_argument("--af-max-excursion-um", type=float, default=5.0, help="Max allowed autofocus excursion from initial Z lock point (µm); set negative to disable")
    parser.add_argument(
        "--calibration-csv",
        default="calibration_sweep.csv",
        help=(
            "Calibration samples CSV path. If present, it is loaded on startup to build the "
            "focus calibration automatically; GUI calibration sweeps also write to this path."
        ),
    )
    parser.add_argument(
        "--calibration-half-range-um",
        type=float,
        default=0.75,
        help="Half-range around current Z for napari calibration sweep",
    )
    parser.add_argument(
        "--calibration-steps",
        type=int,
        default=21,
        help="Number of Z points for napari calibration sweep",
    )
    return parser


def _load_startup_calibration(samples_csv: str | None) -> FocusCalibration:
    if not samples_csv:
        raise ValueError(
            "Calibration CSV path is required. Provide --calibration-csv to reuse a saved sweep."
        )

    csv_path = Path(samples_csv)
    if not csv_path.exists():
        raise ValueError(
            f"Calibration CSV not found: {csv_path}. Run GUI calibration first to generate it."
        )

    samples = load_calibration_samples_csv(csv_path)
    report = fit_linear_calibration_with_report(samples, robust=True)
    print(
        "Loaded calibration "
        f"({csv_path}): slope={report.calibration.error_to_um:+0.4f} um/error, "
        f"error_at_focus={report.calibration.error_at_focus:+0.4f}, "
        f"R^2={report.r2:0.4f}, inliers={report.n_inliers}/{report.n_samples}",
        file=sys.stderr,
    )

    issues = calibration_quality_issues(samples, report)
    if issues:
        raise ValueError(
            "Calibration CSV failed quality checks: " + " ; ".join(issues)
        )

    if report.r2 < 0.9:
        print(
            f"Warning: calibration R^2={report.r2:0.4f} is below 0.9; "
            "fit quality is poor. Consider re-running the calibration sweep "
            "with a smaller Z range or checking that the fiducial is in the ROI.",
            file=sys.stderr,
        )

    try:
        validate_calibration_sign(report.calibration, expected_positive_slope=True)
    except ValueError as sign_err:
        print(f"Warning: {sign_err}", file=sys.stderr)
        print(
            "The calibration slope sign may be inverted for your optical setup. "
            "If autofocus diverges, negate the slope or check your cylindrical "
            "lens orientation.",
            file=sys.stderr,
        )

    return report.calibration


def _build_stage(args, *, mm_core=None) -> StageInterface:
    stage_backend = args.stage or ("simulate" if args.camera == "simulate" else "mcl")

    if stage_backend == "simulate":
        if args.stage_dll is not None or args.stage_wrapper is not None:
            print(
                "Warning: --stage simulate ignores --stage-dll/--stage-wrapper inputs.",
                file=sys.stderr,
            )
        return MclNanoZStage()

    if stage_backend == "micromanager":
        if mm_core is None:
            raise RuntimeError(
                "Micro-Manager stage backend requested but no Micro-Manager core is available. "
                "Use --camera micromanager or choose --stage mcl/simulate."
            )
        if args.stage_dll is not None or args.stage_wrapper is not None:
            print(
                "Warning: --stage micromanager ignores --stage-dll/--stage-wrapper inputs.",
                file=sys.stderr,
            )
        from .micromanager import MicroManagerStage

        return MicroManagerStage(core=mm_core)

    try:
        stage = MclNanoZStage(dll_path=args.stage_dll, wrapper_module=args.stage_wrapper)
    except (NotConnectedError, OSError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"Failed to initialize MCL stage: {exc}. "
            "If the stage is controlled through Micro-Manager, use --stage micromanager. "
            "To run without hardware stage control, use --stage simulate."
        ) from exc
    if args.stage_dll is None and args.stage_wrapper is None:
        print(
            "Warning: no explicit MCL stage backend configured; using in-memory simulated stage.",
            file=sys.stderr,
        )
    return stage


def _build_camera_and_stage(args):
    # Simulated camera + in-memory stage
    if args.camera == "simulate":
        stage = _build_stage(args)
        stage.move_z_um(1.5)
        camera = SimulatedCamera(stage=stage)
        return camera, stage

    # Micro-Manager camera stream (MM owns camera only)
    if args.camera == "micromanager":
        from .micromanager import create_micromanager_frame_source

        mm_source = create_micromanager_frame_source(
            host=args.mm_host,
            port=args.mm_port,
            allow_standalone_core=args.mm_allow_standalone_core,
        )
        stage = _build_stage(args, mm_core=mm_source.core)
        camera = HamamatsuOrcaCamera(
            frame_source=mm_source,
            control_source_lifecycle=False,
        )
        return camera, stage

    # pylablib camera (orca / andor) + package-controlled stage
    stage = _build_stage(args)
    frame_source = create_pylablib_frame_source(args.camera, idx=args.camera_index)
    camera = HamamatsuOrcaCamera(frame_source=frame_source, control_source_lifecycle=True)
    return camera, stage


def main() -> int:
    args = build_parser().parse_args()

    camera, stage = _build_camera_and_stage(args)
    camera_started = False

    try:
        camera.start()
        camera_started = True

        config = AutofocusConfig(
            roi=Roi(x=20, y=20, width=24, height=24),
            loop_hz=args.loop_hz,
            kp=args.kp,
            ki=args.ki,
            max_step_um=args.max_step,
            stage_min_um=args.stage_min_um,
            stage_max_um=args.stage_max_um,
            max_abs_excursion_um=(None if args.af_max_excursion_um < 0 else args.af_max_excursion_um),
        )
        try:
            calibration = _load_startup_calibration(args.calibration_csv)
        except ValueError as exc:
            if not args.show_live:
                raise
            print(f"Warning: {exc}", file=sys.stderr)
            print(
                "Warning: using temporary default calibration for live setup only; "
                "run calibration sweep and restart to use saved calibration.",
                file=sys.stderr,
            )
            calibration = FocusCalibration(error_at_focus=0.0, error_to_um=1.0)

        if args.show_live:
            launch_autofocus_viewer(
                camera,
                stage,
                calibration=calibration,
                default_config=config,
                calibration_output_path=args.calibration_csv,
                calibration_half_range_um=args.calibration_half_range_um,
                calibration_steps=args.calibration_steps,
            )
            return 0

        controller = AstigmaticAutofocusController(
            camera=camera,
            stage=stage,
            config=config,
            calibration=calibration,
        )

        samples = controller.run(duration_s=args.duration)

        if samples:
            final = samples[-1]
            print(
                f"camera={args.camera} steps={len(samples)} final_error={final.error:+0.4f} "
                f"final_error_um={final.error_um:+0.3f} stage={final.commanded_z_um:+0.3f} um"
            )
        return 0
    finally:
        if camera_started:
            camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
