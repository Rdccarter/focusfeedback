import pytest
import time

from orca_focus.autofocus import AstigmaticAutofocusController, AutofocusConfig, AutofocusWorker
from orca_focus.calibration import FocusCalibration
from orca_focus.focus_metric import Roi
from orca_focus.hardware import MclNanoZStage, SimulatedCamera, SimulatedScene
from orca_focus.interfaces import CameraFrame


def test_controller_moves_toward_focal_plane() -> None:
    stage = MclNanoZStage()
    stage.move_z_um(2.0)

    scene = SimulatedScene(focal_plane_um=0.0, alpha_px_per_um=0.25)
    camera = SimulatedCamera(stage=stage, scene=scene)
    camera.start()

    config = AutofocusConfig(
        roi=Roi(x=20, y=20, width=24, height=24),
        kp=0.8,
        ki=0.2,
        max_step_um=0.2,
    )
    calibration = FocusCalibration(error_at_focus=0.0, error_to_um=2.8)
    controller = AstigmaticAutofocusController(
        camera=camera,
        stage=stage,
        config=config,
        calibration=calibration,
    )

    before = abs(stage.get_z_um())
    for _ in range(30):
        controller.run_step()
    after = abs(stage.get_z_um())

    camera.stop()

    assert after < before


def test_controller_respects_stage_limits() -> None:
    stage = MclNanoZStage()
    stage.move_z_um(2.0)
    camera = SimulatedCamera(stage=stage, scene=SimulatedScene(focal_plane_um=0.0, alpha_px_per_um=0.25))
    camera.start()

    config = AutofocusConfig(
        roi=Roi(x=20, y=20, width=24, height=24),
        kp=1.0,
        ki=0.0,
        max_step_um=1.0,
        stage_min_um=1.4,
        stage_max_um=1.5,
    )
    controller = AstigmaticAutofocusController(
        camera=camera,
        stage=stage,
        config=config,
        calibration=FocusCalibration(error_at_focus=0.0, error_to_um=10.0),
    )

    sample = controller.run_step(dt_s=0.01)
    camera.stop()

    assert 1.4 <= sample.commanded_z_um <= 1.5
    assert 1.4 <= stage.get_z_um() <= 1.5


def test_autofocus_worker_runs_and_stops() -> None:
    stage = MclNanoZStage()
    stage.move_z_um(1.0)
    camera = SimulatedCamera(stage=stage, scene=SimulatedScene(focal_plane_um=0.0, alpha_px_per_um=0.2))
    camera.start()

    controller = AstigmaticAutofocusController(
        camera=camera,
        stage=stage,
        config=AutofocusConfig(roi=Roi(x=20, y=20, width=24, height=24), loop_hz=120.0),
        calibration=FocusCalibration(error_at_focus=0.0, error_to_um=2.0),
    )

    samples = []
    worker = AutofocusWorker(controller=controller, on_sample=samples.append)

    worker.start()
    worker.start()  # idempotent start should be safe
    time.sleep(0.05)
    worker.stop()
    camera.stop()

    assert len(samples) >= 1


def test_controller_integrator_freezes_at_stage_limits() -> None:
    """When the commanded Z is clamped to a stage limit the integrator should
    not wind up — subsequent steps should not show a growing integral offset."""
    stage = MclNanoZStage()
    stage.move_z_um(1.5)
    camera = SimulatedCamera(stage=stage, scene=SimulatedScene(focal_plane_um=0.0, alpha_px_per_um=0.25))
    camera.start()

    config = AutofocusConfig(
        roi=Roi(x=20, y=20, width=24, height=24),
        kp=1.0,
        ki=1.0,
        max_step_um=1.0,
        stage_min_um=1.4,
        stage_max_um=1.5,
    )
    controller = AstigmaticAutofocusController(
        camera=camera,
        stage=stage,
        config=config,
        calibration=FocusCalibration(error_at_focus=0.0, error_to_um=10.0),
    )

    # Run many steps while clamped — integrator should stay bounded near zero.
    for _ in range(20):
        controller.run_step(dt_s=0.01)

    integral = controller._integral_um  # noqa: SLF001
    camera.stop()

    # Without anti-windup this would grow large; with it, it should stay small.
    assert abs(integral) < config.integral_limit_um


def test_controller_ema_filter_smooths_error() -> None:
    """With a high alpha the filtered error should lag behind raw changes."""
    stage_raw = MclNanoZStage()
    stage_raw.move_z_um(2.0)
    camera_raw = SimulatedCamera(
        stage=stage_raw,
        scene=SimulatedScene(focal_plane_um=0.0, alpha_px_per_um=0.25),
    )
    camera_raw.start()

    stage_smooth = MclNanoZStage()
    stage_smooth.move_z_um(2.0)
    camera_smooth = SimulatedCamera(
        stage=stage_smooth,
        scene=SimulatedScene(focal_plane_um=0.0, alpha_px_per_um=0.25),
    )
    camera_smooth.start()

    config_raw = AutofocusConfig(
        roi=Roi(x=20, y=20, width=24, height=24),
        kp=0.8, ki=0.0, error_alpha=0.0,
    )
    config_smooth = AutofocusConfig(
        roi=Roi(x=20, y=20, width=24, height=24),
        kp=0.8, ki=0.0, error_alpha=0.8,
    )

    ctrl_raw = AstigmaticAutofocusController(
        camera=camera_raw, stage=stage_raw, config=config_raw,
        calibration=FocusCalibration(error_at_focus=0.0, error_to_um=2.8),
    )
    ctrl_smooth = AstigmaticAutofocusController(
        camera=camera_smooth, stage=stage_smooth, config=config_smooth,
        calibration=FocusCalibration(error_at_focus=0.0, error_to_um=2.8),
    )

    sample_raw = ctrl_raw.run_step()
    sample_smooth = ctrl_smooth.run_step()
    # First step with no history — both should be identical.
    assert sample_raw.error_um == sample_smooth.error_um

    # After a second step the smoothed error should lag toward the previous value.
    sample_raw2 = ctrl_raw.run_step()
    sample_smooth2 = ctrl_smooth.run_step()
    assert abs(sample_smooth2.error_um - sample_raw.error_um) < abs(sample_raw2.error_um - sample_raw.error_um)

    camera_raw.stop()
    camera_smooth.stop()


def test_controller_initial_integral() -> None:
    """Passing initial_integral_um should seed the integrator."""
    stage = MclNanoZStage()
    stage.move_z_um(0.0)
    camera = SimulatedCamera(stage=stage, scene=SimulatedScene(focal_plane_um=0.0, alpha_px_per_um=0.25))
    camera.start()

    controller = AstigmaticAutofocusController(
        camera=camera,
        stage=stage,
        config=AutofocusConfig(roi=Roi(x=20, y=20, width=24, height=24), kp=0.0, ki=1.0),
        calibration=FocusCalibration(error_at_focus=0.0, error_to_um=2.8),
        initial_integral_um=0.5,
    )
    assert controller._integral_um == 0.5  # noqa: SLF001
    camera.stop()


def test_controller_skips_duplicate_frames() -> None:
    """If the camera returns the same timestamp twice, the controller should
    freeze on the second call rather than acting on stale data."""

    frozen_ts = 1000.0
    call_count = 0

    class _StaleCamera:
        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def get_frame(self) -> CameraFrame:
            nonlocal call_count
            call_count += 1
            # Return the same timestamp for the first two calls, then a new one.
            ts = frozen_ts if call_count <= 2 else frozen_ts + 1.0
            return CameraFrame(
                image=[[0.0] * 64 for _ in range(64)],
                timestamp_s=ts,
            )

    stage = MclNanoZStage()
    stage.move_z_um(1.0)
    camera = _StaleCamera()

    controller = AstigmaticAutofocusController(
        camera=camera,
        stage=stage,
        config=AutofocusConfig(roi=Roi(x=20, y=20, width=24, height=24), kp=0.8, ki=0.0),
        calibration=FocusCalibration(error_at_focus=0.0, error_to_um=2.8),
    )

    s1 = controller.run_step()
    assert s1.control_applied is True  # first frame is always fresh

    s2 = controller.run_step()
    assert s2.control_applied is False  # duplicate timestamp → skipped

    s3 = controller.run_step()
    assert s3.control_applied is True  # new timestamp → processed


def test_controller_rejects_invalid_loop_hz() -> None:
    stage = MclNanoZStage()
    camera = SimulatedCamera(stage=stage)
    camera.start()

    with pytest.raises(ValueError, match="loop_hz must be > 0"):
        AstigmaticAutofocusController(
            camera=camera,
            stage=stage,
            config=AutofocusConfig(roi=Roi(x=20, y=20, width=24, height=24), loop_hz=0.0),
            calibration=FocusCalibration(error_at_focus=0.0, error_to_um=2.8),
        )

    camera.stop()


def test_autofocus_worker_stop_nonblocking() -> None:
    stage = MclNanoZStage()
    stage.move_z_um(1.0)
    camera = SimulatedCamera(stage=stage, scene=SimulatedScene(focal_plane_um=0.0, alpha_px_per_um=0.2))
    camera.start()

    controller = AstigmaticAutofocusController(
        camera=camera,
        stage=stage,
        config=AutofocusConfig(roi=Roi(x=20, y=20, width=24, height=24), loop_hz=120.0),
        calibration=FocusCalibration(error_at_focus=0.0, error_to_um=2.0),
    )

    worker = AutofocusWorker(controller=controller)
    worker.start()
    worker.stop(wait=False)
    # allow thread to observe stop event
    time.sleep(0.01)
    camera.stop()
