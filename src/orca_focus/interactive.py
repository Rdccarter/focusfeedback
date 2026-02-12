from __future__ import annotations

import os
import threading
from pathlib import Path

from .autofocus import (
    AstigmaticAutofocusController,
    AutofocusConfig,
    AutofocusSample,
    AutofocusWorker,
)
from .calibration import (
    FocusCalibration,
    auto_calibrate,
    fit_linear_calibration_with_report,
    save_calibration_samples_csv,
)
from .focus_metric import Roi
from .interfaces import CameraInterface, StageInterface


def _prepare_napari_environment() -> None:
    """Set safe defaults to avoid third-party napari plugin crashes.

    Some environments have incompatible external napari plugins (e.g. pydantic
    API mismatches) that can crash viewer startup and interaction. Disabling
    external plugin auto-discovery keeps core viewer behavior stable.
    """
    os.environ.setdefault("NAPARI_DISABLE_PLUGINS", "1")


def launch_autofocus_viewer(
    camera: CameraInterface,
    stage: StageInterface,
    *,
    calibration: FocusCalibration,
    default_config: AutofocusConfig,
    interval_ms: int = 20,
    calibration_output_path: str | None = None,
    calibration_half_range_um: float = 0.75,
    calibration_steps: int = 21,
) -> None:
    """Live napari viewer with interactive ROI selection and background autofocus.

    Controls:
    - Draw ROI rectangle on the "ROI" layer to start/re-target autofocus.
    - Click the "Run Calibration Sweep" button (or press `c`) to sweep Z and save CSV.
    - Press `Escape` to stop and close.
    """

    _prepare_napari_environment()

    try:
        import napari
        import numpy as np
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import QPushButton
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "napari is required for the live viewer. Install with: pip install napari"
        ) from exc

    current_calibration = calibration

    initial = np.asarray(camera.get_frame().image)
    viewer = napari.Viewer(title="Autofocus â€” draw ROI to start")
    image_layer = viewer.add_image(initial, name="camera", blending="opaque")

    roi_layer = viewer.add_shapes(
        name="ROI",
        edge_color="cyan",
        edge_width=2,
        face_color="transparent",
    )
    roi_layer.mode = "add_rectangle"

    status_text = viewer.text_overlay
    status_text.visible = True
    status_text.position = "top_left"
    status_text.font_size = 12
    status_text.text = "Draw a rectangle on the PSF to begin autofocus"

    state: dict = {
        "worker": None,
        "last_sample": None,
        "last_roi": None,
        "calibration_message": "",
        "calibration_busy": False,
    }

    def _roi_from_rectangle(rect_coords) -> Roi:
        arr = np.asarray(rect_coords)
        y_min = int(np.floor(arr[:, 0].min()))
        y_max = int(np.ceil(arr[:, 0].max()))
        x_min = int(np.floor(arr[:, 1].min()))
        x_max = int(np.ceil(arr[:, 1].max()))
        w = max(1, x_max - x_min)
        h = max(1, y_max - y_min)
        return Roi(x=max(0, x_min), y=max(0, y_min), width=w, height=h)

    def _stop_worker() -> None:
        worker = state.get("worker")
        if worker is not None:
            worker.stop()
            state["worker"] = None

    def _on_sample(sample: AutofocusSample) -> None:
        state["last_sample"] = sample

    def _start_autofocus(roi: Roi) -> None:
        _stop_worker()
        config = AutofocusConfig(
            roi=roi,
            loop_hz=default_config.loop_hz,
            kp=default_config.kp,
            ki=default_config.ki,
            max_step_um=default_config.max_step_um,
            integral_limit_um=default_config.integral_limit_um,
            stage_min_um=default_config.stage_min_um,
            stage_max_um=default_config.stage_max_um,
            min_roi_intensity=default_config.min_roi_intensity,
            error_alpha=default_config.error_alpha,
            edge_margin_px=default_config.edge_margin_px,
        )
        controller = AstigmaticAutofocusController(
            camera=camera,
            stage=stage,
            config=config,
            calibration=current_calibration,
        )
        worker = AutofocusWorker(controller=controller, on_sample=_on_sample)
        state["worker"] = worker
        state["last_roi"] = roi
        worker.start()

    def _on_roi_change(_event=None) -> None:
        shapes = roi_layer.data
        if not shapes:
            return
        rect = shapes[-1]
        try:
            roi = _roi_from_rectangle(rect)
        except Exception:
            return
        _start_autofocus(roi)

    def _run_calibration_sweep(roi: Roi) -> None:
        nonlocal current_calibration
        try:
            state["calibration_busy"] = True
            _stop_worker()
            center_z = float(stage.get_z_um())
            z_min = center_z - float(calibration_half_range_um)
            z_max = center_z + float(calibration_half_range_um)
            samples = auto_calibrate(
                camera,
                stage,
                roi,
                z_min_um=z_min,
                z_max_um=z_max,
                n_steps=int(calibration_steps),
            )
            stage.move_z_um(center_z)

            if calibration_output_path:
                out_path = Path(calibration_output_path)
            else:
                out_path = Path.cwd() / "calibration_sweep.csv"
            save_calibration_samples_csv(out_path, samples)

            report = fit_linear_calibration_with_report(samples, robust=True)
            current_calibration = report.calibration
            state["calibration_message"] = (
                "Calibration complete: "
                f"{len(samples)} samples saved to {out_path} | "
                f"slope={report.calibration.error_to_um:+0.4f} um/error, "
                f"error_at_focus={report.calibration.error_at_focus:+0.4f}, "
                f"RÂ²={report.r2:0.4f}"
            )
        except Exception as exc:  # pragma: no cover
            state["calibration_message"] = f"Calibration failed: {exc}"
        finally:
            state["calibration_busy"] = False
            # Use the latest ROI (user may have redrawn during the sweep).
            restart_roi = state.get("last_roi") or roi
            _start_autofocus(restart_roi)

    def _trigger_calibration() -> None:
        if state.get("calibration_busy"):
            return
        roi = state.get("last_roi")
        if roi is None:
            state["calibration_message"] = "Calibration needs an ROI: draw a rectangle first"
            return
        state["calibration_message"] = "Calibration sweep runningâ€¦"
        t = threading.Thread(target=_run_calibration_sweep, args=(roi,), daemon=True)
        t.start()

    calibrate_button = QPushButton("Run Calibration Sweep")
    calibrate_button.setToolTip(
        "Pause autofocus, sweep Z around the current position, and export calibration CSV"
    )
    calibrate_button.clicked.connect(_trigger_calibration)
    viewer.window.add_dock_widget(calibrate_button, area="right", name="Calibration")

    @viewer.bind_key("c")
    def _calibrate(_viewer_ref):  # noqa: ARG001
        _trigger_calibration()

    roi_layer.events.data.connect(_on_roi_change)

    timer = QTimer()

    def _refresh() -> None:
        frame = camera.get_frame()
        image_layer.data = np.asarray(frame.image)

        sample = state.get("last_sample")
        if sample is not None:
            ctrl = "ON " if sample.control_applied else "OFF"
            status_text.text = (
                f"AF {ctrl} | "
                f"err={sample.error:+.4f}  err_um={sample.error_um:+.3f}  "
                f"z={sample.stage_z_um:+.3f} â†’ {sample.commanded_z_um:+.3f} Âµm  "
                f"I={sample.roi_total_intensity:.0f}"
            )
            worker = state.get("worker")
            if worker is not None and worker.last_error is not None:
                status_text.text += f"  âš  {worker.last_error}"
        elif not roi_layer.data:
            status_text.text = "Draw a rectangle on the PSF to begin autofocus"
        else:
            status_text.text = "Starting autofocusâ€¦"

        if state.get("calibration_message"):
            status_text.text += f"\n{state['calibration_message']}"

    timer.timeout.connect(_refresh)
    timer.start(max(1, int(interval_ms)))

    @viewer.bind_key("Escape")
    def _quit(viewer_ref):  # noqa: ARG001
        _stop_worker()
        viewer_ref.close()

    viewer.window._orca_focus_timer = timer  # type: ignore[attr-defined]
    napari.run()
    _stop_worker()


def launch_napari_viewer(camera: CameraInterface, interval_ms: int = 20) -> None:
    """Display a live interactive camera stream using napari."""

    _prepare_napari_environment()

    try:
        import napari
        from qtpy.QtCore import QTimer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "napari is required for the live viewer. Install with: pip install napari"
        ) from exc

    initial = camera.get_frame().image
    viewer = napari.Viewer(title="Live ORCA stream")
    layer = viewer.add_image(initial, name="camera", blending="opaque")

    timer = QTimer()

    def update() -> None:
        frame = camera.get_frame()
        layer.data = frame.image

    timer.timeout.connect(update)
    timer.start(max(1, int(interval_ms)))

    viewer.window._orca_focus_timer = timer  # type: ignore[attr-defined]
    napari.run()


def launch_matplotlib_viewer(camera: CameraInterface) -> None:
    """Legacy matplotlib viewer fallback."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for interactive display. "
            "Install with: pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots()
    initial = camera.get_frame().image
    im = ax.imshow(initial, cmap="gray")
    ax.set_title("Live ORCA stream")

    def update(_: int):
        frame = camera.get_frame()
        im.set_data(frame.image)
        return (im,)

    fig._orca_focus_anim = FuncAnimation(fig, update, interval=40, blit=True)  # type: ignore[attr-defined]
    plt.show()


def launch_live_viewer(camera: CameraInterface) -> None:
    """Launch napari viewer, falling back to matplotlib if needed."""

    try:
        launch_napari_viewer(camera)
    except RuntimeError:
        launch_matplotlib_viewer(camera)
