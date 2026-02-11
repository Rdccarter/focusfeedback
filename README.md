# orca-focus

Real-time astigmatic autofocus for microscopy. Uses the shape of a defocused point spread function (PSF) to measure and correct Z drift with a piezo stage, keeping your sample in focus during long acquisitions.

Designed for Hamamatsu ORCA cameras and Mad City Labs piezo stages, but works with any camera/stage combination through a simple interface. Integrates with Micro-Manager for camera control while running focus correction independently.

## How It Works

A weak cylindrical lens in the detection path introduces astigmatism: the PSF of a bright fiducial (bead, nanoparticle, etc.) appears round at focus, horizontally elongated above focus, and vertically elongated below focus. orca-focus measures this ellipticity in real time and feeds it back to a piezo Z stage through a PI controller.

```
Above focus       In focus        Below focus
   ——              ●               |
  (  )            ( )              |
   ——              ●               |
                                   
error < 0        error ≈ 0        error > 0
```

The pipeline runs at each control loop iteration:

1. Grab a frame from the camera
2. Extract the ROI around the fiducial
3. Compute the anisotropy-based error signal (normalized difference of second moments in X and Y)
4. Convert the optical error to physical Z offset using the calibration
5. Apply PI correction to the stage with bounded step size

## Installation

```bash
git clone <repo-url>
cd orca-focus
pip install -e .
```

### Optional dependencies

```bash
# For the napari interactive viewer
pip install napari[all]

# For Micro-Manager integration
pip install pycromanager

# For pylablib-based direct camera control
pip install pylablib
```

## Quick Start

### Simulated mode (no hardware needed)

```bash
# Headless — runs autofocus for 2 seconds, prints final error
orca-focus --duration 2 --loop-hz 50

# Interactive — opens napari, draw ROI on the PSF to start autofocus
orca-focus --show-live --loop-hz 50
```

### With Micro-Manager

Make sure Micro-Manager is running with live mode on and the pycromanager server enabled (Tools → Options → "Run server on port 4827").

```bash
# Camera frames streamed from Micro-Manager; stage controlled directly by this package
orca-focus --camera micromanager --show-live --loop-hz 30 --stage-dll /path/to/Madlib.dll

# Optional explicit wrapper backend instead of DLL path
orca-focus --camera micromanager --stage-wrapper MCL_Madlib_Wrapper --show-live
```

### Direct camera control (no Micro-Manager)

```bash
# Hamamatsu ORCA via pylablib
orca-focus --camera orca --stage mcl --stage-dll /path/to/Madlib.dll --show-live

# Andor iXon via pylablib
orca-focus --camera andor --stage mcl --stage-wrapper MCL_Madlib_Wrapper --show-live
```

## Interactive Viewer

When launched with `--show-live`, a napari window opens showing the live camera stream:

1. The shapes tool is pre-selected — **draw a rectangle** around your fiducial bead
2. Autofocus starts immediately using that ROI
3. A status overlay shows real-time feedback: error signal, Z position, commanded Z, and intensity
4. **Redraw the rectangle** at any time to retarget a different bead
5. Click **Run Calibration Sweep** (or press **c**) to sweep Z using the current ROI and export CSV
6. The sweep is fit immediately; that fitted model is applied live and the CSV is saved for next runs
7. Press **Escape** or close the window to stop

The status overlay reads like:

```
AF ON | err=+0.0023  err_um=+0.006  z=+0.312 → +0.308 µm  I=48721
```

## Calibration

Before running autofocus on real hardware, you need to calibrate the relationship between the error signal and physical Z displacement. This is done by sweeping the stage through focus and fitting a linear model.

```python
from orca_focus import *
from orca_focus.hardware import MclNanoZStage, HamamatsuOrcaCamera
from orca_focus.micromanager import create_micromanager_frame_source

# Connect to hardware (example using Micro-Manager)
mm = create_micromanager_frame_source()
camera = HamamatsuOrcaCamera(frame_source=mm, control_source_lifecycle=False)
stage = MclNanoZStage(dll_path="/path/to/Madlib.dll")

# Define ROI around your fiducial
roi = Roi(x=120, y=95, width=30, height=30)

# Sweep ±3 µm in 30 steps
samples = auto_calibrate(
    camera, stage, roi,
    z_min_um=-3.0,
    z_max_um=3.0,
    n_steps=30,
)

# Fit with outlier rejection
report = fit_linear_calibration_with_report(samples, robust=True)
print(f"error_to_um = {report.calibration.error_to_um:.3f}")
print(f"error_at_focus = {report.calibration.error_at_focus:.4f}")
print(f"R² = {report.r2:.4f}")
print(f"RMSE = {report.rmse_um:.4f} µm")
print(f"Inliers = {report.n_inliers}/{report.n_samples}")

# Verify slope sign matches your optical setup
validate_calibration_sign(report.calibration, expected_positive_slope=True)

# Persist and reuse samples later
save_calibration_samples_csv("calibration_sweep.csv", samples)
samples2 = load_calibration_samples_csv("calibration_sweep.csv")
```

Typical values: `error_to_um` is usually 2–5 depending on the cylindrical lens strength and objective NA. `error_at_focus` should be close to 0 if the optics are well-aligned.

### Automatic reuse in the GUI/CLI

Use the same `--calibration-csv` path for both calibration and sample runs (default: `calibration_sweep.csv`). On startup, `orca-focus` automatically loads this CSV, fits calibration (`robust=True`), and uses that fitted model for autofocus—no hardcoded calibration constants required.

## CLI Reference

```
orca-focus [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--camera` | `simulate` | Camera backend: `simulate`, `orca`, `andor`, `micromanager` |
| `--camera-index` | `0` | Device index for pylablib cameras |
| `--stage` | auto | Stage backend: `mcl`, `simulate` (camera can still be `micromanager`) |
| `--stage-dll` | — | Path to MCL Madlib DLL |
| `--stage-wrapper` | — | Python module path for MCL wrapper |
| `--mm-host` | `localhost` | Micro-Manager pycromanager host |
| `--mm-port` | `4827` | Micro-Manager pycromanager port |
| `--show-live` | off | Open napari viewer with interactive ROI selection |
| `--duration` | `2.0` | Headless mode runtime in seconds |
| `--loop-hz` | `30.0` | Control loop frequency (Hz) |
| `--kp` | `0.8` | Proportional gain |
| `--ki` | `0.2` | Integral gain |
| `--max-step` | `0.2` | Maximum single correction step (µm) |
| `--calibration-csv` | `calibration_sweep.csv` | Calibration samples CSV to auto-load on startup and overwrite when running a GUI calibration sweep |
| `--calibration-half-range-um` | `0.75` | Half-range around current Z for calibration sweep (µm) |
| `--calibration-steps` | `21` | Number of points in calibration sweep |

## Tuning the PI Controller

The three control parameters interact as follows:

**`kp` (proportional gain)** — How aggressively the controller responds to the current error. Higher values converge faster but can overshoot and oscillate.

**`ki` (integral gain)** — Corrects for accumulated steady-state error. Eliminates the persistent offset that proportional-only control leaves behind. Too high causes slow oscillation.

**`max-step`** — Clamps any single correction to prevent large jumps from noisy frames. Should be set based on your piezo's safe step size and the expected drift rate.

Starting points for common setups:

| Setup | `kp` | `ki` | `max-step` | `loop-hz` |
|---|---|---|---|---|
| ORCA + MCL NanoDrive | 0.5 | 0.1 | 0.15 | 30 |
| Andor iXon + MCL | 0.6 | 0.15 | 0.20 | 50 |
| Simulated | 0.8 | 0.2 | 0.20 | 50 |

If you see oscillation, lower `kp` first. If the PSF settles but stays slightly elliptical, raise `ki`. If convergence is too slow, raise `kp` and `loop-hz`.


## Troubleshooting

- **`Calibration CSV not found: calibration_sweep.csv`**
  - First run is expected to show this warning in `--show-live` mode. Use the GUI calibration button once, which will create the CSV, then restart `orca-focus` so it auto-loads the fitted calibration.
- **Micro-Manager connection error (`Tried pycromanager at localhost:4827`)**
  - Ensure Micro-Manager server is enabled (Tools → Options → Run server on port 4827), or pass `--mm-host/--mm-port` if different.
  - If `python -c "from pycromanager import Core; print(Core())"` works, the package now also attempts `Core()` fallback automatically.
- **napari `PluginManifest ... pydantic` errors**
  - This is a napari/environment mismatch, not autofocus logic. In your conda env, pin compatible versions, e.g. reinstall napari stack:
    `pip install -U "napari[all]" "pydantic<2.11"`

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Micro-Manager GUI                                      │
│  (exposure, ROI crop, recording, live acquisition)      │
└──────────────┬──────────────────────────────────────────┘
               │ pycromanager / MMCorePy
               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│  MicroManagerFrameSource │    │  MicroManagerStage       │
│  (reads circular buffer) │    │  (or MclNanoZStage)      │
└──────────┬───────────────┘    └──────────▲───────────────┘
           │                               │
           ▼                               │
┌──────────────────────────────────────────┴───────────────┐
│  AstigmaticAutofocusController                           │
│                                                          │
│  frame → extract_roi → error_signal → calibration → PI  │
│                                          correction      │
└──────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│  AutofocusWorker (background thread)                     │
│  └── napari viewer (main thread, QTimer refresh)         │
│      └── shapes layer for interactive ROI selection      │
└──────────────────────────────────────────────────────────┘
```

## Module Overview

| Module | Purpose |
|---|---|
| `interfaces.py` | Protocol classes for camera and stage |
| `focus_metric.py` | Astigmatic error signal computation, ROI extraction |
| `calibration.py` | Z sweep, linear fit, outlier rejection, sign validation |
| `autofocus.py` | PI controller and background worker thread |
| `hardware.py` | MCL stage adapter (DLL/wrapper/simulated), simulated camera |
| `micromanager.py` | Micro-Manager frame source and stage adapter |
| `pylablib_camera.py` | pylablib frame source for ORCA and Andor |
| `dcam.py` | DCAM SDK frame source adapter |
| `interactive.py` | napari viewer with ROI selection and live autofocus |
| `viewer.py` | Toolkit-agnostic ROI selection state machine |
| `monitor.py` | Simple frame dispatch loop for custom GUIs |
| `cli.py` | Command-line entry point |

## Python API

### Running autofocus programmatically

```python
from orca_focus import *

# Set up hardware
stage = MclNanoZStage(dll_path="/path/to/Madlib.dll")
camera = HamamatsuOrcaCamera(frame_source=my_source)
camera.start()

# Configure
config = AutofocusConfig(
    roi=Roi(x=120, y=95, width=30, height=30),
    loop_hz=30.0,
    kp=0.5,
    ki=0.1,
    max_step_um=0.15,
    stage_min_um=-10.0,
    stage_max_um=10.0,
    min_roi_intensity=1000.0,  # freeze if bead disappears
)
calibration = FocusCalibration(error_at_focus=0.0, error_to_um=2.8)

# Blocking run
controller = AstigmaticAutofocusController(camera, stage, config, calibration)
samples = controller.run(duration_s=10.0)

# Or background thread
worker = AutofocusWorker(controller, on_sample=lambda s: print(s.error_um))
worker.start()
# ... do other things ...
worker.stop()

camera.stop()
```

### Custom camera or stage

Implement the protocol interfaces:

```python
from orca_focus.interfaces import CameraFrame, CameraInterface, StageInterface

class MyCamera:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_frame(self) -> CameraFrame: ...

class MyStage:
    def get_z_um(self) -> float: ...
    def move_z_um(self, target_z_um: float) -> None: ...
```

No inheritance required — these are `Protocol` classes, so any object with the right methods works.

## Tests

```bash
pytest -v
```

All tests run against the simulated camera and stage — no hardware needed. The test suite covers the controller, calibration fitting, focus metric edge cases, hardware wrapper dispatch, and the interactive viewer.

## Requirements

- Python 3.10+
- numpy (optional but recommended — falls back to pure Python for error signal computation)
- napari (for `--show-live` interactive viewer)
- pycromanager (for `--camera micromanager`)
- pylablib (for `--camera orca` or `--camera andor`)
