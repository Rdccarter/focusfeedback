# orca-focus

Real-time astigmatic autofocus for microscopy.

`orca-focus` measures PSF ellipticity from a bead/fiducial and closes a PI loop on Z to suppress focus drift during long acquisitions. It supports simulation, direct camera access, and Micro-Manager camera streaming.

## Features

- Real-time astigmatic autofocus with bounded PI control.
- Interactive napari workflow (`--show-live`) with ROI drawing.
- **Explicit Start/Stop Autofocus button** in the live viewer.
- Built-in calibration sweeps with robust linear fit.
- **Calibration controls in nanometers** (half-range + step size) directly in the viewer.
- Cancellable calibration sweep with per-step progress reporting.
- Micro-Manager frame-source support with duplicate-frame protection.

## Installation

```bash
git clone <repo-url>
cd orca-focus
pip install -e .
```

Optional extras:

```bash
# napari live UI
pip install "napari[all]"

# Micro-Manager bridge
pip install pycromanager

# Direct ORCA/Andor camera backends
pip install pylablib
```

## Quick Start

### Simulated mode

```bash
# Headless autofocus run
orca-focus --duration 2 --loop-hz 50

# Interactive viewer
orca-focus --show-live --loop-hz 50
```

### Micro-Manager camera + MCL stage control (recommended pattern)

```bash
orca-focus --camera micromanager --show-live --loop-hz 30 --stage-dll /path/to/Madlib.dll
```

Notes:
- Camera frames come from Micro-Manager.
- Stage control remains independent (MCL DLL/wrapper) unless you explicitly set `--stage micromanager`.

### Direct camera control (no Micro-Manager)

```bash
# ORCA via pylablib
orca-focus --camera orca --stage mcl --stage-dll /path/to/Madlib.dll --show-live

# Andor via pylablib
orca-focus --camera andor --stage mcl --stage-wrapper MCL_Madlib_Wrapper --show-live
```

## Live Viewer Workflow (`--show-live`)

1. Draw an ROI rectangle around the fiducial.
2. Autofocus starts using that ROI.
3. Use **Stop Autofocus / Start Autofocus** to pause/resume control.
4. Configure calibration using:
   - **Calibration half-range (nm)**
   - **Calibration step size (nm)**
5. Click **Run Calibration Sweep** (or press `c`) to calibrate.
6. Click the same button while running (**Stop Calibration Sweep**) to cancel.
7. Status text shows current error, measured Z, command Z, intensity, and calibration progress.
- When you re-draw/move ROI to a different bead in the live viewer, autofocus re-anchors the error offset for that ROI while keeping calibration slope, reducing sudden jumps/out-of-focus moves after target changes.

Example status line:

```text
AF ON | err=+0.0023  err_um=+0.006  z(now)=+0.312 → cmd=+0.308 um  I=48721
```

## Calibration Details

Calibration sweeps move across a Z window and fit a linear mapping:

- `z_offset_um ~= error_to_um * (error - error_at_focus)`
- Fitting is done against a local Z reference (center of sweep), so calibration remains valid even when absolute stage coordinates are not near 0.
- The fitted model is used as a **relative move command** (`z_offset_um`), so one calibration can be reused for different beads/targets at different absolute Z positions.
- At runtime, loaded/GUI-applied calibrations use `control_error_at_focus = 0.0` by design, so the CSV primarily provides slope (move scale + direction) and stays reusable across restarts/targets.

In the viewer, sweep bounds are derived from the nm controls around current Z:

- `z_min = z_center - half_range_nm/1000`
- `z_max = z_center + half_range_nm/1000`
- `n_steps` computed from span and nm step size
- each calibration executes an up-and-down (bidirectional) sweep to validate repeatability

The sweep is cancellable and reports per-step target/measured Z. If individual stage moves fail, the sweep can skip bad points; if too few valid points remain, a clear calibration error is raised.

Calibration quality checks are tuned for astigmatic (cylindrical-lens) behavior and now run on a bidirectional sweep (up + down) so backlash/hysteresis can be detected. If quality checks fail, reduce half-range, slow the sweep, and keep the bead centered in ROI.

### Programmatic calibration example

```python
from orca_focus import *
from orca_focus.hardware import HamamatsuOrcaCamera, MclNanoZStage
from orca_focus.micromanager import create_micromanager_frame_source

mm = create_micromanager_frame_source()
camera = HamamatsuOrcaCamera(frame_source=mm, control_source_lifecycle=False)
stage = MclNanoZStage(dll_path="/path/to/Madlib.dll")
roi = Roi(x=120, y=95, width=30, height=30)

samples = auto_calibrate(
    camera,
    stage,
    roi,
    z_min_um=-3.0,
    z_max_um=3.0,
    n_steps=30,
)

report = fit_linear_calibration_with_report(samples, robust=True)
print(report.calibration.error_to_um, report.calibration.error_at_focus)
save_calibration_samples_csv("calibration_sweep.csv", samples)
```

## CLI Reference

```text
orca-focus [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--camera` | `simulate` | `simulate`, `orca`, `andor`, `micromanager` |
| `--camera-index` | `0` | Camera index for pylablib camera backends |
| `--stage` | auto | `mcl`, `simulate`, `micromanager` |
| `--stage-dll` | none | Path to MCL DLL |
| `--stage-wrapper` | none | Python wrapper module for MCL |
| `--mm-host` | `localhost` | Micro-Manager pycromanager host |
| `--mm-port` | `4827` | Micro-Manager pycromanager port |
| `--mm-allow-standalone-core` | off | Allow fallback to standalone `pymmcore/MMCorePy` core |
| `--show-live` | off | Open interactive napari viewer |
| `--duration` | `2.0` | Headless autofocus runtime (s) |
| `--loop-hz` | `30.0` | Control loop rate |
| `--kp` | `0.8` | Proportional gain |
| `--ki` | `0.2` | Integral gain |
| `--max-step` | `0.2` | Max per-step Z command (um) |
| `--command-deadband-um` | `0.02` | Ignore tiny stage commands below this size (um) to reduce Z dithering near lock |
| `--stage-min-um` | none | Hard lower clamp for commanded stage Z (um) |
| `--stage-max-um` | none | Hard upper clamp for commanded stage Z (um) |
| `--af-max-excursion-um` | `5.0` | Max autofocus excursion from initial lock Z (um); set negative to disable this clamp |
| `--calibration-csv` | `calibration_sweep.csv` | Calibration samples CSV path |
| `--calibration-half-range-um` | `0.75` | Initial GUI half-range seed (um) |
| `--calibration-steps` | `21` | Initial GUI step-count seed |

## Troubleshooting

- **Calibration appears to run forever**
  - Use **Stop Calibration Sweep** while running.
  - Increase step size (nm) or decrease half-range (nm) to reduce sweep length.
- **Calibration applies but autofocus runs away/out of focus**
  - This usually means the sweep did not produce a valid local linear model around your operating focus.
  - Re-run calibration with a smaller half-range and a stable, centered ROI bead.
  - Confirm the error trend is roughly monotonic over the sweep; avoid very wide sweeps that include non-linear regions.
  - Set hard limits with `--stage-min-um/--stage-max-um` (or lower `--af-max-excursion-um`) to prevent large absolute Z jumps.
  - If Z chatters around focus, increase `--command-deadband-um` (for example to `0.03`–`0.08`).
  - Status `-6` from `MCL_SingleWriteN` usually indicates out-of-range/invalid moves; verify stage axis and clamp settings.
- **Displayed Z looks off**
  - The viewer now displays `z(now)` from live stage readback when available.
- **Micro-Manager connection error**
  - Ensure MM server is enabled on configured host/port.
  - Use `--mm-allow-standalone-core` only if you intentionally want a standalone core.
- **napari plugin/pydantic manifest warnings**
  - Usually an environment compatibility issue. Use a clean env and compatible napari stack.

## Architecture

```text
Camera frame source -> AutofocusController (error + PI) -> Stage command
                      ^
                      |
                 Calibration model
```

Main modules:

- `focus_metric.py`: ROI extraction + astigmatic error.
- `calibration.py`: sweep acquisition + linear fit/reporting.
- `autofocus.py`: control loop + worker thread.
- `interactive.py`: napari UI, ROI flow, calibration controls.
- `micromanager.py`: MM frame/stage adapters.
- `hardware.py`: MCL stage + simulated hardware.
- `cli.py`: command-line orchestration.
