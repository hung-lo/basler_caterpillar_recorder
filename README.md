# Basler caterpillar recorder

This repository documents and drives the current one-camera recording workflow for the Basler `a2A1920-160ucBAS` camera with serial `40604036` and YAML label `camera1`.

<img src="docs/images/recording_preview.png" alt="Recording preview window" width="760">

The recorder supports:

- setup preview that does not record;
- lightweight recording preview with clip/session progress;
- terminal `STATUS` heartbeats while waiting and recording;
- local-first clip writing;
- verified archive transfer with `.incoming/*.partial` promotion;
- `rsync` on macOS/Linux and `robocopy` on Windows;
- SHA-256 verification before local clip deletion;
- built-in sleep prevention for unattended runs.

If you use a different camera, copy a YAML template and update the model, serial, frame size, and rotation before recording.

## At a glance

- `record_basler.py` is the main CLI for listing cameras, previewing, dry runs, and scheduled recording.
- `validate_session.py` checks clip structure, JSON sidecars, timing, and archive state.
- `test_record_basler.py` covers schedule math, preview sizing, JSON handling, and archive helpers.
- `config_smoke_test.yaml` is a short local test.
- `config_multiclip_smoke_test.yaml` is a short repeated-clip test.
- `config_archive_smoke_test.yaml` is a three-clip archive test.
- `config_windows_test.yaml` is a Windows-local test template.
- `config_windows_long_recording.yaml` is a long-run template, not a fixed-duration promise.
- `config_pilot.yaml` is the main pilot template.
- `QUICKSTART.md` is the short daily checklist after setup is already complete.

The tracked YAML files are templates. Copy one to a local file such as `config_local_macos.yaml` or `config_local_windows.yaml` before you edit it. Git ignores `config_local*.yaml`.

## New computer workflow

1. Install Basler pylon, FFmpeg, Git, and `uv`.
2. Clone this repository.
3. Create the repository `.venv`.
4. Confirm the camera appears in pylon Viewer.
5. Run `--list-cameras`.
6. Copy a YAML template to a local config.
7. Run `--dry-run`.
8. Run `--preview camera1`.
9. Run a short local recording test.
10. Run a three-clip archive smoke test.
11. Validate the session.
12. Only then start the long unattended run.

Do not begin with the long-run template on a new computer.

## Hardware and system setup

Keep the computer on AC power, leave the lid open unless the OS is explicitly configured otherwise, and connect the camera directly to USB 3 when possible. Avoid USB hubs if you can.

### Camera

Install Basler pylon and confirm the camera is visible in pylon Viewer before you record anything.

For the main camera, the expected values are:

```text
model:  a2A1920-160ucBAS
serial: 40604036
label:  camera1
```

### Recording computer

- Keep enough free space on the internal recording disk.
- Mount the external archive drive before archive-enabled runs.
- Do not mix Conda/Anaconda with the repository `.venv` unless you have a specific reason to do so.

## Install software

Use `uv` with the existing `requirements.txt` workflow:

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
uv python install 3.12
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Verify the install with:

```bash
python --version
python -c "import pypylon; print('pypylon OK')"
python -c "import cv2; print('OpenCV', cv2.__version__)"
ffmpeg -version
```

## Confirm the camera

List connected cameras:

```bash
python record_basler.py --list-cameras
```

If the serial differs, update your copied YAML before you continue.

If no camera appears:

1. Confirm it appears in pylon Viewer.
2. Reconnect the USB cable.
3. Try a direct USB 3 port.
4. Close any app already using the camera.
5. Run `--list-cameras` again.

## Create a run-specific config

Copy a template first.

Windows:

```powershell
Copy-Item config_windows_long_recording.yaml config_local_windows.yaml
```

macOS:

```bash
cp config_pilot.yaml config_local_macos.yaml
```

Before every experiment, review:

```yaml
project:
subject:

experiment:
  animal_ids:
  notes:

output_root:

schedule:
  # Optional one-shot local start. Remove or leave null for immediate start.
  # start_at_local: "2026-08-08 05:00"
  clip_duration_s:
  interval_s:
  number_of_clips:
  # or total_duration_h:

cameras:
  - model:
    serial:
    fps:
    width:
    height:
    exposure_us:
    gain:
    rotate:

archive:
  enabled:
  destination_root:
  required_mount_point:
```

For the main camera, the current full-field config uses:

```yaml
model: a2A1920-160ucBAS
serial: "40604036"
fps: 5
width: 1920
height: 1200
offset_x: 0
offset_y: 0
rotate: 270
```

`rotate: 270` means 90 degrees counter-clockwise.

Bounded continuous auto exposure for the current IR setup looks like:

```yaml
exposure_us: 30000
gain: 0
auto_exposure: true
auto_exposure_mode: continuous
auto_exposure_lower_us: 6000
auto_exposure_upper_us: 180000
auto_target_brightness: 0.70
auto_exposure_roi: full
auto_gain: false
```

Keep these warnings in mind:

- Continuous auto exposure changes exposure during recording.
- Keep `auto_gain: false` and `gain: 0` for this experiment.
- `auto_exposure_upper_us` must stay below the frame period.
- At 5 Hz, use `180000 us`, not the full `200000 us`.
- `auto_target_brightness: 0.70` is the current empirically tuned target for the MBL dish/leaf scene, not an exact guarantee.
- Run a blocked-light test before unattended recording.

## Schedule meaning

The recorder now has two scheduling layers:

- Initial start: immediate by default, or optional `schedule.start_at_local` in this computer's local timezone.
- Clip schedule: `clip_duration_s`, `interval_s`, and `total_duration_h` or `number_of_clips`.

Use the values in the YAML to understand the planned span after recording starts.

When a schedule is limited by `number_of_clips`, the nominal span is:

```text
(N - 1) * interval + clip duration
```

When `clip_duration_s` equals `interval_s`, the run is near-continuous with a small clip-boundary overhead.

Do not treat the long-run template as an exact-duration promise. Inspect `clip_duration_s`, `interval_s`, `number_of_clips`, and `total_duration_h` in the copied config you actually plan to use.

One-shot scheduled starts must include both date and time, for example:

```yaml
schedule:
  start_at_local: "2026-08-08 05:00"
  clip_duration_s: 1800
  interval_s: 1800
  total_duration_h: 24
```

Remove or comment `start_at_local` after that experiment if you want the next invocation to start immediately.

## Time And Timestamp Policy

The recorder uses three kinds of time for different purposes:

| Purpose | Time source |
|---|---|
| Folder names, terminal logs, preview, and finish-time display | Local computer time with a numeric UTC offset |
| Scientific metadata and cross-computer alignment | UTC |
| Elapsed time, clip scheduling, and heartbeat timing | Monotonic clock |

A new session folder may look like:

```text
20260804_105301-0400
```

This means the session was created at local time `10:53:01` with UTC offset `-04:00`.

A clip folder may look like:

```text
clip_0000_105302-0400
```

UTC remains the canonical timestamp in JSON metadata and frame timestamp files. Important metadata events also include a human-readable local-time mirror, for example:

```json
{
  "session_start_utc": "2026-08-04T14:53:01.254Z",
  "session_start_local": "2026-08-04T10:53:01.254-04:00"
}
```

The per-frame timestamp file does not repeat a formatted local-time string for every frame. Use `host_utc_ns` for absolute timing and `host_monotonic_ns` or `elapsed_s` for intervals.

Local time is used for operator convenience, but UTC is retained because it is unambiguous across computers, time zones, and daylight-saving transitions.

Older sessions may use UTC-looking folder names without an offset, such as:

```text
20260804_145301
```

The validator and analysis tools continue to support those legacy names.

## Check archive settings

Example Windows archive config:

```yaml
archive:
  enabled: true
  backend: auto
  destination_root: "D:/Hung_MBL"
  required_mount_point: "D:/"
```

Use forward slashes in Windows YAML paths.

`backend: auto` selects `robocopy` on Windows.

Example macOS archive config:

```yaml
archive:
  enabled: true
  backend: auto
  destination_root: "/Volumes/Dr. Rose/Hung_MBL"
  required_mount_point: "/Volumes/Dr. Rose"
```

`backend: auto` selects `rsync` on macOS/Linux.

Archive transfer is local-first:

1. record the clip locally;
2. copy it to `.incoming/<clip>.partial`;
3. compare file names, sizes, and SHA-256 hashes;
4. promote the verified partial directory to the final visible clip directory;
5. delete the local clip only after verification passes.

If transfer fails, the local clip is preserved. Do not manually delete `.incoming/*.partial` until you have inspected the failure.

## Dry run

A dry run checks config, schedule, paths, archive backend, mount point, executables, and free space without opening the camera.

If `schedule.start_at_local` is set, the dry run validates the requested future local start time and reports the planned start and finish, but it does not enter the scheduled wait.

Windows:

```powershell
python record_basler.py --config config_local_windows.yaml --dry-run
```

macOS:

```bash
python record_basler.py --config config_local_macos.yaml --dry-run
```

Do not continue if the dry run reports a failure.

For a scheduled overnight macOS run, the recommended sequence is:

```bash
python record_basler.py --config config_local_macos.yaml --dry-run
caffeinate -i python record_basler.py --config config_local_macos.yaml
```

## Setup preview

Windows:

```powershell
python record_basler.py --config config_local_windows.yaml --preview camera1
```

macOS:

```bash
python record_basler.py --config config_local_macos.yaml --preview camera1
```

`--preview` is preview-only and does not record.

Setup-preview controls:

```text
q or Escape   close preview
s             save a full processed snapshot
p             print camera settings
```

Confirm the full field of view, focus, brightness, rotation, and reflection control before you proceed.

Preview sizing is independent of saved-video resolution. Reducing the preview width or height does not downsample the recorded MP4.

## Recording preview during acquisition

If you want a lightweight monitor window during recording, add:

```yaml
recording_preview:
  enabled: true
  fps: 1
  max_width: 600
  max_height: 720
  show_status: true
  layout: card_panel
```

### Recording window

The `card_panel` layout keeps the camera image unobstructed while showing clip and session progress, measured receive FPS, current exposure, estimated finish time, and recording status.

<img src="docs/images/recording_preview.png" alt="Recording preview window" width="760">

During recording:

- the preview is display-only;
- the preview shows clip and session progress;
- the terminal prints periodic `STATUS` heartbeats;
- `q` hides the recording preview without stopping acquisition;
- `s` saves the raw preview frame without the panel or footer into the clip directory;
- `Ctrl+C` stops the recording cleanly.

### Stopping a recording safely

Press `Ctrl+C` once to stop recording gracefully.

If a clip is currently being recorded, the recorder stops acquisition after the current frame, finalizes the partial MP4 and timestamp sidecar, and marks the clip as intentionally incomplete. On Windows, the recorder isolates FFmpeg from console `Ctrl+C` events so Python handles the stop request and then closes FFmpeg through its stdin pipe for a clean partial-clip finalize. When archiving is enabled, the partial clip is queued for archive and verified before the program exits.

The interrupted clip is preserved but is not counted as a fully completed scheduled clip.

After pressing `Ctrl+C`, keep the terminal open until the recorder reports that pending archive transfers have finished.

Supported layouts:

- `card_panel`: appends a right-side information panel and bottom status footer while keeping the camera image unobstructed.
- `legacy_overlay`: uses the older translucent text overlay on top of the preview image.

The preview-enabled example configs are:

- `config_experiment_day1.yaml`
- `config_multiclip_smoke_test.yaml`
- `config_archive_smoke_test.yaml`

## Run a local recording test

On a new computer, first disable archive in your copied config:

```yaml
archive:
  enabled: false
```

Use a short schedule:

```yaml
schedule:
  clip_duration_s: 10
  interval_s: 10
  number_of_clips: 1
```

Run the short test:

```powershell
python record_basler.py --config config_local_windows.yaml
```

or:

```bash
python record_basler.py --config config_local_macos.yaml
```

Confirm the session contains the expected clip directory, MP4, timestamps, JSON metadata, and log files.

New session and clip folders use local time plus a numeric UTC offset. Older sessions may use the legacy timestamp format without an offset.

## Run the archive smoke test

With the archive drive mounted, run:

```bash
caffeinate -i python record_basler.py --config config_archive_smoke_test.yaml
```

On macOS, `caffeinate -i` is still useful if you want sleep prevention outside the YAML.

Validate the generated session:

```bash
python validate_session.py SESSION_PATH
```

Replace `SESSION_PATH` with the actual session directory you want to check.

Example macOS command to find the newest session and validate it:

```bash
SESSION_PATH=$(find recordings -name session_summary.json -print0 | xargs -0 -n1 dirname | sort | tail -n 1)
python validate_session.py "$SESSION_PATH"
```

Example Windows PowerShell command:

```powershell
$session = Get-ChildItem recordings -Recurse -Filter session_summary.json | Sort-Object LastWriteTime | Select-Object -Last 1
if ($session) { $session = Split-Path $session.FullName -Parent }
python validate_session.py $session
```

`PASS` should be a prerequisite before you rely on unattended long runs on a new setup.

## Start a long run

The long-run template is meant to be reviewed and copied locally before use. Do not assume its duration is still the same after you edit it.

Windows example:

```powershell
python record_basler.py --config config_local_windows.yaml --dry-run
python record_basler.py --config config_local_windows.yaml
```

macOS example:

```bash
python record_basler.py --config config_local_macos.yaml --dry-run
caffeinate -i python record_basler.py --config config_local_macos.yaml
```

For unattended recording, keep the machine on power, leave the lid open if needed, and stop cleanly with `Ctrl+C`.

## Storage check before the long run

Do not rely on a fixed compression estimate. Run a short test first, inspect the MP4 size, and extrapolate from what you actually recorded.

The archive workflow keeps active clips on the local SSD only until verification succeeds, so protect the internal recording disk and mount the external archive drive before you begin.
