# Basler caterpillar recorder: one-camera fast-start

This repository is set up for the currently connected Basler camera:

- model: `a2A1920-160ucBAS`
- serial: `40604036`
- label used in YAML: `camera1`

The configs in this repo cover both full-FoV and downsampled runs. All of them use the same connected camera and the same 90-degree counter-clockwise rotation:

- `config_pilot.yaml`: 24-hour pilot, full sensor FoV, archive enabled.
- `config_experiment_day1.yaml`: day-1 production config, preview enabled, archive enabled, downsampled to about `960 x 1536`.
- `config_smoke_test.yaml`: ten-second smoke test, archive disabled.
- `config_multiclip_smoke_test.yaml`: two-clip preview smoke test, full sensor FoV, archive disabled.
- `config_archive_smoke_test.yaml`: three-clip archive integration test with preview and external-drive transfer.

## On-disk layout

Sessions now use a compact one-second timestamp at the session level, and each clip directory keeps only the clip index and clock time. Camera files use only the camera label.

```text
recordings/
  monarch_behavior_pilot/
    cohort_pilot01/
      20260804_101405/
        config_used.yaml
        session_manifest.json
        session_summary.json
        recorder.log
        clip_0000_101406/
          camera1.mp4
          camera1.timestamps.csv.gz
          camera1.json
          camera1.capture.mkv
          camera1.ffmpeg.log
          camera1.remux.log
          monitor_snapshots/
```

Exact subsecond timing stays in the JSON metadata and the timestamp CSV. The same shortened hierarchy is copied to the archive drive under `/Volumes/Dr. Rose/Hung_MBL` when archive mode is enabled.

## Repository files

- `record_basler.py`: recorder CLI for listing cameras, live preview, dry runs, and scheduled recording.
- `config_pilot.yaml`: the long one-camera pilot configuration for `./recordings`.
- `config_experiment_day1.yaml`: the day-1 production config with preview enabled.
- `config_smoke_test.yaml`: a ten-second one-camera smoke test for `./recordings_test`.
- `config_multiclip_smoke_test.yaml`: a two-clip pre-experiment smoke test with preview enabled.
- `config_archive_smoke_test.yaml`: a three-clip archive smoke test with preview enabled.
- `validate_session.py`: session validator for clip structure, timing, summary state, and leftover temporary files.
- `QUICKSTART.md`: a short setup-and-run checklist.
- `requirements.txt`: Python dependencies for the recorder.

If you swap cameras, update the serial value in the YAML files before recording.

## Setup

Install Basler pylon and FFmpeg, then create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Camera test

List connected cameras:

```bash
python record_basler.py --list-cameras
```

Preview the active camera:

```bash
python record_basler.py --config config_pilot.yaml --preview camera1
```

Preview controls:

- `q` or Escape: quit
- `s`: save a full processed snapshot
- `p`: print current exposure, gain, acquisition rate, and resulting rate

For setup only, `auto_exposure: true` and `auto_gain: true` can help find a usable range. For the real run, use manual exposure and gain so brightness stays stable.

### Preview while recording

You can optionally show a lightweight monitor window during recording by adding:

```yaml
recording_preview:
  enabled: true
  fps: 1
  max_width: 640
  max_height: 720
  show_status: true
```

This preview is display-only. The full transformed frame still goes to FFmpeg. During recording, `q` or Escape hides the preview without stopping acquisition, `s` saves the latest low-resolution monitor frame into the clip directory, and `Ctrl+C` still stops the recording cleanly.

The preview-enabled example configs in this repo are:

- `config_experiment_day1.yaml`
- `config_multiclip_smoke_test.yaml`
- `config_archive_smoke_test.yaml`

## Pre-experiment multi-clip test

Use this before the long pilot to confirm repeated clip creation, remuxing, preview, and session validation:

```bash
python record_basler.py --list-cameras

python record_basler.py \
  --config config_multiclip_smoke_test.yaml \
  --dry-run

caffeinate -i python record_basler.py \
  --config config_multiclip_smoke_test.yaml
```

After the run, validate the generated session:

```bash
python validate_session.py recordings_test/monarch_behavior_multiclip_test/cohort_C01-C04/SESSION_DIRECTORY
```

Notes:

- `--preview camera1` is preview-only and does not record
- `recording_preview.enabled: true` shows the lightweight monitor while actual recording is running
- use `caffeinate -i` on macOS for unattended acquisition
- keep the MacBook plugged in with the lid open
- near-continuous finite clips have a brief boundary gap

## Archive workflow

The archive-enabled configs `config_pilot.yaml`, `config_experiment_day1.yaml`, and `config_archive_smoke_test.yaml` record locally first, then rsync each finished clip directory to `/Volumes/Dr. Rose/Hung_MBL` after confirming that `/Volumes/Dr. Rose` is a real mounted drive.

The recorder keeps the local session metadata, writes transfer ledgers, verifies the external copy, and deletes each local clip directory only after verification succeeds. If the external drive is missing or the archive backlog grows too large, the recorder stops before starting another clip.

Before using an archive config, mount the drive and check that the local SSD still has plenty of free space. The built-in safety defaults are a 50 GB hard clip cap and a 120 GB minimum free-space gate before each new clip.

Run the archive smoke test with:

```bash
caffeinate -i python record_basler.py --config config_archive_smoke_test.yaml
```

## Record and validate

Validate paths and schedule without opening cameras:

```bash
python record_basler.py --config config_pilot.yaml --dry-run
```

Run the ten-second smoke test:

```bash
python record_basler.py --config config_smoke_test.yaml
```

Run the archive smoke test once the external drive is mounted:

```bash
caffeinate -i python record_basler.py --config config_archive_smoke_test.yaml
```

Run the full-FoV day-1 config:

```bash
python record_basler.py --config config_experiment_day1.yaml
```

On macOS, prevent idle sleep during recording with `caffeinate -i`. Keep the MacBook on power, leave the lid open, and disable automatic sleep on the power adapter when possible. For preview-enabled runs, use:

```bash
caffeinate -i python record_basler.py --config config_experiment_day1.yaml
```

Open the MP4 file and inspect the JSON sidecar. Confirm that `success` is true, `grab_failures` is zero, `measured_receive_fps` is close to the requested rate, and `mp4_remux_succeeded` is true. Then start the pilot:

```bash
python record_basler.py --config config_pilot.yaml
```

For long macOS runs, prefer:

```bash
caffeinate -i python record_basler.py --config config_pilot.yaml
```

Stop cleanly with `Ctrl+C`.

Connect the camera directly to a USB 3 port when possible. If frames are incomplete or skipped, shorten or replace the cable, or reduce frame rate/throughput.

The pilot config records 30-minute clips for 24 hours. Equal clip duration and interval produce near-continuous recording. There is a short boundary gap while a clip closes and the next writer starts.

## Storage check before the long run

Compression depends strongly on leaf texture, sensor noise, and movement, so do not rely on a fixed estimate. After the smoke test, run a 10- or 30-minute test and extrapolate from the resulting file sizes.

The archive workflow already stages clips locally and moves them to the external drive after verification, so the main thing to protect is the internal SSD used for active recording. Keep well above the 120 GB free-space threshold before each clip and mount the archive drive before starting a long run.
