# Basler caterpillar recorder: one-camera fast-start

This repository is set up for the currently connected Basler camera:

- model: `a2A1920-160ucBAS`
- serial: `40604036`
- label used in YAML: `camera1`

The active configs use the full native sensor field of view:

- `width: 1920`
- `height: 1200`
- `offset_x: 0`
- `offset_y: 0`

They keep the 90-degree counter-clockwise rotation and software downscaling:

- `rotate: 270`
- `output_width: 960`

That produces an encoded frame of about `960 x 1536` while preserving the full field of view.

## Repository files

- `record_basler.py`: recorder CLI for listing cameras, live preview, dry runs, and scheduled recording.
- `config_pilot.yaml`: the long one-camera full-FoV pilot configuration for `./recordings`.
- `config_smoke_test.yaml`: a ten-second one-camera full-FoV smoke test for `./recordings_test`.
- `config_experiment_day1.yaml`: the full-FoV day-1 production config with preview enabled.
- `config_multiclip_smoke_test.yaml`: a three-clip pre-experiment full-FoV smoke test with preview enabled.
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

## Record and validate

Validate paths and schedule without opening cameras:

```bash
python record_basler.py --config config_pilot.yaml --dry-run
```

Run the ten-second smoke test:

```bash
python record_basler.py --config config_smoke_test.yaml
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

The example config records 30-minute clips for 24 hours. Equal clip duration and interval produce near-continuous recording. There is a short boundary gap while a clip closes and the next writer starts.

## Storage check before the long run

Compression depends strongly on leaf texture, sensor noise, and movement, so do not rely on a fixed estimate. After the smoke test, run a 10- or 30-minute test and extrapolate from the resulting file sizes.

Keep at least 15-20% of the recording disk free. Write directly to a local SSD during acquisition; copy to network or archival storage after clips close.
