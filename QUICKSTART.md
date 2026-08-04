# Tomorrow-start checklist

## Physical setup

- Put one larva per labeled enclosure; use IDs outside the enclosure, not marks on the body.
- Mount the camera overhead, lock focus/aperture, use diffuse visible illumination during the normal light phase, and keep the computer on AC power.
- Connect the camera directly to USB 3 if possible. Record to a local SSD and disable system sleep.

## Software

1. Install the Basler pylon Software Suite.
2. Install FFmpeg and verify `ffmpeg -version` works.
3. In this folder, create a virtual environment and install `requirements.txt`.

macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

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

Preview the current camera:

```bash
python record_basler.py --config config_pilot.yaml --preview camera1
```

Adjust focus first, then illumination/exposure. Aim for no clipped white areas and minimal gain. The smallest larva should be roughly 80-100 pixels long or larger in the saved frame.

Run the ten-second smoke test:

```bash
python record_basler.py --config config_smoke_test.yaml
```

If you want to verify the external-drive archive path, mount `/Volumes/Dr. Rose` first and run:

```bash
caffeinate -i python record_basler.py --config config_archive_smoke_test.yaml
```

If you want a live monitor window during recording, use the full-FoV day-1 config:

```bash
python record_basler.py --config config_experiment_day1.yaml
```

On macOS, keep the machine on power, leave the lid open, and use `caffeinate -i` for unattended runs:

```bash
caffeinate -i python record_basler.py --config config_experiment_day1.yaml
```

Open the MP4 file and inspect the JSON sidecar. In the JSON confirm:

- `success: true`
- `grab_failures: 0`
- `measured_receive_fps` close to 5
- `mp4_remux_succeeded: true`

The archive-enabled configs record locally first, then verify and copy each finished clip directory to `/Volumes/Dr. Rose/Hung_MBL` before deleting the local clip directory.

The short on-disk naming scheme is:

```text
20260804_101405/
  clip_0000_101406/
    camera1.mp4
    camera1.timestamps.csv.gz
    camera1.json
```

## Start the pilot

Edit `project`, `subject`, `animal_ids`, and the schedule in `config_pilot.yaml`, then run:

```bash
caffeinate -i python record_basler.py --config config_pilot.yaml
```

The default is near-continuous 5-fps recording in 30-minute MP4 clips for 24 hours. Stop cleanly with `Ctrl+C`.

Before leaving it unattended, run at least 10-30 minutes and extrapolate daily storage from the actual MP4 sizes. Check the first several JSON files for dropped frames.
