# Tomorrow-start checklist

## Physical setup

- Put one larva per labeled enclosure; use IDs M01-M07 outside the enclosure, not marks on the body.
- Arrange M01-M04 in a fixed 2 x 2 grid under the acA4024 camera.
- Arrange M05-M07 in a fixed 2 x 2 grid under the a2A1920 camera; use the empty fourth position for a scale and neutral reference patch.
- Mount both cameras overhead, lock focus/aperture, use diffuse visible illumination during the normal light phase, and keep the computer on AC power.
- Connect each camera directly to USB 3. Use separate host controllers if available. Record to a local SSD and disable system sleep.

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

```bash
python record_basler.py --list-cameras
```

Copy the printed serial numbers into both YAML files. Then preview each camera:

```bash
python record_basler.py --config config_pilot.yaml --preview arena_A_M01-M04
python record_basler.py --config config_pilot.yaml --preview arena_B_M05-M07
```

Adjust focus first, then illumination/exposure. Aim for no clipped white areas and minimal gain. The smallest larva should be roughly 80-100 pixels long or larger in the saved frame.

Run a ten-second two-camera test:

```bash
python record_basler.py --config config_smoke_test.yaml
```

On macOS, keep the machine on power, leave the lid open, and use `caffeinate -i` for unattended runs so the display can sleep without the system idling out:

```bash
caffeinate -i python record_basler.py --config config_smoke_test.yaml
```

If you want a live monitor window during recording, use the example one-camera preview config:

```bash
python record_basler.py --config config_one_camera_test_rotated_preview.yaml
```

To test a centered native crop on that same camera before committing to it:

```bash
python record_basler.py \
  --config config_one_camera_test_rotated_crop_preview.yaml \
  --preview arena_B_M05-M07
python record_basler.py --config config_one_camera_test_rotated_crop_preview.yaml
```

Open both MP4 files. In each JSON sidecar confirm:

- `success: true`
- `grab_failures: 0`
- `measured_receive_fps` close to 5
- `mp4_remux_succeeded: true`

## Start the pilot

Edit `project`, `subject`, `animal_ids`, and the schedule in `config_pilot.yaml`, then run:

```bash
caffeinate -i python record_basler.py --config config_pilot.yaml
```

The default is near-continuous 5-fps recording in 30-minute MP4 clips for 24 hours. Stop cleanly with `Ctrl+C`.

Before leaving it unattended, run at least 10-30 minutes and extrapolate daily storage from the actual MP4 sizes. Check the first several JSON files for dropped frames.
