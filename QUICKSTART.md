# Daily Checklist

For first-time setup, start with [README.md](./README.md). This file is only the repeatable daily workflow after the environment is already installed.

## Before you start

- Make sure the camera is visible in pylon Viewer.
- Make sure the archive drive is mounted if you are using archive mode.
- Keep the computer on AC power.
- Use `config_local_windows.yaml` or `config_local_macos.yaml`, not the tracked templates.

## Reactivate the environment

Windows PowerShell:

```powershell
cd $HOME\Documents\GitHub\basler_caterpillar_recorder
.venv\Scripts\Activate.ps1
```

macOS:

```bash
cd ~/Documents/GitHub/basler_caterpillar_recorder
source .venv/bin/activate
```

## Daily workflow

1. Copy a template to a local config if needed.
2. Run a dry run.
3. Run `--preview camera1`.
4. Run a short local test.
5. Run the three-clip archive smoke test.
6. Validate the session.
7. Start the unattended run only after validation passes.

Example commands:

```bash
python record_basler.py --list-cameras
python record_basler.py --config config_local_macos.yaml --dry-run
python record_basler.py --config config_local_macos.yaml --preview camera1
python record_basler.py --config config_smoke_test.yaml
caffeinate -i python record_basler.py --config config_archive_smoke_test.yaml
python validate_session.py SESSION_PATH
```

Windows example:

```powershell
python record_basler.py --config config_local_windows.yaml --dry-run
python record_basler.py --config config_local_windows.yaml --preview camera1
python record_basler.py --config config_smoke_test.yaml
python record_basler.py --config config_archive_smoke_test.yaml
python validate_session.py SESSION_PATH
```

## Stop controls

- `q` hides the recording preview without stopping acquisition.
- `Ctrl+C` stops the recording cleanly.

## Final pre-run checklist

- Dry run passed.
- Preview looked correct.
- Short local test passed.
- Three-clip archive test passed if archive mode is enabled.
- `python validate_session.py SESSION_PATH` returned `PASS`.
- Archive drive is mounted.
- Laptop is on power.
- You are using the copied local config, not a tracked template.
