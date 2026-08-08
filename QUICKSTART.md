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

For an optional one-shot scheduled start, set:

```yaml
schedule:
  start_at_local: "2026-08-08 05:00"
```

Then run the usual dry run first. Remove or comment `start_at_local` after that experiment if the next launch should start immediately.

### Timestamp reminder

Folder names and terminal logs use local computer time with a numeric UTC offset. JSON metadata retains canonical UTC timestamps. Elapsed-time calculations use a monotonic clock.

Example commands:

```bash
python record_basler.py --list-cameras
python record_basler.py --config config_local_macos.yaml --dry-run
python record_basler.py --config config_local_macos.yaml --preview camera1
python record_basler.py --config config_smoke_test.yaml
caffeinate -i python record_basler.py --config config_archive_smoke_test.yaml
caffeinate -i python record_basler.py --config config_local_macos.yaml
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

- `q` hides the recording preview for the current clip without stopping acquisition. It reopens automatically at the next clip if preview remains enabled.
- `Ctrl+C` stops the recording cleanly.

### Stopping a recording safely

Press `Ctrl+C` once to stop recording gracefully.

If a clip is in progress, the recorder finalizes the partial MP4 and timestamp sidecar before exiting. On Windows, the recorder isolates FFmpeg from console `Ctrl+C` events so Python handles the stop request and then closes FFmpeg through its stdin pipe for a clean partial-clip finalize. When archiving is enabled, it also waits for the partial clip and any earlier queued clips to finish transfer and verification before the program exits.

The interrupted clip is preserved but is not counted as a fully completed scheduled clip.

After pressing `Ctrl+C`, keep the terminal open until the recorder reports that pending archive transfers have finished.

## Final pre-run checklist

- Dry run passed.
- Preview looked correct.
- Short local test passed.
- Three-clip archive test passed if archive mode is enabled.
- `python validate_session.py SESSION_PATH` returned `PASS`.
- Archive drive is mounted.
- Laptop is on power.
- You are using the copied local config, not a tracked template.
