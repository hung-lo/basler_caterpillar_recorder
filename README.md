# Basler caterpillar recorder: fast-start pilot

This repository is a practical starter for scheduled recording from the two Basler USB cameras:

- **acA4024-29uc**: 4024 x 3036, 12.2 MP, rolling shutter. Best used as the high-resolution overhead overview.
- **a2A1920-160ucBAS**: 1920 x 1200, 2.3 MP, global shutter. Best used as the second fixed overhead arena. Its 160-fps maximum is unnecessary for caterpillar behavior.

The supplied script runs on Windows or recent macOS through Basler pylon/pypylon and uses FFmpeg for consistent H.264/MP4 output.

## Recommended pilot decision

Start with the simplest design that preserves identity and outcome labels:

1. Put each of the seven larvae in a **separately labeled enclosure** within a fixed camera field of view.
2. Do **not** draw directly on the larvae for the first pilot. Marks will be shed at molts, handling can perturb behavior, and body marks complicate segmentation. Put the ID and a scale marker outside each enclosure instead.
3. Split the cohort across two fixed overhead arenas: **four larvae under the acA4024** and **three larvae under the a2A1920**, ideally in 2 x 2 enclosure grids.
4. Record both cameras continuously or nearly continuously at **5 fps**. This gives all seven animals comparable spatial resolution and is preferable to spending the second camera on only one close-up animal.
5. Keep lighting, camera position, milkweed presentation, temperature, and leaf-replacement time as constant as possible.
6. Record through larval development, pupation, and adult eclosion. Score larval survival, successful chrysalis formation, and successful adult emergence separately.

The provided example movie is **1920 x 1200 at 10 fps**. It already resolves body curvature, head direction, and short-timescale posture changes. Because monarch larvae move much more slowly than fish, **5 fps continuous recording** is a good starting compromise. For storage-limited recording, use **2 fps continuously rather than 10-fps sparse bursts**.

A useful framing rule is that the smallest third-instar larva should be at least roughly **80-100 pixels long** in the final encoded overview video. If it is smaller, tighten the field of view or keep the higher encoded resolution.

## What the script saves

Each session contains:

- one H.264 MP4 per camera per clip;
- a compressed `timestamps.csv.gz` with host UTC time, monotonic time, camera timestamp when available, frame/block ID, and skipped-image count;
- one JSON sidecar with requested and actual camera settings, frame count, measured receive rate, and failures;
- a session manifest and an exact copy of the YAML configuration;
- recorder, FFmpeg, and remux logs when something fails.

Frames are acquired through pypylon, converted to BGR, optionally resized, and piped to FFmpeg. Capture is first written to recoverable Matroska (`.mkv`) and then remuxed without recompression to MP4. **The macOS path does not require saving individual TIFF files**; the frames are encoded directly while recording.

An example filename is `monarch_behavior_pilot_cohort_pilot01_arena_A_M01-M04_clip0000_20260803T130000.000000Z.mp4`. Individual animal IDs are also retained in the copied YAML and JSON metadata.

## Installation

### 1. Install Basler pylon

Install the pylon Software Suite appropriate for the operating system and camera interface.

For the fastest first setup, **Windows is the least ambiguous choice** because pylon Viewer and the USB configuration tools are mature there. Python acquisition also works on recent Intel and Apple Silicon Macs, but current Basler pages are slightly inconsistent: pypylon wheels are built for macOS 14 Sonoma or newer, while the pylon 26.06 suite page lists macOS 15. Check the exact installer before relying on the MacBook; use the Windows PC if the Mac is older.

### 2. Install FFmpeg

Install FFmpeg and make sure this succeeds in the same terminal:

```bash
ffmpeg -version
```

### 3. Create a Python environment

Recommended Python version: 3.11 or 3.12.

macOS/Linux:

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

## First-camera test

List connected cameras:

```bash
python record_basler.py --list-cameras
```

The example YAML can match the two unique model names without serial numbers. After the first test, putting the serial numbers into the YAML makes camera identity completely explicit.

Preview arena A:

```bash
python record_basler.py --config config_pilot.yaml --preview arena_A_M01-M04
```

Preview arena B:

```bash
python record_basler.py --config config_pilot.yaml --preview arena_B_M05-M07
```

Preview controls:

- `q` or Escape: quit;
- `s`: save a full processed snapshot;
- `p`: print current exposure, gain, acquisition rate, and resulting rate.

For setup only, `auto_exposure: true` and `auto_gain: true` can help find a usable range. For the actual experiment, return to manual exposure/gain so leaf replacement and animal position do not change image brightness automatically. Use diffuse lighting and the lowest gain that gives a clean image.

## Validate and record

Validate paths and schedule without opening cameras:

```bash
python record_basler.py --config config_pilot.yaml --dry-run
```

Run one ten-second end-to-end test with both cameras:

```bash
python record_basler.py --config config_smoke_test.yaml
```

Open both MP4 files and inspect the JSON sidecars. Confirm that `success` is true, `grab_failures` is zero, `measured_receive_fps` is close to the requested rate, and `mp4_remux_succeeded` is true. Then start the pilot:

```bash
python record_basler.py --config config_pilot.yaml
```

Stop cleanly with `Ctrl+C`.

Connect both cameras directly to USB 3 ports when possible. If frames are incomplete or skipped, move the cameras to ports on separate USB host controllers, shorten/replace the cables, or reduce frame rate/throughput. On Windows, pylon's Bandwidth Manager and USB Configurator are useful for diagnosing whether both ports share the same controller.

The example config records 30-minute clips for 24 hours. Equal clip duration and interval produce **near-continuous** recording. There is a short boundary gap while a clip closes and the next writer starts. For fewer gaps, use longer clips such as 2-6 hours. A future version can use one uninterrupted acquisition process with FFmpeg's segment muxer if exact gapless recording is required.

The two cameras share a scheduled host start and retain per-frame timestamps, but they are not hardware-trigger synchronized. That is sufficient when each camera covers different animals. Use a common hardware trigger only if a later experiment requires frame-level multi-view reconstruction of the same animal.

## Storage check before the long run

Compression depends strongly on leaf texture, sensor noise, and movement, so do not rely on a fixed estimate. After the ten-second smoke test, run a 10- or 30-minute test and extrapolate from the resulting file sizes. As a planning range, these settings will often be on the order of **tens of gigabytes per camera per day**, not hundreds of megabytes. A dedicated 1-TB SSD is a sensible pilot target, but verify the actual rate on your scene before leaving the system unattended.

Keep at least 15-20% of the recording disk free. Write directly to a local SSD during acquisition; copy to network or archival storage after clips close.

## Schedule examples

`interval_s` is the time from one clip start to the next clip start.

Thirty minutes every hour:

```yaml
schedule:
  clip_duration_s: 1800
  interval_s: 3600
  total_duration_h: 24
```

One minute every two minutes:

```yaml
schedule:
  clip_duration_s: 60
  interval_s: 120
  total_duration_h: 24
```

Near-continuous 1-hour files:

```yaml
schedule:
  clip_duration_s: 3600
  interval_s: 3600
  total_duration_h: 24
```

For this biological question, the last option is preferred. Long gaps can miss feeding bouts, molt-associated immobility, pre-pupal wandering, and the transition into the J posture.

## Suggested camera settings

### Arena A: acA4024-29uc

- Layout: four separately housed larvae in a fixed 2 x 2 grid.
- Acquisition: full 4024 x 3036 Bayer 8-bit at 5 fps.
- Encoding: resize to about 2012 x 1518 and H.264 CRF 24-26.
- Exposure starting point: 4-8 ms with bright diffuse illumination.
- Gain: near minimum.

The camera uses a rolling shutter, but this should not be consequential for slow larval motion.

### Arena B: a2A1920-160ucBAS

- Layout: three separately housed larvae in a fixed 2 x 2 grid, leaving one position empty for a scale/lighting reference.
- Acquisition/encoding: 1920 x 1200 at 5 fps.
- Exposure starting point: 2-6 ms.
- Gain: near minimum.

The camera uses a global shutter. Its 160-fps maximum is unnecessary here and would multiply storage and USB load. After the pipeline is stable, a separate brief 10-fps close-up can be collected to validate head and feeding motion if needed.

### Binning versus resizing

The script attempts camera binning only when the requested nodes are exposed by that model/firmware. Binning is not required for the pilot:

- use the camera's full field of view at a low frame rate;
- transmit Bayer 8-bit data to limit USB bandwidth;
- resize on the computer before H.264 encoding.

Cropping reduces field of view; resizing preserves it. Therefore, do not replace full-sensor acquisition with a small central ROI unless all enclosures remain inside that ROI.

## Night recording limitation

Both cameras are color models and contain IR-cut filters as shipped. The acA4024 filter is fixed in its holder; the a2A1920 filter holder can be removed using Basler's documented procedure. Therefore, an 850- or 940-nm lamp may produce little or no useful image with the cameras in their current state.

For a pilot beginning immediately:

- record the light phase first;
- preserve a normal light/dark cycle;
- do not leave visible light on all night merely for video, because nighttime illumination itself can change monarch larval feeding;
- consider modifying only the a2A camera or adding a monochrome/IR-sensitive camera later if nighttime behavior becomes central.

## Pilot experimental layout

Use a fixed grid with one caterpillar per enclosure. Put a large external ID, date, and scale bar in every enclosure crop. Randomize enclosure positions if possible, and rotate positions between cohorts rather than during a single individual's recording.

Record the following manually once or twice daily:

- estimated instar and molt events;
- leaf replacement time and milkweed source/leaf age;
- body length or area from a standardized frame;
- remaining leaf area or estimated leaf area consumed;
- frass count/area as a feeding proxy;
- abnormal coloration, prolonged immobility, or disease signs;
- time of J-hanging, chrysalis formation, and adult eclosion;
- final outcome: larval death, failed pupation, successful chrysalis but failed eclosion, or successful adult emergence.

Do not interpret immobility automatically as poor condition. Caterpillars become very still around molts and before pupation, so the timing and duration of inactivity relative to developmental stage are likely more informative than total activity alone.

## First analysis pass

Start with interpretable features before an unsupervised behavior model:

1. Divide arena A into four fixed crops and arena B into three fixed crops. Identity is then spatial and does not require multi-animal re-identification.
2. Segment the animal or predict a short midline: head, 3-5 body points, and tail.
3. Compute distance traveled, active fraction, median and peak speed, body curvature, head-sweep rate, time on/off milkweed, feeding-bout duration, rest-bout duration, and daily rhythm.
4. Estimate growth from body mask area/length and feeding from leaf-area loss plus head-at-leaf-edge behavior.
5. Align every animal to molt, J-hang, and pupation times, not only chronological clock time.
6. After the tracking is stable, add unsupervised behavioral segmentation to discover recurrent postures or transitions that were not hand-defined.

With seven animals, treat prediction as a demonstration rather than a performance claim. The strongest pilot result would be reliable multi-day acquisition, stable identity, useful segmentation, and individual behavioral trajectories linked to clearly defined metamorphic outcomes.

## References used for the design

- Basler pylon documentation: <https://docs.baslerweb.com/pylon-software-suite>
- Basler pypylon: <https://github.com/basler/pypylon>
- acA4024-29uc documentation: <https://docs.baslerweb.com/aca4024-29uc>
- a2A1920-160ucBAS documentation: <https://docs.baslerweb.com/a2a1920-160ucbas>
- Bedbrook et al., *Lifelong behavioral screen reveals an architecture of vertebrate aging*: <https://www.science.org/doi/10.1126/science.aea9795>
- Associated analysis code: <https://github.com/cnbedbrook/lifelong_behavior>
