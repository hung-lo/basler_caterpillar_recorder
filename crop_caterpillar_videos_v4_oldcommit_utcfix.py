#!/usr/bin/env python3
r"""
MBL Caterpillar Cropper v4 - timestamped output names with oldcommit UTC support.

Current recorder layout:
    ROOT\20260809_134217-0400\clip_0000_134218-0400\camera1.mp4

Old-commit recorder layout:
    ROOT\20260806_133944_oldcommit\clip_0000_133946\camera1.mp4

Important:
- Current-layout timestamps are already Woods Hole/local timestamps.
- Folders ending in "_oldcommit" contain UTC timestamps.
- For this MBL August dataset, oldcommit UTC timestamps are converted to
  Woods Hole EDT, fixed at UTC-04:00.

Output naming for BOTH layouts:
    C01_YYYYMMDD_HHMMSS-0400_clip_0000.mp4

Example oldcommit conversion:
    raw:
      20260806_133944_oldcommit\clip_0000_133946\camera1.mp4

    clip time in raw folder:
      2026-08-06 13:39:46 UTC

    converted Woods Hole time:
      2026-08-06 09:39:46 -0400

    output:
      C01_20260806_093946-0400_clip_0000.mp4

Safety:
- Original videos are never modified or deleted.
- Existing final crops are skipped individually.
- Active/recent raw files are protected by the minimum-age delay.
- Temporary output files are renamed only after ffmpeg succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

VERSION = "MBL Caterpillar Cropper v4 OLDCOMMIT UTC FIX"
DEFAULT_ROOT = Path(r"D:\Hung_MBL\monarch_behavior_windows\new_cohort_C01-08")

EXPECTED_WIDTH = 1200
EXPECTED_HEIGHT = 1920
DEFAULT_MIN_AGE_SECONDS = 180
DEFAULT_POLL_SECONDS = 60

# The oldcommit recordings in this MBL August dataset were written in UTC.
# Woods Hole was EDT = UTC-04:00.
OLDCOMMIT_LOCAL_TZ = timezone(timedelta(hours=-4))

# Fixed 512 x 512 crops measured from the supplied sample frame.
# (x, y, width, height)
CROPS = {
    "C01": (180, 0, 512, 512),
    "C02": (688, 0, 512, 512),
    "C03": (0, 457, 512, 512),
    "C04": (497, 467, 512, 512),
    "C05": (168, 941, 512, 512),
    "C06": (688, 939, 512, 512),
    "C07": (27, 1408, 512, 512),
    "C08": (540, 1408, 512, 512),
}

VIDEO_CODEC = "libx264"
PRESET = "veryfast"
CRF = "18"
ENCODER_THREADS_PER_OUTPUT = "1"

# New/current format, e.g. 20260809_134217-0400
SESSION_CURRENT_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})(?P<tz>[+-]\d{4})$"
)

# Old format, e.g. 20260806_133944_oldcommit
SESSION_OLDCOMMIT_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})_oldcommit$",
    re.IGNORECASE,
)

# Clip folders with or without timezone, e.g.
# clip_0000_134218-0400
# clip_0000_133946
CLIP_RE = re.compile(
    r"^clip_(?P<index>\d+)_(?P<time>\d{6})(?P<tz>[+-]\d{4})?$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=VERSION)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    p.add_argument("--min-age-seconds", type=int, default=DEFAULT_MIN_AGE_SECONDS)
    return p.parse_args()


def creation_flags() -> int:
    if os.name == "nt":
        return getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    return 0


def require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(
            f"ERROR: {name} was not found in PATH. Install FFmpeg and ensure "
            f"{name}.exe is callable from this terminal."
        )


def parse_hhmmss(value: str) -> tuple[int, int, int]:
    return int(value[0:2]), int(value[2:4]), int(value[4:6])


def parse_yyyymmdd(value: str) -> tuple[int, int, int]:
    return int(value[0:4]), int(value[4:6]), int(value[6:8])


def parse_numeric_utc_offset(value: str) -> timezone:
    """Convert '-0400' or '+0530' to datetime.timezone."""
    sign = 1 if value[0] == "+" else -1
    hours = int(value[1:3])
    minutes = int(value[3:5])
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def combine_session_date_and_clip_time(
    session_date: str,
    session_time: str,
    clip_time: str,
    tzinfo: timezone,
) -> datetime:
    """
    Reconstruct the clip datetime from session date + clip HHMMSS.

    The clip folder has no date. If its clock time is earlier than the session
    start clock time, assume the recording crossed midnight and use next day.
    """
    year, month, day = parse_yyyymmdd(session_date)
    sh, sm, ss = parse_hhmmss(session_time)
    ch, cm, cs = parse_hhmmss(clip_time)

    session_dt = datetime(year, month, day, sh, sm, ss, tzinfo=tzinfo)
    clip_dt = datetime(year, month, day, ch, cm, cs, tzinfo=tzinfo)

    if clip_dt < session_dt:
        clip_dt += timedelta(days=1)

    return clip_dt


def source_metadata(source: Path) -> dict[str, str] | None:
    """
    Parse a nested recorder source and return canonical LOCAL clip metadata.

    Returns:
        {
            "date": "20260809",
            "time": "134218",
            "tz": "-0400",
            "clip_index": "0000",
            "layout": "current" | "oldcommit",
        }
    """
    if source.name.lower() != "camera1.mp4":
        return None

    clip_match = CLIP_RE.match(source.parent.name)
    if not clip_match:
        return None

    session_name = source.parent.parent.name
    current_match = SESSION_CURRENT_RE.match(session_name)
    old_match = SESSION_OLDCOMMIT_RE.match(session_name)

    clip_index = clip_match.group("index")
    clip_time = clip_match.group("time")

    if current_match:
        # New/current recordings are already local and timezone-labelled.
        session_tz_text = current_match.group("tz")
        clip_tz_text = clip_match.group("tz") or session_tz_text
        tzinfo = parse_numeric_utc_offset(clip_tz_text)

        local_clip_dt = combine_session_date_and_clip_time(
            session_date=current_match.group("date"),
            session_time=current_match.group("time"),
            clip_time=clip_time,
            tzinfo=tzinfo,
        )

        return {
            "date": local_clip_dt.strftime("%Y%m%d"),
            "time": local_clip_dt.strftime("%H%M%S"),
            "tz": local_clip_dt.strftime("%z"),
            "clip_index": clip_index,
            "layout": "current",
        }

    if old_match:
        # oldcommit timestamps are UTC. Reconstruct the actual UTC clip
        # datetime, including any midnight rollover, then convert to Woods Hole.
        utc_clip_dt = combine_session_date_and_clip_time(
            session_date=old_match.group("date"),
            session_time=old_match.group("time"),
            clip_time=clip_time,
            tzinfo=timezone.utc,
        )
        local_clip_dt = utc_clip_dt.astimezone(OLDCOMMIT_LOCAL_TZ)

        return {
            "date": local_clip_dt.strftime("%Y%m%d"),
            "time": local_clip_dt.strftime("%H%M%S"),
            "tz": local_clip_dt.strftime("%z"),
            "clip_index": clip_index,
            "layout": "oldcommit",
        }

    return None


def discover_sources(root: Path) -> list[Path]:
    """Find flat clips plus nested recorder camera1.mp4 clips."""
    found: set[Path] = set()

    for p in root.glob("clip_*.mp4"):
        if p.is_file():
            found.add(p)

    for p in root.rglob("camera1.mp4"):
        if p.is_file() and p.parent.name.lower().startswith("clip_"):
            if "cropped_by_caterpillar" not in {part.lower() for part in p.parts}:
                found.add(p)

    def safe_sort_key(p: Path):
        try:
            return (p.stat().st_mtime, str(p).lower())
        except OSError:
            return (float("inf"), str(p).lower())

    return sorted(found, key=safe_sort_key)


def recording_stem(source: Path) -> str:
    r"""
    Return the canonical identifying stem for the cropped output.

    Current:
      20260809_134217-0400\clip_0000_134218-0400\camera1.mp4
      -> 20260809_134218-0400_clip_0000

    Oldcommit:
      20260806_133944_oldcommit\clip_0000_133946\camera1.mp4
      -> 20260806_093946-0400_clip_0000
    """
    metadata = source_metadata(source)
    if metadata:
        return (
            f"{metadata['date']}_{metadata['time']}{metadata['tz']}"
            f"_clip_{metadata['clip_index']}"
        )

    # Preserve the legacy behavior for unexpected/flat inputs instead of
    # inventing timestamp information.
    if source.name.lower() == "camera1.mp4":
        return source.parent.name
    return source.stem


def output_path(output_dir: Path, caterpillar_id: str, source: Path) -> Path:
    return output_dir / f"{caterpillar_id}_{recording_stem(source)}.mp4"


def temp_path(output_dir: Path, caterpillar_id: str, source: Path) -> Path:
    return output_dir / f".{caterpillar_id}_{recording_stem(source)}.partial.mp4"


def missing_ids(output_dir: Path, source: Path) -> list[str]:
    return [
        cid
        for cid in CROPS
        if not output_path(output_dir, cid, source).exists()
    ]


def age_seconds(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def probe_resolution(path: Path) -> tuple[int, int] | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        creationflags=creation_flags(),
    )
    if result.returncode != 0:
        return None

    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None


def font_for_ffmpeg() -> str | None:
    for p in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    ):
        if p.exists():
            return p.as_posix().replace(":", r"\:")
    return None


def build_ffmpeg(source: Path, output_dir: Path, ids: list[str]):
    n = len(ids)
    font = font_for_ffmpeg()
    filters: list[str] = []

    if n == 1:
        inputs = ["[0:v]"]
    else:
        labels = "".join(f"[s{i}]" for i in range(n))
        filters.append(f"[0:v]split={n}{labels}")
        inputs = [f"[s{i}]" for i in range(n)]

    pairs: list[tuple[Path, Path]] = []

    for i, cid in enumerate(ids):
        x, y, w, h = CROPS[cid]

        if font:
            draw = (
                f"drawtext=fontfile='{font}':text='{cid}':"
                "x=14:y=14:fontsize=30:"
                "fontcolor=white:borderw=2:bordercolor=black"
            )
        else:
            draw = (
                f"drawtext=text='{cid}':"
                "x=14:y=14:fontsize=30:"
                "fontcolor=white:borderw=2:bordercolor=black"
            )

        filters.append(
            f"{inputs[i]}crop={w}:{h}:{x}:{y},{draw}[out{i}]"
        )
        pairs.append(
            (
                temp_path(output_dir, cid, source),
                output_path(output_dir, cid, source),
            )
        )

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-i",
        str(source),
        "-filter_complex",
        ";".join(filters),
    ]

    for i, (tmp, _final) in enumerate(pairs):
        cmd += [
            "-map",
            f"[out{i}]",
            "-an",
            "-c:v",
            VIDEO_CODEC,
            "-preset",
            PRESET,
            "-crf",
            CRF,
            "-pix_fmt",
            "yuv420p",
            "-threads",
            ENCODER_THREADS_PER_OUTPUT,
            "-movflags",
            "+faststart",
            "-y",
            str(tmp),
        ]

    return cmd, pairs


def process_source(source: Path, output_dir: Path) -> bool:
    ids = missing_ids(output_dir, source)

    if not ids:
        print(f"SKIP already complete: {source}")
        return False

    res = probe_resolution(source)

    if res is None:
        print(f"SKIP ffprobe cannot read yet: {source}")
        return False

    if res != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        print(
            f"SKIP unexpected resolution {res[0]}x{res[1]} "
            f"(expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}): {source}"
        )
        return False

    meta = source_metadata(source)
    if meta and meta["layout"] == "oldcommit":
        print(f"PROCESS OLDCOMMIT UTC->-0400: {source}")
    else:
        print(f"PROCESS {source}")

    print(
        "  outputs: "
        + ", ".join(
            output_path(output_dir, cid, source).name for cid in ids
        )
    )

    for cid in ids:
        tmp = temp_path(output_dir, cid, source)
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass

    cmd, pairs = build_ffmpeg(source, output_dir, ids)
    result = subprocess.run(cmd, creationflags=creation_flags())

    if result.returncode != 0:
        print(f"ERROR ffmpeg failed: {source}")
        for tmp, _ in pairs:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        return False

    for tmp, final in pairs:
        if not tmp.exists():
            print(f"ERROR missing temporary output: {tmp}")
            return False
        os.replace(tmp, final)

    print(f"DONE {recording_stem(source)}")
    return True


def scan_once(
    root: Path,
    min_age: int,
    dry_run: bool,
    max_process: int | None = None,
) -> int:
    output_dir = root / "cropped_by_caterpillar"
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(root)
    print(f"Discovered source videos: {len(sources)}")

    if not sources:
        print("No matching inputs found.")
        return 0

    processed = 0

    for source in sources:
        age = age_seconds(source)

        if age is None:
            print(f"SKIP cannot stat: {source}")
            continue

        ids = missing_ids(output_dir, source)

        if not ids:
            print(f"SKIP already complete: {source}")
            continue

        if age < min_age:
            print(
                f"SKIP too recent ({age:.0f}s old; need {min_age}s): "
                f"{source}"
            )
            continue

        meta = source_metadata(source)
        mode_note = ""
        if meta and meta["layout"] == "oldcommit":
            mode_note = " [oldcommit UTC -> Woods Hole -0400]"

        if dry_run:
            print(f"WOULD PROCESS ({age:.0f}s old){mode_note}: {source}")
            print(
                "  would create: "
                + ", ".join(
                    output_path(output_dir, cid, source).name
                    for cid in ids
                )
            )
            processed += 1
        else:
            if process_source(source, output_dir):
                processed += 1

        if max_process is not None and processed >= max_process:
            break

    return processed


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    print("=" * 72)
    print(VERSION)
    print("=" * 72)
    print(f"Source: {root}")
    print(f"Output: {root / 'cropped_by_caterpillar'}")
    print("Naming: C01_YYYYMMDD_HHMMSS-TZ_clip_0000.mp4")
    print("Oldcommit rule: raw UTC -> Woods Hole UTC-04:00")
    print(r"Input search: clip_*.mp4 plus recursive clip_*\camera1.mp4")
    print(f"Safety delay: {args.min_age_seconds} seconds after last write")

    if not root.exists():
        print(f"ERROR root folder does not exist: {root}", file=sys.stderr)
        return 2

    if not args.dry_run:
        require_program("ffmpeg")
        require_program("ffprobe")

    if not args.watch:
        count = scan_once(root, args.min_age_seconds, args.dry_run)
        label = "eligible source clips" if args.dry_run else "source clips processed"
        print(f"Finished scan. {label}: {count}")
        return 0

    print(
        f"Watch mode: scanning every {args.poll_seconds}s; "
        "max one source per cycle."
    )

    try:
        while True:
            scan_once(
                root,
                args.min_age_seconds,
                False,
                max_process=1,
            )
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\nCrop watcher stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
