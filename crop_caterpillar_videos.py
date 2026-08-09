#!/usr/bin/env python3
r"""
MBL Caterpillar Cropper v3 - timestamped output names.

Supports both:
    ROOT\clip_0000_152652.mp4

and the recorder layout:
    ROOT\20260809_134217-0400\clip_0000_134218-0400\camera1.mp4

For the nested recorder layout, output names use:
    caterpillar ID + clip date + actual clip start time + timezone + raw clip index

Example:
    C01_20260809_134218-0400_clip_0000.mp4

The date is taken from the session directory.
The time/timezone and clip index are taken from the clip directory.

Original videos are never modified or deleted.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from caterpillar_clip_naming import ClipParseError, resolve_source_naming

VERSION = "MBL Caterpillar Cropper v3 TIMESTAMPED"
DEFAULT_ROOT = Path(r"D:\Hung_MBL\monarch_behavior_windows\new_cohort_C01-08")

EXPECTED_WIDTH = 1200
EXPECTED_HEIGHT = 1920
DEFAULT_MIN_AGE_SECONDS = 180
DEFAULT_POLL_SECONDS = 60

# Fixed 512x512 crops measured from the supplied sample frame.
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


def discover_sources(root: Path) -> list[Path]:
    """Find flat clips plus nested recorder camera1.mp4 clips."""
    found: set[Path] = set()

    # Original flat layout.
    for p in root.glob("clip_*.mp4"):
        if p.is_file():
            found.add(p)

    # Recorder layout: session/clip_xxxx_xxxxxx/camera1.mp4.
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


def output_path(
    output_dir: Path,
    caterpillar_id: str,
    naming,
) -> Path:
    if naming.stem is None:
        raise ClipParseError(naming.error or "missing canonical stem")
    return output_dir / f"{caterpillar_id}_{naming.stem}.mp4"


def temp_path(
    output_dir: Path,
    caterpillar_id: str,
    naming,
) -> Path:
    if naming.stem is None:
        raise ClipParseError(naming.error or "missing canonical stem")
    return output_dir / f".{caterpillar_id}_{naming.stem}.partial.mp4"


def missing_ids(output_dir: Path, naming) -> list[str]:
    return [
        cid
        for cid in CROPS
        if not output_path(output_dir, cid, naming).exists()
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


def build_ffmpeg(source: Path, output_dir: Path, ids: list[str], naming):
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
                temp_path(output_dir, cid, naming),
                output_path(output_dir, cid, naming),
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


def process_source(source: Path, output_dir: Path, naming, ids: list[str]) -> bool:

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

    print(f"PROCESS {source}")
    print(
        "  outputs: "
        + ", ".join(
            output_path(output_dir, cid, naming).name for cid in ids
        )
    )

    for cid in ids:
        tmp = temp_path(output_dir, cid, naming)
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass

    cmd, pairs = build_ffmpeg(source, output_dir, ids, naming)
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

    print(f"DONE {naming.stem}")
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
        print(
            r"Expected nested layout like: "
            r"20260809_134217-0400\clip_0000_134218-0400\camera1.mp4"
        )
        return 0

    processed = 0

    for source in sources:
        naming = resolve_source_naming(source)
        if naming.error:
            print(f"SKIP unparsable source: {source}")
            print(f"  reason: {naming.error}")
            continue

        age = age_seconds(source)
        if age is None:
            print(f"SKIP cannot stat: {source}")
            continue

        ids = missing_ids(output_dir, naming)

        if not ids:
            print(f"SKIP already complete: {source}")
            continue

        if age < min_age:
            print(
                f"SKIP too recent ({age:.0f}s old; need {min_age}s): "
                f"{source}"
            )
            continue

        if dry_run:
            print(f"WOULD PROCESS ({age:.0f}s old): {source}")
            print(
                "  would create: "
                + ", ".join(
                    output_path(output_dir, cid, naming).name
                    for cid in ids
                )
            )
            processed += 1
        else:
            if process_source(source, output_dir, naming, ids):
                processed += 1

        if max_process is not None and processed >= max_process:
            break

    return processed


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    print("=" * 68)
    print(VERSION)
    print("=" * 68)
    print(f"Source: {root}")
    print(f"Output: {root / 'cropped_by_caterpillar'}")
    print("Naming: CXX_YYYYMMDD_HHMMSS-TZ_clip_NNNN.mp4")
    print("Current timestamps: already local")
    print("oldcommit timestamps: UTC -> Woods Hole local")
    print(
        r"Input search: clip_*.mp4 plus recursive clip_*\camera1.mp4"
    )
    print(
        f"Safety delay: {args.min_age_seconds} seconds after last write"
    )

    if not root.exists():
        print(
            f"ERROR root folder does not exist: {root}",
            file=sys.stderr,
        )
        return 2

    if not args.dry_run:
        require_program("ffmpeg")
        require_program("ffprobe")

    if not args.watch:
        count = scan_once(
            root,
            args.min_age_seconds,
            args.dry_run,
        )
        label = (
            "eligible source clips"
            if args.dry_run
            else "source clips processed"
        )
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
