#!/usr/bin/env python3
r"""
MBL Caterpillar Cropper v5 - unified timestamped output names.

Supports flat legacy clips:
    ROOT\clip_0000_152652.mp4

and nested recorder layouts:
    ROOT\20260809_134217-0400\clip_0000_134218-0400\camera1.mp4
    ROOT\20260806_133944_oldcommit\clip_0000_133946\camera1.mp4
    ROOT\20260804_145301\clip_0000_145302\camera1.mp4

Canonical nested output names use:
    caterpillar ID + final clip date + final clip time + timezone + raw clip index

Example:
    C01_20260809_134218-0400_clip_0000.mp4

Nested current-offset timestamps stay local.
Legacy `_oldcommit` and no-offset nested recorder layouts are treated as UTC
and converted to Woods Hole local time.

Original videos are never modified or deleted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

VERSION = "MBL Caterpillar Cropper v5 UNIFIED TIMESTAMPS"
DEFAULT_ROOT = Path(r"D:\Hung_MBL\monarch_behavior_windows\new_cohort_C01-08")

EXPECTED_WIDTH = 1200
EXPECTED_HEIGHT = 1920
DEFAULT_MIN_AGE_SECONDS = 180
DEFAULT_POLL_SECONDS = 60

SESSION_CURRENT_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})(?P<tz>[+-]\d{4})$"
)
SESSION_OLDCOMMIT_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})(?:[_-]old(?:[_-]?commit))$",
    re.IGNORECASE,
)
SESSION_LEGACY_UTC_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})$"
)
CLIP_RE = re.compile(
    r"^clip_(?P<index>\d+)_(?P<time>\d{6})(?P<tz>[+-]\d{4})?$",
    re.IGNORECASE,
)

WOODS_HOLE_LEGACY_TZ = dt.timezone(dt.timedelta(hours=-4))


@dataclass(frozen=True)
class ClipMetadata:
    local_datetime: dt.datetime
    clip_index: str
    source_format: str

    @property
    def stem(self) -> str:
        return f"{self.local_datetime.strftime('%Y%m%d_%H%M%S%z')}_clip_{self.clip_index}"


@dataclass(frozen=True)
class SourceNaming:
    stem: str | None
    layout: str | None = None
    error: str | None = None
    session_name: str | None = None
    clip_name: str | None = None
    metadata: ClipMetadata | None = None


class ClipParseError(ValueError):
    """Raised when a nested clip path cannot be interpreted safely."""

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


def _parse_datetime(date_text: str, time_text: str, tzinfo: dt.tzinfo) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(
            f"{date_text}{time_text}", "%Y%m%d%H%M%S"
        ).replace(tzinfo=tzinfo)
    except ValueError:
        return None


def _parse_offset(text: str | None) -> dt.tzinfo | None:
    if not text or len(text) != 5 or text[0] not in "+-":
        return None

    try:
        hours = int(text[1:3])
        minutes = int(text[3:5])
    except ValueError:
        return None

    sign = 1 if text[0] == "+" else -1
    return dt.timezone(sign * dt.timedelta(hours=hours, minutes=minutes))


def _session_layout(session_name: str) -> str | None:
    if SESSION_CURRENT_RE.fullmatch(session_name):
        return "current"
    if SESSION_OLDCOMMIT_RE.fullmatch(session_name):
        return "oldcommit"
    if SESSION_LEGACY_UTC_RE.fullmatch(session_name):
        return "legacy_utc"
    return None


def parse_clip_metadata(source: Path) -> ClipMetadata:
    """Parse a nested recorder source into canonical local clip metadata."""

    if source.name.lower() != "camera1.mp4":
        raise ClipParseError(f"not a nested camera1.mp4 source: {source}")

    if not source.parent.name.lower().startswith("clip_"):
        raise ClipParseError(
            f"nested camera1.mp4 is not inside a clip_ folder: {source}"
        )

    clip_match = CLIP_RE.fullmatch(source.parent.name)
    if not clip_match:
        raise ClipParseError(f"unrecognized clip folder name: {source.parent.name}")

    session_name = source.parent.parent.name
    session_match = (
        SESSION_CURRENT_RE.fullmatch(session_name)
        or SESSION_OLDCOMMIT_RE.fullmatch(session_name)
        or SESSION_LEGACY_UTC_RE.fullmatch(session_name)
    )
    if not session_match:
        raise ClipParseError(f"unrecognized session folder name: {session_name}")

    clip_index = clip_match.group("index")
    clip_time = clip_match.group("time")
    clip_tz = clip_match.group("tz")
    session_date = session_match.group("date")
    session_time = session_match.group("time")

    layout = _session_layout(session_name)
    if layout is None:
        raise ClipParseError(f"unrecognized session folder name: {session_name}")

    if layout == "current":
        session_tz_text = session_match.group("tz")
        if session_tz_text is None:
            raise ClipParseError(
                f"missing timezone offset in session folder: {session_name}"
            )
        session_tz = _parse_offset(session_tz_text)
        if session_tz is None:
            raise ClipParseError(
                f"invalid timezone offset in session folder: {session_name}"
            )
        clip_tz_text = clip_tz or session_tz_text
        clip_tzinfo = _parse_offset(clip_tz_text)
        if clip_tzinfo is None:
            raise ClipParseError(
                f"invalid timezone offset in clip folder: {source.parent.name}"
            )
        session_dt = _parse_datetime(session_date, session_time, session_tz)
        clip_dt = _parse_datetime(session_date, clip_time, clip_tzinfo)
        if session_dt is None or clip_dt is None:
            raise ClipParseError(
                f"invalid numeric timestamp in nested source: {source}"
            )
        if clip_dt < session_dt:
            clip_dt += dt.timedelta(days=1)
        return ClipMetadata(local_datetime=clip_dt, clip_index=clip_index, source_format="current")

    if layout in {"oldcommit", "legacy_utc"}:
        session_dt = _parse_datetime(session_date, session_time, dt.timezone.utc)
        clip_dt = _parse_datetime(session_date, clip_time, dt.timezone.utc)
        if session_dt is None or clip_dt is None:
            raise ClipParseError(
                f"invalid numeric timestamp in nested source: {source}"
            )
        if clip_dt < session_dt:
            clip_dt += dt.timedelta(days=1)
        local_dt = clip_dt.astimezone(WOODS_HOLE_LEGACY_TZ)
        return ClipMetadata(
            local_datetime=local_dt,
            clip_index=clip_index,
            source_format=layout,
        )

    raise ClipParseError(f"unrecognized session folder name: {session_name}")


def source_metadata(source: Path) -> ClipMetadata | None:
    """Return parsed nested metadata or None when the layout is unsafe."""

    if source.name.lower() != "camera1.mp4" or not source.parent.name.lower().startswith("clip_"):
        return None
    try:
        return parse_clip_metadata(source)
    except ClipParseError:
        return None


def canonical_nested_recording_stem(source: Path) -> str | None:
    metadata = source_metadata(source)
    return None if metadata is None else metadata.stem


def resolve_source_naming(source: Path) -> SourceNaming:
    if source.name.lower() != "camera1.mp4":
        return SourceNaming(stem=source.stem, layout="flat_legacy")

    session_name = source.parent.parent.name
    clip_name = source.parent.name

    if not source.parent.name.lower().startswith("clip_"):
        return SourceNaming(
            stem=None,
            layout=None,
            error=f"nested camera1.mp4 is not inside a clip_ folder: {source}",
            session_name=session_name,
            clip_name=clip_name,
        )

    try:
        metadata = parse_clip_metadata(source)
    except ClipParseError as exc:
        layout = _session_layout(session_name)
        return SourceNaming(
            stem=None,
            layout=layout,
            error=str(exc),
            session_name=session_name,
            clip_name=clip_name,
        )

    return SourceNaming(
        stem=metadata.stem,
        layout=metadata.source_format,
        metadata=metadata,
        session_name=session_name,
        clip_name=clip_name,
    )


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

    layout_tag = f"[{naming.layout}] " if naming.layout and naming.layout != "flat_legacy" else ""
    print(f"PROCESS {layout_tag}{source}")
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
            print(f"SKIP unrecognized recorder timestamp layout: {source}")
            if naming.session_name is not None:
                print(f"  session: {naming.session_name}")
            if naming.clip_name is not None:
                print(f"  clip: {naming.clip_name}")
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
            layout_tag = f"[{naming.layout}] " if naming.layout and naming.layout != "flat_legacy" else ""
            print(f"WOULD PROCESS {layout_tag}({age:.0f}s old): {source}")
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
    print(
        "Timestamp rules: current offset timestamps preserved; "
        "legacy/oldcommit UTC -> Woods Hole -0400"
    )
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
