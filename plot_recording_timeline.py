#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
from collections import defaultdict
import gzip
import hashlib
import io
import json
import logging
import os
import tempfile
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Optional, Sequence
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as patheffects
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import NullFormatter, NullLocator
    from matplotlib.transforms import blended_transform_factory
except ImportError as exc:  # pragma: no cover - import-time guard for missing optional deps
    raise SystemExit(
        "plot_recording_timeline.py requires matplotlib. Install dependencies with "
        "`python -m pip install -r requirements.txt`."
    ) from exc

import numpy as np

from analysis_timing import (
    DEFAULT_TIMEZONE,
    TIMESTAMP_SUFFIXES,
    UTC,
    coerce_int,
    format_local,
    format_utc,
    load_timezone,
    open_text_file,
    parse_iso_datetime,
    parse_timestamp_row,
    parse_utc_value,
    timezone_label,
    to_plot_local,
    utc_from_ns,
)

LOG = logging.getLogger("plot_recording_timeline")

DEFAULT_ANIMALS = [f"C{index:02d}" for index in range(1, 9)]
ANIMAL_ORDER = [f"C{index:02d}" for index in range(1, 9)]
MAX_CLIP_ANNOTATIONS = 60
POINT_EVENT_DURATION_SECONDS = 1
SHORT_EVENT_THRESHOLD_SECONDS = 300
DEFAULT_POINT_MARKER_SIZE = 70
SHED_MARKER_SIZE = 100
STIM_MARKER_SIZE = 180
STIM_MARKER_ZORDER = 100
DEATH_MARKER_SIZE = 120
MOTION_IMMOBILE_COLOR = "#e2e2e2"
MOTION_MOBILE_COLOR = "#59a14f"
MOTION_IMMOBILE_ALPHA = 0.38
MOTION_MOBILE_ALPHA = 0.48
RECORDING_COLOR = "#4c78a8"
GAP_SHADE_COLOR = "#fafafa"
GAP_BOUNDARY_COLOR = "#d1d5db"
ROW_SEPARATOR_COLOR = "#e5e7eb"
ROW_BACKGROUND_COLOR = "#fbfbfc"
MAJOR_GAP_MIN_SECONDS = 5 * 60
MAX_MAJOR_GAP_LABELS = 6
GOOGLE_SHEETS_HOST = "docs.google.com"
GOOGLE_SHEETS_TIMEOUT_SECONDS = 30
GOOGLE_SHEETS_USER_AGENT = "basler-caterpillar-recorder/1.0"
GLOBAL_EVENT_ALIASES = {"all", "global", "*"}
VIDEO_QUALITY_LOW_COLOR = "#F59E0B"
VIDEO_QUALITY_LOW_ALPHA = 0.08
GENERIC_GLOBAL_EVENT_COLOR = "#94a3b8"
GENERIC_GLOBAL_EVENT_ALPHA = 0.08
MIN_GLOBAL_EVENT_LABEL_SECONDS = 5 * 60
FOOD_UNAVAILABLE_COLOR = "#FB7185"
FOOD_UNAVAILABLE_ALPHA = 0.08
FOOD_UNAVAILABLE_EVENT_NAMES = {
    "food_unavailable",
}

TIMELINE_LEGEND_ORDER = [
    "Shed / molt",
    "J-hang",
    "Pupation",
    "\u26a1 Electrical stimulation",
    "Death",
    "Automatic feeding bouts",
    "Motion-derived immobile",
    "Motion-derived mobile",
    "Low video quality",
    "Food unavailable",
    "Manual interval",
]

SHED_EVENT_NAMES = {
    "shed",
    "shed_found",
    "molt",
    "molting",
}
STIM_EVENT_NAMES = {
    "electrical_stimulation",
    "electric_stimulation",
    "electrical_shock",
    "electric_shock",
    "shock",
}
DEATH_EVENT_NAMES = {
    "death",
    "dead",
}
J_HANG_EVENT_NAMES = {
    "j_hang",
}
PUPATION_EVENT_NAMES = {
    "pupation",
}
LOW_MOBILITY_EVENT_NAMES = {
    "low_mobility",
}
VIDEO_QUALITY_GLOBAL_EVENT_NAMES = {
    "video_quality_low",
    "poor_video_quality",
    "bad_video_quality",
    "bad_lighting",
    "poor_lighting",
}

SHED_COLOR = "#d97706"
J_HANG_COLOR = SHED_COLOR
PUPATION_COLOR = SHED_COLOR
STIM_COLOR = "#dc2626"
DEATH_COLOR = "#111827"
DEFAULT_POINT_COLOR = "#475569"
INTERVAL_BAR_COLOR = "#8aa1c7"
FEEDING_COLOR = "#7c3aed"
J_HANG_MARKER = "v"
PUPATION_MARKER = "D"
J_HANG_MARKER_SIZE = SHED_MARKER_SIZE
PUPATION_MARKER_SIZE = SHED_MARKER_SIZE
RECORDING_COVERAGE_FIELDNAMES = [
    "clip_id",
    "timestamp_file",
    "video_file",
    "start_utc",
    "end_utc",
    "start_local",
    "end_local",
    "duration_s",
    "frames",
    "timestamp_size_bytes",
    "timestamp_mtime_ns",
]


@dataclasses.dataclass(frozen=True)
class RecordingClip:
    clip_id: str
    timestamp_file: Path
    video_file: Path
    camera_label: str
    start_utc: dt.datetime
    end_utc: dt.datetime
    duration_s: float
    frames: int
    timestamp_size_bytes: Optional[int] = None
    timestamp_mtime_ns: Optional[int] = None

    @property
    def start_local(self) -> dt.datetime:
        return self.start_utc.astimezone()

    @property
    def end_local(self) -> dt.datetime:
        return self.end_utc.astimezone()


@dataclasses.dataclass(frozen=True)
class BehaviorEvent:
    animal_id: str
    start_local: dt.datetime
    end_local: dt.datetime
    event: str
    kind: str
    notes: str
    is_point: bool = False


@dataclasses.dataclass(frozen=True)
class MotionState:
    animal_id: str
    clip_key: str
    start_utc: dt.datetime
    end_utc: dt.datetime
    state: str
    threshold: float
    threshold_source: str
    mean_motion_energy: float
    peak_motion_energy: float
    n_windows: int


@dataclasses.dataclass(frozen=True)
class MotionEnergySample:
    animal_id: str
    clip_key: str
    timestamp_utc: dt.datetime
    motion_energy: float


@dataclasses.dataclass(frozen=True)
class GlobalEvent:
    start_local: dt.datetime
    end_local: dt.datetime
    event: str
    kind: str
    notes: str
    approximate: bool = False


@dataclasses.dataclass(frozen=True)
class GoogleSheetSource:
    source_url: str
    sheet_id: str
    gid: str
    export_url: str

def strip_timestamp_suffix(name: str) -> str:
    for suffix in TIMESTAMP_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"Unsupported timestamp filename: {name}")


def video_file_for_timestamp_file(timestamp_file: Path) -> Path:
    base = strip_timestamp_suffix(timestamp_file.name)
    return timestamp_file.with_name(f"{base}.mp4")




def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_bytes(data)
    temp_path.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def recording_path_keys(path: Path) -> tuple[str, str]:
    raw = str(path)
    resolved = str(path.resolve(strict=False))
    if resolved == raw:
        return (raw, raw)
    return (raw, resolved)


def recording_clip_identity(timestamp_file: Path, root: Path) -> tuple[str, str]:
    try:
        rel_dir = timestamp_file.parent.relative_to(root)
        clip_id_prefix = rel_dir.as_posix()
    except ValueError:
        clip_id_prefix = timestamp_file.parent.name
    if clip_id_prefix == ".":
        clip_id_prefix = ""
    camera_label = strip_timestamp_suffix(timestamp_file.name)
    clip_id = f"{clip_id_prefix}/{camera_label}" if clip_id_prefix else camera_label
    return clip_id, camera_label


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def parse_google_sheet_url(value: str) -> GoogleSheetSource:
    parsed = urllib_parse.urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Google Sheet URL must start with http:// or https://")
    if parsed.netloc.lower() != GOOGLE_SHEETS_HOST:
        raise ValueError("Only supported Google Sheets URLs from docs.google.com are accepted")

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 3 or path_parts[0] != "spreadsheets" or path_parts[1] != "d":
        raise ValueError("Unsupported Google Sheets URL format")
    sheet_id = path_parts[2].strip()
    if not sheet_id:
        raise ValueError("Google Sheets URL is missing a sheet ID")

    query = urllib_parse.parse_qs(parsed.query)
    fragment = urllib_parse.parse_qs(parsed.fragment)
    gid = (
        (query.get("gid") or [None])[0]
        or (fragment.get("gid") or [None])[0]
        or "0"
    )
    if gid == "0":
        LOG.warning("Google Sheets URL did not include a tab gid; defaulting to gid=0")

    export_query = urllib_parse.urlencode({"format": "csv", "gid": gid})
    export_url = f"https://{GOOGLE_SHEETS_HOST}/spreadsheets/d/{sheet_id}/export?{export_query}"
    return GoogleSheetSource(
        source_url=value,
        sheet_id=sheet_id,
        gid=gid,
        export_url=export_url,
    )


def normalized_event_terms(event_name: str, kind_name: str) -> set[str]:
    return {
        term
        for term in {
            normalize_event_name(event_name),
            normalize_event_name(kind_name),
        }
        if term
    }


def looks_like_html_document(text: str) -> bool:
    prefix = text.lstrip()[:256].lower()
    return prefix.startswith("<!doctype html") or prefix.startswith("<html")


def fetch_google_sheet_csv(source: GoogleSheetSource) -> bytes:
    request = urllib_request.Request(
        source.export_url,
        headers={"User-Agent": GOOGLE_SHEETS_USER_AGENT},
    )
    try:
        with urllib_request.urlopen(request, timeout=GOOGLE_SHEETS_TIMEOUT_SECONDS) as response:
            csv_bytes = response.read()
    except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Google Sheet could not be downloaded as CSV. "
            "Make sure the sheet is accessible to the plotting computer without interactive login, "
            "or export Animal_event_log as CSV and use a local path."
        ) from exc

    try:
        decoded = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RuntimeError("Downloaded Google Sheet CSV could not be decoded as UTF-8") from exc
    if looks_like_html_document(decoded):
        raise RuntimeError(
            "Google Sheet could not be downloaded as CSV. "
            "Make sure the sheet is accessible to the plotting computer without interactive login, "
            "or export Animal_event_log as CSV and use a local path."
        )
    return csv_bytes


def write_google_sheet_event_snapshot(
    *,
    root: Path,
    source: GoogleSheetSource,
    csv_bytes: bytes,
    behavior_events: list[BehaviorEvent],
    global_events: list[GlobalEvent],
) -> None:
    snapshot_path = root / "behavior_events_used.csv"
    metadata_path = root / "behavior_events_source.json"
    atomic_write_bytes(snapshot_path, csv_bytes)
    metadata = {
        "source_type": "google_sheet",
        "source_url": source.source_url,
        "sheet_id": source.sheet_id,
        "gid": source.gid,
        "export_url": source.export_url,
        "fetched_at_utc": format_utc(dt.datetime.now(UTC)),
        "sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "rows": len(behavior_events) + len(global_events),
        "behavior_events": len(behavior_events),
        "global_events": len(global_events),
        "snapshot_csv": relative_to_root(snapshot_path, root),
    }
    atomic_write_text(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    LOG.info("Saved event snapshot: %s", snapshot_path)
    LOG.info("Saved event source metadata: %s", metadata_path)


def discover_timestamp_files(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    discovered: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name for name in dirnames if name.lower() != "cropped_by_caterpillar"
        )
        for filename in sorted(filenames):
            if filename.endswith(TIMESTAMP_SUFFIXES):
                discovered.append(Path(dirpath) / filename)
    return discovered


def read_timestamp_clip(timestamp_file: Path, root: Path) -> RecordingClip:
    stat_result = timestamp_file.stat()
    start_utc: Optional[dt.datetime] = None
    end_utc: Optional[dt.datetime] = None
    frames = 0
    with open_text_file(timestamp_file) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("missing CSV header")
        for row_index, row in enumerate(reader, start=2):
            try:
                timestamp = parse_timestamp_row(row)
                if start_utc is None or timestamp < start_utc:
                    start_utc = timestamp
                if end_utc is None or timestamp > end_utc:
                    end_utc = timestamp
                frames += 1
            except Exception as exc:
                raise ValueError(f"row {row_index}: {exc}") from exc

    if frames == 0 or start_utc is None or end_utc is None:
        raise ValueError("no timestamp rows")

    video_file = video_file_for_timestamp_file(timestamp_file)
    if not video_file.exists():
        raise ValueError(f"missing video sidecar: {video_file.name}")

    clip_id, camera_label = recording_clip_identity(timestamp_file, root)

    return RecordingClip(
        clip_id=clip_id,
        timestamp_file=timestamp_file,
        video_file=video_file,
        camera_label=camera_label,
        start_utc=start_utc,
        end_utc=end_utc,
        duration_s=(end_utc - start_utc).total_seconds(),
        frames=frames,
        timestamp_size_bytes=stat_result.st_size,
        timestamp_mtime_ns=stat_result.st_mtime_ns,
    )


def load_recording_coverage_cache(
    coverage_cache: Path,
) -> tuple[dict[str, dict[str, str]], list[tuple[str, dict[str, str]]]]:
    cache_index: dict[str, dict[str, str]] = {}
    cache_entries: list[tuple[str, dict[str, str]]] = []
    if not coverage_cache.exists():
        return cache_index, cache_entries
    try:
        with coverage_cache.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return cache_index, cache_entries
            for row in reader:
                timestamp_text = str(row.get("timestamp_file") or "").strip()
                if not timestamp_text:
                    continue
                timestamp_path = Path(timestamp_text)
                primary_key = str(timestamp_path)
                cache_entries.append((primary_key, row))
                for key in recording_path_keys(timestamp_path):
                    cache_index[key] = row
    except Exception as exc:
        LOG.warning("Ignoring recording coverage cache %s: %s", coverage_cache, exc)
        return {}, []
    return cache_index, cache_entries


def recording_clip_from_cache_row(timestamp_file: Path, root: Path, row: dict[str, str]) -> RecordingClip:
    stat_result = timestamp_file.stat()
    size_text = str(row.get("timestamp_size_bytes") or "").strip()
    mtime_text = str(row.get("timestamp_mtime_ns") or "").strip()
    if not size_text or not mtime_text:
        raise ValueError("missing cache metadata")
    if int(size_text) != stat_result.st_size:
        raise ValueError("timestamp sidecar size changed")
    if int(mtime_text) != stat_result.st_mtime_ns:
        raise ValueError("timestamp sidecar mtime changed")

    start_utc = parse_utc_value(str(row.get("start_utc") or "").strip())
    end_utc = parse_utc_value(str(row.get("end_utc") or "").strip())
    duration_s_text = str(row.get("duration_s") or "").strip()
    if not duration_s_text:
        raise ValueError("missing cached duration")
    cached_duration_s = float(duration_s_text)
    computed_duration_s = (end_utc - start_utc).total_seconds()
    if abs(cached_duration_s - computed_duration_s) > 1e-6:
        raise ValueError("cached duration does not match the stored timestamps")
    frames = int(str(row.get("frames") or "").strip())

    video_file = video_file_for_timestamp_file(timestamp_file)
    if not video_file.exists():
        raise ValueError(f"missing video sidecar: {video_file.name}")

    clip_id, camera_label = recording_clip_identity(timestamp_file, root)
    return RecordingClip(
        clip_id=clip_id,
        timestamp_file=timestamp_file,
        video_file=video_file,
        camera_label=camera_label,
        start_utc=start_utc,
        end_utc=end_utc,
        duration_s=(end_utc - start_utc).total_seconds(),
        frames=frames,
        timestamp_size_bytes=stat_result.st_size,
        timestamp_mtime_ns=stat_result.st_mtime_ns,
    )


def collect_recording_clips(
    root: Path,
    *,
    coverage_cache: Optional[Path] = None,
) -> tuple[list[RecordingClip], list[str]]:
    clips: list[RecordingClip] = []
    warnings: list[str] = []
    cache_index: dict[str, dict[str, str]] = {}
    cache_entries: list[tuple[str, dict[str, str]]] = []
    if coverage_cache is not None:
        cache_index, cache_entries = load_recording_coverage_cache(coverage_cache)
    discovered_keys: set[str] = set()
    cache_hits = 0
    cache_misses = 0
    for timestamp_file in discover_timestamp_files(root):
        discovered_keys.update(recording_path_keys(timestamp_file))
        cache_row = None
        if cache_index:
            for key in recording_path_keys(timestamp_file):
                cache_row = cache_index.get(key)
                if cache_row is not None:
                    break
        if cache_row is not None:
            try:
                clips.append(recording_clip_from_cache_row(timestamp_file, root, cache_row))
                cache_hits += 1
                continue
            except Exception as exc:
                LOG.debug("Recording coverage cache miss for %s: %s", timestamp_file, exc)
        cache_misses += 1
        try:
            clips.append(read_timestamp_clip(timestamp_file, root))
        except Exception as exc:
            warning = f"Skipping malformed timestamp file {timestamp_file}: {exc}"
            LOG.warning(warning)
            warnings.append(warning)
    clips.sort(key=lambda clip: (clip.start_utc, clip.clip_id))
    if coverage_cache is not None:
        stale_rows = 0
        for primary_key, _row in cache_entries:
            row_keys = recording_path_keys(Path(primary_key))
            if not any(key in discovered_keys for key in row_keys):
                stale_rows += 1
        LOG.info(
            "Recording coverage cache: %d hit(s), %d miss(es), %d stale row(s)",
            cache_hits,
            cache_misses,
            stale_rows,
        )
    return clips, warnings


def clip_row(clip: RecordingClip, tz: dt.tzinfo) -> dict[str, str]:
    return {
        "clip_id": clip.clip_id,
        "timestamp_file": str(clip.timestamp_file),
        "video_file": str(clip.video_file),
        "start_utc": format_utc(clip.start_utc),
        "end_utc": format_utc(clip.end_utc),
        "start_local": format_local(clip.start_utc, tz),
        "end_local": format_local(clip.end_utc, tz),
        "duration_s": f"{clip.duration_s:.6f}",
        "frames": str(clip.frames),
        "timestamp_size_bytes": "" if clip.timestamp_size_bytes is None else str(clip.timestamp_size_bytes),
        "timestamp_mtime_ns": "" if clip.timestamp_mtime_ns is None else str(clip.timestamp_mtime_ns),
    }


def write_coverage_csv(clips: list[RecordingClip], output_path: Path, tz: dt.tzinfo) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=RECORDING_COVERAGE_FIELDNAMES)
    writer.writeheader()
    for clip in clips:
        writer.writerow(clip_row(clip, tz))
    atomic_write_text(output_path, buffer.getvalue())


def parse_local_datetime(value: str, tz: dt.tzinfo) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.strip())
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=tz)
    return parsed.astimezone(tz)


def parse_event_time_fields(
    row: dict[str, Any],
    *,
    local_field: str,
    utc_field: str,
    tz: dt.tzinfo,
) -> Optional[dt.datetime]:
    utc_text = str(row.get(utc_field) or "").strip()
    if utc_text:
        return parse_utc_value(utc_text).astimezone(tz)
    local_text = str(row.get(local_field) or "").strip()
    if local_text:
        return parse_local_datetime(local_text, tz)
    return None


def parse_boolish(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def is_global_event_alias(animal_id: str) -> bool:
    return animal_id.strip().lower() in GLOBAL_EVENT_ALIASES


def load_event_tables_from_text(
    text: str,
    tz: dt.tzinfo,
    *,
    source_name: str,
) -> tuple[list[BehaviorEvent], list[GlobalEvent]]:
    events: list[BehaviorEvent] = []
    global_events: list[GlobalEvent] = []
    with io.StringIO(text) as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=2):
            try:
                animal_id = str(row.get("animal_id") or "").strip()
                if not animal_id:
                    raise ValueError("missing animal_id")
                start_local = parse_event_time_fields(
                    row,
                    local_field="start_local",
                    utc_field="start_utc",
                    tz=tz,
                )
                end_local = parse_event_time_fields(
                    row,
                    local_field="end_local",
                    utc_field="end_utc",
                    tz=tz,
                )
                if start_local is None:
                    raise ValueError("missing start_local")
                event_name = str(row.get("event") or "").strip() or "event"
                kind_name = str(row.get("kind") or "").strip() or "event"
                notes = str(row.get("notes") or "").strip()
                approximate = parse_boolish(row.get("approximate"))
                if is_global_event_alias(animal_id):
                    if end_local is None:
                        raise ValueError("global event intervals require end_local")
                    if end_local <= start_local:
                        raise ValueError("end_local must be after start_local")
                    global_events.append(
                        GlobalEvent(
                            start_local=start_local,
                            end_local=end_local,
                            event=event_name,
                            kind=kind_name,
                            notes=notes,
                            approximate=approximate,
                        )
                    )
                    continue
                if end_local is not None:
                    if end_local <= start_local:
                        raise ValueError("end_local must be after start_local")
                    is_point = False
                else:
                    is_point = True
                    end_local = start_local + dt.timedelta(
                        seconds=POINT_EVENT_DURATION_SECONDS
                    )
                events.append(
                    BehaviorEvent(
                        animal_id=animal_id,
                        start_local=start_local,
                        end_local=end_local,
                        event=event_name,
                        kind=kind_name,
                        notes=notes,
                        is_point=is_point,
                    )
                )
            except Exception as exc:
                LOG.warning(
                    "Skipping malformed behavior event row %d in %s: %s",
                    row_index,
                    source_name,
                    exc,
                )
    return events, global_events


def load_behavior_events(path: Optional[Path], tz: dt.tzinfo) -> list[BehaviorEvent]:
    if path is None or not path.exists():
        return []
    text = path.read_text(encoding="utf-8-sig")
    events, _global_events = load_event_tables_from_text(text, tz, source_name=str(path))
    return events


def load_event_tables(
    path: Optional[Path],
    tz: dt.tzinfo,
) -> tuple[list[BehaviorEvent], list[GlobalEvent]]:
    if path is None or not path.exists():
        return [], []
    text = path.read_text(encoding="utf-8-sig")
    return load_event_tables_from_text(text, tz, source_name=str(path))


def load_event_tables_source(
    source: Optional[str],
    *,
    root: Path,
    tz: dt.tzinfo,
) -> tuple[list[BehaviorEvent], list[GlobalEvent]]:
    if source is None:
        LOG.info("Loaded 0 per-animal behavior event(s)")
        LOG.info("Loaded 0 global event interval(s)")
        return [], []

    parsed = urllib_parse.urlparse(source)
    if parsed.scheme in {"http", "https"}:
        google_source = parse_google_sheet_url(source)
        LOG.info("Fetching behavior events from Google Sheet gid=%s", google_source.gid)
        csv_bytes = fetch_google_sheet_csv(google_source)
        text = csv_bytes.decode("utf-8-sig")
        events, global_events = load_event_tables_from_text(
            text,
            tz,
            source_name=google_source.source_url,
        )
        write_google_sheet_event_snapshot(
            root=root,
            source=google_source,
            csv_bytes=csv_bytes,
            behavior_events=events,
            global_events=global_events,
        )
        LOG.info("Loaded %d per-animal behavior event(s)", len(events))
        LOG.info("Loaded %d global event interval(s)", len(global_events))
        return events, global_events

    if "://" in source:
        raise ValueError(
            "Only local CSV paths and supported Google Sheets URLs from docs.google.com are accepted for --events"
        )

    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"behavior event CSV does not exist: {path}")
    events, global_events = load_event_tables(path, tz)
    LOG.info("Loaded %d per-animal behavior event(s)", len(events))
    LOG.info("Loaded %d global event interval(s)", len(global_events))
    return events, global_events


def resolve_behavior_event_source(
    root: Path,
    explicit_source: Optional[str],
) -> tuple[Optional[str], str]:
    if explicit_source:
        return explicit_source, "explicit"

    metadata_path = root / "behavior_events_source.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("source_type") == "google_sheet":
                source_url = str(metadata.get("source_url") or "").strip()
                if source_url:
                    parse_google_sheet_url(source_url)
                    return source_url, "saved_google_sheet_source"
        except Exception as exc:
            LOG.warning("Ignoring malformed behavior event source metadata %s: %s", metadata_path, exc)

    animal_event_log = root / "animal_event_log.csv"
    if animal_event_log.exists():
        return str(animal_event_log.resolve()), "animal_event_log.csv"

    behavior_events = root / "behavior_events.csv"
    if behavior_events.exists():
        return str(behavior_events.resolve()), "behavior_events.csv"

    return None, "none"


def load_behavior_events_source(
    source: Optional[str],
    *,
    root: Path,
    tz: dt.tzinfo,
) -> list[BehaviorEvent]:
    events, _global_events = load_event_tables_source(source, root=root, tz=tz)
    return events


def load_motion_states(path: Optional[Path]) -> list[MotionState]:
    if path is None or not path.exists():
        return []

    states: list[MotionState] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=2):
            try:
                animal_id = str(row.get("animal_id") or "").strip()
                if not animal_id:
                    raise ValueError("missing animal_id")
                state_name = str(row.get("state") or "").strip().lower()
                if state_name not in {"mobile", "immobile"}:
                    raise ValueError(f"invalid state {state_name!r}")
                start_utc = parse_utc_value(row.get("start_utc"))
                end_utc = parse_utc_value(row.get("end_utc"))
                if end_utc <= start_utc:
                    raise ValueError("end_utc must be after start_utc")
                states.append(
                    MotionState(
                        animal_id=animal_id,
                        clip_key=str(row.get("clip_key") or "").strip(),
                        start_utc=start_utc,
                        end_utc=end_utc,
                        state=state_name,
                        threshold=float(str(row.get("threshold") or "").strip()),
                        threshold_source=str(row.get("threshold_source") or "").strip() or "manual",
                        mean_motion_energy=float(str(row.get("mean_motion_energy") or "").strip()),
                        peak_motion_energy=float(str(row.get("peak_motion_energy") or "").strip()),
                        n_windows=int(str(row.get("n_windows") or "").strip()),
                    )
                )
            except Exception as exc:
                LOG.warning("Skipping malformed motion state row %d in %s: %s", row_index, path, exc)
    return states


def infer_animals(events: Iterable[BehaviorEvent]) -> list[str]:
    animals: list[str] = []
    seen: set[str] = set()
    for event in events:
        if event.animal_id not in seen:
            seen.add(event.animal_id)
            animals.append(event.animal_id)
    return animals


def find_major_recording_gaps(
    clips: list[RecordingClip],
    *,
    min_gap_seconds: int = MAJOR_GAP_MIN_SECONDS,
) -> list[tuple[dt.datetime, dt.datetime]]:
    gaps: list[tuple[dt.datetime, dt.datetime]] = []
    if len(clips) < 2:
        return gaps
    for previous_clip, current_clip in zip(clips, clips[1:]):
        if current_clip.start_utc <= previous_clip.end_utc:
            continue
        gap_s = (current_clip.start_utc - previous_clip.end_utc).total_seconds()
        if gap_s >= min_gap_seconds:
            gaps.append((previous_clip.end_utc, current_clip.start_utc))
    return gaps


def format_gap_duration(duration_s: float) -> str:
    duration_min = duration_s / 60.0
    if duration_min >= 120:
        return f"{duration_s / 3600.0:.1f} h\ngap"
    return f"{duration_min:.0f} min\ngap"


def major_tick_interval_hours(start_local: dt.datetime, end_local: dt.datetime) -> int:
    span_h = max((end_local - start_local).total_seconds(), 0.0) / 3600.0
    if span_h <= 48:
        return 6
    if span_h <= 120:
        return 12
    return 24


def configure_time_axis(axis, start_local: dt.datetime, end_local: dt.datetime) -> None:
    span_h = max((end_local - start_local).total_seconds(), 0.0) / 3600.0
    if span_h > 48:
        axis.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axis.xaxis.set_minor_locator(mdates.HourLocator(interval=12))
        axis.xaxis.set_minor_formatter(NullFormatter())
    else:
        interval_h = 6 if span_h <= 24 else 12
        axis.xaxis.set_major_locator(mdates.HourLocator(interval=interval_h))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.xaxis.set_minor_formatter(NullFormatter())
    axis.tick_params(axis="x", labelrotation=0)
    for label in axis.get_xticklabels():
        label.set_ha("center")


def unique_timezone_abbreviations(
    tz: dt.tzinfo,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
) -> set[str]:
    abbreviations = {
        label
        for label in {
            start_utc.astimezone(tz).tzname(),
            end_utc.astimezone(tz).tzname(),
        }
        if label
    }
    return abbreviations


def subtitle_time_label(
    tz: dt.tzinfo,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
) -> str:
    abbreviations = sorted(unique_timezone_abbreviations(tz, start_utc, end_utc))
    if timezone_label(tz) == DEFAULT_TIMEZONE:
        base_label = "Woods Hole local time"
    else:
        base_label = "Local time"
    if len(abbreviations) == 1:
        return f"{base_label} ({abbreviations[0]})"
    return base_label


def x_axis_time_label(
    tz: dt.tzinfo,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
) -> str:
    abbreviations = sorted(unique_timezone_abbreviations(tz, start_utc, end_utc))
    label = f"Local time - {timezone_label(tz)}"
    if len(abbreviations) == 1:
        return f"{label} ({abbreviations[0]})"
    return label


def _bar_left_width(start_utc: dt.datetime, end_utc: dt.datetime, tz: dt.tzinfo) -> tuple[float, float]:
    left = mdates.date2num(to_plot_local(start_utc, tz))
    width = max((end_utc - start_utc).total_seconds() / 86400.0, 1.0 / 86400.0 / 24.0)
    return left, width


def _event_duration_s(event: BehaviorEvent) -> float:
    return max((event.end_local - event.start_local).total_seconds(), 0.0)


def normalize_event_name(event_name: str) -> str:
    return event_name.strip().lower().replace(" ", "_").replace("-", "_")


def behavior_event_terms(event: BehaviorEvent) -> set[str]:
    return normalized_event_terms(event.event, event.kind)


def global_event_terms(event: GlobalEvent) -> set[str]:
    return normalized_event_terms(event.event, event.kind)


def behavior_event_style(event: BehaviorEvent) -> Optional[tuple[str, float, str]]:
    terms = behavior_event_terms(event)
    if terms & SHED_EVENT_NAMES:
        return "^", SHED_MARKER_SIZE, SHED_COLOR
    if terms & J_HANG_EVENT_NAMES:
        return J_HANG_MARKER, J_HANG_MARKER_SIZE, J_HANG_COLOR
    if terms & PUPATION_EVENT_NAMES:
        return PUPATION_MARKER, PUPATION_MARKER_SIZE, PUPATION_COLOR
    if terms & STIM_EVENT_NAMES:
        return "*", STIM_MARKER_SIZE, STIM_COLOR
    if terms & DEATH_EVENT_NAMES:
        return "X", DEATH_MARKER_SIZE, DEATH_COLOR
    return None


def is_supported_behavior_point_event(event: BehaviorEvent) -> bool:
    return event.is_point and behavior_event_style(event) is not None


def is_stimulation_event(event: BehaviorEvent) -> bool:
    return bool(behavior_event_terms(event) & STIM_EVENT_NAMES)


def is_food_unavailable_behavior_event(event: BehaviorEvent) -> bool:
    return bool(behavior_event_terms(event) & FOOD_UNAVAILABLE_EVENT_NAMES)


def is_low_mobility_event(event: BehaviorEvent) -> bool:
    return bool(behavior_event_terms(event) & LOW_MOBILITY_EVENT_NAMES)


def behavior_event_is_visible(event: BehaviorEvent) -> bool:
    if is_low_mobility_event(event):
        return False
    return not event.is_point or is_supported_behavior_point_event(event)


def behavior_event_display_span(event: BehaviorEvent) -> Optional[tuple[dt.datetime, dt.datetime]]:
    if not behavior_event_is_visible(event):
        return None
    start = event.start_local.replace(tzinfo=None)
    if event.is_point:
        return start, start
    return start, event.end_local.replace(tzinfo=None)


def is_video_quality_global_event(event: GlobalEvent) -> bool:
    terms = global_event_terms(event)
    return bool(terms & (VIDEO_QUALITY_GLOBAL_EVENT_NAMES | {"video_quality"}))


def is_food_unavailable_global_event(event: GlobalEvent) -> bool:
    terms = global_event_terms(event)
    return bool(terms & FOOD_UNAVAILABLE_EVENT_NAMES)


def rendered_global_event_style(event: GlobalEvent) -> Optional[tuple[str, float]]:
    if is_video_quality_global_event(event):
        return VIDEO_QUALITY_LOW_COLOR, VIDEO_QUALITY_LOW_ALPHA
    if is_food_unavailable_global_event(event):
        return FOOD_UNAVAILABLE_COLOR, FOOD_UNAVAILABLE_ALPHA
    return None


def global_event_band_style(event: GlobalEvent) -> Optional[tuple[str, float]]:
    return rendered_global_event_style(event)


def global_event_label(event: GlobalEvent) -> Optional[str]:
    if not is_video_quality_global_event(event):
        return None
    terms = global_event_terms(event)
    if terms & {"bad_lighting", "poor_lighting"}:
        return "Poor lighting"
    return "Low video quality"


def global_annotation_legend_handles(global_events: list[GlobalEvent]) -> list[Patch]:
    handles: list[Patch] = []
    if any(is_video_quality_global_event(event) for event in global_events):
        handles.append(
            Patch(
                facecolor=VIDEO_QUALITY_LOW_COLOR,
                alpha=VIDEO_QUALITY_LOW_ALPHA,
                edgecolor="none",
                label="Low video quality",
            )
        )
    return handles


def add_recording_gap_band(axis, gap_start: dt.datetime, gap_end: dt.datetime, timezone: dt.tzinfo) -> None:
    start_local = to_plot_local(gap_start, timezone)
    end_local = to_plot_local(gap_end, timezone)
    axis.axvspan(start_local, end_local, facecolor=GAP_SHADE_COLOR, alpha=1.0, zorder=0.05)
    axis.axvline(start_local, color=GAP_BOUNDARY_COLOR, alpha=0.9, linewidth=0.6, zorder=0.06)
    axis.axvline(end_local, color=GAP_BOUNDARY_COLOR, alpha=0.9, linewidth=0.6, zorder=0.06)


def timeline_bounds_utc(
    clips: list[RecordingClip],
    events: list[BehaviorEvent],
    motion_states: list[MotionState],
    global_events: list[GlobalEvent],
    timezone: dt.tzinfo,
    terminal_cutoffs_local: Optional[dict[str, dt.datetime]] = None,
) -> Optional[tuple[dt.datetime, dt.datetime]]:
    starts_utc: list[dt.datetime] = []
    ends_utc: list[dt.datetime] = []
    for clip in clips:
        starts_utc.append(clip.start_utc)
        ends_utc.append(clip.end_utc)
    for event in events:
        if not behavior_event_is_visible(event):
            continue
        terminal_cutoff_local = terminal_cutoffs_local.get(event.animal_id) if terminal_cutoffs_local else None
        if terminal_cutoff_local is not None:
            terms = behavior_event_terms(event)
            is_terminal_activity = bool(terms & (J_HANG_EVENT_NAMES | PUPATION_EVENT_NAMES | DEATH_EVENT_NAMES))
            if is_terminal_activity:
                if event.start_local != terminal_cutoff_local:
                    continue
            elif event.start_local >= terminal_cutoff_local:
                continue
        starts_utc.append(event.start_local.astimezone(UTC))
        if event.is_point:
            ends_utc.append(event.start_local.astimezone(UTC))
        else:
            ends_utc.append(event.end_local.astimezone(UTC))
    for state in motion_states:
        terminal_cutoff_local = terminal_cutoffs_local.get(state.animal_id) if terminal_cutoffs_local else None
        if terminal_cutoff_local is not None:
            if terminal_cutoff_local.tzinfo is None:
                terminal_cutoff_local = terminal_cutoff_local.replace(tzinfo=timezone)
            terminal_cutoff_utc = terminal_cutoff_local.astimezone(UTC)
            if state.start_utc >= terminal_cutoff_utc:
                continue
            state_end_utc = min(state.end_utc, terminal_cutoff_utc)
        else:
            state_end_utc = state.end_utc
        starts_utc.append(state.start_utc)
        ends_utc.append(state_end_utc)
    for event in global_events:
        style = rendered_global_event_style(event)
        if style is None:
            continue
        starts_utc.append(event.start_local.astimezone(UTC))
        ends_utc.append(event.end_local.astimezone(UTC))
    if not starts_utc or not ends_utc:
        return None
    return min(starts_utc), max(ends_utc)


def merge_intervals(
    intervals: Iterable[tuple[dt.datetime, dt.datetime]],
) -> list[tuple[dt.datetime, dt.datetime]]:
    normalized = sorted(
        ((start, end) for start, end in intervals if end > start),
        key=lambda pair: (pair[0], pair[1]),
    )
    if not normalized:
        return []

    merged: list[tuple[dt.datetime, dt.datetime]] = []
    current_start, current_end = normalized[0]
    for start, end in normalized[1:]:
        if start < current_end:
            if end > current_end:
                current_end = end
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def subtract_intervals(
    start: dt.datetime,
    end: dt.datetime,
    exclusions: Iterable[tuple[dt.datetime, dt.datetime]],
) -> list[tuple[dt.datetime, dt.datetime]]:
    if end <= start:
        return []

    remaining: list[tuple[dt.datetime, dt.datetime]] = [(start, end)]
    for exclusion_start, exclusion_end in merge_intervals(exclusions):
        next_remaining: list[tuple[dt.datetime, dt.datetime]] = []
        for piece_start, piece_end in remaining:
            if exclusion_end <= piece_start or exclusion_start >= piece_end:
                next_remaining.append((piece_start, piece_end))
                continue
            if exclusion_start > piece_start:
                next_remaining.append((piece_start, exclusion_start))
            if exclusion_end < piece_end:
                next_remaining.append((exclusion_end, piece_end))
        remaining = next_remaining
        if not remaining:
            break
    return [(piece_start, piece_end) for piece_start, piece_end in remaining if piece_end > piece_start]


def video_quality_mask_intervals_utc(
    global_events: Iterable[GlobalEvent],
) -> list[tuple[dt.datetime, dt.datetime]]:
    intervals = [
        (event.start_local.astimezone(UTC), event.end_local.astimezone(UTC))
        for event in global_events
        if is_video_quality_global_event(event)
    ]
    return merge_intervals(intervals)


def get_point_event_style(event_name: str) -> tuple[str, float, str]:
    normalized = normalize_event_name(event_name)

    if normalized in SHED_EVENT_NAMES:
        return "^", SHED_MARKER_SIZE, SHED_COLOR

    if normalized in J_HANG_EVENT_NAMES:
        return J_HANG_MARKER, J_HANG_MARKER_SIZE, J_HANG_COLOR

    if normalized in PUPATION_EVENT_NAMES:
        return PUPATION_MARKER, PUPATION_MARKER_SIZE, PUPATION_COLOR

    if normalized in STIM_EVENT_NAMES:
        return "*", STIM_MARKER_SIZE, STIM_COLOR

    if normalized in DEATH_EVENT_NAMES:
        return "X", DEATH_MARKER_SIZE, DEATH_COLOR

    return "o", DEFAULT_POINT_MARKER_SIZE, DEFAULT_POINT_COLOR


def is_stimulation_event_name(event_name: str) -> bool:
    return normalize_event_name(event_name) in STIM_EVENT_NAMES


def event_bar_color(event: BehaviorEvent) -> str:
    if is_feeding_event(event):
        return FEEDING_COLOR
    if is_food_unavailable_behavior_event(event):
        return FOOD_UNAVAILABLE_COLOR
    style = behavior_event_style(event)
    if style is not None:
        return style[2]
    return INTERVAL_BAR_COLOR


def is_death_event(event: BehaviorEvent) -> bool:
    return bool(behavior_event_terms(event) & DEATH_EVENT_NAMES)


def is_feeding_event(event: BehaviorEvent) -> bool:
    return "feeding" in behavior_event_terms(event)


def death_cutoffs_local(events: Iterable[BehaviorEvent]) -> dict[str, dt.datetime]:
    cutoffs: dict[str, dt.datetime] = {}
    for event in events:
        if not is_death_event(event):
            continue
        cutoff = cutoffs.get(event.animal_id)
        if cutoff is None or event.start_local < cutoff:
            cutoffs[event.animal_id] = event.start_local
    return cutoffs


def terminal_activity_cutoff_by_animal(events: Iterable[BehaviorEvent]) -> dict[str, dt.datetime]:
    cutoffs: dict[str, dt.datetime] = {}
    for event in events:
        terms = behavior_event_terms(event)
        if not (terms & (J_HANG_EVENT_NAMES | PUPATION_EVENT_NAMES | DEATH_EVENT_NAMES)):
            continue
        cutoff = cutoffs.get(event.animal_id)
        if cutoff is None or event.start_local < cutoff:
            cutoffs[event.animal_id] = event.start_local
    return cutoffs


def load_motion_energy_samples(path: Optional[Path]) -> list[MotionEnergySample]:
    if path is None or not path.exists():
        return []
    samples: list[MotionEnergySample] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=2):
            try:
                animal_id = str(row.get("animal_id") or "").strip()
                if animal_id not in ANIMAL_ORDER:
                    raise ValueError(f"unknown animal_id {animal_id!r}")
                samples.append(
                    MotionEnergySample(
                        animal_id=animal_id,
                        clip_key=str(row.get("clip_key") or "").strip(),
                        timestamp_utc=parse_utc_value(row.get("timestamp_utc") or row.get("end_utc")),
                        motion_energy=float(str(row.get("motion_energy") or "").strip()),
                    )
                )
            except Exception as exc:
                LOG.warning("Skipping malformed motion-energy row %d in %s: %s", row_index, path, exc)
    return samples


def aggregate_motion_energy_samples(
    samples: Sequence[MotionEnergySample],
    *,
    bin_minutes: int,
    stat: str,
) -> dict[str, list[tuple[dt.datetime, float, str]]]:
    if bin_minutes <= 0:
        raise ValueError("bin_minutes must be > 0")
    groups: dict[str, dict[tuple[dt.datetime, str], list[float]]] = {animal_id: {} for animal_id in ANIMAL_ORDER}
    bin_delta = dt.timedelta(minutes=bin_minutes)
    for sample in samples:
        seconds = int(sample.timestamp_utc.timestamp())
        bin_seconds = bin_minutes * 60
        bucket = dt.datetime.fromtimestamp((seconds // bin_seconds) * bin_seconds, tz=UTC)
        groups[sample.animal_id].setdefault((bucket, sample.clip_key), []).append(sample.motion_energy)

    def reduce_values(values: list[float]) -> float:
        if stat == "median":
            return float(np.median(values))
        if stat == "mean":
            return float(np.mean(values))
        if stat == "p90":
            return float(np.percentile(values, 90))
        if stat == "max":
            return float(np.max(values))
        raise ValueError(f"unsupported motion plot stat: {stat}")

    aggregated: dict[str, list[tuple[dt.datetime, float, str]]] = {}
    for animal_id in ANIMAL_ORDER:
        rows: list[tuple[dt.datetime, float, str]] = []
        for (bucket, clip_key), values in sorted(groups[animal_id].items()):
            rows.append((bucket + bin_delta, reduce_values(values), clip_key))
        aggregated[animal_id] = rows
    return aggregated


def motion_legend_handles() -> list[Patch]:
    return [
        Patch(
            facecolor=MOTION_IMMOBILE_COLOR,
            alpha=MOTION_IMMOBILE_ALPHA,
            edgecolor="none",
            label="Motion-derived immobile",
        ),
        Patch(
            facecolor=MOTION_MOBILE_COLOR,
            alpha=MOTION_MOBILE_ALPHA,
            edgecolor="none",
            label="Motion-derived mobile",
        ),
    ]


def manual_interval_legend_handle() -> Patch:
    return Patch(facecolor=INTERVAL_BAR_COLOR, alpha=0.72, edgecolor="none", label="Manual interval")


def manual_annotation_legend_handles(events: Sequence[BehaviorEvent] | None = None) -> list[Line2D | Patch]:
    if events is None:
        return [
            manual_interval_legend_handle(),
            Line2D(
                [0],
                [0],
                marker="^",
                linestyle="None",
                markersize=8,
                markerfacecolor=SHED_COLOR,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label="Shed / molt",
            ),
            Line2D(
                [0],
                [0],
                marker=J_HANG_MARKER,
                linestyle="None",
                markersize=8,
                markerfacecolor=J_HANG_COLOR,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label="J-hang",
            ),
            Line2D(
                [0],
                [0],
                marker=PUPATION_MARKER,
                linestyle="None",
                markersize=7.5,
                markerfacecolor=PUPATION_COLOR,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label="Pupation",
            ),
            Line2D(
                [0],
                [0],
                linestyle="None",
                label="\u26a1 Electrical stimulation",
                color=STIM_COLOR,
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                linestyle="None",
                markersize=8,
                markerfacecolor=DEATH_COLOR,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label="Death",
            ),
        ]

    handles: list[Line2D | Patch] = []
    if any(is_manual_interval_event(event) for event in events):
        handles.append(manual_interval_legend_handle())
    if any(behavior_event_terms(event) & SHED_EVENT_NAMES for event in events):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="^",
                linestyle="None",
                markersize=8,
                markerfacecolor=SHED_COLOR,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label="Shed / molt",
            )
        )
    if any(behavior_event_terms(event) & J_HANG_EVENT_NAMES for event in events):
        handles.append(
            Line2D(
                [0],
                [0],
                marker=J_HANG_MARKER,
                linestyle="None",
                markersize=8,
                markerfacecolor=J_HANG_COLOR,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label="J-hang",
            )
        )
    if any(behavior_event_terms(event) & PUPATION_EVENT_NAMES for event in events):
        handles.append(
            Line2D(
                [0],
                [0],
                marker=PUPATION_MARKER,
                linestyle="None",
                markersize=7.5,
                markerfacecolor=PUPATION_COLOR,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label="Pupation",
            )
        )
    if any(is_stimulation_event(event) for event in events):
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                label="\u26a1 Electrical stimulation",
                color=STIM_COLOR,
            )
        )
    if any(is_death_event(event) for event in events):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="X",
                linestyle="None",
                markersize=8,
                markerfacecolor=DEATH_COLOR,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label="Death",
            )
        )
    return handles


def food_unavailable_legend_handle(
    events: Sequence[BehaviorEvent],
    global_events: Sequence[GlobalEvent],
) -> Optional[Patch]:
    has_behavior_food_unavailable = any(is_food_unavailable_behavior_event(event) for event in events)
    has_global_food_unavailable = any(is_food_unavailable_global_event(event) for event in global_events)
    if not has_behavior_food_unavailable and not has_global_food_unavailable:
        return None
    alpha = 0.74 if has_behavior_food_unavailable else FOOD_UNAVAILABLE_ALPHA
    return Patch(
        facecolor=FOOD_UNAVAILABLE_COLOR,
        alpha=alpha,
        edgecolor="none",
        label="Food unavailable",
    )


def feeding_legend_handles(events: Sequence[BehaviorEvent]) -> list[Patch]:
    if not any(is_feeding_event(event) for event in events):
        return []
    return [
        Patch(
            facecolor=FEEDING_COLOR,
            alpha=0.88,
            edgecolor="none",
            label="Automatic feeding bouts",
        )
    ]


def is_manual_interval_event(event: BehaviorEvent) -> bool:
    if event.is_point:
        return False
    if (
        is_feeding_event(event)
        or is_stimulation_event(event)
        or is_food_unavailable_behavior_event(event)
        or is_low_mobility_event(event)
    ):
        return False
    return True


def display_time_bounds(
    plot_start: dt.datetime,
    plot_end: dt.datetime,
) -> tuple[dt.datetime, dt.datetime]:
    plot_span = plot_end - plot_start
    display_pad = max(dt.timedelta(minutes=10), plot_span * 0.003)
    return plot_start - display_pad, plot_end + display_pad


def timeline_legend_handles(
    *,
    events: Sequence[BehaviorEvent],
    motion_states: Sequence[MotionState],
    global_events: Sequence[GlobalEvent],
) -> list[Line2D | Patch]:
    handle_map = timeline_legend_handle_map(
        events=events,
        motion_states=motion_states,
        global_events=global_events,
    )
    handles: list[Line2D | Patch] = []
    for label in TIMELINE_LEGEND_ORDER:
        handle = handle_map.get(label)
        if handle is not None:
            handles.append(handle)
    return handles


def timeline_legend_handle_map(
    *,
    events: Sequence[BehaviorEvent],
    motion_states: Sequence[MotionState],
    global_events: Sequence[GlobalEvent],
) -> dict[str, Line2D | Patch]:
    handles: dict[str, Line2D | Patch] = {}
    if motion_states:
        for handle in motion_legend_handles():
            handles[handle.get_label()] = handle
    for handle in feeding_legend_handles(events):
        handles[handle.get_label()] = handle
    for handle in global_annotation_legend_handles(list(global_events)):
        handles[handle.get_label()] = handle
    food_unavailable_handle = food_unavailable_legend_handle(events, global_events)
    if food_unavailable_handle is not None:
        handles[food_unavailable_handle.get_label()] = food_unavailable_handle
    for handle in manual_annotation_legend_handles(events):
        handles[handle.get_label()] = handle
    return handles


def draw_timeline_legend(axis, legend_handle_map: dict[str, Line2D | Patch]) -> Optional[object]:
    handles: list[Line2D | Patch] = []
    labels: list[str] = []
    for label in TIMELINE_LEGEND_ORDER:
        handle = legend_handle_map.get(label)
        if handle is None:
            continue
        handles.append(handle)
        labels.append(label)

    axis.axis("off")
    if not handles:
        return None

    return axis.legend(
        handles=handles,
        labels=labels,
        loc="center left",
        bbox_to_anchor=(0.0, 0.52),
        frameon=False,
        fontsize=8.0,
        handlelength=1.25,
        handletextpad=0.45,
        columnspacing=1.05,
        borderaxespad=0.0,
        labelspacing=0.4,
        ncol=len(handles),
    )


def alternating_row_backgrounds(axis, row_count: int) -> None:
    for row_index in range(row_count):
        if row_index % 2 == 0:
            axis.axhspan(
                row_index - 0.5,
                row_index + 0.5,
                facecolor=ROW_BACKGROUND_COLOR,
                alpha=1.0,
                zorder=0.02,
            )


def plot_recording_timeline(
    clips: list[RecordingClip],
    events: list[BehaviorEvent],
    motion_states: list[MotionState],
    animals: list[str],
    *,
    global_events: Optional[list[GlobalEvent]] = None,
    motion_energy_samples: Optional[list[MotionEnergySample]] = None,
    motion_plot_bin_minutes: int = 1,
    motion_plot_stat: str = "p90",
    timezone: dt.tzinfo,
    output_path: Path,
    annotate_clips: bool,
    profile_timing: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    global_events = list(global_events or [])
    motion_energy_samples = list(motion_energy_samples or [])
    clip_annotations = annotate_clips and len(clips) <= MAX_CLIP_ANNOTATIONS
    if annotate_clips and not clip_annotations:
        LOG.info("Too many clips for per-clip annotations; disabling clip labels")

    timing_enabled = profile_timing
    timing_start = perf_counter() if timing_enabled else 0.0
    timing_sections: list[tuple[str, float]] = []

    def record_timing(label: str, started_at: float) -> None:
        if timing_enabled:
            timing_sections.append((label, perf_counter() - started_at))

    prep_start = perf_counter() if timing_enabled else 0.0
    terminal_cutoffs_local = terminal_activity_cutoff_by_animal(events)
    terminal_cutoffs_utc: dict[str, dt.datetime] = {}
    for animal_id, cutoff_local in terminal_cutoffs_local.items():
        if cutoff_local.tzinfo is None:
            cutoff_local = cutoff_local.replace(tzinfo=timezone)
        terminal_cutoffs_utc[animal_id] = cutoff_local.astimezone(UTC)

    behavior_rows = list(ANIMAL_ORDER)
    behavior_index = {animal: index for index, animal in enumerate(behavior_rows)}

    motion_energy_samples_for_plot = [
        sample
        for sample in motion_energy_samples
        if sample.animal_id in behavior_index
        and (
            sample.animal_id not in terminal_cutoffs_utc
            or sample.timestamp_utc < terminal_cutoffs_utc[sample.animal_id]
        )
    ]

    bounds = timeline_bounds_utc(
        clips,
        events,
        motion_states,
        global_events,
        timezone,
        terminal_cutoffs_local,
    )
    if clips:
        first_utc = min(clip.start_utc for clip in clips)
        last_utc = max(clip.end_utc for clip in clips)
        recorded_duration_s = sum(clip.duration_s for clip in clips)
        elapsed_duration_s = max((last_utc - first_utc).total_seconds(), 0.0)
        recorded_fraction = recorded_duration_s / elapsed_duration_s if elapsed_duration_s > 0 else 1.0
    elif bounds is not None:
        first_utc, last_utc = bounds
        recorded_duration_s = 0.0
        elapsed_duration_s = max((last_utc - first_utc).total_seconds(), 0.0)
        recorded_fraction = 0.0
    else:
        first_utc = last_utc = dt.datetime.now(UTC)
        recorded_duration_s = 0.0
        elapsed_duration_s = 0.0
        recorded_fraction = 0.0
    unknown_animals = sorted(
        {
            *(event.animal_id for event in events if event.animal_id not in behavior_index),
            *(state.animal_id for state in motion_states if state.animal_id not in behavior_index),
        }
    )
    for animal_id in unknown_animals:
        LOG.warning("Skipping timeline row for unknown animal ID: %s", animal_id)

    has_motion_plot = bool(motion_energy_samples_for_plot)
    fig = plt.figure(figsize=(17, 10 if has_motion_plot else 9), constrained_layout=False)
    if has_motion_plot:
        gs = fig.add_gridspec(4, 1, height_ratios=[0.86, 1.92, 0.42, 6.1], hspace=0.08)
    else:
        gs = fig.add_gridspec(3, 1, height_ratios=[0.86, 0.42, 6.1], hspace=0.08)
    ax_cov = fig.add_subplot(gs[0])
    ax_motion = fig.add_subplot(gs[1], sharex=ax_cov) if has_motion_plot else None
    ax_legend = fig.add_subplot(gs[2] if has_motion_plot else gs[1])
    ax_beh = fig.add_subplot(gs[3] if has_motion_plot else gs[2], sharex=ax_cov)
    fig.subplots_adjust(top=0.87, bottom=0.11, left=0.08, right=0.985)
    fig.patch.set_facecolor("white")
    ax_cov.set_facecolor("white")
    if ax_motion is not None:
        ax_motion.set_facecolor("white")
    ax_legend.set_facecolor("white")
    ax_beh.set_facecolor("white")
    ax_legend.axis("off")

    total_recorded_h = recorded_duration_s / 3600.0
    elapsed_h = elapsed_duration_s / 3600.0
    subtitle = f"Elapsed {elapsed_h:.1f} h | recorded {total_recorded_h:.1f} h | coverage {recorded_fraction:.1%}"
    fig.suptitle(
        "Continuous behavioral monitoring of monarch caterpillars",
        x=0.08,
        y=0.975,
        ha="left",
        fontsize=14.5,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.08,
        0.942,
        subtitle,
        ha="left",
        va="top",
        fontsize=10,
        color="#475569",
    )

    plot_starts: list[dt.datetime] = []
    plot_ends: list[dt.datetime] = []
    if clips:
        plot_starts.append(to_plot_local(first_utc, timezone))
        plot_ends.append(to_plot_local(last_utc, timezone))
    for event in events:
        if event.animal_id in behavior_index and behavior_event_is_visible(event):
            start_local = event.start_local.replace(tzinfo=None)
            end_local = start_local if event.is_point else event.end_local.replace(tzinfo=None)
            plot_starts.append(start_local)
            plot_ends.append(end_local)
    for state in motion_states:
        if state.animal_id in behavior_index:
            plot_starts.append(to_plot_local(state.start_utc, timezone))
            plot_ends.append(to_plot_local(state.end_utc, timezone))
    for event in global_events:
        if rendered_global_event_style(event) is not None:
            plot_starts.append(event.start_local.replace(tzinfo=None))
            plot_ends.append(event.end_local.replace(tzinfo=None))
    for sample in motion_energy_samples_for_plot:
        if sample.animal_id in behavior_index:
            sample_local = to_plot_local(sample.timestamp_utc, timezone)
            plot_starts.append(sample_local)
            plot_ends.append(sample_local)

    x_min: Optional[dt.datetime] = None
    x_max: Optional[dt.datetime] = None
    if plot_starts and plot_ends:
        x_min = min(plot_starts)
        x_max = max(plot_ends)
        if x_max <= x_min:
            x_max = x_min + dt.timedelta(minutes=1)
        display_x_min, display_x_max = display_time_bounds(x_min, x_max)
        ax_cov.set_xlim(display_x_min, display_x_max)

    major_gaps = find_major_recording_gaps(clips)
    video_quality_masks_utc = video_quality_mask_intervals_utc(global_events)
    death_cutoffs = death_cutoffs_local(events)
    record_timing("plot data preparation", prep_start)
    artist_start = perf_counter() if timing_enabled else 0.0
    for gap_start_utc, gap_end_utc in major_gaps:
        add_recording_gap_band(ax_cov, gap_start_utc, gap_end_utc, timezone)
        if ax_motion is not None:
            add_recording_gap_band(ax_motion, gap_start_utc, gap_end_utc, timezone)
        add_recording_gap_band(ax_beh, gap_start_utc, gap_end_utc, timezone)

    for event in global_events:
        style = global_event_band_style(event)
        if style is None:
            continue
        band_start = event.start_local.replace(tzinfo=None)
        band_end = event.end_local.replace(tzinfo=None)
        facecolor, alpha = style
        ax_cov.axvspan(band_start, band_end, facecolor=facecolor, alpha=alpha, zorder=0.35)
        ax_beh.axvspan(band_start, band_end, facecolor=facecolor, alpha=alpha, zorder=0.45)
        label = global_event_label(event)
        if (
            label
            and (event.end_local - event.start_local).total_seconds() >= MIN_GLOBAL_EVENT_LABEL_SECONDS
        ):
            ax_cov.text(
                mdates.date2num(band_start + (band_end - band_start) / 2),
                0.86,
                label,
                transform=blended_transform_factory(ax_cov.transData, ax_cov.transAxes),
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#92400e",
                zorder=4,
                clip_on=True,
            )

    if clips:
        recording_segments = [_bar_left_width(clip.start_utc, clip.end_utc, timezone) for clip in clips]
        ax_cov.broken_barh(
            recording_segments,
            (0.18, 0.64),
            facecolors=RECORDING_COLOR,
            alpha=0.9,
            zorder=2,
        )
        if clip_annotations:
            for clip, (left, width) in zip(clips, recording_segments):
                center = left + width / 2.0
                duration_min = clip.duration_s / 60.0
                ax_cov.text(
                    center,
                    0.86,
                    f"{clip.camera_label}  {duration_min:.1f}m",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#1f2937",
                    clip_on=True,
                )

        ax_cov.set_ylim(0, 1)
        ax_cov.set_yticks([])
        ax_cov.set_ylabel("Recording", rotation=0, labelpad=34, va="center", fontsize=10, color="#334155")
        ax_cov.grid(True, axis="x", color="#cbd5e1", alpha=0.18, linewidth=0.45)
    else:
        ax_cov.text(
            0.5,
            0.5,
            "No recording clips found",
            transform=ax_cov.transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
        ax_cov.set_yticks([])
        ax_cov.grid(False)

    if major_gaps and len(major_gaps) <= MAX_MAJOR_GAP_LABELS:
        for gap_start_utc, gap_end_utc in major_gaps:
            center = mdates.date2num(
                to_plot_local(gap_start_utc + (gap_end_utc - gap_start_utc) / 2, timezone)
            )
            ax_cov.text(
                center,
                0.5,
                format_gap_duration((gap_end_utc - gap_start_utc).total_seconds()),
                ha="center",
                va="center",
                fontsize=8,
                color="#475569",
                zorder=1,
            )

    ax_cov.set_title("Recording coverage", loc="left", fontsize=11, color="#0f172a", pad=8)
    ax_cov.tick_params(axis="x", labelbottom=False)

    if ax_motion is not None:
        aggregated_motion = aggregate_motion_energy_samples(
            motion_energy_samples_for_plot,
            bin_minutes=motion_plot_bin_minutes,
            stat=motion_plot_stat,
        )
        ax_motion.set_title("Motion energy", loc="left", fontsize=11, color="#0f172a", pad=8)
        ax_motion.set_yticks(range(len(behavior_rows)))
        ax_motion.set_yticklabels(behavior_rows)
        ax_motion.set_ylabel("Animal")
        ax_motion.set_ylim(-0.5, len(behavior_rows) - 0.5)
        ax_motion.invert_yaxis()
        ax_motion.grid(True, axis="x", color="#cbd5e1", alpha=0.18, linewidth=0.45)
        ax_motion.tick_params(axis="x", labelbottom=False)
        alternating_row_backgrounds(ax_motion, len(behavior_rows))
        for y in range(len(behavior_rows) + 1):
            ax_motion.axhline(y - 0.5, color=ROW_SEPARATOR_COLOR, linewidth=0.7, zorder=0.2)
        for animal_id, y in behavior_index.items():
            rows = aggregated_motion.get(animal_id, [])
            if not rows:
                continue
            values = np.array([value for _bucket, value, _clip_key in rows], dtype=np.float64)
            p5 = float(np.percentile(values, 5))
            p95 = float(np.percentile(values, 95))
            if p95 <= p5:
                normalized_values = np.full(values.shape, 0.5, dtype=np.float64)
            else:
                normalized_values = np.clip((values - p5) / (p95 - p5), 0.0, 1.0)
            bin_delta = dt.timedelta(minutes=motion_plot_bin_minutes)
            current_segment_x: list[dt.datetime] = []
            current_segment_y: list[float] = []
            previous_bucket: Optional[dt.datetime] = None
            previous_clip_key: Optional[str] = None
            tolerance = dt.timedelta(seconds=10)
            for (bucket_utc, _raw_value, clip_key), normalized in zip(rows, normalized_values):
                if (
                    previous_bucket is not None
                    and (clip_key != previous_clip_key or bucket_utc - previous_bucket > bin_delta + tolerance)
                    and current_segment_x
                ):
                    ax_motion.plot(current_segment_x, current_segment_y, color="#0f766e", linewidth=1.2, zorder=2)
                    current_segment_x = []
                    current_segment_y = []
                current_segment_x.append(to_plot_local(bucket_utc, timezone))
                current_segment_y.append(y + 0.35 - (0.7 * float(normalized)))
                previous_bucket = bucket_utc
                previous_clip_key = clip_key
            if current_segment_x:
                ax_motion.plot(current_segment_x, current_segment_y, color="#0f766e", linewidth=1.2, zorder=2)

    legend_handle_map = timeline_legend_handle_map(
        events=events,
        motion_states=motion_states,
        global_events=global_events,
    )
    draw_timeline_legend(ax_legend, legend_handle_map)

    ax_beh.set_title("Behavior annotations", loc="left", fontsize=11, color="#0f172a", pad=5)
    ax_beh.set_yticks(range(len(behavior_rows)))
    ax_beh.set_yticklabels(behavior_rows)
    ax_beh.set_ylabel("Animal")
    ax_beh.set_ylim(-0.5, len(behavior_rows) - 0.5)
    ax_beh.invert_yaxis()
    ax_beh.grid(True, axis="x", color="#cbd5e1", alpha=0.18, linewidth=0.45)
    ax_beh.set_xlabel(x_axis_time_label(timezone, first_utc, last_utc))
    alternating_row_backgrounds(ax_beh, len(behavior_rows))

    for y in range(len(behavior_rows) + 1):
        ax_beh.axhline(y - 0.5, color=ROW_SEPARATOR_COLOR, linewidth=0.7, zorder=0.35)

    if x_min is not None and x_max is not None:
        configure_time_axis(ax_beh, x_min, x_max)
    if video_quality_masks_utc:
        LOG.info(
            "Applying %d video-quality visualization mask(s) to motion-derived states",
            len(video_quality_masks_utc),
        )
    suppressed_motion_spans = 0
    clipped_motion_spans = 0
    motion_segments_by_group: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for state in motion_states:
        if state.animal_id not in behavior_index:
            continue
        terminal_cutoff_utc = terminal_cutoffs_utc.get(state.animal_id)
        state_start_utc = state.start_utc
        state_end_utc = state.end_utc
        if terminal_cutoff_utc is not None:
            if state_start_utc >= terminal_cutoff_utc:
                suppressed_motion_spans += 1
                continue
            if state_end_utc > terminal_cutoff_utc:
                state_end_utc = terminal_cutoff_utc
        death_cutoff_local = death_cutoffs.get(state.animal_id)
        if death_cutoff_local is not None:
            if death_cutoff_local.tzinfo is None:
                death_cutoff_local = death_cutoff_local.replace(tzinfo=timezone)
            death_cutoff_utc = death_cutoff_local.astimezone(UTC)
            if state_start_utc >= death_cutoff_utc:
                suppressed_motion_spans += 1
                continue
            if state_end_utc > death_cutoff_utc:
                state_end_utc = death_cutoff_utc
        visible_segments = subtract_intervals(state_start_utc, state_end_utc, video_quality_masks_utc)
        if not visible_segments:
            suppressed_motion_spans += 1
            continue
        if len(visible_segments) != 1 or visible_segments[0] != (state_start_utc, state_end_utc):
            clipped_motion_spans += 1
        for segment_start_utc, segment_end_utc in visible_segments:
            left = mdates.date2num(to_plot_local(segment_start_utc, timezone))
            width = max((segment_end_utc - segment_start_utc).total_seconds() / 86400.0, 1e-9)
            motion_segments_by_group[(state.animal_id, state.state)].append((left, width))

    for (animal_id, state_name), segments in motion_segments_by_group.items():
        y = behavior_index[animal_id]
        color = MOTION_MOBILE_COLOR if state_name == "mobile" else MOTION_IMMOBILE_COLOR
        ax_beh.broken_barh(
            segments,
            (y - 0.31, 0.62),
            facecolors=color,
            alpha=MOTION_MOBILE_ALPHA if state_name == "mobile" else MOTION_IMMOBILE_ALPHA,
            zorder=1,
        )

    if video_quality_masks_utc:
        LOG.info(
            "Suppressed %d motion state span(s) and clipped %d span(s) due to video-quality masks",
            suppressed_motion_spans,
            clipped_motion_spans,
        )

    stim_event_count = sum(1 for event in events if is_stimulation_event(event))
    LOG.info("Electrical stimulation events available: %d", stim_event_count)
    stimulation_markers: list[tuple[str, float, int, dt.datetime]] = []
    feeding_segments_by_animal: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for event in events:
        if event.animal_id not in behavior_index:
            continue
        terminal_cutoff_local = terminal_cutoffs_local.get(event.animal_id)
        if terminal_cutoff_local is not None:
            terms = behavior_event_terms(event)
            is_terminal_activity = bool(terms & (J_HANG_EVENT_NAMES | PUPATION_EVENT_NAMES | DEATH_EVENT_NAMES))
            if is_terminal_activity:
                if event.start_local != terminal_cutoff_local:
                    continue
            elif event.start_local >= terminal_cutoff_local:
                continue
        y = behavior_index[event.animal_id]
        if not behavior_event_is_visible(event):
            continue
        event_end_local = event.end_local
        if terminal_cutoff_local is not None and not (
            behavior_event_terms(event) & (J_HANG_EVENT_NAMES | PUPATION_EVENT_NAMES | DEATH_EVENT_NAMES)
        ) and event_end_local > terminal_cutoff_local:
            event_end_local = terminal_cutoff_local
        death_cutoff_local = death_cutoffs.get(event.animal_id)
        if death_cutoff_local is not None and not is_death_event(event) and event_end_local > death_cutoff_local:
            event_end_local = death_cutoff_local
        duration_s = max((event_end_local - event.start_local).total_seconds(), 0.0)
        left = mdates.date2num(event.start_local.replace(tzinfo=None))

        if is_stimulation_event(event):
            if not event.is_point and duration_s > 0:
                width = max(duration_s / 86400.0, 1e-9)
                color = event_bar_color(event)
                ax_beh.broken_barh(
                    [(left, width)],
                    (y - 0.22, 0.44),
                    facecolors=color,
                    alpha=0.74,
                    zorder=3,
                )
            stimulation_markers.append((event.animal_id, left, y, event.start_local))
            continue

        if is_feeding_event(event) and not event.is_point:
            if duration_s <= 0:
                continue
            width = max(duration_s / 86400.0, 1e-9)
            feeding_segments_by_animal[event.animal_id].append((left, width))
            continue

        if event.is_point:
            style = behavior_event_style(event)
            if style is None:
                continue
            marker, marker_size, point_color = style
            if marker == "*":
                ax_beh.text(
                    left,
                    y,
                    "\u26a1",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color=point_color,
                    fontweight="bold",
                    zorder=6,
                )
            else:
                ax_beh.scatter(
                    left,
                    y,
                    s=marker_size,
                    marker=marker,
                    color=point_color,
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=6,
                )
            continue

        if duration_s <= 0:
            continue
        width = max(duration_s / 86400.0, 1e-9)
        color = event_bar_color(event)
        ax_beh.broken_barh(
            [(left, width)],
            (y - 0.22, 0.44),
            facecolors=color,
            alpha=0.74,
            zorder=3,
        )

    for animal_id, segments in feeding_segments_by_animal.items():
        y = behavior_index[animal_id]
        ax_beh.broken_barh(
            segments,
            (y - 0.22, 0.44),
            facecolors=FEEDING_COLOR,
            alpha=0.74,
            zorder=3,
        )

    rendered_stimulation_markers = 0
    for animal_id, left, y, event_start_local in stimulation_markers:
        LOG.debug("stimulation %s at %s", animal_id, event_start_local.isoformat(sep=" "))
        ax_beh.text(
            left,
            y,
            "\u26a1",
            ha="center",
            va="center",
            fontsize=15,
            color=STIM_COLOR,
            fontweight="bold",
            zorder=STIM_MARKER_ZORDER,
            clip_on=False,
            path_effects=[patheffects.withStroke(linewidth=2.2, foreground="white")],
        )
        rendered_stimulation_markers += 1

    LOG.info("Electrical stimulation markers rendered: %d", rendered_stimulation_markers)
    if stim_event_count > 0 and rendered_stimulation_markers == 0:
        LOG.warning(
            "Loaded %d electrical stimulation event(s) but rendered 0 marker(s)",
            stim_event_count,
        )

    axes_to_style = [ax_cov, ax_beh]
    if ax_motion is not None:
        axes_to_style.append(ax_motion)
    for ax in axes_to_style:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(axis="both", colors="#334155")

    for label in ax_beh.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")
    if timing_enabled:
        record_timing("Matplotlib artist creation", artist_start)
        savefig_start = perf_counter()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if timing_enabled:
        record_timing("savefig", savefig_start)
        record_timing("plot total", timing_start)
        for label, elapsed in timing_sections:
            LOG.info("Timing: %s: %.1f s", label, elapsed)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot recording coverage and behavior annotations from timestamp sidecars.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Dataset root to scan recursively for *.timestamps.csv(.gz) files.",
    )
    parser.add_argument(
        "--animals",
        nargs="*",
        help="Animal rows to show in the behavior panel, for example: --animals C01 C02 ...",
    )
    parser.add_argument(
        "--events",
        help=(
            "Behavior event source: either a local CSV path or a supported Google Sheets URL "
            "(defaults: explicit source, saved Google Sheet metadata, "
            "<root>/animal_event_log.csv, then <root>/behavior_events.csv)."
        ),
    )
    parser.add_argument(
        "--feeding-events",
        help="Optional local CSV of automatic feeding events to overlay without modifying manual events.",
    )
    parser.add_argument(
        "--no-feeding-events",
        action="store_true",
        help="Disable automatic feeding_events.csv loading even when the default file exists.",
    )
    parser.add_argument(
        "--motion-states",
        type=Path,
        help=(
            "Path to motion_states.csv. Defaults to "
            "<root>/cropped_by_caterpillar/motion_energy/motion_states.csv when present."
        ),
    )
    parser.add_argument(
        "--motion-energy",
        type=Path,
        help="Optional motion_energy_timeseries.csv for a quantitative motion panel.",
    )
    parser.add_argument(
        "--motion-plot-bin-minutes",
        type=int,
        default=1,
        help="Bin size in minutes for the optional motion-energy panel (default: 1).",
    )
    parser.add_argument(
        "--motion-plot-stat",
        choices=["median", "mean", "p90", "max"],
        default="p90",
        help="Aggregation statistic for the optional motion-energy panel (default: p90).",
    )
    parser.add_argument(
        "--no-motion-states",
        action="store_true",
        help="Disable automatic motion_states.csv loading.",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help=f"Timezone used for local display labels (default: {DEFAULT_TIMEZONE}).",
    )
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        help="Output CSV path for recording_coverage.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path for recording_behavior_timeline.png.",
    )
    parser.add_argument(
        "--annotate-clips",
        dest="annotate_clips",
        action="store_true",
        help="Show small clip labels on the recording coverage bars.",
    )
    parser.add_argument(
        "--no-annotate-clips",
        dest="annotate_clips",
        action="store_false",
        help="Hide per-clip labels on the recording coverage bars.",
    )
    parser.add_argument(
        "--profile-timing",
        action="store_true",
        help="Log coarse timing information for clip collection, event loading, and plotting.",
    )
    parser.set_defaults(annotate_clips=True)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        parser.error(f"root must be an existing directory: {root}")
    if args.motion_states is not None and args.no_motion_states:
        parser.error("--motion-states and --no-motion-states cannot be used together")

    local_tz = load_timezone(args.timezone)
    coverage_csv = args.coverage_csv or (root / "recording_coverage.csv")
    output_png = args.output or (root / "recording_behavior_timeline.png")
    total_start = perf_counter() if args.profile_timing else 0.0

    coverage_start = perf_counter() if args.profile_timing else 0.0
    clips, warnings = collect_recording_clips(root, coverage_cache=coverage_csv)
    if args.profile_timing:
        LOG.info(
            "Timing: recording coverage discovery/load: %.1f s",
            perf_counter() - coverage_start,
        )
    write_coverage_csv(clips, coverage_csv, local_tz)

    events_start = perf_counter() if args.profile_timing else 0.0
    events_source, events_source_reason = resolve_behavior_event_source(root, args.events)
    if events_source is not None:
        LOG.info("Behavior event source (%s): %s", events_source_reason, events_source)
    else:
        LOG.info("No behavior event source found")
    try:
        events, global_events = load_event_tables_source(
            events_source,
            root=root,
            tz=local_tz,
        )
        feeding_events_path: Optional[Path] = None
        if args.feeding_events is not None:
            feeding_events_path = Path(args.feeding_events).expanduser().resolve()
            if not feeding_events_path.exists():
                parser.error(f"feeding events file does not exist: {feeding_events_path}")
        elif not args.no_feeding_events:
            auto_feeding_events = (
                root / "cropped_by_caterpillar" / "leaf_feeding" / "feeding_events.csv"
            )
            if auto_feeding_events.exists():
                feeding_events_path = auto_feeding_events

        if feeding_events_path is not None:
            feeding_start = perf_counter() if args.profile_timing else 0.0
            feeding_events, feeding_global_events = load_event_tables(feeding_events_path, local_tz)
            if args.profile_timing:
                LOG.info("Timing: feeding events: %.1f s", perf_counter() - feeding_start)
            events.extend(feeding_events)
            global_events.extend(feeding_global_events)
            LOG.info(
                "Loaded %d automatic feeding event(s) from %s",
                len(feeding_events),
                feeding_events_path,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOG.error("%s", exc)
        return 1
    if args.profile_timing:
        LOG.info(
            "Timing: behavior/global events: %.1f s",
            perf_counter() - events_start,
        )

    motion_states_start = perf_counter() if args.profile_timing else 0.0
    motion_states_path: Optional[Path] = None
    if not args.no_motion_states:
        if args.motion_states is not None:
            motion_states_path = args.motion_states
            if not motion_states_path.exists():
                parser.error(f"motion states file does not exist: {motion_states_path}")
        else:
            auto_motion_states = root / "cropped_by_caterpillar" / "motion_energy" / "motion_states.csv"
            motion_states_path = auto_motion_states if auto_motion_states.exists() else None
    motion_states = load_motion_states(motion_states_path)
    if args.profile_timing:
        LOG.info("Timing: motion states: %.1f s", perf_counter() - motion_states_start)
    motion_energy_start = perf_counter() if args.profile_timing else 0.0
    motion_energy_path = args.motion_energy
    motion_energy_samples = load_motion_energy_samples(motion_energy_path)
    if args.profile_timing:
        LOG.info("Timing: motion energy: %.1f s", perf_counter() - motion_energy_start)

    animals = args.animals if args.animals else (infer_animals(events) or DEFAULT_ANIMALS)
    if not clips:
        LOG.warning("No recording clips were found under %s", root)

    plot_recording_timeline(
        clips,
        events,
        motion_states,
        animals,
        global_events=global_events,
        motion_energy_samples=motion_energy_samples,
        motion_plot_bin_minutes=args.motion_plot_bin_minutes,
        motion_plot_stat=args.motion_plot_stat,
        timezone=local_tz,
        output_path=output_png,
        annotate_clips=args.annotate_clips,
        profile_timing=args.profile_timing,
    )

    LOG.info("Wrote coverage CSV: %s", coverage_csv)
    LOG.info("Wrote timeline PNG: %s", output_png)
    if warnings:
        LOG.info("Skipped %d malformed timestamp file(s)", len(warnings))
    if args.profile_timing:
        LOG.info("Timing: total: %.1f s", perf_counter() - total_start)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
