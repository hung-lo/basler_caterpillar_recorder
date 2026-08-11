#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import gzip
import hashlib
import io
import json
import logging
import os
import tempfile
import sys
from pathlib import Path
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
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
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
DEATH_MARKER_SIZE = 120
MOTION_IMMOBILE_COLOR = "#bdbdbd"
MOTION_MOBILE_COLOR = "#59a14f"
RECORDING_COLOR = "#4c78a8"
GAP_SHADE_COLOR = "#efefef"
ROW_SEPARATOR_COLOR = "#e5e7eb"
MAJOR_GAP_MIN_SECONDS = 5 * 60
MAX_MAJOR_GAP_LABELS = 6
GOOGLE_SHEETS_HOST = "docs.google.com"
GOOGLE_SHEETS_TIMEOUT_SECONDS = 30
GOOGLE_SHEETS_USER_AGENT = "basler-caterpillar-recorder/1.0"
GLOBAL_EVENT_ALIASES = {"all", "global", "*"}
VIDEO_QUALITY_LOW_COLOR = "#F59E0B"
VIDEO_QUALITY_LOW_ALPHA = 0.12
GENERIC_GLOBAL_EVENT_COLOR = "#94a3b8"
GENERIC_GLOBAL_EVENT_ALPHA = 0.1
MIN_GLOBAL_EVENT_LABEL_SECONDS = 5 * 60
FOOD_UNAVAILABLE_COLOR = "#FB7185"
FOOD_UNAVAILABLE_ALPHA = 0.10

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
}
DEATH_EVENT_NAMES = {
    "death",
    "dead",
}
VIDEO_QUALITY_GLOBAL_EVENT_NAMES = {
    "video_quality_low",
    "poor_video_quality",
    "bad_video_quality",
    "bad_lighting",
    "poor_lighting",
}

SHED_COLOR = "#d97706"
STIM_COLOR = "#dc2626"
DEATH_COLOR = "#111827"
DEFAULT_POINT_COLOR = "#475569"
INTERVAL_BAR_COLOR = "#8aa1c7"
FEEDING_COLOR = "#7c3aed"


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
    timestamps: list[dt.datetime] = []
    with open_text_file(timestamp_file) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("missing CSV header")
        for row_index, row in enumerate(reader, start=2):
            try:
                timestamps.append(parse_timestamp_row(row))
            except Exception as exc:
                raise ValueError(f"row {row_index}: {exc}") from exc

    if not timestamps:
        raise ValueError("no timestamp rows")

    start_utc = min(timestamps)
    end_utc = max(timestamps)
    if end_utc < start_utc:
        raise ValueError("timestamps are out of order")

    video_file = video_file_for_timestamp_file(timestamp_file)
    if not video_file.exists():
        raise ValueError(f"missing video sidecar: {video_file.name}")

    try:
        rel_dir = timestamp_file.parent.relative_to(root)
        clip_id_prefix = rel_dir.as_posix()
    except ValueError:
        clip_id_prefix = timestamp_file.parent.name
    if clip_id_prefix == ".":
        clip_id_prefix = ""
    camera_label = strip_timestamp_suffix(timestamp_file.name)
    clip_id = f"{clip_id_prefix}/{camera_label}" if clip_id_prefix else camera_label

    return RecordingClip(
        clip_id=clip_id,
        timestamp_file=timestamp_file,
        video_file=video_file,
        camera_label=camera_label,
        start_utc=start_utc,
        end_utc=end_utc,
        duration_s=(end_utc - start_utc).total_seconds(),
        frames=len(timestamps),
    )


def collect_recording_clips(root: Path) -> tuple[list[RecordingClip], list[str]]:
    clips: list[RecordingClip] = []
    warnings: list[str] = []
    for timestamp_file in discover_timestamp_files(root):
        try:
            clips.append(read_timestamp_clip(timestamp_file, root))
        except Exception as exc:
            warning = f"Skipping malformed timestamp file {timestamp_file}: {exc}"
            LOG.warning(warning)
            warnings.append(warning)
    clips.sort(key=lambda clip: (clip.start_utc, clip.clip_id))
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
    }


def write_coverage_csv(clips: list[RecordingClip], output_path: Path, tz: dt.tzinfo) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip_id",
                "timestamp_file",
                "video_file",
                "start_utc",
                "end_utc",
                "start_local",
                "end_local",
                "duration_s",
                "frames",
            ],
        )
        writer.writeheader()
        for clip in clips:
            writer.writerow(clip_row(clip, tz))


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
    interval_h = major_tick_interval_hours(start_local, end_local)
    axis.xaxis.set_major_locator(mdates.HourLocator(interval=interval_h))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
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
    if terms & STIM_EVENT_NAMES:
        return "*", STIM_MARKER_SIZE, STIM_COLOR
    if terms & DEATH_EVENT_NAMES:
        return "X", DEATH_MARKER_SIZE, DEATH_COLOR
    return None


def is_supported_behavior_point_event(event: BehaviorEvent) -> bool:
    return event.is_point and behavior_event_style(event) is not None


def behavior_event_is_visible(event: BehaviorEvent) -> bool:
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
    return "food_unavailable" in terms


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
    if any(is_food_unavailable_global_event(event) for event in global_events):
        handles.append(
            Patch(
                facecolor=FOOD_UNAVAILABLE_COLOR,
                alpha=FOOD_UNAVAILABLE_ALPHA,
                edgecolor="none",
                label="food_unavailable",
            )
        )
    return handles


def timeline_bounds_utc(
    clips: list[RecordingClip],
    events: list[BehaviorEvent],
    motion_states: list[MotionState],
    global_events: list[GlobalEvent],
) -> Optional[tuple[dt.datetime, dt.datetime]]:
    starts_utc: list[dt.datetime] = []
    ends_utc: list[dt.datetime] = []
    for clip in clips:
        starts_utc.append(clip.start_utc)
        ends_utc.append(clip.end_utc)
    for event in events:
        if not behavior_event_is_visible(event):
            continue
        starts_utc.append(event.start_local.astimezone(UTC))
        if event.is_point:
            ends_utc.append(event.start_local.astimezone(UTC))
        else:
            ends_utc.append(event.end_local.astimezone(UTC))
    for state in motion_states:
        starts_utc.append(state.start_utc)
        ends_utc.append(state.end_utc)
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
            alpha=0.68,
            edgecolor="none",
            label="Motion-derived immobile",
        ),
        Patch(
            facecolor=MOTION_MOBILE_COLOR,
            alpha=0.68,
            edgecolor="none",
            label="Motion-derived mobile",
        ),
    ]


def manual_annotation_legend_handles() -> list[Line2D | Patch]:
    return [
        Patch(facecolor=INTERVAL_BAR_COLOR, alpha=0.72, edgecolor="none", label="Duration / state"),
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


def feeding_legend_handles(events: Sequence[BehaviorEvent]) -> list[Patch]:
    if not any(is_feeding_event(event) for event in events):
        return []
    return [
        Patch(
            facecolor=FEEDING_COLOR,
            alpha=0.82,
            edgecolor="none",
            label="Automatic feeding",
        )
    ]


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
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    global_events = list(global_events or [])
    motion_energy_samples = list(motion_energy_samples or [])
    clip_annotations = annotate_clips and len(clips) <= MAX_CLIP_ANNOTATIONS
    if annotate_clips and not clip_annotations:
        LOG.info("Too many clips for per-clip annotations; disabling clip labels")

    bounds = timeline_bounds_utc(clips, events, motion_states, global_events)
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

    behavior_rows = list(ANIMAL_ORDER)
    behavior_index = {animal: index for index, animal in enumerate(behavior_rows)}
    unknown_animals = sorted(
        {
            *(event.animal_id for event in events if event.animal_id not in behavior_index),
            *(state.animal_id for state in motion_states if state.animal_id not in behavior_index),
        }
    )
    for animal_id in unknown_animals:
        LOG.warning("Skipping timeline row for unknown animal ID: %s", animal_id)

    has_motion_plot = bool(motion_energy_samples)
    fig = plt.figure(figsize=(17, 10 if has_motion_plot else 9), constrained_layout=False)
    if has_motion_plot:
        gs = fig.add_gridspec(4, 1, height_ratios=[1.05, 2.2, 0.75, 5.5], hspace=0.06)
    else:
        gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 0.75, 5.5], hspace=0.04)
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
    subtitle = (
        f"{subtitle_time_label(timezone, first_utc, last_utc)} | "
        f"elapsed {elapsed_h:.1f} h | recorded {total_recorded_h:.1f} h | "
        f"coverage {recorded_fraction:.1%}"
    )
    fig.suptitle(
        "Monarch caterpillar recording + behavior timeline",
        x=0.08,
        y=0.975,
        ha="left",
        fontsize=15,
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
    for sample in motion_energy_samples:
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
        ax_cov.set_xlim(x_min, x_max)

    major_gaps = find_major_recording_gaps(clips)
    for gap_start_utc, gap_end_utc in major_gaps:
        gap_start = to_plot_local(gap_start_utc, timezone)
        gap_end = to_plot_local(gap_end_utc, timezone)
        ax_cov.axvspan(gap_start, gap_end, facecolor=GAP_SHADE_COLOR, alpha=0.9, zorder=0.1)
        ax_beh.axvspan(gap_start, gap_end, facecolor=GAP_SHADE_COLOR, alpha=0.9, zorder=0.1)

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
                0.5,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="#92400e",
                zorder=3,
            )

    if clips:
        for clip in clips:
            left, width = _bar_left_width(clip.start_utc, clip.end_utc, timezone)
            ax_cov.broken_barh(
                [(left, width)],
                (0.18, 0.64),
                facecolors=RECORDING_COLOR,
                alpha=0.9,
                zorder=2,
            )
            if clip_annotations:
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
        ax_cov.grid(True, axis="x", color="#cbd5e1", alpha=0.4, linewidth=0.6)
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
            motion_energy_samples,
            bin_minutes=motion_plot_bin_minutes,
            stat=motion_plot_stat,
        )
        ax_motion.set_title("Motion energy", loc="left", fontsize=11, color="#0f172a", pad=8)
        ax_motion.set_yticks(range(len(behavior_rows)))
        ax_motion.set_yticklabels(behavior_rows)
        ax_motion.set_ylabel("Animal")
        ax_motion.set_ylim(-0.5, len(behavior_rows) - 0.5)
        ax_motion.invert_yaxis()
        ax_motion.grid(True, axis="x", color="#cbd5e1", alpha=0.35, linewidth=0.6)
        ax_motion.tick_params(axis="x", labelbottom=False)
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

    if motion_states:
        motion_legend = ax_legend.legend(
            handles=motion_legend_handles(),
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=2,
            frameon=False,
            fontsize=8,
            title="Motion-derived states",
            title_fontsize=9,
            handlelength=1.3,
            handletextpad=0.5,
            columnspacing=1.4,
            borderaxespad=0.0,
        )
        ax_legend.add_artist(motion_legend)

    global_handles = global_annotation_legend_handles(global_events)
    if global_handles:
        global_legend = ax_legend.legend(
            handles=global_handles,
            loc="upper left",
            bbox_to_anchor=(0.34 if motion_states else 0.0, 1.0),
            ncol=len(global_handles),
            frameon=False,
            fontsize=8,
            title="Global intervals",
            title_fontsize=9,
            handlelength=1.3,
            handletextpad=0.5,
            columnspacing=1.2,
            borderaxespad=0.0,
        )
        ax_legend.add_artist(global_legend)

    feeding_handles = feeding_legend_handles(events)
    if feeding_handles:
        feeding_legend = ax_legend.legend(
            handles=feeding_handles,
            loc="upper left",
            bbox_to_anchor=(
                0.58 if motion_states and global_handles else
                0.26 if global_handles else
                0.32 if motion_states else
                0.0,
                1.0,
            ),
            ncol=1,
            frameon=False,
            fontsize=8,
            title="Feeding",
            title_fontsize=9,
            handlelength=1.3,
            handletextpad=0.5,
            columnspacing=1.2,
            borderaxespad=0.0,
        )
        ax_legend.add_artist(feeding_legend)

    ax_legend.legend(
        handles=manual_annotation_legend_handles(),
        loc="upper left",
        bbox_to_anchor=(
            0.72 if feeding_handles and motion_states and global_handles else
            0.42 if feeding_handles and global_handles else
            0.46 if feeding_handles and motion_states else
            0.14 if feeding_handles else
            0.58 if motion_states and global_handles else
            0.26 if global_handles else
            0.32 if motion_states else
            0.0,
            1.0,
        ),
        ncol=5,
        frameon=False,
        fontsize=8,
        title="Manual annotations",
        title_fontsize=9,
        handlelength=1.3,
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    ax_beh.set_title("Behavior annotations", loc="left", fontsize=11, color="#0f172a", pad=10)
    ax_beh.set_yticks(range(len(behavior_rows)))
    ax_beh.set_yticklabels(behavior_rows)
    ax_beh.set_ylabel("Animal")
    ax_beh.set_ylim(-0.5, len(behavior_rows) - 0.5)
    ax_beh.invert_yaxis()
    ax_beh.grid(True, axis="x", color="#cbd5e1", alpha=0.4, linewidth=0.6)
    ax_beh.set_xlabel(x_axis_time_label(timezone, first_utc, last_utc))

    for y in range(len(behavior_rows) + 1):
        ax_beh.axhline(y - 0.5, color=ROW_SEPARATOR_COLOR, linewidth=0.7, zorder=0.35)

    if x_min is not None and x_max is not None:
        configure_time_axis(ax_beh, x_min, x_max)

    video_quality_masks_utc = video_quality_mask_intervals_utc(global_events)
    death_cutoffs = death_cutoffs_local(events)
    if video_quality_masks_utc:
        LOG.info(
            "Applying %d video-quality visualization mask(s) to motion-derived states",
            len(video_quality_masks_utc),
        )
    suppressed_motion_spans = 0
    clipped_motion_spans = 0
    for state in motion_states:
        if state.animal_id not in behavior_index:
            continue
        death_cutoff_local = death_cutoffs.get(state.animal_id)
        state_start_utc = state.start_utc
        state_end_utc = state.end_utc
        if death_cutoff_local is not None:
            if death_cutoff_local.tzinfo is None:
                death_cutoff_local = death_cutoff_local.replace(tzinfo=timezone)
            death_cutoff_utc = death_cutoff_local.astimezone(UTC)
            if state_start_utc >= death_cutoff_utc:
                suppressed_motion_spans += 1
                continue
            if state_end_utc > death_cutoff_utc:
                state_end_utc = death_cutoff_utc
        y = behavior_index[state.animal_id]
        color = MOTION_MOBILE_COLOR if state.state == "mobile" else MOTION_IMMOBILE_COLOR
        visible_segments = subtract_intervals(state_start_utc, state_end_utc, video_quality_masks_utc)
        if not visible_segments:
            suppressed_motion_spans += 1
            continue
        if len(visible_segments) != 1 or visible_segments[0] != (state_start_utc, state_end_utc):
            clipped_motion_spans += 1
        for segment_start_utc, segment_end_utc in visible_segments:
            left = mdates.date2num(to_plot_local(segment_start_utc, timezone))
            width = max((segment_end_utc - segment_start_utc).total_seconds() / 86400.0, 1e-9)
            ax_beh.broken_barh(
                [(left, width)],
                (y - 0.31, 0.62),
                facecolors=color,
                alpha=0.68,
                zorder=1,
            )

    if video_quality_masks_utc:
        LOG.info(
            "Suppressed %d motion state span(s) and clipped %d span(s) due to video-quality masks",
            suppressed_motion_spans,
            clipped_motion_spans,
        )

    for event in events:
        if event.animal_id not in behavior_index:
            continue
        death_cutoff_local = death_cutoffs.get(event.animal_id)
        if death_cutoff_local is not None:
            if is_death_event(event):
                if event.start_local != death_cutoff_local:
                    continue
            else:
                if event.start_local >= death_cutoff_local:
                    continue
        y = behavior_index[event.animal_id]
        if event.is_point:
            style = behavior_event_style(event)
            if style is None:
                continue
            left = mdates.date2num(event.start_local.replace(tzinfo=None))
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

        event_end_local = event.end_local
        if death_cutoff_local is not None and not is_death_event(event) and event_end_local > death_cutoff_local:
            event_end_local = death_cutoff_local
        duration_s = max((event_end_local - event.start_local).total_seconds(), 0.0)
        if duration_s <= 0:
            continue
        left = mdates.date2num(event.start_local.replace(tzinfo=None))
        width = max(duration_s / 86400.0, 1e-9)
        color = event_bar_color(event)
        ax_beh.broken_barh(
            [(left, width)],
            (y - 0.22, 0.44),
            facecolors=color,
            alpha=0.74,
            zorder=3,
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
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
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
    clips, warnings = collect_recording_clips(root)

    coverage_csv = args.coverage_csv or (root / "recording_coverage.csv")
    output_png = args.output or (root / "recording_behavior_timeline.png")
    write_coverage_csv(clips, coverage_csv, local_tz)

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
            feeding_events, feeding_global_events = load_event_tables(feeding_events_path, local_tz)
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
    motion_energy_path = args.motion_energy
    motion_energy_samples = load_motion_energy_samples(motion_energy_path)

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
    )

    LOG.info("Wrote coverage CSV: %s", coverage_csv)
    LOG.info("Wrote timeline PNG: %s", output_png)
    if warnings:
        LOG.info("Skipped %d malformed timestamp file(s)", len(warnings))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
