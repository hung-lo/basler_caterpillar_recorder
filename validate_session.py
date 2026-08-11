#!/usr/bin/env python3
"""Validate a scheduled recording session directory.

The validator checks clip structure, per-camera artifact presence, JSON metadata,
timestamp coverage, clip duration, and gaps between back-to-back clips.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import gzip
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

import yaml


SESSION_NAME_RE = re.compile(r"^\d{8}_\d{6}(?:[+-]\d{4})?(?:_\d{2})?$")
CLIP_NAME_RE = re.compile(r"^clip_\d{4}_\d{6}(?:[+-]\d{4})?(?:_\d{2})?$")


def utc_from_ns(value_ns: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(value_ns / 1e9, tz=dt.timezone.utc)


def iso_from_ns(value_ns: int) -> str:
    return utc_from_ns(value_ns).isoformat()


def parse_iso_ns(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return int(dt.datetime.fromisoformat(text).timestamp() * 1e9)
        except ValueError:
            return None
    return None


def parse_iso_datetime(value: str) -> dt.datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = dt.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("Timestamp is missing a timezone offset")
    return parsed


def first_present(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def load_json_mapping(path: Path) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        return None, f"could not read JSON: {exc}"
    if not isinstance(data, dict):
        return None, "JSON root is not an object"
    return data, None


def coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1", "on"}:
            return True
        if text in {"false", "no", "0", "off"}:
            return False
    return None


def sanitize_token(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    cleaned = "".join(ch if ch in allowed else "-" for ch in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-_") or "unnamed"


@dataclasses.dataclass
class TimestampStats:
    row_count: int
    first_utc_ns: Optional[int]
    last_utc_ns: Optional[int]
    first_mono_ns: Optional[int]
    last_mono_ns: Optional[int]
    median_receive_fps: Optional[float]

    @property
    def observed_duration_s(self) -> Optional[float]:
        if self.first_utc_ns is None or self.last_utc_ns is None:
            return None
        return (self.last_utc_ns - self.first_utc_ns) / 1e9


@dataclasses.dataclass
class ClipCameraReport:
    clip_index: int
    camera_label: str
    json_path: Path
    mp4_path: Path
    timestamps_path: Path
    success: bool
    grab_failures: Optional[int]
    mp4_remux_succeeded: Optional[bool]
    frame_count: Optional[int]
    expected_frames: Optional[float]
    expected_duration_s: Optional[float]
    requested_fps: Optional[float]
    first_utc_ns: Optional[int]
    last_utc_ns: Optional[int]
    observed_duration_s: Optional[float]
    median_receive_fps: Optional[float]
    issues: list[str]


def load_config_used(session_dir: Path) -> tuple[dict[str, Any], Path]:
    config_path = session_dir / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_used.yaml in {session_dir}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config_used.yaml must contain a YAML mapping")
    return config, config_path


def read_timestamp_stats(path: Path) -> TimestampStats:
    row_count = 0
    first_utc_ns: Optional[int] = None
    last_utc_ns: Optional[int] = None
    first_mono_ns: Optional[int] = None
    last_mono_ns: Optional[int] = None
    mono_series: list[int] = []

    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            utc_ns = coerce_int(first_present(row, "host_utc_ns", "utc_ns", "timestamp_ns"))
            if utc_ns is None:
                utc_ns = parse_iso_ns(first_present(row, "host_utc_iso", "utc_iso", "timestamp_iso"))
            mono_ns = coerce_int(first_present(row, "host_monotonic_ns", "monotonic_ns", "mono_ns"))

            if first_utc_ns is None and utc_ns is not None:
                first_utc_ns = utc_ns
            if first_mono_ns is None and mono_ns is not None:
                first_mono_ns = mono_ns
            if utc_ns is not None:
                last_utc_ns = utc_ns
            if mono_ns is not None:
                last_mono_ns = mono_ns
                mono_series.append(mono_ns)

    median_receive_fps: Optional[float] = None
    if len(mono_series) >= 3:
        deltas_s = [
            (later - earlier) / 1e9
            for earlier, later in zip(mono_series, mono_series[1:])
            if later > earlier
        ]
        if deltas_s:
            median_delta_s = statistics.median(deltas_s)
            if median_delta_s > 0:
                median_receive_fps = 1.0 / median_delta_s

    return TimestampStats(
        row_count=row_count,
        first_utc_ns=first_utc_ns,
        last_utc_ns=last_utc_ns,
        first_mono_ns=first_mono_ns,
        last_mono_ns=last_mono_ns,
        median_receive_fps=median_receive_fps,
    )


def expected_label_order(config: dict[str, Any]) -> list[str]:
    cameras = config.get("cameras")
    if not isinstance(cameras, list):
        return []
    labels: list[str] = []
    for item in cameras:
        if isinstance(item, dict):
            label = str(item.get("label") or "").strip()
            if label:
                labels.append(label)
    return labels


def parse_session_summary(session_dir: Path) -> tuple[Optional[dict[str, Any]], list[str]]:
    summary_path = session_dir / "session_summary.json"
    issues: list[str] = []
    if not summary_path.exists():
        return None, [f"missing session summary: {summary_path.name}"]
    summary, error = load_json_mapping(summary_path)
    if error is not None:
        return None, [f"{summary_path.name}: {error}"]
    return summary, issues


def parse_session_manifest(session_dir: Path) -> tuple[Optional[dict[str, Any]], list[str]]:
    manifest_path = session_dir / "session_manifest.json"
    issues: list[str] = []
    if not manifest_path.exists():
        return None, [f"missing session manifest: {manifest_path.name}"]
    manifest, error = load_json_mapping(manifest_path)
    if error is not None:
        return None, [f"{manifest_path.name}: {error}"]
    return manifest, issues


def parse_archive_summary(session_dir: Path) -> tuple[Optional[dict[str, Any]], list[str]]:
    summary_path = session_dir / "archive_summary.json"
    issues: list[str] = []
    if not summary_path.exists():
        return None, [f"missing archive summary: {summary_path.name}"]
    summary, error = load_json_mapping(summary_path)
    if error is not None:
        return None, [f"{summary_path.name}: {error}"]
    return summary, issues


def find_leftover_capture_mkvs(session_dir: Path) -> list[Path]:
    return sorted(session_dir.rglob("*.capture.mkv"))


def format_path_for_report(path: Path, session_dir: Path) -> str:
    try:
        return str(path.relative_to(session_dir))
    except ValueError:
        return str(path)


def find_clip_directories(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        [item for item in root.iterdir() if item.is_dir() and item.name.startswith("clip_")],
        key=lambda path: path.name,
    )


def detect_session_timestamp_naming(name: str) -> Optional[str]:
    if not SESSION_NAME_RE.fullmatch(name):
        return None
    if re.search(r"[+-]\d{4}(?:_\d{2})?$", name):
        return "local_with_offset"
    return "legacy"


def detect_clip_timestamp_naming(name: str) -> Optional[str]:
    if not CLIP_NAME_RE.fullmatch(name):
        return None
    if re.search(r"[+-]\d{4}(?:_\d{2})?$", name):
        return "local_with_offset"
    return "legacy"


def validate_timestamp_pair(
    mapping: dict[str, Any],
    *,
    utc_field: str,
    local_field: str,
    issues: list[str],
    required_local: bool = False,
) -> None:
    utc_value = mapping.get(utc_field)
    local_value = mapping.get(local_field)
    if utc_value in (None, ""):
        return
    if not isinstance(utc_value, str):
        issues.append(f"{utc_field} is not a string")
        return
    try:
        utc_dt = parse_iso_datetime(utc_value)
    except Exception as exc:
        issues.append(f"{utc_field} is invalid: {exc}")
        return

    if local_value in (None, ""):
        if required_local:
            issues.append(f"{local_field} is missing")
        return
    if not isinstance(local_value, str):
        issues.append(f"{local_field} is not a string")
        return
    try:
        local_dt = parse_iso_datetime(local_value)
    except Exception as exc:
        issues.append(f"{local_field} is invalid: {exc}")
        return

    if utc_dt.astimezone(dt.timezone.utc) != local_dt.astimezone(dt.timezone.utc):
        issues.append(f"{utc_field} and {local_field} do not represent the same instant")


def resolve_clip_camera_artifacts(clip_dir: Path, label: str) -> tuple[Optional[Path], Optional[str], list[str]]:
    stem = sanitize_token(label)
    exact_json = clip_dir / f"{stem}.json"
    if exact_json.exists():
        return exact_json, "v2", []

    legacy_json_matches = sorted(clip_dir.glob(f"*_{stem}_clip*.json"))
    if len(legacy_json_matches) == 1:
        return legacy_json_matches[0], "legacy", []
    if len(legacy_json_matches) == 0:
        return None, None, [f"missing JSON sidecar for {label!r}"]
    return (
        None,
        None,
        [
            f"expected exactly one legacy JSON sidecar for {label!r}, "
            f"found {len(legacy_json_matches)}",
        ],
    )


def format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def validate_clip_camera(
    clip_index: int,
    expected_duration_s: Optional[float],
    camera_cfg: dict[str, Any],
    json_path: Path,
) -> ClipCameraReport:
    issues: list[str] = []
    mp4_path = json_path.with_suffix(".mp4")
    timestamps_path = json_path.with_suffix(".timestamps.csv.gz")

    if not mp4_path.exists():
        issues.append(f"missing MP4: {mp4_path.name}")
    if not timestamps_path.exists():
        issues.append(f"missing timestamps: {timestamps_path.name}")

    metadata, error = load_json_mapping(json_path)
    if error is not None:
        return ClipCameraReport(
            clip_index=clip_index,
            camera_label=camera_cfg.get("label", ""),
            json_path=json_path,
            mp4_path=mp4_path,
            timestamps_path=timestamps_path,
            success=False,
            grab_failures=None,
            mp4_remux_succeeded=None,
            frame_count=None,
            expected_frames=None,
            expected_duration_s=expected_duration_s,
            requested_fps=coerce_float(camera_cfg.get("fps")),
            first_utc_ns=None,
            last_utc_ns=None,
            observed_duration_s=None,
            median_receive_fps=None,
            issues=issues + [error],
        )

    requested_settings = metadata.get("requested_settings")
    actual_settings = metadata.get("actual_settings")
    if not isinstance(requested_settings, dict):
        requested_settings = {}
        issues.append("requested_settings missing or invalid")
    if not isinstance(actual_settings, dict):
        actual_settings = {}
    timestamp_policy = metadata.get("timestamp_policy")
    requires_local_mirrors = isinstance(timestamp_policy, dict)

    label = str(
        first_present(metadata, "label")
        or first_present(requested_settings, "label")
        or camera_cfg.get("label")
        or ""
    ).strip()
    success = coerce_bool(first_present(metadata, "success"))
    if success is None:
        success = False
        issues.append("success missing or invalid")
    grab_failures = coerce_int(first_present(metadata, "grab_failures", "grabFailures", "grab_failures_count"))
    mp4_remux_succeeded = metadata.get("mp4_remux_succeeded")
    if mp4_remux_succeeded is None:
        mp4_remux_succeeded = metadata.get("mp4RemuxSucceeded")
    mp4_remux_succeeded = coerce_bool(mp4_remux_succeeded)
    frame_count = coerce_int(first_present(metadata, "frame_count", "frames"))
    requested_fps = coerce_float(first_present(requested_settings, "fps"))
    if requested_fps is None:
        requested_fps = coerce_float(
            first_present(actual_settings, "AcquisitionFrameRate", "AcquisitionFrameRateAbs")
        )
    if requested_fps is not None and expected_duration_s is not None:
        expected_frames = requested_fps * expected_duration_s
    else:
        expected_frames = None

    if label and label != camera_cfg.get("label"):
        issues.append(f"label mismatch: expected {camera_cfg.get('label')!r}, found {label!r}")

    validate_timestamp_pair(
        metadata,
        utc_field="planned_start_utc",
        local_field="planned_start_local",
        issues=issues,
        required_local=requires_local_mirrors,
    )
    validate_timestamp_pair(
        metadata,
        utc_field="planned_finish_utc",
        local_field="planned_finish_local",
        issues=issues,
        required_local=False,
    )
    validate_timestamp_pair(
        metadata,
        utc_field="actual_start_utc",
        local_field="actual_start_local",
        issues=issues,
        required_local=False,
    )
    validate_timestamp_pair(
        metadata,
        utc_field="actual_stop_utc",
        local_field="actual_stop_local",
        issues=issues,
        required_local=False,
    )
    validate_timestamp_pair(
        metadata,
        utc_field="completed_utc",
        local_field="completed_local",
        issues=issues,
        required_local=requires_local_mirrors,
    )

    timestamp_stats: Optional[TimestampStats] = None
    if timestamps_path.exists():
        try:
            timestamp_stats = read_timestamp_stats(timestamps_path)
        except Exception as exc:
            issues.append(f"could not read timestamps: {exc}")
    else:
        timestamp_stats = TimestampStats(
            row_count=0,
            first_utc_ns=None,
            last_utc_ns=None,
            first_mono_ns=None,
            last_mono_ns=None,
            median_receive_fps=None,
        )

    if frame_count is None and timestamp_stats is not None:
        frame_count = timestamp_stats.row_count

    if timestamp_stats is not None and frame_count is not None and timestamp_stats.row_count != frame_count:
        issues.append(
            f"timestamp row count {timestamp_stats.row_count} does not match frame_count {frame_count}"
        )

    if frame_count is not None and expected_frames is not None:
        lower = expected_frames * 0.9
        upper = expected_frames * 1.1
        if not (lower <= frame_count <= upper):
            issues.append(
                f"frame_count {frame_count} outside 10% tolerance of expected {expected_frames:.1f}"
            )

    observed_duration_s = timestamp_stats.observed_duration_s if timestamp_stats else None
    if observed_duration_s is not None and expected_duration_s is not None:
        if abs(observed_duration_s - expected_duration_s) > 2.0:
            issues.append(
                f"observed duration {observed_duration_s:.2f}s outside 2s tolerance for {expected_duration_s:.2f}s clip"
            )

    if not success:
        issues.append("success is false")
    if grab_failures != 0:
        issues.append(f"grab_failures is {grab_failures}")
    if mp4_remux_succeeded is not True:
        issues.append(f"mp4_remux_succeeded is {mp4_remux_succeeded!r}")

    return ClipCameraReport(
        clip_index=clip_index,
        camera_label=camera_cfg.get("label", ""),
        json_path=json_path,
        mp4_path=mp4_path,
        timestamps_path=timestamps_path,
        success=success,
        grab_failures=grab_failures,
        mp4_remux_succeeded=mp4_remux_succeeded,
        frame_count=frame_count,
        expected_frames=expected_frames,
        expected_duration_s=expected_duration_s,
        requested_fps=requested_fps,
        first_utc_ns=timestamp_stats.first_utc_ns if timestamp_stats else None,
        last_utc_ns=timestamp_stats.last_utc_ns if timestamp_stats else None,
        observed_duration_s=observed_duration_s,
        median_receive_fps=timestamp_stats.median_receive_fps if timestamp_stats else None,
        issues=issues,
    )


def validate_review_snapshots(
    *,
    clip_dir: Path,
    camera_json_path: Path,
    camera_metadata: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    review_snapshots = camera_metadata.get("review_snapshots")
    if not isinstance(review_snapshots, dict) or review_snapshots.get("enabled") is not True:
        return warnings

    review_dir_name = str(review_snapshots.get("directory") or "review_snapshots").strip() or "review_snapshots"
    review_dir = clip_dir / review_dir_name
    writer_started = coerce_bool(review_snapshots.get("writer_started"))
    writer_finalized = coerce_bool(review_snapshots.get("writer_finalized"))
    operational = coerce_bool(review_snapshots.get("operational"))
    index_csv_written = coerce_bool(review_snapshots.get("index_csv_written"))
    saved = coerce_int(review_snapshots.get("saved"))
    failed = coerce_int(review_snapshots.get("failed"))
    dropped_queue_full = coerce_int(review_snapshots.get("dropped_queue_full"))
    missed_due_acquisition_gap = coerce_int(review_snapshots.get("missed_due_acquisition_gap"))
    unreached_targets = coerce_int(review_snapshots.get("unreached_targets"))
    requested_per_full_clip = coerce_int(review_snapshots.get("requested_per_full_clip"))

    if operational is False and writer_started is False:
        warnings.append(f"{camera_json_path.name}: review snapshot subsystem was requested but could not start")
        return warnings

    if writer_started and writer_finalized is False:
        warnings.append(f"{camera_json_path.name}: review snapshot writer did not finalize safely")
    elif writer_started is False and writer_finalized is True and operational is not False:
        warnings.append(f"{camera_json_path.name}: review snapshot writer was not started")

    if not review_dir.exists():
        warnings.append(f"{camera_json_path.name}: review snapshot directory is missing: {review_dir_name}")
        return warnings

    temp_review_files = sorted(path for path in review_dir.rglob("*") if path.is_file() and path.name.endswith(".tmp"))
    if temp_review_files:
        warnings.append(
            f"{camera_json_path.name}: review snapshot temporary files remain: "
            + ", ".join(path.name for path in temp_review_files)
        )

    index_csv_rel = str(review_snapshots.get("index_csv") or "").strip()
    index_csv_path = clip_dir / index_csv_rel if index_csv_rel else review_dir / f"{camera_json_path.stem}_snapshots.csv"
    if index_csv_written is True and not index_csv_path.exists():
        warnings.append(f"{camera_json_path.name}: review snapshot index CSV is missing: {index_csv_path.name}")
    elif index_csv_written is False:
        warnings.append(f"{camera_json_path.name}: review snapshot index CSV was not finalized")

    if not index_csv_path.exists():
        return warnings

    try:
        with index_csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            required_fields = {"snapshot_index", "filename", "frame_index"}
            if not required_fields.issubset(set(fieldnames)):
                warnings.append(
                    f"{camera_json_path.name}: review snapshot index CSV is missing required columns "
                    f"{sorted(required_fields - set(fieldnames))}"
                )
                return warnings

            rows = list(reader)
    except Exception as exc:
        warnings.append(f"{camera_json_path.name}: review snapshot index CSV could not be parsed: {exc}")
        return warnings

    if saved is not None and saved >= 0 and len(rows) != saved:
        warnings.append(
            f"{camera_json_path.name}: review snapshot metadata saved={saved} but index contains {len(rows)} rows"
        )

    snapshot_indices: set[int] = set()
    frame_indices: set[int] = set()
    for row_index, row in enumerate(rows, start=2):
        try:
            snapshot_index = int(str(row.get("snapshot_index") or "").strip())
            frame_index = int(str(row.get("frame_index") or "").strip())
        except Exception:
            warnings.append(f"{camera_json_path.name}: review snapshot CSV row {row_index} has invalid indices")
            continue

        if snapshot_index in snapshot_indices:
            warnings.append(f"{camera_json_path.name}: review snapshot index {snapshot_index} is duplicated")
        else:
            snapshot_indices.add(snapshot_index)

        if frame_index in frame_indices:
            warnings.append(f"{camera_json_path.name}: review snapshot frame index {frame_index} is duplicated")
        else:
            frame_indices.add(frame_index)

        filename = str(row.get("filename") or "").strip()
        jpeg_path = review_dir / filename if filename else None
        if jpeg_path is None or not jpeg_path.exists() or not jpeg_path.is_file():
            warnings.append(
                f"{camera_json_path.name}: review snapshot JPEG is missing: "
                f"{filename or '(blank filename)'}"
            )

    if any(value is not None and value < 0 for value in [saved, failed, dropped_queue_full, missed_due_acquisition_gap, unreached_targets]):
        warnings.append(f"{camera_json_path.name}: review snapshot counts must be non-negative")

    if (
        requested_per_full_clip is not None
        and saved is not None
        and failed is not None
        and dropped_queue_full is not None
        and missed_due_acquisition_gap is not None
        and unreached_targets is not None
    ):
        if saved + failed + dropped_queue_full + missed_due_acquisition_gap + unreached_targets != requested_per_full_clip:
            warnings.append(
                f"{camera_json_path.name}: review snapshot counts do not add up to requested_per_full_clip"
            )

    return warnings


def validate_session(session_dir: Path) -> int:
    problems: list[str] = []
    warnings: list[str] = []

    try:
        config, config_path = load_config_used(session_dir)
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1

    schedule = config.get("schedule") if isinstance(config.get("schedule"), dict) else {}
    number_of_clips = coerce_int(schedule.get("number_of_clips")) if isinstance(schedule, dict) else None
    clip_duration_s = coerce_float(schedule.get("clip_duration_s")) if isinstance(schedule, dict) else None
    camera_configs = config.get("cameras") if isinstance(config.get("cameras"), list) else []
    expected_labels = expected_label_order(config)
    archive_config = config.get("archive") if isinstance(config.get("archive"), dict) else {}
    manifest, manifest_issues = parse_session_manifest(session_dir)
    archive_enabled = bool(archive_config.get("enabled", False)) if isinstance(archive_config, dict) else False
    summary, summary_issues = parse_session_summary(session_dir)
    archive_summary, archive_summary_issues = (
        parse_archive_summary(session_dir) if archive_enabled else (None, [])
    )
    archive_session_dir: Optional[Path] = None
    local_clip_dirs = find_clip_directories(session_dir)
    archived_clip_dirs: list[Path] = []
    incoming_partial_dirs: list[Path] = []
    duplicate_clip_names: list[str] = []

    if archive_enabled and archive_summary is not None:
        archive_session_dir_value = archive_summary.get("archive_session_dir")
        if isinstance(archive_session_dir_value, str) and archive_session_dir_value.strip():
            archive_session_dir = Path(archive_session_dir_value).expanduser()
            if archive_session_dir.exists() and archive_session_dir.is_dir():
                archived_clip_dirs = find_clip_directories(archive_session_dir)
                incoming_dir = archive_session_dir / ".incoming"
                if incoming_dir.exists() and incoming_dir.is_dir():
                    incoming_partial_dirs = sorted(incoming_dir.glob("*.partial"))

    clip_dirs_by_name: dict[str, Path] = {}
    for clip_dir in archived_clip_dirs + local_clip_dirs:
        if clip_dir.name in clip_dirs_by_name:
            duplicate_clip_names.append(clip_dir.name)
            continue
        clip_dirs_by_name[clip_dir.name] = clip_dir
    clip_dirs = sorted(clip_dirs_by_name.values(), key=lambda path: path.name)
    leftover_mkvs = find_leftover_capture_mkvs(session_dir)
    session_timestamp_naming = detect_session_timestamp_naming(session_dir.name)
    clip_timestamp_namings: set[str] = set()
    requires_local_mirrors = False

    print(f"Session: {session_dir}")
    print(f"Config: {config.get('project')!r} / {config.get('subject')!r}")
    print(f"Local clip dirs: {len(local_clip_dirs)}")
    if archive_session_dir is not None:
        print(f"Archive session dir: {archive_session_dir}")
        print(f"Archived clip dirs: {len(archived_clip_dirs)}")
        print(f"Archive partial dirs: {len(incoming_partial_dirs)}")
    print(f"Visible clip dirs: {len(clip_dirs)}")
    print(f"Temp MKVs: {len(leftover_mkvs)}")
    if session_timestamp_naming is not None:
        print(f"Session timestamp naming: {session_timestamp_naming}")
    else:
        warnings.append(
            f"session directory name does not match a supported timestamp format: {session_dir.name}"
        )

    if manifest is not None:
        naming = manifest.get("naming")
        naming_version = coerce_int(naming.get("version")) if isinstance(naming, dict) else None
        timestamp_policy = manifest.get("timestamp_policy")
        requires_local_mirrors = naming_version is not None and naming_version >= 3
        if not requires_local_mirrors and isinstance(timestamp_policy, dict):
            requires_local_mirrors = True
        validate_timestamp_pair(
            manifest,
            utc_field="created_utc",
            local_field="created_local",
            issues=problems,
            required_local=requires_local_mirrors,
        )
        validate_timestamp_pair(
            manifest,
            utc_field="session_start_utc",
            local_field="session_start_local",
            issues=problems,
            required_local=requires_local_mirrors,
        )
        recording_plan = manifest.get("recording_plan")
        if isinstance(recording_plan, dict):
            validate_timestamp_pair(
                recording_plan,
                utc_field="schedule_start_utc",
                local_field="schedule_start_local",
                issues=problems,
                required_local=requires_local_mirrors,
            )
            validate_timestamp_pair(
                recording_plan,
                utc_field="planned_finish_utc",
                local_field="planned_finish_local",
                issues=problems,
                required_local=requires_local_mirrors,
            )
    else:
        problems.extend(manifest_issues)

    if duplicate_clip_names:
        problems.append(f"duplicate clip directories found across local/archive views: {sorted(duplicate_clip_names)}")

    if number_of_clips is not None and len(clip_dirs) != number_of_clips:
        problems.append(
            f"expected {number_of_clips} clip directories from config_used.yaml, found {len(clip_dirs)}"
        )

    session_succeeded = False
    if summary is not None:
        completed_clips = coerce_int(summary.get("completed_clips"))
        requested_clips = coerce_int(summary.get("requested_clips"))
        stopped_by_signal = coerce_bool(summary.get("stopped_by_signal"))
        any_failure = coerce_bool(summary.get("any_failure"))
        exit_status = summary.get("exit_status")
        finished_utc = summary.get("finished_utc")
        unexpected_exception = summary.get("unexpected_exception")

        print(
            "Summary: "
            f"completed_clips={format_optional(completed_clips)} "
            f"requested_clips={format_optional(requested_clips)} "
            f"stopped_by_signal={format_optional(stopped_by_signal)} "
            f"any_failure={format_optional(any_failure)} "
            f"exit_status={format_optional(exit_status)} "
            f"finished_utc={format_optional(finished_utc)} "
            f"unexpected_exception={format_optional(unexpected_exception)}"
        )
        validate_timestamp_pair(
            summary,
            utc_field="finished_utc",
            local_field="finished_local",
            issues=problems,
            required_local=requires_local_mirrors,
        )

        if completed_clips is None:
            problems.append("session_summary.json: completed_clips missing or invalid")
        elif completed_clips != len(clip_dirs):
            problems.append(
                f"session_summary.json: completed_clips {completed_clips} does not match the {len(clip_dirs)} clip directories found"
            )

        if number_of_clips is not None and requested_clips != number_of_clips:
            problems.append(
                f"session_summary.json: requested_clips {requested_clips} does not match config_used.yaml number_of_clips {number_of_clips}"
            )

        if any_failure is None:
            problems.append("session_summary.json: any_failure missing or invalid")
        elif any_failure:
            problems.append("session_summary.json: any_failure is true")

        if exit_status != "success":
            problems.append(f"session_summary.json: exit_status is {exit_status!r}, expected 'success'")

        if stopped_by_signal is None:
            problems.append("session_summary.json: stopped_by_signal missing or invalid")
        elif stopped_by_signal:
            problems.append("session_summary.json: stopped_by_signal is true")

        if unexpected_exception not in (None, ""):
            problems.append(f"session_summary.json: unexpected_exception is {unexpected_exception!r}")
        session_succeeded = (
            completed_clips is not None
            and completed_clips == len(clip_dirs)
            and requested_clips == number_of_clips
            and stopped_by_signal is False
            and any_failure is False
            and exit_status == "success"
            and unexpected_exception in (None, "")
        )
    else:
        problems.extend(summary_issues)

    if archive_enabled:
        if archive_summary is not None:
            queued_clips = coerce_int(archive_summary.get("queued_clips"))
            processed_clips = coerce_int(archive_summary.get("processed_clips"))
            successful_clips = coerce_int(archive_summary.get("successful_clips"))
            failed_clips = coerce_int(archive_summary.get("failed_clips"))
            failure_detected = coerce_bool(archive_summary.get("failure_detected"))
            completed_utc = archive_summary.get("completed_utc")

            print(
                "Archive summary: "
                f"queued_clips={format_optional(queued_clips)} "
                f"processed_clips={format_optional(processed_clips)} "
                f"successful_clips={format_optional(successful_clips)} "
                f"failed_clips={format_optional(failed_clips)} "
                f"failure_detected={format_optional(failure_detected)} "
                f"completed_utc={format_optional(completed_utc)}"
            )
            validate_timestamp_pair(
                archive_summary,
                utc_field="completed_utc",
                local_field="completed_local",
                issues=problems,
                required_local=False,
            )

            if queued_clips is None:
                problems.append("archive_summary.json: queued_clips missing or invalid")
            elif queued_clips != len(clip_dirs):
                problems.append(
                    f"archive_summary.json: queued_clips {queued_clips} does not match the {len(clip_dirs)} clip directories found"
                )

            if processed_clips is None:
                problems.append("archive_summary.json: processed_clips missing or invalid")
            elif processed_clips != len(clip_dirs):
                problems.append(
                    f"archive_summary.json: processed_clips {processed_clips} does not match the {len(clip_dirs)} clip directories found"
                )

            if successful_clips is None:
                problems.append("archive_summary.json: successful_clips missing or invalid")
            elif successful_clips != len(clip_dirs):
                problems.append(
                    f"archive_summary.json: successful_clips {successful_clips} does not match the {len(clip_dirs)} clip directories found"
                )

            if failed_clips is None:
                problems.append("archive_summary.json: failed_clips missing or invalid")
            elif failed_clips != 0:
                problems.append(f"archive_summary.json: failed_clips is {failed_clips}")

            if failure_detected is None:
                problems.append("archive_summary.json: failure_detected missing or invalid")
            elif failure_detected:
                problems.append("archive_summary.json: failure_detected is true")

            results = archive_summary.get("results")
            if not isinstance(results, list):
                problems.append("archive_summary.json: results missing or invalid")
            elif len(results) != len(clip_dirs):
                problems.append(
                    f"archive_summary.json: results length {len(results)} does not match the {len(clip_dirs)} clip directories found"
                )
            else:
                for index, result in enumerate(results):
                    if not isinstance(result, dict):
                        problems.append(f"archive_summary.json: result {index} is not an object")
                        continue
                    validate_timestamp_pair(
                        result,
                        utc_field="started_utc",
                        local_field="started_local",
                        issues=problems,
                        required_local=False,
                    )
                    validate_timestamp_pair(
                        result,
                        utc_field="completed_utc",
                        local_field="completed_local",
                        issues=problems,
                        required_local=False,
                    )
                    result_success = coerce_bool(result.get("success"))
                    verification_succeeded = coerce_bool(result.get("verification_succeeded"))
                    destination_path_value = result.get("destination_path")
                    destination_path = (
                        Path(destination_path_value).expanduser()
                        if isinstance(destination_path_value, str) and destination_path_value.strip()
                        else None
                    )
                    if session_succeeded and result_success is not True:
                        problems.append(f"archive_summary.json: result {index} success is {result_success!r}")
                    if session_succeeded and verification_succeeded is not True:
                        problems.append(
                            f"archive_summary.json: result {index} verification_succeeded is {verification_succeeded!r}"
                        )
                    if result_success is True and destination_path is not None and not destination_path.is_dir():
                        problems.append(
                            f"archive_summary.json: successful transfer missing final visible clip directory {destination_path}"
                        )
        else:
            problems.extend(archive_summary_issues)

        if archive_session_dir is None:
            if session_succeeded:
                problems.append("archive session directory could not be resolved from archive_summary.json")
        elif not archive_session_dir.exists():
            if session_succeeded:
                problems.append(f"archive session directory is missing: {archive_session_dir}")
            else:
                warnings.append(f"archive session directory is unavailable: {archive_session_dir}")

        if incoming_partial_dirs:
            if session_succeeded:
                problems.append(
                    "incomplete archive clip remains: "
                    + ", ".join(str(path) for path in incoming_partial_dirs)
                )
            else:
                warnings.append(
                    "preserved archive partial directories: "
                    + ", ".join(str(path) for path in incoming_partial_dirs)
                )

    if leftover_mkvs:
        problems.append(f"found {len(leftover_mkvs)} leftover temporary MKV file(s)")
        for path in leftover_mkvs:
            problems.append(f"leftover temporary MKV: {path}")

    clip_reports: list[ClipCameraReport] = []
    per_camera_reports: dict[str, list[ClipCameraReport]] = {label: [] for label in expected_labels}
    naming_layouts: set[str] = set()

    for clip_dir in clip_dirs:
        clip_timestamp_naming = detect_clip_timestamp_naming(clip_dir.name)
        if clip_timestamp_naming is None:
            warnings.append(
                f"{clip_dir.name}: clip directory name does not match a supported timestamp format"
            )
        else:
            clip_timestamp_namings.add(clip_timestamp_naming)
        try:
            clip_index = int(clip_dir.name.split("_", 2)[1])
        except Exception:
            clip_index = -1
            warnings.append(f"{clip_dir.name}: could not parse clip index")

        json_files = sorted(clip_dir.glob("*.json"))
        if len(json_files) != len(expected_labels):
            problems.append(
                f"{clip_dir.name}: expected {len(expected_labels)} JSON sidecars, found {len(json_files)}"
            )

        reports_by_label: dict[str, ClipCameraReport] = {}
        clip_layouts: set[str] = set()

        for json_path in json_files:
            try:
                with json_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except Exception as exc:
                problems.append(f"{format_path_for_report(json_path, session_dir)}: could not read JSON: {exc}")
                continue

            if not isinstance(metadata, dict):
                problems.append(f"{format_path_for_report(json_path, session_dir)}: JSON root is not an object")
                continue

            requested_settings = metadata.get("requested_settings")
            if not isinstance(requested_settings, dict):
                requested_settings = {}

            label = str(
                first_present(
                    metadata,
                    "label",
                )
                or first_present(requested_settings, "label")
                or ""
            ).strip()
            if not label:
                problems.append(f"{format_path_for_report(json_path, session_dir)}: missing camera label")
                continue
            if label in reports_by_label:
                problems.append(f"{clip_dir.name}: duplicate camera label {label!r}")
                continue
            if label not in expected_labels:
                problems.append(f"{clip_dir.name}: unexpected camera label {label!r}")
                continue

            warnings.extend(
                validate_review_snapshots(
                    clip_dir=clip_dir,
                    camera_json_path=json_path,
                    camera_metadata=metadata,
                )
            )

            report = validate_clip_camera(
                clip_index=clip_index,
                expected_duration_s=clip_duration_s,
                camera_cfg=next((item for item in camera_configs if str(item.get("label")) == label), {"label": label}),
                json_path=json_path,
            )
            reports_by_label[label] = report
            clip_reports.append(report)
            per_camera_reports.setdefault(label, []).append(report)

        for expected_label in expected_labels:
            json_path, layout, resolution_issues = resolve_clip_camera_artifacts(clip_dir, expected_label)
            if layout is not None:
                clip_layouts.add(layout)
            for issue in resolution_issues:
                problems.append(f"{clip_dir.name}: {issue}")
            if json_path is None:
                problems.append(f"{clip_dir.name}: missing camera report for {expected_label!r}")
                continue
            if expected_label not in reports_by_label:
                report = validate_clip_camera(
                    clip_index=clip_index,
                    expected_duration_s=clip_duration_s,
                    camera_cfg=next(
                        (item for item in camera_configs if str(item.get("label")) == expected_label),
                        {"label": expected_label},
                    ),
                    json_path=json_path,
                )
                reports_by_label[expected_label] = report
                clip_reports.append(report)
                per_camera_reports.setdefault(expected_label, []).append(report)

        if clip_layouts:
            naming_layouts.update(clip_layouts)
            if len(clip_layouts) > 1:
                problems.append(f"{clip_dir.name}: mixed naming layouts detected: {sorted(clip_layouts)}")

    for label in expected_labels:
        reports = sorted(per_camera_reports.get(label, []), key=lambda item: item.clip_index)
        for report in reports:
            if report.issues:
                for issue in report.issues:
                    problems.append(f"{format_path_for_report(report.json_path, session_dir)}: {issue}")
            first = iso_from_ns(report.first_utc_ns) if report.first_utc_ns is not None else "n/a"
            last = iso_from_ns(report.last_utc_ns) if report.last_utc_ns is not None else "n/a"
            observed = f"{report.observed_duration_s:.2f}s" if report.observed_duration_s is not None else "n/a"
            fps = f"{report.median_receive_fps:.2f}" if report.median_receive_fps is not None else "n/a"
            expected_frames = f"{report.expected_frames:.1f}" if report.expected_frames is not None else "n/a"
            print(
                f"clip {report.clip_index:04d} {label}: "
                f"frames={report.frame_count} expected={expected_frames} "
                f"duration={observed} first={first} last={last} median_fps={fps}"
            )

        for prev, nxt in zip(reports, reports[1:]):
            if prev.last_utc_ns is None or nxt.first_utc_ns is None:
                problems.append(
                    f"{label} clip {prev.clip_index:04d}->{nxt.clip_index:04d}: missing timestamps for boundary gap"
                )
                continue
            gap_s = (nxt.first_utc_ns - prev.last_utc_ns) / 1e9
            gap_line = f"{label} clip {prev.clip_index:04d}->{nxt.clip_index:04d}: gap={gap_s:.2f}s"
            if gap_s > 15.0:
                problems.append(gap_line + " exceeds 15s failure threshold")
            elif gap_s > 5.0:
                warnings.append(gap_line + " exceeds 5s warning threshold")
            else:
                print(gap_line)

    if naming_layouts == {"v2"}:
        print("Naming layout: v2")
    elif naming_layouts == {"legacy"}:
        print("Naming layout: legacy")
    elif naming_layouts:
        print(f"Naming layout: mixed ({', '.join(sorted(naming_layouts))})")
    if clip_timestamp_namings == {"local_with_offset"}:
        print("Timestamp naming: local_with_offset")
    elif clip_timestamp_namings == {"legacy"}:
        print("Timestamp naming: legacy")
    elif clip_timestamp_namings:
        print(f"Timestamp naming: mixed ({', '.join(sorted(clip_timestamp_namings))})")

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")

    if problems:
        print("FAIL:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path, help="Path to the recorded session directory")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    session_dir = args.session_dir.expanduser()
    if not session_dir.exists():
        print(f"FAIL: session directory does not exist: {session_dir}")
        return 1
    if not session_dir.is_dir():
        print(f"FAIL: not a directory: {session_dir}")
        return 1
    return validate_session(session_dir)


if __name__ == "__main__":
    raise SystemExit(main())
