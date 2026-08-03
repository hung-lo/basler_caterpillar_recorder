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
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

import yaml


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


def first_present(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


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
            utc_ns = coerce_int(first_present(row, "host_utc_ns"))
            if utc_ns is None:
                utc_ns = parse_iso_ns(first_present(row, "host_utc_iso"))
            mono_ns = coerce_int(first_present(row, "host_monotonic_ns"))

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

    try:
        with json_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception as exc:
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
            issues=issues + [f"could not read JSON: {exc}"],
        )

    if not isinstance(metadata, dict):
        issues.append("JSON sidecar does not contain an object")
        metadata = {}

    label = str(first_present(metadata, "label") or first_present(metadata.get("requested_settings", {}) if isinstance(metadata.get("requested_settings"), dict) else {}, "label") or camera_cfg.get("label") or "").strip()
    requested_settings = metadata.get("requested_settings")
    actual_settings = metadata.get("actual_settings")
    if not isinstance(requested_settings, dict):
        requested_settings = {}
        issues.append("requested_settings missing or invalid")
    if not isinstance(actual_settings, dict):
        actual_settings = {}

    success = bool(metadata.get("success")) if "success" in metadata else False
    grab_failures = coerce_int(first_present(metadata, "grab_failures", "grabFailures"))
    mp4_remux_succeeded = metadata.get("mp4_remux_succeeded")
    if mp4_remux_succeeded is None:
        mp4_remux_succeeded = metadata.get("mp4RemuxSucceeded")
    if mp4_remux_succeeded is not None:
        mp4_remux_succeeded = bool(mp4_remux_succeeded)
    frame_count = coerce_int(first_present(metadata, "frame_count", "frames"))
    requested_fps = coerce_float(first_present(requested_settings, "fps"))
    if requested_fps is None:
        requested_fps = coerce_float(first_present(actual_settings, "AcquisitionFrameRate", "AcquisitionFrameRateAbs"))
    if requested_fps is not None and expected_duration_s is not None:
        expected_frames = requested_fps * expected_duration_s
    else:
        expected_frames = None

    if label and label != camera_cfg.get("label"):
        issues.append(f"label mismatch: expected {camera_cfg.get('label')!r}, found {label!r}")

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

    clip_dirs = sorted(
        [item for item in session_dir.iterdir() if item.is_dir() and item.name.startswith("clip_")],
        key=lambda path: path.name,
    )

    print(f"Session: {session_dir}")
    print(f"Config: {config.get('project')!r} / {config.get('subject')!r}")
    print(f"Clip dirs: {len(clip_dirs)}")

    if number_of_clips is not None and len(clip_dirs) != number_of_clips:
        problems.append(
            f"expected {number_of_clips} clip directories from config_used.yaml, found {len(clip_dirs)}"
        )

    clip_reports: list[ClipCameraReport] = []
    per_camera_reports: dict[str, list[ClipCameraReport]] = {label: [] for label in expected_labels}

    for clip_dir in clip_dirs:
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
        for json_path in json_files:
            try:
                with json_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except Exception as exc:
                problems.append(f"{json_path.relative_to(session_dir)}: could not read JSON: {exc}")
                continue

            if not isinstance(metadata, dict):
                problems.append(f"{json_path.relative_to(session_dir)}: JSON root is not an object")
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
                problems.append(f"{json_path.relative_to(session_dir)}: missing camera label")
                continue
            if label in reports_by_label:
                problems.append(f"{clip_dir.name}: duplicate camera label {label!r}")
                continue
            if label not in expected_labels:
                problems.append(f"{clip_dir.name}: unexpected camera label {label!r}")
                continue

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
            if expected_label not in reports_by_label:
                problems.append(f"{clip_dir.name}: missing camera report for {expected_label!r}")

    for label in expected_labels:
        reports = sorted(per_camera_reports.get(label, []), key=lambda item: item.clip_index)
        for report in reports:
            if report.issues:
                for issue in report.issues:
                    problems.append(f"{report.json_path.relative_to(session_dir)}: {issue}")
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
