#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import logging
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence, TextIO

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from analysis_timing import (
    DEFAULT_TIMEZONE,
    UTC,
    format_local,
    format_utc,
    load_timestamp_series,
    load_timezone,
    timezone_label,
    to_plot_local,
)
from extract_motion_energy import ANIMAL_ORDER

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

LOG = logging.getLogger("analyze_leaf_feeding")

LEAF_ESTIMATE_INTERVAL_MINUTES = 5
BURST_DURATION_SECONDS = 60
BURST_STEP_SECONDS = 10
FEEDING_START_LOSS_PX2 = 750.0
FEEDING_CONTINUE_LOSS_PX2 = 300.0
FEEDING_MERGE_GAP_MINUTES = 5
FEEDING_MIN_BOUT_MINUTES = 5
DEFAULT_ALLOWED_GAP_EXTRA_MINUTES = 1
GRID_TOLERANCE = dt.timedelta(seconds=1)
LEAF_RESET_EVENT_NAMES = {"leaf_added", "leaf_replaced", "leaf_change"}
PROGRESS_BAR_WIDTH = 28
PROGRESS_UPDATE_SECONDS = 0.25

LEAF_AREA_FIELDS = [
    "animal_id",
    "clip_key",
    "timestamp_utc",
    "timestamp_local",
    "leaf_epoch",
    "leaf_area_proxy_px",
    "relative_leaf_area",
    "loss_prev_estimate_pct",
    "delta_area_5min_px2",
    "feeding_valid",
    "feeding_raw",
    "feeding",
    "video_quality_excluded",
    "n_sampled_frames",
    "n_valid_frames",
    "qc_flag",
]
FEEDING_EVENT_FIELDS = [
    "animal_id",
    "start_utc",
    "end_utc",
    "start_local",
    "end_local",
    "event",
    "kind",
    "notes",
]
SUMMARY_FIELDS = [
    "animal_id",
    "clip_key",
    "cropped_video",
    "timestamp_file",
    "timestamp_rows",
    "video_frame_count",
    "selected_leaf_frames",
    "decoded_leaf_frames",
    "leaf_estimates",
    "status",
    "error",
]


@dataclasses.dataclass(frozen=True)
class ManifestEntry:
    animal_id: str
    clip_key: str
    cropped_video: Path
    timestamp_file: Path


@dataclasses.dataclass(frozen=True)
class LeafSampleTarget:
    frame_index: int
    timestamp_utc: dt.datetime
    estimate_bucket_utc: dt.datetime


@dataclasses.dataclass(frozen=True)
class LeafAreaEstimate:
    animal_id: str
    clip_key: str
    timestamp_utc: dt.datetime
    leaf_area_proxy_px: float
    sample_areas_px: tuple[float, ...]
    n_sampled_frames: int
    n_valid_frames: int
    video_quality_excluded: bool = False
    qc_flag: str = ""


@dataclasses.dataclass(frozen=True)
class LeafAreaRow:
    animal_id: str
    clip_key: str
    timestamp_utc: dt.datetime
    leaf_epoch: int
    leaf_area_proxy_px: float
    relative_leaf_area: float
    loss_prev_estimate_pct: Optional[float]
    delta_area_5min_px2: Optional[float]
    feeding_valid: bool
    feeding_raw: bool
    feeding: bool
    video_quality_excluded: bool
    n_sampled_frames: int
    n_valid_frames: int
    qc_flag: str = ""


@dataclasses.dataclass(frozen=True)
class ClipFeedingResult:
    estimates: list[LeafAreaEstimate]
    status: str
    timestamp_rows: int
    video_frame_count: Optional[int]
    selected_leaf_frames: int
    decoded_leaf_frames: int
    error: str = ""


def format_progress_bar(completed: int, total: int, *, width: int = PROGRESS_BAR_WIDTH) -> str:
    if total <= 0:
        fraction = 0.0
    else:
        fraction = min(max(completed / total, 0.0), 1.0)
    filled = min(width, int(round(width * fraction)))
    return f"[{'#' * filled}{'-' * (width - filled)}] {fraction * 100:5.1f}%"


def build_progress_line(
    *,
    clip_index: int,
    total_clips: int,
    entry: ManifestEntry,
    decoded_leaf_frames: int,
    total_leaf_frames: int,
    completed_estimates: int,
    total_estimates: int,
    status: str,
) -> str:
    bar = format_progress_bar(decoded_leaf_frames, total_leaf_frames)
    return (
        f"{bar} clip {clip_index}/{total_clips} "
        f"{entry.animal_id} {entry.clip_key} "
        f"samples {decoded_leaf_frames}/{total_leaf_frames} "
        f"estimates {completed_estimates}/{total_estimates} "
        f"{status}"
    )


@dataclasses.dataclass
class ProgressReporter:
    total_clips: int
    enabled: bool = dataclasses.field(default_factory=lambda: sys.stderr.isatty())
    stream: TextIO = dataclasses.field(default_factory=lambda: sys.stderr)
    update_interval_s: float = PROGRESS_UPDATE_SECONDS
    current_clip_index: int = 0
    current_entry: Optional[ManifestEntry] = None
    current_total_leaf_frames: int = 0
    current_total_estimates: int = 0
    last_render_time: float = 0.0

    def start_clip(
        self,
        clip_index: int,
        entry: ManifestEntry,
        *,
        source_frames: int,
        selected_leaf_frames: int,
        total_estimates: int,
    ) -> None:
        self.current_clip_index = clip_index
        self.current_entry = entry
        self.current_total_leaf_frames = max(selected_leaf_frames, 0)
        self.current_total_estimates = max(total_estimates, 0)
        self.last_render_time = 0.0
        fraction = (selected_leaf_frames / source_frames) if source_frames > 0 else 0.0
        LOG.info(
            "%s %s: %d source frames, %d selected leaf frames (%.1f%%)",
            entry.animal_id,
            entry.clip_key,
            source_frames,
            selected_leaf_frames,
            fraction * 100.0,
        )
        self.update(decoded_leaf_frames=0, completed_estimates=0, status="starting", force=True)

    def update(
        self,
        *,
        decoded_leaf_frames: int,
        completed_estimates: int,
        status: str,
        force: bool = False,
    ) -> None:
        if not self.enabled or self.current_entry is None:
            return
        now = time.monotonic()
        if not force and now - self.last_render_time < self.update_interval_s:
            return
        line = build_progress_line(
            clip_index=self.current_clip_index,
            total_clips=self.total_clips,
            entry=self.current_entry,
            decoded_leaf_frames=decoded_leaf_frames,
            total_leaf_frames=self.current_total_leaf_frames,
            completed_estimates=completed_estimates,
            total_estimates=self.current_total_estimates,
            status=status,
        )
        self.stream.write(f"\r{line}")
        self.stream.flush()
        self.last_render_time = now

    def finish_clip(
        self,
        *,
        decoded_leaf_frames: int,
        completed_estimates: int,
        status: str,
    ) -> None:
        if self.current_entry is not None:
            LOG.info(
                "Finished clip %d/%d: %s %s -> %s (%d estimate bucket(s))",
                self.current_clip_index,
                self.total_clips,
                self.current_entry.animal_id,
                self.current_entry.clip_key,
                status,
                completed_estimates,
            )
        self.update(
            decoded_leaf_frames=decoded_leaf_frames,
            completed_estimates=completed_estimates,
            status=status,
            force=True,
        )
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temp_path.replace(path)


def load_manifest_entries(root: Path, *, animals: Optional[set[str]]) -> list[ManifestEntry]:
    manifest_path = root / "cropped_by_caterpillar" / "crop_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"missing crop manifest: {manifest_path}. Run prepare_cropped_timestamps.py first."
        )
    entries: list[ManifestEntry] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            animal_id = str(row.get("animal_id") or "").strip()
            if animal_id not in ANIMAL_ORDER:
                continue
            if animals and animal_id not in animals:
                continue
            clip_key = str(row.get("clip_key") or "").strip()
            cropped_rel = str(row.get("cropped_video") or "").strip()
            timestamp_rel = str(row.get("copied_timestamp_file") or "").strip()
            if not clip_key or not cropped_rel or not timestamp_rel:
                continue
            entries.append(
                ManifestEntry(
                    animal_id=animal_id,
                    clip_key=clip_key,
                    cropped_video=(root / cropped_rel).resolve(),
                    timestamp_file=(root / timestamp_rel).resolve(),
                )
            )
    entries.sort(key=lambda entry: (entry.clip_key, entry.animal_id))
    return entries


def floor_utc_interval(value: dt.datetime, interval_minutes: int) -> dt.datetime:
    value = value.astimezone(UTC)
    floored_minute = value.minute - (value.minute % interval_minutes)
    return value.replace(minute=floored_minute, second=0, microsecond=0)


def interval_buckets(
    timestamps: Sequence[dt.datetime],
    *,
    interval_minutes: int,
) -> list[dt.datetime]:
    if not timestamps:
        return []
    start_bucket = floor_utc_interval(timestamps[0], interval_minutes)
    end_bucket = floor_utc_interval(timestamps[-1], interval_minutes)
    buckets: list[dt.datetime] = []
    current = start_bucket
    step = dt.timedelta(minutes=interval_minutes)
    while current <= end_bucket:
        buckets.append(current)
        current += step
    return buckets


def select_leaf_sample_targets(
    timestamps_utc: Sequence[dt.datetime],
    *,
    estimate_interval_minutes: int = LEAF_ESTIMATE_INTERVAL_MINUTES,
    burst_duration_seconds: int = BURST_DURATION_SECONDS,
    burst_step_seconds: int = BURST_STEP_SECONDS,
) -> list[LeafSampleTarget]:
    if not timestamps_utc:
        return []
    if estimate_interval_minutes <= 0:
        raise ValueError("estimate_interval_minutes must be > 0")
    if burst_duration_seconds <= 0:
        raise ValueError("burst_duration_seconds must be > 0")
    if burst_step_seconds <= 0:
        raise ValueError("burst_step_seconds must be > 0")

    ordered_timestamps = sorted(timestamp.astimezone(UTC) for timestamp in timestamps_utc)
    timestamp_ns = np.array([int(timestamp.timestamp() * 1e9) for timestamp in ordered_timestamps], dtype=np.int64)
    burst_end_delta = dt.timedelta(seconds=burst_duration_seconds)
    targets: list[LeafSampleTarget] = []
    seen: set[tuple[int, dt.datetime]] = set()
    for bucket in interval_buckets(ordered_timestamps, interval_minutes=estimate_interval_minutes):
        bucket_end = bucket + burst_end_delta
        for offset_seconds in range(0, burst_duration_seconds, burst_step_seconds):
            target_time = bucket + dt.timedelta(seconds=offset_seconds)
            target_index = int(np.searchsorted(timestamp_ns, int(target_time.timestamp() * 1e9), side="left"))
            if target_index >= len(ordered_timestamps):
                continue
            timestamp_utc = ordered_timestamps[target_index]
            if timestamp_utc >= bucket_end:
                continue
            key = (target_index, bucket)
            if key in seen:
                continue
            seen.add(key)
            targets.append(
                LeafSampleTarget(
                    frame_index=target_index,
                    timestamp_utc=timestamp_utc,
                    estimate_bucket_utc=bucket,
                )
            )
    targets.sort(key=lambda target: (target.frame_index, target.estimate_bucket_utc, target.timestamp_utc))
    return targets


def segment_leaf_area(
    frame_bgr: np.ndarray,
    *,
    hue_low: int,
    hue_high: int,
    sat_min: int,
    value_min: int,
    min_component_px: int,
    morph_kernel: int,
) -> tuple[float, np.ndarray]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([hue_low, sat_min, value_min], dtype=np.uint8)
    upper = np.array([hue_high, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    area_sum = 0.0
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_px:
            continue
        filtered[labels == label] = 255
        area_sum += area
    return area_sum, filtered


def classify_feeding_hysteresis(
    losses_px2: Iterable[Optional[float]],
    *,
    start_loss_px2: float,
    continue_loss_px2: float,
) -> list[bool]:
    feeding = False
    output: list[bool] = []
    for value in losses_px2:
        if value is None or math.isnan(value):
            feeding = False
            output.append(False)
            continue
        if not feeding:
            feeding = value >= start_loss_px2
        else:
            feeding = value >= continue_loss_px2
        output.append(feeding)
    return output


def merge_short_gaps(
    flags: list[bool],
    *,
    step_minutes: int,
    max_gap_minutes: int,
) -> list[bool]:
    if max_gap_minutes <= 0 or step_minutes <= 0:
        return list(flags)
    merged = list(flags)
    i = 0
    while i < len(merged):
        if merged[i]:
            i += 1
            continue
        gap_start = i
        while i < len(merged) and not merged[i]:
            i += 1
        gap_end = i
        gap_minutes = (gap_end - gap_start) * step_minutes
        if gap_start > 0 and gap_end < len(merged) and merged[gap_start - 1] and merged[gap_end] and gap_minutes <= max_gap_minutes:
            for index in range(gap_start, gap_end):
                merged[index] = True
    return merged


def remove_short_bouts(
    flags: list[bool],
    *,
    step_minutes: int,
    min_bout_minutes: int,
) -> list[bool]:
    if min_bout_minutes <= step_minutes or step_minutes <= 0:
        return list(flags)
    cleaned = list(flags)
    i = 0
    while i < len(cleaned):
        if not cleaned[i]:
            i += 1
            continue
        bout_start = i
        while i < len(cleaned) and cleaned[i]:
            i += 1
        bout_end = i
        bout_minutes = (bout_end - bout_start) * step_minutes
        if bout_minutes < min_bout_minutes:
            for index in range(bout_start, bout_end):
                cleaned[index] = False
    return cleaned


def relative_change_percent(previous: float, current: float) -> float:
    if previous <= 0:
        return 0.0
    return 100.0 * (current - previous) / previous


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
        if start <= current_end:
            if end > current_end:
                current_end = end
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def overlaps_any_interval(
    value: dt.datetime,
    intervals_utc: Sequence[tuple[dt.datetime, dt.datetime]],
) -> bool:
    for start_utc, end_utc in intervals_utc:
        if start_utc <= value < end_utc:
            return True
    return False


def completed_estimate_count(
    grouped_sampled: dict[dt.datetime, int],
    *,
    min_valid_frames: int,
    grouped_valid: dict[dt.datetime, int],
) -> int:
    completed = 0
    for bucket in grouped_sampled:
        if grouped_valid.get(bucket, 0) >= min_valid_frames:
            completed += 1
    return completed


def seek_and_read_frame(capture: cv2.VideoCapture, frame_index: int) -> tuple[str, Optional[np.ndarray]]:
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index)):
        return "seek_failure", None
    ok, frame = capture.read()
    if not ok or frame is None:
        return "seek_failure", None
    position_after_read = capture.get(cv2.CAP_PROP_POS_FRAMES)
    if position_after_read > 0 and abs(position_after_read - float(frame_index + 1)) > 1.5:
        return "seek_mismatch", None
    return "ok", frame


def sparse_sample_frames(
    capture: cv2.VideoCapture,
    targets: Sequence[LeafSampleTarget],
) -> tuple[str, list[tuple[LeafSampleTarget, np.ndarray]]]:
    samples: list[tuple[LeafSampleTarget, np.ndarray]] = []
    for target in targets:
        status, frame = seek_and_read_frame(capture, target.frame_index)
        if status != "ok" or frame is None:
            return status, []
        samples.append((target, frame))
    return "ok", samples


def sequential_sample_frames(
    capture: cv2.VideoCapture,
    targets: Sequence[LeafSampleTarget],
) -> tuple[str, list[tuple[LeafSampleTarget, np.ndarray]]]:
    samples: list[tuple[LeafSampleTarget, np.ndarray]] = []
    if not targets:
        return "ok", samples
    ordered_targets = sorted(targets, key=lambda target: target.frame_index)
    target_index = 0
    frame_index = 0
    while target_index < len(ordered_targets):
        ok, frame = capture.read()
        if not ok or frame is None:
            return "seek_failure", []
        current_target = ordered_targets[target_index]
        if frame_index == current_target.frame_index:
            samples.append((current_target, frame))
            target_index += 1
        frame_index += 1
    return "ok", samples


def extract_clip_leaf_estimates(
    entry: ManifestEntry,
    *,
    clip_index: int = 0,
    estimate_interval_minutes: int,
    burst_duration_seconds: int,
    burst_step_seconds: int,
    leaf_area_percentile: float,
    min_valid_frames: int,
    hue_low: int,
    hue_high: int,
    sat_min: int,
    value_min: int,
    min_component_px: int,
    morph_kernel: int,
    frame_access: str,
    video_quality_intervals_utc: Sequence[tuple[dt.datetime, dt.datetime]],
    progress_reporter: Optional[ProgressReporter] = None,
) -> ClipFeedingResult:
    timestamps = load_timestamp_series(entry.timestamp_file)
    timestamp_rows = len(timestamps)
    targets = select_leaf_sample_targets(
        timestamps,
        estimate_interval_minutes=estimate_interval_minutes,
        burst_duration_seconds=burst_duration_seconds,
        burst_step_seconds=burst_step_seconds,
    )
    total_estimates = len({target.estimate_bucket_utc for target in targets})
    if progress_reporter is not None:
        progress_reporter.start_clip(
            clip_index,
            entry,
            source_frames=timestamp_rows,
            selected_leaf_frames=len(targets),
            total_estimates=total_estimates,
        )

    capture = cv2.VideoCapture(str(entry.cropped_video))
    if not capture.isOpened():
        if progress_reporter is not None:
            progress_reporter.finish_clip(decoded_leaf_frames=0, completed_estimates=0, status="failed")
        return ClipFeedingResult([], "failed", timestamp_rows, None, len(targets), 0, f"could not open {entry.cropped_video}")

    video_frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if video_frame_count > 0 and video_frame_count != timestamp_rows:
        capture.release()
        if progress_reporter is not None:
            progress_reporter.finish_clip(decoded_leaf_frames=0, completed_estimates=0, status="frame_mismatch")
        return ClipFeedingResult(
            [],
            "frame_mismatch",
            timestamp_rows,
            video_frame_count,
            len(targets),
            0,
            (
                f"frame count mismatch for {entry.cropped_video}: "
                f"video reports {video_frame_count} frames, timestamps contain {timestamp_rows} rows"
            ),
        )

    if frame_access == "sparse":
        read_status, sampled_frames = sparse_sample_frames(capture, targets)
    elif frame_access == "sequential":
        read_status, sampled_frames = sequential_sample_frames(capture, targets)
    else:
        capture.release()
        raise ValueError(f"unsupported frame access mode: {frame_access}")
    capture.release()

    if read_status != "ok":
        if progress_reporter is not None:
            progress_reporter.finish_clip(decoded_leaf_frames=0, completed_estimates=0, status=read_status)
        return ClipFeedingResult(
            [],
            read_status,
            timestamp_rows,
            video_frame_count,
            len(targets),
            0,
            f"{read_status} while sampling {entry.cropped_video}",
        )

    grouped_sampled: dict[dt.datetime, int] = defaultdict(int)
    grouped_valid: dict[dt.datetime, int] = defaultdict(int)
    grouped_areas: dict[dt.datetime, list[float]] = defaultdict(list)
    grouped_excluded: dict[dt.datetime, bool] = defaultdict(bool)
    decoded_leaf_frames = 0
    for target, frame in sampled_frames:
        area, _mask = segment_leaf_area(
            frame,
            hue_low=hue_low,
            hue_high=hue_high,
            sat_min=sat_min,
            value_min=value_min,
            min_component_px=min_component_px,
            morph_kernel=morph_kernel,
        )
        decoded_leaf_frames += 1
        bucket = target.estimate_bucket_utc
        grouped_sampled[bucket] += 1
        if overlaps_any_interval(target.timestamp_utc, video_quality_intervals_utc):
            grouped_excluded[bucket] = True
        if area > 0:
            grouped_areas[bucket].append(float(area))
            grouped_valid[bucket] += 1
        if progress_reporter is not None:
            progress_reporter.update(
                decoded_leaf_frames=decoded_leaf_frames,
                completed_estimates=completed_estimate_count(
                    grouped_sampled,
                    min_valid_frames=min_valid_frames,
                    grouped_valid=grouped_valid,
                ),
                status="analyzing",
            )

    estimates: list[LeafAreaEstimate] = []
    for bucket in sorted(grouped_sampled):
        areas = grouped_areas.get(bucket, [])
        valid_count = grouped_valid.get(bucket, 0)
        if valid_count < min_valid_frames or not areas:
            continue
        area_array = np.array(areas, dtype=np.float64)
        estimate = float(np.percentile(area_array, leaf_area_percentile))
        estimates.append(
            LeafAreaEstimate(
                animal_id=entry.animal_id,
                clip_key=entry.clip_key,
                timestamp_utc=bucket,
                leaf_area_proxy_px=estimate,
                sample_areas_px=tuple(float(value) for value in area_array.tolist()),
                n_sampled_frames=grouped_sampled[bucket],
                n_valid_frames=valid_count,
                video_quality_excluded=grouped_excluded.get(bucket, False),
                qc_flag="",
            )
        )

    if progress_reporter is not None:
        progress_reporter.finish_clip(
            decoded_leaf_frames=decoded_leaf_frames,
            completed_estimates=len(estimates),
            status="done",
        )
    return ClipFeedingResult(
        estimates,
        "computed",
        timestamp_rows,
        video_frame_count if video_frame_count > 0 else timestamp_rows,
        len(targets),
        decoded_leaf_frames,
    )


def load_analysis_events(
    *,
    root: Path,
    source: Optional[str],
    timezone: dt.tzinfo,
) -> tuple[dict[str, list[dt.datetime]], list[tuple[dt.datetime, dt.datetime]]]:
    if source is None:
        return {}, []
    from plot_recording_timeline import load_event_tables_source, video_quality_mask_intervals_utc

    events, global_events = load_event_tables_source(source, root=root, tz=timezone)
    resets: dict[str, list[dt.datetime]] = defaultdict(list)
    for event in events:
        event_name = (event.event or "").strip().lower()
        if event.animal_id in ANIMAL_ORDER and event_name in LEAF_RESET_EVENT_NAMES:
            resets[event.animal_id].append(event.start_local.astimezone(UTC))
    for animal_id in list(resets):
        resets[animal_id].sort()
    return dict(resets), video_quality_mask_intervals_utc(global_events)


def consolidate_leaf_estimates(
    estimates: list[LeafAreaEstimate],
    *,
    leaf_area_percentile: float,
) -> list[LeafAreaEstimate]:
    grouped: dict[tuple[str, dt.datetime], list[LeafAreaEstimate]] = defaultdict(list)
    for estimate in estimates:
        grouped[(estimate.animal_id, estimate.timestamp_utc)].append(estimate)

    consolidated: list[LeafAreaEstimate] = []
    for _key, duplicates in sorted(grouped.items(), key=lambda item: (ANIMAL_ORDER.index(item[0][0]), item[0][1])):
        if len(duplicates) == 1:
            consolidated.append(duplicates[0])
            continue
        sample_areas = [area for duplicate in duplicates for area in duplicate.sample_areas_px]
        if not sample_areas:
            continue
        clip_keys = sorted({duplicate.clip_key for duplicate in duplicates})
        area_proxy = float(np.percentile(np.array(sample_areas, dtype=np.float64), leaf_area_percentile))
        consolidated.append(
            LeafAreaEstimate(
                animal_id=duplicates[0].animal_id,
                clip_key="+".join(clip_keys),
                timestamp_utc=duplicates[0].timestamp_utc,
                leaf_area_proxy_px=area_proxy,
                sample_areas_px=tuple(float(area) for area in sample_areas),
                n_sampled_frames=sum(duplicate.n_sampled_frames for duplicate in duplicates),
                n_valid_frames=sum(duplicate.n_valid_frames for duplicate in duplicates),
                video_quality_excluded=any(duplicate.video_quality_excluded for duplicate in duplicates),
                qc_flag=";".join(sorted({duplicate.qc_flag for duplicate in duplicates if duplicate.qc_flag})),
            )
        )
    return consolidated


def finalize_leaf_rows(
    estimates: list[LeafAreaEstimate],
    *,
    timezone: dt.tzinfo,
    leaf_area_percentile: float,
    estimate_interval_minutes: int,
    allowed_gap_minutes: int,
    start_loss_px2: float,
    continue_loss_px2: float,
    merge_gap_minutes: int,
    min_bout_minutes: int,
    leaf_reset_increase_pct: float,
    explicit_resets: Optional[dict[str, list[dt.datetime]]] = None,
) -> list[LeafAreaRow]:
    consolidated_estimates = consolidate_leaf_estimates(estimates, leaf_area_percentile=leaf_area_percentile)
    explicit_resets = explicit_resets or {}
    grouped: dict[str, list[LeafAreaEstimate]] = defaultdict(list)
    for estimate in consolidated_estimates:
        grouped[estimate.animal_id].append(estimate)

    rows: list[LeafAreaRow] = []
    allowed_gap = dt.timedelta(minutes=allowed_gap_minutes)
    estimate_interval = dt.timedelta(minutes=estimate_interval_minutes)
    for animal_id in ANIMAL_ORDER:
        animal_estimates = sorted(grouped.get(animal_id, []), key=lambda estimate: (estimate.timestamp_utc, estimate.clip_key))
        if not animal_estimates:
            continue
        current_epoch = 1
        previous: Optional[LeafAreaEstimate] = None
        baseline_area: Optional[float] = None
        reset_index = 0
        reset_times = explicit_resets.get(animal_id, [])
        animal_rows: list[LeafAreaRow] = []
        for estimate in animal_estimates:
            while reset_index < len(reset_times) and reset_times[reset_index] <= estimate.timestamp_utc:
                current_epoch += 1
                baseline_area = None
                previous = None
                reset_index += 1
            if previous is not None:
                if estimate.timestamp_utc - previous.timestamp_utc > allowed_gap:
                    current_epoch += 1
                    baseline_area = None
                    previous = None
                else:
                    increase_pct = relative_change_percent(previous.leaf_area_proxy_px, estimate.leaf_area_proxy_px)
                    if increase_pct > leaf_reset_increase_pct:
                        current_epoch += 1
                        baseline_area = None
                        previous = None
            if baseline_area is None:
                baseline_area = estimate.leaf_area_proxy_px
            relative_area = estimate.leaf_area_proxy_px / max(baseline_area, 1.0)
            loss_prev_estimate_pct: Optional[float] = None
            delta_area_5min_px2: Optional[float] = None
            feeding_valid = not estimate.video_quality_excluded
            if previous is not None and previous.timestamp_utc + estimate_interval - GRID_TOLERANCE <= estimate.timestamp_utc <= previous.timestamp_utc + estimate_interval + GRID_TOLERANCE:
                loss_prev_estimate_pct = 100.0 * (
                    previous.leaf_area_proxy_px - estimate.leaf_area_proxy_px
                ) / max(previous.leaf_area_proxy_px, 1.0)
                if feeding_valid and not previous.video_quality_excluded:
                    delta_area_5min_px2 = previous.leaf_area_proxy_px - estimate.leaf_area_proxy_px
            animal_rows.append(
                LeafAreaRow(
                    animal_id=estimate.animal_id,
                    clip_key=estimate.clip_key,
                    timestamp_utc=estimate.timestamp_utc,
                    leaf_epoch=current_epoch,
                    leaf_area_proxy_px=estimate.leaf_area_proxy_px,
                    relative_leaf_area=relative_area,
                    loss_prev_estimate_pct=loss_prev_estimate_pct,
                    delta_area_5min_px2=delta_area_5min_px2,
                    feeding_valid=feeding_valid,
                    feeding_raw=False,
                    feeding=False,
                    video_quality_excluded=estimate.video_quality_excluded,
                    n_sampled_frames=estimate.n_sampled_frames,
                    n_valid_frames=estimate.n_valid_frames,
                    qc_flag=estimate.qc_flag,
                )
            )
            previous = estimate

        by_epoch: dict[int, list[LeafAreaRow]] = defaultdict(list)
        for row in animal_rows:
            by_epoch[row.leaf_epoch].append(row)
        for epoch in sorted(by_epoch):
            epoch_rows = by_epoch[epoch]
            raw_flags = classify_feeding_hysteresis(
                [row.delta_area_5min_px2 if row.feeding_valid else None for row in epoch_rows],
                start_loss_px2=start_loss_px2,
                continue_loss_px2=continue_loss_px2,
            )
            final_flags = list(raw_flags)
            segment_start = 0
            while segment_start < len(epoch_rows):
                if not epoch_rows[segment_start].feeding_valid:
                    segment_start += 1
                    continue
                segment_end = segment_start + 1
                while segment_end < len(epoch_rows):
                    previous_row = epoch_rows[segment_end - 1]
                    current_row = epoch_rows[segment_end]
                    gap = current_row.timestamp_utc - previous_row.timestamp_utc
                    if not current_row.feeding_valid or abs(gap - estimate_interval) > GRID_TOLERANCE:
                        break
                    segment_end += 1
                segment_flags = raw_flags[segment_start:segment_end]
                cleaned_flags = remove_short_bouts(
                    merge_short_gaps(
                        segment_flags,
                        step_minutes=estimate_interval_minutes,
                        max_gap_minutes=merge_gap_minutes,
                    ),
                    step_minutes=estimate_interval_minutes,
                    min_bout_minutes=min_bout_minutes,
                )
                final_flags[segment_start:segment_end] = cleaned_flags
                segment_start = segment_end
            for row, raw_flag, final_flag in zip(epoch_rows, raw_flags, final_flags):
                rows.append(dataclasses.replace(row, feeding_raw=raw_flag, feeding=final_flag))
    rows.sort(key=lambda row: (ANIMAL_ORDER.index(row.animal_id), row.timestamp_utc))
    return rows


def leaf_rows_to_dicts(rows: list[LeafAreaRow], timezone: dt.tzinfo) -> list[dict[str, str]]:
    serialized: list[dict[str, str]] = []
    for row in rows:
        serialized.append(
            {
                "animal_id": row.animal_id,
                "clip_key": row.clip_key,
                "timestamp_utc": format_utc(row.timestamp_utc),
                "timestamp_local": format_local(row.timestamp_utc, timezone),
                "leaf_epoch": str(row.leaf_epoch),
                "leaf_area_proxy_px": f"{row.leaf_area_proxy_px:.6f}",
                "relative_leaf_area": f"{row.relative_leaf_area:.6f}",
                "loss_prev_estimate_pct": "" if row.loss_prev_estimate_pct is None else f"{row.loss_prev_estimate_pct:.6f}",
                "delta_area_5min_px2": "" if row.delta_area_5min_px2 is None else f"{row.delta_area_5min_px2:.6f}",
                "feeding_valid": "TRUE" if row.feeding_valid else "FALSE",
                "feeding_raw": "TRUE" if row.feeding_raw else "FALSE",
                "feeding": "TRUE" if row.feeding else "FALSE",
                "video_quality_excluded": "TRUE" if row.video_quality_excluded else "FALSE",
                "n_sampled_frames": str(row.n_sampled_frames),
                "n_valid_frames": str(row.n_valid_frames),
                "qc_flag": row.qc_flag,
            }
        )
    return serialized


def feeding_event_dicts(
    rows: list[LeafAreaRow],
    timezone: dt.tzinfo,
    *,
    estimate_interval_minutes: int,
    start_loss_px2: float,
    continue_loss_px2: float,
) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    grouped: dict[str, list[LeafAreaRow]] = defaultdict(list)
    for row in rows:
        grouped[row.animal_id].append(row)
    estimate_interval = dt.timedelta(minutes=estimate_interval_minutes)
    notes = (
        f"coarse {estimate_interval_minutes}-min absolute leaf-area detector; "
        f"start>={start_loss_px2:g}px2; continue>={continue_loss_px2:g}px2"
    )
    for animal_id in ANIMAL_ORDER:
        animal_rows = sorted(grouped.get(animal_id, []), key=lambda row: row.timestamp_utc)
        active_start: Optional[LeafAreaRow] = None
        previous: Optional[LeafAreaRow] = None
        for row in animal_rows:
            contiguous = (
                previous is not None
                and row.leaf_epoch == previous.leaf_epoch
                and abs((row.timestamp_utc - previous.timestamp_utc) - estimate_interval) <= GRID_TOLERANCE
            )
            if row.feeding and active_start is None:
                active_start = row
            elif active_start is not None and previous is not None and (not row.feeding or not contiguous):
                start_utc = active_start.timestamp_utc
                end_utc = previous.timestamp_utc + estimate_interval
                events.append(
                    {
                        "animal_id": animal_id,
                        "start_utc": format_utc(start_utc),
                        "end_utc": format_utc(end_utc),
                        "start_local": format_local(start_utc, timezone),
                        "end_local": format_local(end_utc, timezone),
                        "event": "feeding",
                        "kind": "feeding",
                        "notes": notes,
                    }
                )
                active_start = row if row.feeding else None
            previous = row
        if active_start is not None and previous is not None:
            start_utc = active_start.timestamp_utc
            end_utc = previous.timestamp_utc + estimate_interval
            events.append(
                {
                    "animal_id": animal_id,
                    "start_utc": format_utc(start_utc),
                    "end_utc": format_utc(end_utc),
                    "start_local": format_local(start_utc, timezone),
                    "end_local": format_local(end_utc, timezone),
                    "event": "feeding",
                    "kind": "feeding",
                    "notes": notes,
                }
            )
    return events


def summary_row(root: Path, entry: ManifestEntry, result: ClipFeedingResult) -> dict[str, str]:
    return {
        "animal_id": entry.animal_id,
        "clip_key": entry.clip_key,
        "cropped_video": str(entry.cropped_video.relative_to(root)),
        "timestamp_file": str(entry.timestamp_file.relative_to(root)),
        "timestamp_rows": str(result.timestamp_rows),
        "video_frame_count": "" if result.video_frame_count is None else str(result.video_frame_count),
        "selected_leaf_frames": str(result.selected_leaf_frames),
        "decoded_leaf_frames": str(result.decoded_leaf_frames),
        "leaf_estimates": str(len(result.estimates)),
        "status": result.status,
        "error": result.error,
    }


def write_leaf_qc(
    root: Path,
    rows: list[LeafAreaRow],
    timezone: dt.tzinfo,
    *,
    estimate_interval_minutes: int,
) -> None:
    qc_dir = root / "cropped_by_caterpillar" / "leaf_feeding" / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[LeafAreaRow]] = defaultdict(list)
    for row in rows:
        grouped[row.animal_id].append(row)
    estimate_interval = dt.timedelta(minutes=estimate_interval_minutes)
    for animal_id in ANIMAL_ORDER:
        animal_rows = grouped.get(animal_id, [])
        if not animal_rows:
            continue
        fig, ax = plt.subplots(figsize=(12, 3))
        x_values = [to_plot_local(row.timestamp_utc, timezone) for row in animal_rows]
        y_values = [row.relative_leaf_area for row in animal_rows]
        ax.plot(x_values, y_values, color="#166534", linewidth=1.4)
        for row in animal_rows:
            if row.video_quality_excluded:
                ax.axvspan(
                    to_plot_local(row.timestamp_utc, timezone),
                    to_plot_local(row.timestamp_utc + estimate_interval, timezone),
                    facecolor="#f59e0b",
                    alpha=0.12,
                    zorder=0.1,
                )
            if row.feeding:
                ax.axvspan(
                    to_plot_local(row.timestamp_utc, timezone),
                    to_plot_local(row.timestamp_utc + estimate_interval, timezone),
                    facecolor="#7c3aed",
                    alpha=0.18,
                    zorder=0.2,
                )
        ax.set_title(f"{animal_id} relative leaf area")
        ax.set_ylabel("Relative area")
        ax.set_xlabel(f"Local time - {timezone_label(timezone)}")
        ax.grid(True, axis="y", color="#e5e7eb")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        fig.savefig(qc_dir / f"{animal_id}_leaf_area_qc.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze coarse leaf-area feeding from cropped caterpillar videos.")
    parser.add_argument("root", nargs="?", default=".", help="Dataset root.")
    parser.add_argument("--animals", nargs="*", help="Optional list of animal IDs to analyze.")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--events", help="Optional local CSV path or supported Google Sheets URL for behavior/global events.")
    parser.add_argument("--leaf-estimate-minutes", type=int, default=LEAF_ESTIMATE_INTERVAL_MINUTES)
    parser.add_argument("--burst-duration-seconds", type=int, default=BURST_DURATION_SECONDS)
    parser.add_argument("--burst-step-seconds", type=int, default=BURST_STEP_SECONDS)
    parser.add_argument("--frame-access", choices=["sparse", "sequential"], default="sparse")
    parser.add_argument("--leaf-area-percentile", type=float, default=95.0)
    parser.add_argument("--min-valid-frames", type=int, default=3)
    parser.add_argument("--allowed-gap-minutes", type=int)
    parser.add_argument("--feeding-start-loss-px2", type=float, default=FEEDING_START_LOSS_PX2)
    parser.add_argument("--feeding-continue-loss-px2", type=float, default=FEEDING_CONTINUE_LOSS_PX2)
    parser.add_argument("--feeding-merge-gap-minutes", type=int, default=FEEDING_MERGE_GAP_MINUTES)
    parser.add_argument("--feeding-min-bout-minutes", type=int, default=FEEDING_MIN_BOUT_MINUTES)
    parser.add_argument("--leaf-reset-increase-pct", type=float, default=20.0)
    parser.add_argument("--hue-low", type=int, default=25)
    parser.add_argument("--hue-high", type=int, default=95)
    parser.add_argument("--sat-min", type=int, default=35)
    parser.add_argument("--value-min", type=int, default=25)
    parser.add_argument("--min-component-px", type=int, default=1500)
    parser.add_argument("--morph-kernel", type=int, default=5)
    parser.add_argument("--qc", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        parser.error(f"root must be an existing directory: {root}")
    if args.leaf_estimate_minutes <= 0:
        parser.error("--leaf-estimate-minutes must be > 0")
    timezone = load_timezone(args.timezone)
    animal_filter = set(args.animals) if args.animals else None
    entries = load_manifest_entries(root, animals=animal_filter)
    if not entries:
        parser.error("no cropped manifest rows with copied timestamps were found")

    events_source = args.events
    if events_source is None:
        default_events = root / "behavior_events.csv"
        events_source = str(default_events) if default_events.exists() else None
    try:
        explicit_resets, video_quality_intervals_utc = load_analysis_events(
            root=root,
            source=events_source,
            timezone=timezone,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOG.error("%s", exc)
        return 1

    output_root = root / "cropped_by_caterpillar" / "leaf_feeding"
    output_root.mkdir(parents=True, exist_ok=True)
    all_estimates: list[LeafAreaEstimate] = []
    summary_rows: list[dict[str, str]] = []
    progress_reporter = ProgressReporter(total_clips=len(entries))
    status_counts: dict[str, int] = defaultdict(int)
    source_frames_total = 0
    decoded_leaf_frames_total = 0
    allowed_gap_minutes = (
        args.allowed_gap_minutes
        if args.allowed_gap_minutes is not None
        else args.leaf_estimate_minutes + DEFAULT_ALLOWED_GAP_EXTRA_MINUTES
    )

    for clip_index, entry in enumerate(entries, start=1):
        result = extract_clip_leaf_estimates(
            entry,
            clip_index=clip_index,
            estimate_interval_minutes=args.leaf_estimate_minutes,
            burst_duration_seconds=args.burst_duration_seconds,
            burst_step_seconds=args.burst_step_seconds,
            leaf_area_percentile=args.leaf_area_percentile,
            min_valid_frames=args.min_valid_frames,
            hue_low=args.hue_low,
            hue_high=args.hue_high,
            sat_min=args.sat_min,
            value_min=args.value_min,
            min_component_px=args.min_component_px,
            morph_kernel=args.morph_kernel,
            frame_access=args.frame_access,
            video_quality_intervals_utc=video_quality_intervals_utc,
            progress_reporter=progress_reporter,
        )
        if result.error:
            LOG.error(result.error)
        all_estimates.extend(result.estimates)
        summary_rows.append(summary_row(root, entry, result))
        status_counts[result.status] += 1
        source_frames_total += result.timestamp_rows
        decoded_leaf_frames_total += result.decoded_leaf_frames

    finalized_rows = finalize_leaf_rows(
        all_estimates,
        timezone=timezone,
        leaf_area_percentile=args.leaf_area_percentile,
        estimate_interval_minutes=args.leaf_estimate_minutes,
        allowed_gap_minutes=allowed_gap_minutes,
        start_loss_px2=args.feeding_start_loss_px2,
        continue_loss_px2=args.feeding_continue_loss_px2,
        merge_gap_minutes=args.feeding_merge_gap_minutes,
        min_bout_minutes=args.feeding_min_bout_minutes,
        leaf_reset_increase_pct=args.leaf_reset_increase_pct,
        explicit_resets=explicit_resets,
    )
    feeding_events = feeding_event_dicts(
        finalized_rows,
        timezone,
        estimate_interval_minutes=args.leaf_estimate_minutes,
        start_loss_px2=args.feeding_start_loss_px2,
        continue_loss_px2=args.feeding_continue_loss_px2,
    )
    write_csv(output_root / "leaf_area_timeseries.csv", LEAF_AREA_FIELDS, leaf_rows_to_dicts(finalized_rows, timezone))
    write_csv(output_root / "feeding_events.csv", FEEDING_EVENT_FIELDS, feeding_events)
    write_csv(output_root / "leaf_feeding_summary.csv", SUMMARY_FIELDS, summary_rows)
    if args.qc:
        write_leaf_qc(
            root,
            finalized_rows,
            timezone,
            estimate_interval_minutes=args.leaf_estimate_minutes,
        )

    LOG.info("Leaf feeding analysis complete")
    LOG.info("  clips selected: %d", len(entries))
    LOG.info("  clips analyzed: %d", status_counts.get("computed", 0))
    LOG.info("  frame mismatches: %d", status_counts.get("frame_mismatch", 0))
    LOG.info("  seek failures: %d", status_counts.get("seek_failure", 0))
    LOG.info("  seek mismatches: %d", status_counts.get("seek_mismatch", 0))
    LOG.info("  source frames represented: %d", source_frames_total)
    LOG.info("  leaf frames decoded: %d", decoded_leaf_frames_total)
    LOG.info("  %d-min leaf estimates: %d", args.leaf_estimate_minutes, len(finalized_rows))
    LOG.info("  coarse feeding events: %d", len(feeding_events))
    LOG.info("Wrote leaf area timeseries: %s", output_root / "leaf_area_timeseries.csv")
    LOG.info("Wrote feeding events: %s", output_root / "feeding_events.csv")
    LOG.info("Wrote leaf feeding summary: %s", output_root / "leaf_feeding_summary.csv")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
