#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import logging
import math
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, TextIO

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
    parse_utc_value,
    timezone_label,
    to_plot_local,
)
from extract_motion_energy import ANIMAL_ORDER

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

LOG = logging.getLogger("analyze_leaf_feeding")

LEAF_AREA_FIELDS = [
    "animal_id",
    "clip_key",
    "timestamp_utc",
    "timestamp_local",
    "leaf_epoch",
    "leaf_area_proxy_px",
    "relative_leaf_area",
    "loss_prev_min_pct",
    "feeding_raw",
    "feeding_final",
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
    "decoded_frames",
    "sampled_frames",
    "minute_estimates",
    "status",
    "error",
]
LEAF_RESET_EVENT_NAMES = {"leaf_added", "leaf_replaced", "leaf_change"}
PROGRESS_BAR_WIDTH = 28
PROGRESS_UPDATE_SECONDS = 0.25
ESTIMATE_INTERVAL = dt.timedelta(minutes=1)


@dataclasses.dataclass(frozen=True)
class ManifestEntry:
    animal_id: str
    clip_key: str
    cropped_video: Path
    timestamp_file: Path


@dataclasses.dataclass(frozen=True)
class MinuteLeafEstimate:
    animal_id: str
    clip_key: str
    timestamp_utc: dt.datetime
    leaf_area_proxy_px: float
    n_sampled_frames: int
    n_valid_frames: int
    qc_flag: str = ""


@dataclasses.dataclass(frozen=True)
class LeafAreaRow:
    animal_id: str
    clip_key: str
    timestamp_utc: dt.datetime
    leaf_epoch: int
    leaf_area_proxy_px: float
    relative_leaf_area: float
    loss_prev_min_pct: Optional[float]
    feeding_raw: bool
    feeding_final: bool
    n_sampled_frames: int
    n_valid_frames: int
    qc_flag: str = ""


@dataclasses.dataclass(frozen=True)
class ClipFeedingResult:
    estimates: list[MinuteLeafEstimate]
    status: str
    timestamp_rows: int
    decoded_frames: Optional[int]
    sampled_frames: int
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
    decoded_frames: int,
    total_frames: int,
    minute_bins: int,
    status: str,
) -> str:
    bar = format_progress_bar(decoded_frames, total_frames)
    return (
        f"{bar} clip {clip_index}/{total_clips} "
        f"{entry.animal_id} {entry.clip_key} "
        f"{decoded_frames}/{total_frames} frames "
        f"{minute_bins} bins "
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
    current_total_frames: int = 0
    last_render_time: float = 0.0

    def start_clip(self, clip_index: int, entry: ManifestEntry, total_frames: int) -> None:
        self.current_clip_index = clip_index
        self.current_entry = entry
        self.current_total_frames = max(total_frames, 0)
        self.last_render_time = 0.0
        LOG.info(
            "Processing clip %d/%d: %s %s (%d timestamp rows)",
            clip_index,
            self.total_clips,
            entry.animal_id,
            entry.clip_key,
            total_frames,
        )
        self.update(decoded_frames=0, minute_bins=0, status="starting", force=True)

    def update(
        self,
        *,
        decoded_frames: int,
        minute_bins: int,
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
            decoded_frames=decoded_frames,
            total_frames=self.current_total_frames,
            minute_bins=minute_bins,
            status=status,
        )
        self.stream.write(f"\r{line}")
        self.stream.flush()
        self.last_render_time = now

    def finish_clip(
        self,
        *,
        decoded_frames: int,
        minute_bins: int,
        status: str,
    ) -> None:
        if self.current_entry is not None:
            LOG.info(
                "Finished clip %d/%d: %s %s -> %s (%d minute bins)",
                self.current_clip_index,
                self.total_clips,
                self.current_entry.animal_id,
                self.current_entry.clip_key,
                status,
                minute_bins,
            )
        self.update(decoded_frames=decoded_frames, minute_bins=minute_bins, status=status, force=True)
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


def floor_utc_minute(value: dt.datetime) -> dt.datetime:
    return value.astimezone(UTC).replace(second=0, microsecond=0)


def minute_targets(
    timestamps: list[dt.datetime],
    *,
    burst_step_seconds: int,
) -> list[dt.datetime]:
    if not timestamps:
        return []
    start_minute = floor_utc_minute(timestamps[0])
    end_minute = floor_utc_minute(timestamps[-1])
    targets: list[dt.datetime] = []
    current = start_minute
    while current <= end_minute:
        for second in range(0, 60, burst_step_seconds):
            targets.append(current + dt.timedelta(seconds=second))
        current += dt.timedelta(minutes=1)
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
    loss_values: Iterable[Optional[float]],
    *,
    start_loss_pct: float,
    continue_loss_pct: float,
) -> list[bool]:
    feeding = False
    output: list[bool] = []
    for value in loss_values:
        if value is None or math.isnan(value):
            feeding = False
            output.append(False)
            continue
        if not feeding:
            feeding = value >= start_loss_pct
        else:
            feeding = value >= continue_loss_pct
        output.append(feeding)
    return output


def merge_short_gaps(flags: list[bool], *, max_gap_minutes: int) -> list[bool]:
    if max_gap_minutes <= 0:
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
        if gap_start > 0 and gap_end < len(merged) and merged[gap_start - 1] and merged[gap_end] and (gap_end - gap_start) <= max_gap_minutes:
            for j in range(gap_start, gap_end):
                merged[j] = True
    return merged


def remove_short_bouts(flags: list[bool], *, min_bout_minutes: int) -> list[bool]:
    if min_bout_minutes <= 1:
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
        if bout_end - bout_start < min_bout_minutes:
            for j in range(bout_start, bout_end):
                cleaned[j] = False
    return cleaned


def relative_change_percent(previous: float, current: float) -> float:
    if previous <= 0:
        return 0.0
    return 100.0 * (current - previous) / previous


def extract_clip_leaf_estimates(
    entry: ManifestEntry,
    *,
    clip_index: int = 0,
    timezone: dt.tzinfo,
    burst_step_seconds: int,
    leaf_area_percentile: float,
    min_valid_frames: int,
    hue_low: int,
    hue_high: int,
    sat_min: int,
    value_min: int,
    min_component_px: int,
    morph_kernel: int,
    progress_reporter: Optional[ProgressReporter] = None,
) -> ClipFeedingResult:
    timestamps = load_timestamp_series(entry.timestamp_file)
    target_times = minute_targets(timestamps, burst_step_seconds=burst_step_seconds)
    timestamp_rows = len(timestamps)
    if progress_reporter is not None:
        progress_reporter.start_clip(clip_index, entry, timestamp_rows)
    capture = cv2.VideoCapture(str(entry.cropped_video))
    if not capture.isOpened():
        if progress_reporter is not None:
            progress_reporter.finish_clip(decoded_frames=0, minute_bins=0, status="error")
        return ClipFeedingResult([], "failed", timestamp_rows, None, 0, f"could not open {entry.cropped_video}")

    grouped_areas: dict[dt.datetime, list[float]] = defaultdict(list)
    grouped_sampled: dict[dt.datetime, int] = defaultdict(int)
    grouped_valid: dict[dt.datetime, int] = defaultdict(int)
    decoded_frames = 0
    sampled_frames = 0
    target_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if decoded_frames >= len(timestamps):
                break
            timestamp = timestamps[decoded_frames]
            should_sample = False
            while target_index < len(target_times) and timestamp >= target_times[target_index]:
                should_sample = True
                target_index += 1
            if should_sample:
                sampled_frames += 1
                bucket = floor_utc_minute(timestamp)
                grouped_sampled[bucket] += 1
                area, _mask = segment_leaf_area(
                    frame,
                    hue_low=hue_low,
                    hue_high=hue_high,
                    sat_min=sat_min,
                    value_min=value_min,
                    min_component_px=min_component_px,
                    morph_kernel=morph_kernel,
                )
                if area > 0:
                    grouped_areas[bucket].append(area)
                    grouped_valid[bucket] += 1
            decoded_frames += 1
            if progress_reporter is not None:
                progress_reporter.update(
                    decoded_frames=decoded_frames,
                    minute_bins=len(grouped_sampled),
                    status="analyzing",
                )
    finally:
        capture.release()

    if decoded_frames != timestamp_rows:
        if progress_reporter is not None:
            progress_reporter.finish_clip(
                decoded_frames=decoded_frames,
                minute_bins=len(grouped_sampled),
                status="error",
            )
        return ClipFeedingResult(
            [],
            "frame_mismatch",
            timestamp_rows,
            decoded_frames,
            sampled_frames,
            f"frame count mismatch for {entry.cropped_video}: decoded {decoded_frames}, timestamps {timestamp_rows}",
        )

    estimates: list[MinuteLeafEstimate] = []
    for bucket in sorted(grouped_sampled):
        areas = grouped_areas.get(bucket, [])
        valid_count = grouped_valid.get(bucket, 0)
        qc_flag = ""
        if valid_count < min_valid_frames or not areas:
            qc_flag = "too_few_valid_frames"
            continue
        estimate = float(np.percentile(np.array(areas, dtype=np.float64), leaf_area_percentile))
        estimates.append(
            MinuteLeafEstimate(
                animal_id=entry.animal_id,
                clip_key=entry.clip_key,
                timestamp_utc=bucket + dt.timedelta(minutes=1),
                leaf_area_proxy_px=estimate,
                n_sampled_frames=grouped_sampled[bucket],
                n_valid_frames=valid_count,
                qc_flag=qc_flag,
            )
        )
    if progress_reporter is not None:
        progress_reporter.finish_clip(
            decoded_frames=decoded_frames,
            minute_bins=len(estimates),
            status="done",
        )
    return ClipFeedingResult(estimates, "computed", timestamp_rows, decoded_frames, sampled_frames)


def load_leaf_reset_events(path: Optional[Path], tz: dt.tzinfo) -> dict[str, list[dt.datetime]]:
    if path is None or not path.exists():
        return {}
    from plot_recording_timeline import load_event_tables

    events, _global = load_event_tables(path, tz)
    resets: dict[str, list[dt.datetime]] = defaultdict(list)
    for event in events:
        if event.animal_id not in ANIMAL_ORDER:
            continue
        if (event.event or "").strip().lower() in LEAF_RESET_EVENT_NAMES:
            resets[event.animal_id].append(event.start_local.astimezone(UTC))
    for animal_id in list(resets):
        resets[animal_id].sort()
    return dict(resets)


def finalize_leaf_rows(
    estimates: list[MinuteLeafEstimate],
    *,
    timezone: dt.tzinfo,
    allowed_gap_minutes: int,
    start_loss_pct: float,
    continue_loss_pct: float,
    merge_gap_minutes: int,
    min_bout_minutes: int,
    leaf_reset_increase_pct: float,
    explicit_resets: Optional[dict[str, list[dt.datetime]]] = None,
) -> list[LeafAreaRow]:
    consolidated_estimates = consolidate_minute_estimates(estimates)
    explicit_resets = explicit_resets or {}
    grouped: dict[str, list[MinuteLeafEstimate]] = defaultdict(list)
    for estimate in consolidated_estimates:
        grouped[estimate.animal_id].append(estimate)
    rows: list[LeafAreaRow] = []
    allowed_gap = dt.timedelta(minutes=allowed_gap_minutes)
    for animal_id in ANIMAL_ORDER:
        animal_rows = sorted(grouped.get(animal_id, []), key=lambda row: (row.timestamp_utc, row.clip_key))
        if not animal_rows:
            continue
        current_epoch = 1
        previous: Optional[MinuteLeafEstimate] = None
        baseline_area: Optional[float] = None
        reset_index = 0
        reset_times = explicit_resets.get(animal_id, [])
        draft_rows: list[LeafAreaRow] = []
        for estimate in animal_rows:
            while reset_index < len(reset_times) and reset_times[reset_index] <= estimate.timestamp_utc:
                current_epoch += 1
                baseline_area = None
                previous = None
                reset_index += 1
            if baseline_area is None:
                baseline_area = estimate.leaf_area_proxy_px
            loss_prev_min_pct: Optional[float] = None
            if previous is not None:
                if estimate.timestamp_utc - previous.timestamp_utc > allowed_gap:
                    current_epoch += 1
                    baseline_area = estimate.leaf_area_proxy_px
                    previous = None
                else:
                    increase_pct = relative_change_percent(previous.leaf_area_proxy_px, estimate.leaf_area_proxy_px)
                    if increase_pct > leaf_reset_increase_pct:
                        current_epoch += 1
                        baseline_area = estimate.leaf_area_proxy_px
                        previous = None
                    else:
                        loss_prev_min_pct = 100.0 * (
                            previous.leaf_area_proxy_px - estimate.leaf_area_proxy_px
                        ) / max(previous.leaf_area_proxy_px, 1.0)
            relative_area = estimate.leaf_area_proxy_px / max(baseline_area or estimate.leaf_area_proxy_px, 1.0)
            draft_rows.append(
                LeafAreaRow(
                    animal_id=estimate.animal_id,
                    clip_key=estimate.clip_key,
                    timestamp_utc=estimate.timestamp_utc,
                    leaf_epoch=current_epoch,
                    leaf_area_proxy_px=estimate.leaf_area_proxy_px,
                    relative_leaf_area=relative_area,
                    loss_prev_min_pct=loss_prev_min_pct,
                    feeding_raw=False,
                    feeding_final=False,
                    n_sampled_frames=estimate.n_sampled_frames,
                    n_valid_frames=estimate.n_valid_frames,
                    qc_flag=estimate.qc_flag,
                )
            )
            previous = estimate

        by_epoch: dict[int, list[LeafAreaRow]] = defaultdict(list)
        for row in draft_rows:
            by_epoch[row.leaf_epoch].append(row)
        for epoch in sorted(by_epoch):
            epoch_rows = by_epoch[epoch]
            losses = [row.loss_prev_min_pct for row in epoch_rows]
            feeding_raw = classify_feeding_hysteresis(
                losses,
                start_loss_pct=start_loss_pct,
                continue_loss_pct=continue_loss_pct,
            )
            feeding_final = remove_short_bouts(
                merge_short_gaps(feeding_raw, max_gap_minutes=merge_gap_minutes),
                min_bout_minutes=min_bout_minutes,
            )
            for row, raw, final in zip(epoch_rows, feeding_raw, feeding_final):
                rows.append(dataclasses.replace(row, feeding_raw=raw, feeding_final=final))
    rows.sort(key=lambda row: (ANIMAL_ORDER.index(row.animal_id), row.timestamp_utc))
    return rows


def consolidate_minute_estimates(estimates: list[MinuteLeafEstimate]) -> list[MinuteLeafEstimate]:
    grouped: dict[tuple[str, dt.datetime], list[MinuteLeafEstimate]] = defaultdict(list)
    for estimate in estimates:
        grouped[(estimate.animal_id, estimate.timestamp_utc)].append(estimate)

    consolidated: list[MinuteLeafEstimate] = []
    for _key, duplicates in sorted(grouped.items(), key=lambda item: (ANIMAL_ORDER.index(item[0][0]), item[0][1])):
        if len(duplicates) == 1:
            consolidated.append(duplicates[0])
            continue
        chosen = max(duplicates, key=lambda row: (row.leaf_area_proxy_px, row.timestamp_utc, row.clip_key))
        consolidated.append(
            MinuteLeafEstimate(
                animal_id=chosen.animal_id,
                clip_key=chosen.clip_key,
                timestamp_utc=chosen.timestamp_utc,
                leaf_area_proxy_px=max(row.leaf_area_proxy_px for row in duplicates),
                n_sampled_frames=sum(row.n_sampled_frames for row in duplicates),
                n_valid_frames=sum(row.n_valid_frames for row in duplicates),
                qc_flag=";".join(sorted({row.qc_flag for row in duplicates if row.qc_flag})),
            )
        )
    return consolidated


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
                "loss_prev_min_pct": "" if row.loss_prev_min_pct is None else f"{row.loss_prev_min_pct:.6f}",
                "feeding_raw": "TRUE" if row.feeding_raw else "FALSE",
                "feeding_final": "TRUE" if row.feeding_final else "FALSE",
                "n_sampled_frames": str(row.n_sampled_frames),
                "n_valid_frames": str(row.n_valid_frames),
                "qc_flag": row.qc_flag,
            }
        )
    return serialized


def feeding_event_dicts(rows: list[LeafAreaRow], timezone: dt.tzinfo, *, sample_minutes: int) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    grouped: dict[str, list[LeafAreaRow]] = defaultdict(list)
    for row in rows:
        grouped[row.animal_id].append(row)
    for animal_id, animal_rows in grouped.items():
        active_start: Optional[LeafAreaRow] = None
        previous: Optional[LeafAreaRow] = None
        for row in animal_rows:
            contiguous = (
                previous is not None
                and row.leaf_epoch == previous.leaf_epoch
                and row.timestamp_utc - previous.timestamp_utc <= dt.timedelta(minutes=sample_minutes + 1)
            )
            if row.feeding_final and active_start is None:
                active_start = row
            elif (not row.feeding_final or not contiguous) and active_start is not None and previous is not None:
                start_utc = active_start.timestamp_utc - ESTIMATE_INTERVAL
                end_utc = previous.timestamp_utc
                events.append(
                    {
                        "animal_id": animal_id,
                        "start_utc": format_utc(start_utc),
                        "end_utc": format_utc(end_utc),
                        "start_local": format_local(start_utc, timezone),
                        "end_local": format_local(end_utc, timezone),
                        "event": "feeding",
                        "kind": "feeding",
                        "notes": "automatic leaf-area detection",
                    }
                )
                active_start = row if row.feeding_final else None
            previous = row
        if active_start is not None and previous is not None:
            start_utc = active_start.timestamp_utc - ESTIMATE_INTERVAL
            end_utc = previous.timestamp_utc
            events.append(
                {
                    "animal_id": animal_id,
                    "start_utc": format_utc(start_utc),
                    "end_utc": format_utc(end_utc),
                    "start_local": format_local(start_utc, timezone),
                    "end_local": format_local(end_utc, timezone),
                    "event": "feeding",
                    "kind": "feeding",
                    "notes": "automatic leaf-area detection",
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
        "decoded_frames": "" if result.decoded_frames is None else str(result.decoded_frames),
        "sampled_frames": str(result.sampled_frames),
        "minute_estimates": str(len(result.estimates)),
        "status": result.status,
        "error": result.error,
    }


def write_leaf_qc(root: Path, rows: list[LeafAreaRow], timezone: dt.tzinfo) -> None:
    qc_dir = root / "cropped_by_caterpillar" / "leaf_feeding" / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[LeafAreaRow]] = defaultdict(list)
    for row in rows:
        grouped[row.animal_id].append(row)
    for animal_id in ANIMAL_ORDER:
        animal_rows = grouped.get(animal_id, [])
        if not animal_rows:
            continue
        fig, ax = plt.subplots(figsize=(12, 3))
        x_values = [to_plot_local(row.timestamp_utc, timezone) for row in animal_rows]
        y_values = [row.relative_leaf_area for row in animal_rows]
        ax.plot(x_values, y_values, color="#166534", linewidth=1.4)
        for row in animal_rows:
            if row.feeding_final:
                ax.axvspan(
                    to_plot_local(row.timestamp_utc, timezone),
                    to_plot_local(row.timestamp_utc + dt.timedelta(minutes=1), timezone),
                    facecolor="#f59e0b",
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
    parser = argparse.ArgumentParser(description="Analyze leaf-area feeding from cropped caterpillar videos.")
    parser.add_argument("root", nargs="?", default=".", help="Dataset root.")
    parser.add_argument("--animals", nargs="*", help="Optional list of animal IDs to analyze.")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--sample-minutes", type=int, default=1)
    parser.add_argument("--burst-step-seconds", type=int, default=10)
    parser.add_argument("--leaf-area-percentile", type=float, default=95.0)
    parser.add_argument("--min-valid-frames", type=int, default=3)
    parser.add_argument("--allowed-gap-minutes", type=int, default=2)
    parser.add_argument("--start-loss-pct", type=float, default=2.0)
    parser.add_argument("--continue-loss-pct", type=float, default=1.0)
    parser.add_argument("--merge-gap-minutes", type=int, default=2)
    parser.add_argument("--min-bout-minutes", type=int, default=2)
    parser.add_argument("--leaf-reset-increase-pct", type=float, default=20.0)
    parser.add_argument("--leaf-events", type=Path, help="Optional local CSV of manual leaf reset events.")
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
    if args.sample_minutes != 1:
        parser.error("--sample-minutes currently supports only 1-minute estimates in v1")
    timezone = load_timezone(args.timezone)
    animal_filter = set(args.animals) if args.animals else None
    entries = load_manifest_entries(root, animals=animal_filter)
    if not entries:
        parser.error("no cropped manifest rows with copied timestamps were found")

    output_root = root / "cropped_by_caterpillar" / "leaf_feeding"
    output_root.mkdir(parents=True, exist_ok=True)
    all_estimates: list[MinuteLeafEstimate] = []
    summary_rows: list[dict[str, str]] = []
    progress_reporter = ProgressReporter(total_clips=len(entries))
    status_counts: dict[str, int] = defaultdict(int)
    for clip_index, entry in enumerate(entries, start=1):
        result = extract_clip_leaf_estimates(
            entry,
            clip_index=clip_index,
            timezone=timezone,
            burst_step_seconds=args.burst_step_seconds,
            leaf_area_percentile=args.leaf_area_percentile,
            min_valid_frames=args.min_valid_frames,
            hue_low=args.hue_low,
            hue_high=args.hue_high,
            sat_min=args.sat_min,
            value_min=args.value_min,
            min_component_px=args.min_component_px,
            morph_kernel=args.morph_kernel,
            progress_reporter=progress_reporter,
        )
        if result.error:
            LOG.error(result.error)
        all_estimates.extend(result.estimates)
        summary_rows.append(summary_row(root, entry, result))
        status_counts[result.status] += 1

    explicit_resets = load_leaf_reset_events(args.leaf_events, timezone)
    finalized_rows = finalize_leaf_rows(
        all_estimates,
        timezone=timezone,
        allowed_gap_minutes=args.allowed_gap_minutes,
        start_loss_pct=args.start_loss_pct,
        continue_loss_pct=args.continue_loss_pct,
        merge_gap_minutes=args.merge_gap_minutes,
        min_bout_minutes=args.min_bout_minutes,
        leaf_reset_increase_pct=args.leaf_reset_increase_pct,
        explicit_resets=explicit_resets,
    )
    feeding_events = feeding_event_dicts(finalized_rows, timezone, sample_minutes=args.sample_minutes)
    write_csv(output_root / "leaf_area_timeseries.csv", LEAF_AREA_FIELDS, leaf_rows_to_dicts(finalized_rows, timezone))
    write_csv(output_root / "feeding_events.csv", FEEDING_EVENT_FIELDS, feeding_events)
    write_csv(output_root / "leaf_feeding_summary.csv", SUMMARY_FIELDS, summary_rows)
    if args.qc:
        write_leaf_qc(root, finalized_rows, timezone)
    LOG.info("Leaf feeding analysis complete")
    LOG.info("  clips selected: %d", len(entries))
    LOG.info("  clips analyzed: %d", status_counts.get("computed", 0))
    LOG.info("  frame mismatches: %d", status_counts.get("frame_mismatch", 0))
    LOG.info("  failed opens: %d", status_counts.get("failed", 0))
    LOG.info("  minute estimates: %d", len(finalized_rows))
    LOG.info("  feeding events: %d", len(feeding_events))
    LOG.info("Wrote leaf area timeseries: %s", output_root / "leaf_area_timeseries.csv")
    LOG.info("Wrote feeding events: %s", output_root / "feeding_events.csv")
    LOG.info("Wrote leaf feeding summary: %s", output_root / "leaf_feeding_summary.csv")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
