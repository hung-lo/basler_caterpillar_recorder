#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import gzip
import logging
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, TextIO

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from plot_recording_timeline import (
    ANIMAL_ORDER,
    DEFAULT_TIMEZONE,
    UTC,
    format_local,
    format_utc,
    load_timezone,
    open_text_file,
    parse_timestamp_row,
    parse_utc_value,
    timezone_label,
)

LOG = logging.getLogger("extract_motion_energy")

TRACE_FIELDS = [
    "animal_id",
    "clip_key",
    "frame_index_start",
    "frame_index_end",
    "start_utc",
    "end_utc",
    "start_local",
    "end_local",
    "motion_energy",
    "motion_mean",
    "global_luminance_shift",
]
THRESHOLD_FIELDS = [
    "animal_id",
    "threshold",
    "threshold_source",
    "median",
    "mad",
    "p90",
    "p95",
    "p99",
    "n_windows",
]
STATE_FIELDS = [
    "animal_id",
    "clip_key",
    "start_utc",
    "end_utc",
    "start_local",
    "end_local",
    "state",
    "threshold",
    "threshold_source",
    "mean_motion_energy",
    "peak_motion_energy",
    "n_windows",
]
SUMMARY_FIELDS = [
    "animal_id",
    "clip_key",
    "cropped_video",
    "timestamp_file",
    "timestamp_rows",
    "decoded_frames",
    "sample_windows",
    "status",
    "error",
]
TRACE_HEADER_REQUIRED = set(TRACE_FIELDS)
MOTION_WIDTH = 96
MOTION_HEIGHT = 96
BLUR_KERNEL = (5, 5)
TOP_FRACTION = 0.05
ROBUST_SIGMA_SCALE = 1.4826
THRESHOLD_SIGMA_MULTIPLIER = 6.0
DEFAULT_SAMPLE_HZ = 1.0
DEFAULT_MERGE_GAP_S = 2.0
DEFAULT_MIN_MOBILE_S = 1.0
CONTIGUITY_TOLERANCE_SECONDS = 0.5
DISPLAY_MAX_POINTS = 1500
PROGRESS_BAR_WIDTH = 28
PROGRESS_UPDATE_SECONDS = 0.25


@dataclasses.dataclass(frozen=True)
class ManifestEntry:
    animal_id: str
    clip_key: str
    cropped_video: Path
    timestamp_file: Path


@dataclasses.dataclass(frozen=True)
class MotionTraceRow:
    animal_id: str
    clip_key: str
    frame_index_start: int
    frame_index_end: int
    start_utc: dt.datetime
    end_utc: dt.datetime
    motion_energy: float
    motion_mean: float
    global_luminance_shift: float


@dataclasses.dataclass(frozen=True)
class MotionThreshold:
    animal_id: str
    threshold: Optional[float]
    threshold_source: str
    median: Optional[float]
    mad: Optional[float]
    p90: Optional[float]
    p95: Optional[float]
    p99: Optional[float]
    n_windows: int


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
class ExtractionResult:
    trace_path: Path
    status: str
    timestamp_rows: int
    decoded_frames: Optional[int]
    sample_windows: int
    error: str = ""


@dataclasses.dataclass
class _StateAccumulator:
    animal_id: str
    clip_key: str
    start_utc: dt.datetime
    end_utc: dt.datetime
    state: str
    threshold: float
    threshold_source: str
    motion_sum: float
    peak_motion: float
    n_windows: int

    @property
    def duration_s(self) -> float:
        return max((self.end_utc - self.start_utc).total_seconds(), 0.0)

    def as_state(self) -> MotionState:
        return MotionState(
            animal_id=self.animal_id,
            clip_key=self.clip_key,
            start_utc=self.start_utc,
            end_utc=self.end_utc,
            state=self.state,
            threshold=self.threshold,
            threshold_source=self.threshold_source,
            mean_motion_energy=self.motion_sum / max(self.n_windows, 1),
            peak_motion_energy=self.peak_motion,
            n_windows=self.n_windows,
        )


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
    sample_windows: int,
    status: str,
) -> str:
    bar = format_progress_bar(decoded_frames, total_frames)
    return (
        f"{bar} clip {clip_index}/{total_clips} "
        f"{entry.animal_id} {entry.clip_key} "
        f"{decoded_frames}/{total_frames} frames "
        f"{sample_windows} windows "
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
        self.update(decoded_frames=0, sample_windows=0, status="starting", force=True)

    def update(
        self,
        *,
        decoded_frames: int,
        sample_windows: int,
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
            sample_windows=sample_windows,
            status=status,
        )
        self.stream.write(f"\r{line}")
        self.stream.flush()
        self.last_render_time = now

    def finish_clip(
        self,
        *,
        decoded_frames: int,
        sample_windows: int,
        status: str,
    ) -> None:
        if self.current_entry is not None:
            LOG.info(
                "Finished clip %d/%d: %s %s -> %s (%d windows)",
                self.current_clip_index,
                self.total_clips,
                self.current_entry.animal_id,
                self.current_entry.clip_key,
                status,
                sample_windows,
            )
        self.update(
            decoded_frames=decoded_frames,
            sample_windows=sample_windows,
            status=status,
            force=True,
        )
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()


@dataclasses.dataclass(frozen=True)
class CachedTraceInventory:
    trace_rows_by_animal: dict[str, list[MotionTraceRow]]
    trace_files_by_animal: dict[str, int]
    valid_windows_by_animal: dict[str, int]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract low-resolution motion energy from cropped_by_caterpillar/*.mp4, "
            "cache per-crop traces, and generate motion-derived mobile/immobile states."
        )
    )
    parser.add_argument("root", nargs="?", default=".", help="Dataset root.")
    parser.add_argument("--animals", nargs="*", help="Limit processing to specific animal IDs.")
    parser.add_argument(
        "--limit-clips",
        type=int,
        help="Limit the number of cropped video clips processed after filtering.",
    )
    parser.add_argument(
        "--sample-hz",
        type=float,
        default=DEFAULT_SAMPLE_HZ,
        help=f"Sampling rate for motion analysis (default: {DEFAULT_SAMPLE_HZ}).",
    )
    parser.add_argument(
        "--motion-width",
        type=int,
        default=MOTION_WIDTH,
        help=f"Width of the low-resolution motion image (default: {MOTION_WIDTH}).",
    )
    parser.add_argument(
        "--motion-height",
        type=int,
        default=MOTION_HEIGHT,
        help=f"Height of the low-resolution motion image (default: {MOTION_HEIGHT}).",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=TOP_FRACTION,
        help=f"Top residual-pixel fraction used for the motion-energy score (default: {TOP_FRACTION}).",
    )
    parser.add_argument(
        "--merge-gap-s",
        type=float,
        default=DEFAULT_MERGE_GAP_S,
        help=f"Merge short immobile gaps between mobile bouts up to this duration (default: {DEFAULT_MERGE_GAP_S}).",
    )
    parser.add_argument(
        "--min-mobile-s",
        type=float,
        default=DEFAULT_MIN_MOBILE_S,
        help=f"Minimum duration for a mobile bout to remain mobile (default: {DEFAULT_MIN_MOBILE_S}).",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help=f"Timezone used for convenience local timestamps (default: {DEFAULT_TIMEZONE}).",
    )
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Reuse cached traces and thresholds without decoding MP4 files again.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute cached motion traces even when valid trace files already exist.",
    )
    parser.add_argument(
        "--reset-thresholds",
        action="store_true",
        help="Regenerate motion_thresholds.csv from cached traces.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Optional temporary global threshold override used for classification.",
    )
    return parser


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def open_csv_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def write_gzip_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with gzip.open(temp_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temp_path.replace(path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temp_path.replace(path)


def load_manifest_entries(
    root: Path,
    *,
    animals: Optional[set[str]],
    limit_clips: Optional[int],
) -> list[ManifestEntry]:
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
    if limit_clips is not None:
        entries = entries[: max(limit_clips, 0)]
    return entries


def load_timestamp_series(path: Path) -> list[dt.datetime]:
    timestamps: list[dt.datetime] = []
    with open_text_file(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header in {path}")
        for row_index, row in enumerate(reader, start=2):
            try:
                timestamps.append(parse_timestamp_row(row))
            except Exception as exc:
                raise ValueError(f"row {row_index}: {exc}") from exc
    if not timestamps:
        raise ValueError(f"no timestamp rows in {path}")
    return timestamps


def preprocess_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, BLUR_KERNEL, 0)
    return blurred.astype(np.float32, copy=False)


def compute_motion_metrics(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    *,
    top_fraction: float,
) -> tuple[float, float, float]:
    diff = current_frame - previous_frame
    global_shift = float(np.median(diff))
    residual = diff - global_shift
    abs_residual = np.abs(residual)
    motion_mean = float(abs_residual.mean())
    flat = abs_residual.ravel()
    k = max(1, int(round(top_fraction * flat.size)))
    top = np.partition(flat, flat.size - k)[-k:]
    motion_energy = float(top.mean())
    return motion_energy, motion_mean, global_shift


def trace_path_for_entry(root: Path, entry: ManifestEntry) -> Path:
    trace_dir = root / "cropped_by_caterpillar" / "motion_energy" / "traces"
    return trace_dir / f"{entry.cropped_video.stem}.motion.csv.gz"


def trace_rows_to_dicts(
    rows: list[MotionTraceRow],
    *,
    timezone: dt.tzinfo,
) -> list[dict[str, str]]:
    serialized: list[dict[str, str]] = []
    for row in rows:
        serialized.append(
            {
                "animal_id": row.animal_id,
                "clip_key": row.clip_key,
                "frame_index_start": str(row.frame_index_start),
                "frame_index_end": str(row.frame_index_end),
                "start_utc": format_utc(row.start_utc),
                "end_utc": format_utc(row.end_utc),
                "start_local": format_local(row.start_utc, timezone),
                "end_local": format_local(row.end_utc, timezone),
                "motion_energy": f"{row.motion_energy:.6f}",
                "motion_mean": f"{row.motion_mean:.6f}",
                "global_luminance_shift": f"{row.global_luminance_shift:.6f}",
            }
        )
    return serialized


def load_motion_trace_rows(path: Path) -> list[MotionTraceRow]:
    rows: list[MotionTraceRow] = []
    with open_csv_text(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not TRACE_HEADER_REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"invalid motion trace header in {path}")
        for row_index, row in enumerate(reader, start=2):
            try:
                rows.append(
                    MotionTraceRow(
                        animal_id=str(row.get("animal_id") or "").strip(),
                        clip_key=str(row.get("clip_key") or "").strip(),
                        frame_index_start=int(str(row.get("frame_index_start") or "").strip()),
                        frame_index_end=int(str(row.get("frame_index_end") or "").strip()),
                        start_utc=parse_utc_value(row.get("start_utc")),
                        end_utc=parse_utc_value(row.get("end_utc")),
                        motion_energy=float(str(row.get("motion_energy") or "").strip()),
                        motion_mean=float(str(row.get("motion_mean") or "").strip()),
                        global_luminance_shift=float(
                            str(row.get("global_luminance_shift") or "").strip()
                        ),
                    )
                )
            except Exception as exc:
                raise ValueError(f"invalid motion trace row {row_index} in {path}: {exc}") from exc
    return rows


def valid_cached_trace(path: Path, entry: ManifestEntry) -> bool:
    if not path.exists():
        return False
    try:
        rows = load_motion_trace_rows(path)
    except Exception as exc:
        LOG.warning("Ignoring invalid cached trace %s: %s", path, exc)
        return False
    return all(row.animal_id == entry.animal_id and row.clip_key == entry.clip_key for row in rows)


def extract_trace_for_entry(
    root: Path,
    entry: ManifestEntry,
    *,
    clip_index: int,
    timezone: dt.tzinfo,
    sample_hz: float,
    motion_width: int,
    motion_height: int,
    top_fraction: float,
    force: bool,
    progress_reporter: Optional[ProgressReporter] = None,
) -> ExtractionResult:
    trace_path = trace_path_for_entry(root, entry)
    timestamps = load_timestamp_series(entry.timestamp_file)
    timestamp_rows = len(timestamps)
    if progress_reporter is not None:
        progress_reporter.start_clip(clip_index, entry, timestamp_rows)

    if valid_cached_trace(trace_path, entry) and not force:
        sample_windows = len(load_motion_trace_rows(trace_path))
        if progress_reporter is not None:
            progress_reporter.finish_clip(
                decoded_frames=timestamp_rows,
                sample_windows=sample_windows,
                status="cached",
            )
        return ExtractionResult(
            trace_path=trace_path,
            status="cached",
            timestamp_rows=timestamp_rows,
            decoded_frames=timestamp_rows,
            sample_windows=sample_windows,
        )

    capture = cv2.VideoCapture(str(entry.cropped_video))
    if not capture.isOpened():
        if progress_reporter is not None:
            progress_reporter.finish_clip(
                decoded_frames=0,
                sample_windows=0,
                status="failed",
            )
        return ExtractionResult(
            trace_path=trace_path,
            status="failed",
            timestamp_rows=timestamp_rows,
            decoded_frames=None,
            sample_windows=0,
            error=f"could not open cropped video: {entry.cropped_video}",
        )

    sample_period_s = 1.0 / sample_hz
    sample_period = dt.timedelta(seconds=sample_period_s)
    max_gap_s = sample_period_s * 2.5
    next_sample_utc = timestamps[0]
    decoded_frames = 0
    previous_sample: Optional[tuple[int, dt.datetime, np.ndarray]] = None
    trace_rows: list[MotionTraceRow] = []

    try:
        while True:
            current_timestamp = timestamps[decoded_frames] if decoded_frames < len(timestamps) else None
            should_process = current_timestamp is not None and current_timestamp >= next_sample_utc
            if should_process:
                ok, frame = capture.read()
            else:
                ok = capture.grab()
                frame = None
            if not ok:
                break

            if should_process and frame is not None and current_timestamp is not None:
                processed = preprocess_frame(frame, motion_width, motion_height)
                if previous_sample is not None:
                    previous_index, previous_timestamp, previous_frame = previous_sample
                    gap_s = (current_timestamp - previous_timestamp).total_seconds()
                    if 0.0 < gap_s <= max_gap_s:
                        motion_energy, motion_mean, global_shift = compute_motion_metrics(
                            previous_frame,
                            processed,
                            top_fraction=top_fraction,
                        )
                        trace_rows.append(
                            MotionTraceRow(
                                animal_id=entry.animal_id,
                                clip_key=entry.clip_key,
                                frame_index_start=previous_index,
                                frame_index_end=decoded_frames,
                                start_utc=previous_timestamp,
                                end_utc=current_timestamp,
                                motion_energy=motion_energy,
                                motion_mean=motion_mean,
                                global_luminance_shift=global_shift,
                            )
                        )
                previous_sample = (decoded_frames, current_timestamp, processed)
                next_sample_utc = current_timestamp + sample_period

            decoded_frames += 1
            if progress_reporter is not None:
                progress_reporter.update(
                    decoded_frames=decoded_frames,
                    sample_windows=len(trace_rows),
                    status="processing",
                )
    finally:
        capture.release()

    if decoded_frames != timestamp_rows:
        if progress_reporter is not None:
            progress_reporter.finish_clip(
                decoded_frames=decoded_frames,
                sample_windows=len(trace_rows),
                status="frame_mismatch",
            )
        return ExtractionResult(
            trace_path=trace_path,
            status="frame_mismatch",
            timestamp_rows=timestamp_rows,
            decoded_frames=decoded_frames,
            sample_windows=len(trace_rows),
            error=(
                f"frame count mismatch for {entry.cropped_video}: "
                f"decoded {decoded_frames}, timestamps {timestamp_rows}"
            ),
        )

    write_gzip_csv(
        trace_path,
        TRACE_FIELDS,
        trace_rows_to_dicts(trace_rows, timezone=timezone),
    )
    if progress_reporter is not None:
        progress_reporter.finish_clip(
            decoded_frames=decoded_frames,
            sample_windows=len(trace_rows),
            status="computed",
        )
    return ExtractionResult(
        trace_path=trace_path,
        status="computed",
        timestamp_rows=timestamp_rows,
        decoded_frames=decoded_frames,
        sample_windows=len(trace_rows),
    )


def percentile(values: np.ndarray, value: float) -> float:
    return float(np.percentile(values, value)) if values.size else float("nan")

def compute_auto_thresholds(
    trace_rows_by_animal: dict[str, list[MotionTraceRow]],
) -> dict[str, MotionThreshold]:
    thresholds: dict[str, MotionThreshold] = {}
    for animal_id in ANIMAL_ORDER:
        values = np.array(
            [row.motion_energy for row in trace_rows_by_animal.get(animal_id, [])],
            dtype=np.float64,
        )
        if values.size == 0:
            thresholds[animal_id] = MotionThreshold(
                animal_id=animal_id,
                threshold=None,
                threshold_source="auto",
                median=None,
                mad=None,
                p90=None,
                p95=None,
                p99=None,
                n_windows=0,
            )
            continue

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        p90 = percentile(values, 90)
        p95 = percentile(values, 95)
        p99 = percentile(values, 99)
        robust_sigma = ROBUST_SIGMA_SCALE * mad
        if robust_sigma > 1e-9:
            threshold = median + THRESHOLD_SIGMA_MULTIPLIER * robust_sigma
        else:
            # When nearly every value is identical, fall back to upper percentiles.
            threshold = max(p95, p99)

        thresholds[animal_id] = MotionThreshold(
            animal_id=animal_id,
            threshold=float(threshold),
            threshold_source="auto",
            median=median,
            mad=mad,
            p90=p90,
            p95=p95,
            p99=p99,
            n_windows=int(values.size),
        )
    return thresholds


def threshold_to_row(threshold: MotionThreshold) -> dict[str, str]:
    def maybe_number(value: Optional[float]) -> str:
        return "" if value is None else f"{value:.6f}"

    return {
        "animal_id": threshold.animal_id,
        "threshold": maybe_number(threshold.threshold),
        "threshold_source": threshold.threshold_source,
        "median": maybe_number(threshold.median),
        "mad": maybe_number(threshold.mad),
        "p90": maybe_number(threshold.p90),
        "p95": maybe_number(threshold.p95),
        "p99": maybe_number(threshold.p99),
        "n_windows": str(threshold.n_windows),
    }


def write_thresholds(path: Path, thresholds: dict[str, MotionThreshold]) -> None:
    rows = [threshold_to_row(thresholds[animal_id]) for animal_id in ANIMAL_ORDER]
    write_csv(path, THRESHOLD_FIELDS, rows)


def load_thresholds(path: Path) -> dict[str, MotionThreshold]:
    thresholds: dict[str, MotionThreshold] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            animal_id = str(row.get("animal_id") or "").strip()
            if not animal_id:
                continue
            threshold_text = str(row.get("threshold") or "").strip()
            thresholds[animal_id] = MotionThreshold(
                animal_id=animal_id,
                threshold=None if not threshold_text else float(threshold_text),
                threshold_source=str(row.get("threshold_source") or "").strip() or "manual",
                median=_parse_optional_float(row.get("median")),
                mad=_parse_optional_float(row.get("mad")),
                p90=_parse_optional_float(row.get("p90")),
                p95=_parse_optional_float(row.get("p95")),
                p99=_parse_optional_float(row.get("p99")),
                n_windows=int(str(row.get("n_windows") or "0").strip() or "0"),
            )
    return thresholds


def _parse_optional_float(value: object) -> Optional[float]:
    text = str(value or "").strip()
    return None if not text else float(text)


def merge_thresholds(
    *,
    auto_thresholds: dict[str, MotionThreshold],
    threshold_path: Path,
    reset_thresholds: bool,
    threshold_override: Optional[float],
) -> dict[str, MotionThreshold]:
    existing: dict[str, MotionThreshold] = {}
    if threshold_path.exists() and not reset_thresholds:
        existing = load_thresholds(threshold_path)

    merged: dict[str, MotionThreshold] = {}
    for animal_id in ANIMAL_ORDER:
        fresh = auto_thresholds[animal_id]

        if threshold_override is not None:
            merged[animal_id] = dataclasses.replace(
                fresh,
                threshold=threshold_override,
                threshold_source="override",
            )
            continue

        if reset_thresholds:
            merged[animal_id] = fresh
            continue

        old = existing.get(animal_id)
        if old is not None and old.threshold_source == "manual" and old.threshold is not None:
            merged[animal_id] = dataclasses.replace(
                fresh,
                threshold=old.threshold,
                threshold_source="manual",
            )
            continue

        merged[animal_id] = fresh

    return merged


def log_threshold_summary(thresholds: dict[str, MotionThreshold]) -> None:
    summary = ", ".join(
        (
            f"{animal_id}="
            f"{'NA' if thresholds[animal_id].threshold is None else f'{thresholds[animal_id].threshold:.6f}'} "
            f"({thresholds[animal_id].threshold_source}, n={thresholds[animal_id].n_windows})"
        )
        for animal_id in ANIMAL_ORDER
    )
    LOG.info("Motion thresholds: %s", summary)


def collapse_segments(segments: list[_StateAccumulator]) -> list[_StateAccumulator]:
    if not segments:
        return []
    collapsed = [segments[0]]
    tolerance = dt.timedelta(seconds=CONTIGUITY_TOLERANCE_SECONDS)
    for segment in segments[1:]:
        current = collapsed[-1]
        if (
            segment.state == current.state
            and segment.clip_key == current.clip_key
            and segment.start_utc <= current.end_utc + tolerance
        ):
            current.end_utc = max(current.end_utc, segment.end_utc)
            current.motion_sum += segment.motion_sum
            current.peak_motion = max(current.peak_motion, segment.peak_motion)
            current.n_windows += segment.n_windows
            continue
        collapsed.append(segment)
    return collapsed


def merge_short_immobile_gaps(
    segments: list[_StateAccumulator],
    *,
    merge_gap_s: float,
) -> list[_StateAccumulator]:
    if len(segments) < 3:
        return segments
    merged: list[_StateAccumulator] = []
    index = 0
    tolerance = dt.timedelta(seconds=CONTIGUITY_TOLERANCE_SECONDS)
    while index < len(segments):
        if index + 2 < len(segments):
            left = segments[index]
            middle = segments[index + 1]
            right = segments[index + 2]
            if (
                left.clip_key == middle.clip_key == right.clip_key
                and left.state == "mobile"
                and middle.state == "immobile"
                and right.state == "mobile"
                and (middle.end_utc - middle.start_utc).total_seconds() <= merge_gap_s
                and middle.start_utc <= left.end_utc + tolerance
                and right.start_utc <= middle.end_utc + tolerance
            ):
                merged.append(
                    _StateAccumulator(
                        animal_id=left.animal_id,
                        clip_key=left.clip_key,
                        start_utc=left.start_utc,
                        end_utc=right.end_utc,
                        state="mobile",
                        threshold=left.threshold,
                        threshold_source=left.threshold_source,
                        motion_sum=left.motion_sum + middle.motion_sum + right.motion_sum,
                        peak_motion=max(left.peak_motion, middle.peak_motion, right.peak_motion),
                        n_windows=left.n_windows + middle.n_windows + right.n_windows,
                    )
                )
                index += 3
                continue
        merged.append(segments[index])
        index += 1
    return collapse_segments(merged)


def apply_minimum_mobile_duration(
    segments: list[_StateAccumulator],
    *,
    min_mobile_s: float,
) -> list[_StateAccumulator]:
    adjusted: list[_StateAccumulator] = []
    for segment in segments:
        if segment.state == "mobile" and segment.duration_s < min_mobile_s:
            adjusted.append(
                dataclasses.replace(
                    segment,
                    state="immobile",
                )
            )
        else:
            adjusted.append(segment)
    return collapse_segments(adjusted)


def classify_motion_states(
    trace_rows: list[MotionTraceRow],
    *,
    threshold: MotionThreshold,
    merge_gap_s: float,
    min_mobile_s: float,
) -> list[MotionState]:
    if threshold.threshold is None:
        return []
    sorted_rows = sorted(trace_rows, key=lambda row: (row.clip_key, row.start_utc, row.end_utc))
    per_clip: dict[str, list[_StateAccumulator]] = defaultdict(list)
    for row in sorted_rows:
        state = "mobile" if row.motion_energy >= threshold.threshold else "immobile"
        per_clip[row.clip_key].append(
            _StateAccumulator(
                animal_id=row.animal_id,
                clip_key=row.clip_key,
                start_utc=row.start_utc,
                end_utc=row.end_utc,
                state=state,
                threshold=threshold.threshold,
                threshold_source=threshold.threshold_source,
                motion_sum=row.motion_energy,
                peak_motion=row.motion_energy,
                n_windows=1,
            )
        )

    states: list[MotionState] = []
    for clip_key in sorted(per_clip):
        segments = collapse_segments(per_clip[clip_key])
        segments = merge_short_immobile_gaps(segments, merge_gap_s=merge_gap_s)
        segments = apply_minimum_mobile_duration(segments, min_mobile_s=min_mobile_s)
        states.extend(segment.as_state() for segment in segments)
    return states


def state_to_row(state: MotionState, timezone: dt.tzinfo) -> dict[str, str]:
    return {
        "animal_id": state.animal_id,
        "clip_key": state.clip_key,
        "start_utc": format_utc(state.start_utc),
        "end_utc": format_utc(state.end_utc),
        "start_local": format_local(state.start_utc, timezone),
        "end_local": format_local(state.end_utc, timezone),
        "state": state.state,
        "threshold": f"{state.threshold:.6f}",
        "threshold_source": state.threshold_source,
        "mean_motion_energy": f"{state.mean_motion_energy:.6f}",
        "peak_motion_energy": f"{state.peak_motion_energy:.6f}",
        "n_windows": str(state.n_windows),
    }


def write_motion_states(path: Path, states: list[MotionState], timezone: dt.tzinfo) -> None:
    ordered = sorted(
        states,
        key=lambda state: (
            ANIMAL_ORDER.index(state.animal_id) if state.animal_id in ANIMAL_ORDER else len(ANIMAL_ORDER),
            state.start_utc,
            state.clip_key,
        ),
    )
    write_csv(path, STATE_FIELDS, [state_to_row(state, timezone) for state in ordered])


def load_motion_states(path: Optional[Path]) -> list[MotionState]:
    if path is None or not path.exists():
        return []
    states: list[MotionState] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=2):
            try:
                animal_id = str(row.get("animal_id") or "").strip()
                state_name = str(row.get("state") or "").strip().lower()
                if animal_id not in ANIMAL_ORDER:
                    raise ValueError(f"unknown animal_id {animal_id}")
                if state_name not in {"mobile", "immobile"}:
                    raise ValueError(f"invalid state {state_name}")
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


def discover_cached_trace_files(root: Path) -> list[Path]:
    trace_dir = root / "cropped_by_caterpillar" / "motion_energy" / "traces"
    if not trace_dir.exists():
        return []
    return sorted(trace_dir.glob("*.motion.csv.gz"))


def infer_animal_id_from_trace_path(path: Path) -> str:
    prefix = path.name.split("_", 1)[0].strip()
    return prefix


def load_cached_trace_inventory(root: Path) -> CachedTraceInventory:
    grouped: dict[str, list[MotionTraceRow]] = defaultdict(list)
    trace_files_by_animal: dict[str, int] = {animal_id: 0 for animal_id in ANIMAL_ORDER}
    valid_windows_by_animal: dict[str, int] = {animal_id: 0 for animal_id in ANIMAL_ORDER}

    for trace_path in discover_cached_trace_files(root):
        inferred_animal_id = infer_animal_id_from_trace_path(trace_path)
        if inferred_animal_id in trace_files_by_animal:
            trace_files_by_animal[inferred_animal_id] += 1

        try:
            rows = load_motion_trace_rows(trace_path)
        except Exception as exc:
            LOG.warning("Ignoring invalid cached trace %s: %s", trace_path, exc)
            continue

        if not rows:
            continue

        animal_id = rows[0].animal_id
        grouped[animal_id].extend(rows)
        if animal_id in valid_windows_by_animal:
            valid_windows_by_animal[animal_id] += len(rows)
        else:
            valid_windows_by_animal[animal_id] = len(rows)
            trace_files_by_animal.setdefault(animal_id, 0)

    for animal_id in grouped:
        grouped[animal_id].sort(key=lambda row: (row.start_utc, row.end_utc, row.clip_key))

    return CachedTraceInventory(
        trace_rows_by_animal=dict(grouped),
        trace_files_by_animal=trace_files_by_animal,
        valid_windows_by_animal=valid_windows_by_animal,
    )


def log_cached_trace_inventory(inventory: CachedTraceInventory) -> None:
    availability = ", ".join(
        f"{animal_id}={inventory.valid_windows_by_animal.get(animal_id, 0)}"
        for animal_id in ANIMAL_ORDER
    )
    LOG.info("Motion windows available: %s", availability)

    thresholds_summary = ", ".join(
        f"{animal_id} files={inventory.trace_files_by_animal.get(animal_id, 0)}"
        for animal_id in ANIMAL_ORDER
    )
    LOG.info("Cached trace files discovered: %s", thresholds_summary)

    for animal_id in ANIMAL_ORDER:
        file_count = inventory.trace_files_by_animal.get(animal_id, 0)
        window_count = inventory.valid_windows_by_animal.get(animal_id, 0)
        if file_count > 0 and window_count == 0:
            LOG.warning(
                "%s has %d cached trace files but zero valid motion windows after parsing",
                animal_id,
                file_count,
            )


def downsample_trace_rows(rows: list[MotionTraceRow], max_points: int) -> list[MotionTraceRow]:
    if len(rows) <= max_points:
        return rows
    chunk_size = max(1, math.ceil(len(rows) / max_points))
    sampled: list[MotionTraceRow] = []
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        sampled.append(max(chunk, key=lambda row: row.motion_energy))
    return sampled


def write_motion_diagnostics(
    path: Path,
    trace_rows_by_animal: dict[str, list[MotionTraceRow]],
    thresholds: dict[str, MotionThreshold],
    timezone: dt.tzinfo,
) -> None:
    fig, axes = plt.subplots(len(ANIMAL_ORDER), 1, figsize=(16, 14), sharex=True)
    fig.subplots_adjust(top=0.95, bottom=0.06, left=0.08, right=0.99, hspace=0.24)
    fig.suptitle(
        "Motion-energy diagnostics (motion-derived activity proxy)",
        x=0.08,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )

    for axis, animal_id in zip(axes, ANIMAL_ORDER):
        rows = trace_rows_by_animal.get(animal_id, [])
        threshold = thresholds.get(animal_id)
        sampled = downsample_trace_rows(rows, DISPLAY_MAX_POINTS)
        if sampled:
            x_values = [row.end_utc.astimezone(timezone).replace(tzinfo=None) for row in sampled]
            y_values = [row.motion_energy for row in sampled]
            axis.scatter(x_values, y_values, s=7, color="#14532d", alpha=0.72, linewidths=0)
        if threshold and threshold.threshold is not None:
            axis.axhline(threshold.threshold, color="#dc2626", linewidth=1.2, linestyle="--")
        axis.set_ylabel(animal_id, rotation=0, labelpad=24)
        axis.grid(True, axis="y", color="#e2e8f0", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#cbd5e1")
        axis.spines["bottom"].set_color("#cbd5e1")

    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    axes[-1].set_xlabel(f"Local time - {timezone_label(timezone)}")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summary_row(
    *,
    root: Path,
    entry: ManifestEntry,
    result: ExtractionResult,
) -> dict[str, str]:
    return {
        "animal_id": entry.animal_id,
        "clip_key": entry.clip_key,
        "cropped_video": relative_to_root(entry.cropped_video, root),
        "timestamp_file": relative_to_root(entry.timestamp_file, root),
        "timestamp_rows": str(result.timestamp_rows),
        "decoded_frames": "" if result.decoded_frames is None else str(result.decoded_frames),
        "sample_windows": str(result.sample_windows),
        "status": result.status,
        "error": result.error,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        parser.error(f"root must be an existing directory: {root}")
    if args.sample_hz <= 0:
        parser.error("--sample-hz must be > 0")
    if args.motion_width <= 0 or args.motion_height <= 0:
        parser.error("--motion-width and --motion-height must be > 0")
    if not (0.0 < args.top_fraction <= 1.0):
        parser.error("--top-fraction must be in (0, 1]")

    timezone = load_timezone(args.timezone)
    animal_filter = set(args.animals) if args.animals else None
    entries: list[ManifestEntry] = []
    if not args.classify_only:
        entries = load_manifest_entries(
            root,
            animals=animal_filter,
            limit_clips=args.limit_clips,
        )
        if not entries:
            parser.error("no cropped manifest rows with copied timestamps were found")

    motion_root = root / "cropped_by_caterpillar" / "motion_energy"
    motion_root.mkdir(parents=True, exist_ok=True)
    thresholds_path = motion_root / "motion_thresholds.csv"
    states_path = motion_root / "motion_states.csv"
    summary_path = motion_root / "motion_summary.csv"
    diagnostics_path = motion_root / "motion_energy_diagnostics.png"

    summary_rows: list[dict[str, str]] = []
    if not args.classify_only:
        progress_reporter = ProgressReporter(total_clips=len(entries))
        for clip_index, entry in enumerate(entries, start=1):
            result = extract_trace_for_entry(
                root,
                entry,
                clip_index=clip_index,
                timezone=timezone,
                sample_hz=args.sample_hz,
                motion_width=args.motion_width,
                motion_height=args.motion_height,
                top_fraction=args.top_fraction,
                force=args.force,
                progress_reporter=progress_reporter,
            )
            if result.error:
                LOG.error(result.error)
            summary_rows.append(summary_row(root=root, entry=entry, result=result))
        write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
        LOG.info("Wrote motion summary: %s", summary_path)

    inventory = load_cached_trace_inventory(root)
    if not any(inventory.trace_files_by_animal.values()):
        if args.classify_only:
            parser.error("no cached motion trace files were found under cropped_by_caterpillar/motion_energy/traces")
        LOG.warning("No cached motion trace files were found under %s", motion_root / "traces")
    log_cached_trace_inventory(inventory)

    auto_thresholds = compute_auto_thresholds(inventory.trace_rows_by_animal)
    thresholds = merge_thresholds(
        auto_thresholds=auto_thresholds,
        threshold_path=thresholds_path,
        reset_thresholds=args.reset_thresholds,
        threshold_override=args.threshold,
    )
    write_thresholds(thresholds_path, thresholds)
    log_threshold_summary(thresholds)
    LOG.info("Wrote refreshed thresholds: %s", thresholds_path)

    LOG.info("Rendering motion-energy diagnostics...")
    write_motion_diagnostics(diagnostics_path, inventory.trace_rows_by_animal, thresholds, timezone)
    LOG.info("Wrote diagnostics PNG: %s", diagnostics_path)

    LOG.info("Classifying motion-derived states...")
    all_states: list[MotionState] = []
    for animal_id in ANIMAL_ORDER:
        all_states.extend(
            classify_motion_states(
                inventory.trace_rows_by_animal.get(animal_id, []),
                threshold=thresholds[animal_id],
                merge_gap_s=args.merge_gap_s,
                min_mobile_s=args.min_mobile_s,
            )
        )
    write_motion_states(states_path, all_states, timezone)
    LOG.info("Wrote motion states CSV: %s", states_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
