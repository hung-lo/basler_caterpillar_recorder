#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import gzip
import logging
import os
import tempfile
import sys
from pathlib import Path
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError as exc:  # pragma: no cover - import-time guard for missing optional deps
    raise SystemExit(
        "plot_recording_timeline.py requires matplotlib. Install dependencies with "
        "`python -m pip install -r requirements.txt`."
    ) from exc


LOG = logging.getLogger("plot_recording_timeline")

UTC = dt.timezone.utc
DEFAULT_TIMEZONE = "America/New_York"
DEFAULT_ANIMALS = [f"C{index:02d}" for index in range(1, 9)]
TIMESTAMP_SUFFIXES = (".timestamps.csv.gz", ".timestamps.csv")
MAX_CLIP_ANNOTATIONS = 60

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)


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


def utc_from_ns(value_ns: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(value_ns / 1e9, tz=UTC)


def coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_iso_datetime(value: str) -> dt.datetime:
    text = value.strip()
    if not text:
        raise ValueError("timestamp is empty")
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    parsed = dt.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("timestamp is missing a timezone offset")
    return parsed


def parse_utc_value(value: Any) -> dt.datetime:
    if value is None:
        raise ValueError("missing UTC timestamp")
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = utc_from_ns(int(value))
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("missing UTC timestamp")
        try:
            parsed = utc_from_ns(int(text))
        except ValueError:
            parsed = parse_iso_datetime(text)
    return parsed.astimezone(UTC)


def parse_timestamp_row(row: dict[str, Any]) -> dt.datetime:
    for field in ("host_utc_ns", "utc_ns", "timestamp_ns"):
        value = row.get(field)
        if value not in (None, ""):
            parsed_ns = coerce_int(value)
            if parsed_ns is None:
                raise ValueError(f"{field} is not an integer")
            return utc_from_ns(parsed_ns)
    for field in ("host_utc_iso", "utc_iso", "timestamp_iso"):
        value = row.get(field)
        if value not in (None, ""):
            return parse_utc_value(value)
    raise ValueError("missing UTC timestamp columns")


def open_text_file(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def strip_timestamp_suffix(name: str) -> str:
    for suffix in TIMESTAMP_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"Unsupported timestamp filename: {name}")


def video_file_for_timestamp_file(timestamp_file: Path) -> Path:
    base = strip_timestamp_suffix(timestamp_file.name)
    return timestamp_file.with_name(f"{base}.mp4")


def timezone_label(tz: dt.tzinfo) -> str:
    label = getattr(tz, "key", None)
    if label:
        return str(label)
    sample = dt.datetime.now(UTC).astimezone(tz)
    name = sample.tzname()
    return name or str(tz)


def load_timezone(name: str) -> dt.tzinfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        local_tz = dt.datetime.now().astimezone().tzinfo
        if local_tz is None:
            LOG.warning(
                "Could not load timezone %s and the local timezone is unavailable; falling back to UTC",
                name,
            )
            return UTC
        LOG.warning(
            "Could not load timezone %s; falling back to the local system timezone (%s)",
            name,
            timezone_label(local_tz),
        )
        return local_tz


def to_plot_local(value_utc: dt.datetime, tz: dt.tzinfo) -> dt.datetime:
    return value_utc.astimezone(tz).replace(tzinfo=None)


def format_utc(value_utc: dt.datetime) -> str:
    return value_utc.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def format_local(value_utc: dt.datetime, tz: dt.tzinfo) -> str:
    return value_utc.astimezone(tz).isoformat(sep=" ", timespec="microseconds")


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


def load_behavior_events(path: Optional[Path], tz: dt.tzinfo) -> list[BehaviorEvent]:
    if path is None or not path.exists():
        return []

    events: list[BehaviorEvent] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=2):
            try:
                animal_id = str(row.get("animal_id") or "").strip()
                if not animal_id:
                    raise ValueError("missing animal_id")
                start_text = str(row.get("start_local") or "").strip()
                end_text = str(row.get("end_local") or "").strip()
                if not start_text or not end_text:
                    raise ValueError("missing start_local or end_local")
                start_local = parse_local_datetime(start_text, tz)
                end_local = parse_local_datetime(end_text, tz)
                if end_local <= start_local:
                    raise ValueError("end_local must be after start_local")
                events.append(
                    BehaviorEvent(
                        animal_id=animal_id,
                        start_local=start_local,
                        end_local=end_local,
                        event=str(row.get("event") or "").strip() or "event",
                        kind=str(row.get("kind") or "").strip() or "event",
                        notes=str(row.get("notes") or "").strip(),
                    )
                )
            except Exception as exc:
                LOG.warning("Skipping malformed behavior event row %d in %s: %s", row_index, path, exc)
    return events


def infer_animals(events: Iterable[BehaviorEvent]) -> list[str]:
    animals: list[str] = []
    seen: set[str] = set()
    for event in events:
        if event.animal_id not in seen:
            seen.add(event.animal_id)
            animals.append(event.animal_id)
    return animals


def _bar_left_width(start_utc: dt.datetime, end_utc: dt.datetime, tz: dt.tzinfo) -> tuple[float, float]:
    left = mdates.date2num(to_plot_local(start_utc, tz))
    width = max((end_utc - start_utc).total_seconds() / 86400.0, 1.0 / 86400.0 / 24.0)
    return left, width


def plot_recording_timeline(
    clips: list[RecordingClip],
    events: list[BehaviorEvent],
    animals: list[str],
    *,
    timezone: dt.tzinfo,
    output_path: Path,
    annotate_clips: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip_annotations = annotate_clips and len(clips) <= MAX_CLIP_ANNOTATIONS
    if annotate_clips and not clip_annotations:
        LOG.info("Too many clips for per-clip annotations; disabling clip labels")

    if clips:
        first_utc = min(clip.start_utc for clip in clips)
        last_utc = max(clip.end_utc for clip in clips)
        recorded_duration_s = sum(clip.duration_s for clip in clips)
        elapsed_duration_s = max((last_utc - first_utc).total_seconds(), 0.0)
        recorded_fraction = recorded_duration_s / elapsed_duration_s if elapsed_duration_s > 0 else 1.0
    else:
        first_utc = last_utc = dt.datetime.now(UTC)
        recorded_duration_s = 0.0
        elapsed_duration_s = 0.0
        recorded_fraction = 0.0

    behavior_rows = animals or DEFAULT_ANIMALS
    behavior_height = max(1.8, 0.42 * len(behavior_rows))
    fig_height = 4.2 + behavior_height if behavior_rows else 4.2
    fig = plt.figure(figsize=(17, fig_height), constrained_layout=False)
    if behavior_rows:
        gs = fig.add_gridspec(2, 1, height_ratios=[1.3, max(1.0, 0.45 * len(behavior_rows))], hspace=0.18)
        ax_cov = fig.add_subplot(gs[0])
        ax_beh = fig.add_subplot(gs[1], sharex=ax_cov)
    else:
        ax_cov = fig.add_subplot(1, 1, 1)
        ax_beh = None

    if clips:
        colors = {"camera": "#3b82f6"}
        for index, clip in enumerate(clips):
            left, width = _bar_left_width(clip.start_utc, clip.end_utc, timezone)
            ax_cov.broken_barh([(left, width)], (0.25, 0.5), facecolors=colors["camera"], alpha=0.72)
            if clip_annotations:
                center = left + width / 2.0
                duration_min = clip.duration_s / 60.0
                ax_cov.text(
                    center,
                    0.64,
                    f"{clip.camera_label}  {duration_min:.1f}m",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#1f2937",
                    clip_on=True,
                )

        ax_cov.set_ylim(0, 1)
        ax_cov.set_yticks([])
        ax_cov.set_ylabel("Recording", rotation=0, labelpad=46, va="center")
        ax_cov.grid(True, axis="x", alpha=0.25)

        summary_text = (
            f"first recorded: {format_local(first_utc, timezone)}\n"
            f"last recorded:  {format_local(last_utc, timezone)}\n"
            f"total recorded: {recorded_duration_s / 60.0:.1f} min\n"
            f"recorded / elapsed: {recorded_fraction:.1%}"
        )
        ax_cov.text(
            0.01,
            0.98,
            summary_text,
            transform=ax_cov.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.78, edgecolor="#cbd5e1"),
        )
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

    if clips:
        x_min = to_plot_local(first_utc, timezone)
        x_max = to_plot_local(last_utc, timezone)
        if x_max <= x_min:
            x_max = x_min + dt.timedelta(minutes=1)
        ax_cov.set_xlim(x_min, x_max)
    ax_cov.set_title("Recording coverage")
    locator = mdates.AutoDateLocator()
    ax_cov.xaxis.set_major_locator(locator)
    ax_cov.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax_cov.set_xlabel(f"Local time ({timezone_label(timezone)})")

    if ax_beh is not None:
        ax_beh.set_title("Behavior annotations")
        ax_beh.set_yticks(range(len(behavior_rows)))
        ax_beh.set_yticklabels(behavior_rows)
        ax_beh.set_ylabel("Animal")
        ax_beh.grid(True, axis="x", alpha=0.2)
        ax_beh.set_ylim(-0.5, len(behavior_rows) - 0.5)
        ax_beh.invert_yaxis()

        color_cycle = plt.get_cmap("tab20")
        categories: dict[str, str] = {}
        legend_handles: list[Patch] = []
        for event in events:
            if event.animal_id not in behavior_rows:
                continue
            y = behavior_rows.index(event.animal_id)
            category = event.event or event.kind
            if category not in categories:
                categories[category] = color_cycle(len(categories) % color_cycle.N)
            color = categories[category]
            left = mdates.date2num(event.start_local.replace(tzinfo=None))
            width = max((event.end_local - event.start_local).total_seconds() / 86400.0, 1.0 / 86400.0 / 24.0)
            ax_beh.broken_barh([(left, width)], (y - 0.32, 0.64), facecolors=color, alpha=0.8)

        if categories:
            legend_handles = [Patch(facecolor=color, label=label, alpha=0.8) for label, color in categories.items()]
            if len(legend_handles) <= 10:
                ax_beh.legend(
                    handles=legend_handles,
                    loc="upper right",
                    fontsize=8,
                    frameon=True,
                    title="Behavior",
                    title_fontsize=8,
                )
    elif behavior_rows:
        ax_cov.text(
            0.99,
            0.02,
            "No behavior rows to display",
            transform=ax_cov.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#6b7280",
        )

    fig.autofmt_xdate()
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
        type=Path,
        help="Path to behavior_events.csv (defaults to <root>/behavior_events.csv when present).",
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

    local_tz = load_timezone(args.timezone)
    clips, warnings = collect_recording_clips(root)

    coverage_csv = args.coverage_csv or (root / "recording_coverage.csv")
    output_png = args.output or (root / "recording_behavior_timeline.png")
    write_coverage_csv(clips, coverage_csv, local_tz)

    events_path = args.events
    if events_path is None:
        default_events = root / "behavior_events.csv"
        events_path = default_events if default_events.exists() else None
    events = load_behavior_events(events_path, local_tz)

    animals = args.animals if args.animals else (infer_animals(events) or DEFAULT_ANIMALS)
    if not clips:
        LOG.warning("No recording clips were found under %s", root)

    plot_recording_timeline(
        clips,
        events,
        animals,
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
