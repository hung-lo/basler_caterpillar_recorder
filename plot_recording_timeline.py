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


LOG = logging.getLogger("plot_recording_timeline")

UTC = dt.timezone.utc
DEFAULT_TIMEZONE = "America/New_York"
DEFAULT_ANIMALS = [f"C{index:02d}" for index in range(1, 9)]
ANIMAL_ORDER = [f"C{index:02d}" for index in range(1, 9)]
TIMESTAMP_SUFFIXES = (".timestamps.csv.gz", ".timestamps.csv")
MAX_CLIP_ANNOTATIONS = 60
SHORT_EVENT_THRESHOLD_SECONDS = 300
DEFAULT_POINT_MARKER_SIZE = 70
SHED_MARKER_SIZE = 100
STIM_MARKER_SIZE = 180
DEATH_MARKER_SIZE = 120

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

SHED_COLOR = "#d97706"
STIM_COLOR = "#dc2626"
DEATH_COLOR = "#111827"
DEFAULT_POINT_COLOR = "#475569"
INTERVAL_BAR_COLOR = "#8aa1c7"
ROW_BAND_COLOR = "#f8fafc"


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
                if not start_text:
                    raise ValueError("missing start_local")
                start_local = parse_local_datetime(start_text, tz)
                if end_text:
                    end_local = parse_local_datetime(end_text, tz)
                    if end_local <= start_local:
                        raise ValueError("end_local must be after start_local")
                else:
                    end_local = start_local + dt.timedelta(seconds=1)
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


def _event_duration_s(event: BehaviorEvent) -> float:
    return max((event.end_local - event.start_local).total_seconds(), 0.0)


def normalize_event_name(event_name: str) -> str:
    return event_name.strip().lower().replace(" ", "_").replace("-", "_")


def get_point_event_style(event_name: str) -> tuple[str, float, str]:
    normalized = normalize_event_name(event_name)

    if normalized in SHED_EVENT_NAMES:
        return "^", SHED_MARKER_SIZE, SHED_COLOR

    if normalized in STIM_EVENT_NAMES:
        return "*", STIM_MARKER_SIZE, STIM_COLOR

    if normalized in DEATH_EVENT_NAMES:
        return "X", DEATH_MARKER_SIZE, DEATH_COLOR

    return "o", DEFAULT_POINT_MARKER_SIZE, DEFAULT_POINT_COLOR


def event_bar_color(event: BehaviorEvent) -> str:
    marker, _marker_size, color = get_point_event_style(event.event or event.kind)
    normalized = normalize_event_name(event.event or event.kind)
    if normalized in SHED_EVENT_NAMES | STIM_EVENT_NAMES | DEATH_EVENT_NAMES:
        return color
    return INTERVAL_BAR_COLOR


def semantic_legend_handles() -> list[Line2D | Patch]:
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
            marker="*",
            linestyle="None",
            markersize=11,
            markerfacecolor=STIM_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label="Electrical stimulation",
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
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor=DEFAULT_POINT_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label="Other point event",
        ),
    ]


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

    behavior_rows = list(ANIMAL_ORDER)
    behavior_index = {animal: index for index, animal in enumerate(behavior_rows)}
    unknown_animals = sorted({event.animal_id for event in events if event.animal_id not in behavior_index})
    for animal_id in unknown_animals:
        LOG.warning("Skipping behavior event for unknown animal ID: %s", animal_id)

    fig_height = 7.6
    fig = plt.figure(figsize=(18, fig_height), constrained_layout=False)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 4.5], hspace=0.12)
    ax_cov = fig.add_subplot(gs[0])
    ax_beh = fig.add_subplot(gs[1], sharex=ax_cov)
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.08, right=0.985)
    fig.patch.set_facecolor("white")

    total_recorded_h = recorded_duration_s / 3600.0
    elapsed_h = elapsed_duration_s / 3600.0
    subtitle = (
        f"Local time - {timezone_label(timezone)} | "
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

    if clips:
        colors = {"camera": "#0f766e"}
        for index, clip in enumerate(clips):
            left, width = _bar_left_width(clip.start_utc, clip.end_utc, timezone)
            ax_cov.broken_barh([(left, width)], (0.3, 0.42), facecolors=colors["camera"], alpha=0.82)
            if clip_annotations:
                center = left + width / 2.0
                duration_min = clip.duration_s / 60.0
                ax_cov.text(
                    center,
                    0.77,
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
        ax_cov.grid(True, axis="x", color="#cbd5e1", alpha=0.45, linewidth=0.6)
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

    plot_starts: list[dt.datetime] = []
    plot_ends: list[dt.datetime] = []
    if clips:
        plot_starts.append(to_plot_local(first_utc, timezone))
        plot_ends.append(to_plot_local(last_utc, timezone))
    for event in events:
        if event.animal_id in behavior_index:
            plot_starts.append(event.start_local.replace(tzinfo=None))
            plot_ends.append(event.end_local.replace(tzinfo=None))
    if plot_starts and plot_ends:
        x_min = min(plot_starts)
        x_max = max(plot_ends)
        if x_max <= x_min:
            x_max = x_min + dt.timedelta(minutes=1)
        ax_cov.set_xlim(x_min, x_max)

    ax_cov.set_title("Recording coverage", loc="left", fontsize=11, color="#0f172a", pad=8)
    ax_cov.tick_params(axis="x", labelbottom=False)

    ax_beh.set_title("Behavior annotations", loc="left", fontsize=11, color="#0f172a", pad=18)
    ax_beh.set_yticks(range(len(behavior_rows)))
    ax_beh.set_yticklabels(behavior_rows)
    ax_beh.set_ylabel("Animal")
    ax_beh.set_ylim(-0.5, len(behavior_rows) - 0.5)
    ax_beh.invert_yaxis()
    ax_beh.grid(True, axis="x", color="#cbd5e1", alpha=0.45, linewidth=0.6)
    ax_beh.set_xlabel(f"Local time - {timezone_label(timezone)}")

    for y in range(len(behavior_rows)):
        ax_beh.axhspan(y - 0.46, y + 0.46, facecolor=ROW_BAND_COLOR, alpha=0.9, zorder=0)

    locator = mdates.AutoDateLocator()
    ax_beh.xaxis.set_major_locator(locator)
    ax_beh.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))

    for event in events:
        if event.animal_id not in behavior_index:
            continue
        y = behavior_index[event.animal_id]
        left = mdates.date2num(event.start_local.replace(tzinfo=None))
        duration_s = _event_duration_s(event)
        width = max(duration_s / 86400.0, 1e-9)
        color = event_bar_color(event)
        ax_beh.broken_barh(
            [(left, width)],
            (y - 0.33, 0.66),
            facecolors=color,
            alpha=0.74,
            zorder=2,
        )
        if duration_s <= SHORT_EVENT_THRESHOLD_SECONDS:
            marker, marker_size, point_color = get_point_event_style(event.event or event.kind)
            ax_beh.scatter(
                left,
                y,
                s=marker_size,
                marker=marker,
                color=point_color,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )

    ax_beh.legend(
        handles=semantic_legend_handles(),
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=5,
        frameon=False,
        fontsize=8,
        handlelength=1.3,
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    for ax in (ax_cov, ax_beh):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(axis="both", colors="#334155")

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
