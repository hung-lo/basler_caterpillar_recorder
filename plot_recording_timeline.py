#!/usr/bin/env python3
"""Plot recording coverage and manually annotated caterpillar behavior intervals.

The recorder stores per-frame UTC timestamps in files such as:
    camera1.timestamps.csv.gz

This script recursively finds those files, takes the first and last
`host_utc_iso` timestamp from each clip, converts them to local time
(default: America/New_York / Woods Hole), writes a clip summary CSV,
and plots:
  1. overall recording coverage on top (gaps remain visible)
  2. one row per caterpillar with manually editable behavior intervals

Only matplotlib is required beyond the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


TIMESTAMP_PATTERNS = ("*.timestamps.csv.gz", "*.timestamps.csv")
STATE_EVENTS = {"mobile", "immobile", "upside_down_on_lid", "premolt_upside_down"}


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def parse_iso(value: str, local_tz: ZoneInfo) -> datetime:
    """Parse ISO time. Naive values are interpreted as local time."""
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=local_tz)
    return dt


def first_last_timestamp(path: Path) -> tuple[datetime, datetime, int]:
    """Return first UTC timestamp, last UTC timestamp, and frame count."""
    first = None
    last = None
    count = 0
    with open_text(path) as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "host_utc_iso" not in reader.fieldnames:
            raise ValueError("missing host_utc_iso column")
        for row in reader:
            raw = (row.get("host_utc_iso") or "").strip()
            if not raw:
                continue
            dt = parse_iso(raw, ZoneInfo("UTC")).astimezone(timezone.utc)
            if first is None:
                first = dt
            last = dt
            count += 1
    if first is None or last is None:
        raise ValueError("no valid host_utc_iso timestamps")
    return first, last, count


def adjacent_video(timestamp_path: Path) -> str:
    videos = sorted(timestamp_path.parent.glob("*.mp4"))
    if not videos:
        return ""
    preferred = [p for p in videos if p.stem == timestamp_path.name.split(".timestamps")[0]]
    return (preferred[0] if preferred else videos[0]).name


def discover_clips(root: Path, local_tz: ZoneInfo) -> list[dict]:
    paths = []
    for pattern in TIMESTAMP_PATTERNS:
        paths.extend(root.rglob(pattern))
    paths = sorted(set(paths))

    clips = []
    for path in paths:
        try:
            start_utc, end_utc, frames = first_last_timestamp(path)
        except Exception as exc:
            print(f"WARNING: skipping {path}: {exc}")
            continue
        start_local = start_utc.astimezone(local_tz)
        end_local = end_utc.astimezone(local_tz)
        clips.append(
            {
                "clip_id": path.parent.name,
                "timestamp_file": str(path),
                "video_file": adjacent_video(path),
                "start_utc": start_utc,
                "end_utc": end_utc,
                "start_local": start_local,
                "end_local": end_local,
                "duration_s": (end_utc - start_utc).total_seconds(),
                "frames": frames,
            }
        )
    return sorted(clips, key=lambda x: x["start_utc"])


def write_coverage_csv(clips: list[dict], output_path: Path) -> None:
    fields = [
        "clip_id",
        "timestamp_file",
        "video_file",
        "start_utc",
        "end_utc",
        "start_local",
        "end_local",
        "duration_s",
        "frames",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for clip in clips:
            row = clip.copy()
            for key in ("start_utc", "end_utc", "start_local", "end_local"):
                row[key] = row[key].isoformat()
            row["duration_s"] = f"{row['duration_s']:.3f}"
            writer.writerow(row)


def ensure_event_csv(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["animal_id", "start_local", "end_local", "event", "kind", "notes"])
    print(f"Created empty manual annotation file: {path}")


def read_events(path: Path, local_tz: ZoneInfo) -> list[dict]:
    events = []
    if not path.exists():
        return events
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, start=2):
            animal = (row.get("animal_id") or "").strip()
            event = (row.get("event") or "").strip()
            start_raw = (row.get("start_local") or "").strip()
            end_raw = (row.get("end_local") or "").strip()
            if not (animal and event and start_raw):
                continue
            try:
                start = parse_iso(start_raw, local_tz).astimezone(local_tz)
                end = parse_iso(end_raw, local_tz).astimezone(local_tz) if end_raw else start
            except ValueError as exc:
                print(f"WARNING: bad event time on line {line_no}: {exc}")
                continue
            if end < start:
                print(f"WARNING: end before start on line {line_no}; skipping")
                continue
            kind = (row.get("kind") or "").strip().lower()
            if not kind:
                kind = "state" if event.lower() in STATE_EVENTS else "event"
            events.append(
                {
                    "animal_id": animal,
                    "start": start,
                    "end": end,
                    "event": event,
                    "kind": kind,
                    "notes": (row.get("notes") or "").strip(),
                }
            )
    return events


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def plot_timeline(
    clips: list[dict],
    events: list[dict],
    animals: list[str],
    local_tz: ZoneInfo,
    output_path: Path,
    annotate_clips: bool,
) -> None:
    if not clips:
        raise SystemExit("No valid timestamp files found.")

    all_event_animals = {e["animal_id"] for e in events}
    if animals:
        animal_order = animals
    else:
        animal_order = sorted(all_event_animals)
    if not animal_order:
        animal_order = ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08"]

    fig_height = max(6.5, 2.5 + 0.65 * len(animal_order))
    fig = plt.figure(figsize=(16, fig_height))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, max(2.5, len(animal_order))], hspace=0.10)
    ax_cov = fig.add_subplot(gs[0])
    ax_evt = fig.add_subplot(gs[1], sharex=ax_cov)

    # ---- Overall recording coverage ----
    for clip in clips:
        start_num = mdates.date2num(clip["start_local"])
        width_days = clip["duration_s"] / 86400.0
        ax_cov.barh(0, width_days, left=start_num, height=0.55, align="center")
        if annotate_clips:
            ax_cov.text(
                start_num + width_days / 2,
                0.36,
                fmt_duration(clip["duration_s"]),
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45 if len(clips) > 25 else 0,
            )

    start_all = clips[0]["start_local"]
    end_all = max(c["end_local"] for c in clips)
    span_s = (end_all - start_all).total_seconds()
    recorded_s = sum(c["duration_s"] for c in clips)
    coverage_pct = 100 * recorded_s / span_s if span_s > 0 else 100.0
    ax_cov.set_yticks([0])
    ax_cov.set_yticklabels(["recording"])
    ax_cov.set_title(
        f"Recording coverage | {start_all:%Y-%m-%d %H:%M} to {end_all:%Y-%m-%d %H:%M} "
        f"({fmt_duration(recorded_s)} recorded; {coverage_pct:.1f}% of elapsed span)"
    )
    ax_cov.grid(axis="x", alpha=0.25)
    ax_cov.tick_params(axis="x", labelbottom=False)

    # ---- Caterpillar behavior rows ----
    y_for_animal = {animal: i for i, animal in enumerate(animal_order)}
    event_names = sorted({e["event"] for e in events})
    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    event_color = {name: cycle_colors[i % len(cycle_colors)] for i, name in enumerate(event_names)}
    legend_handles = {}

    for event in events:
        animal = event["animal_id"]
        if animal not in y_for_animal:
            continue
        y = y_for_animal[animal]
        start_num = mdates.date2num(event["start"])
        duration_s = max(0.0, (event["end"] - event["start"]).total_seconds())
        width_days = duration_s / 86400.0
        color = event_color[event["event"]]

        if duration_s == 0:
            handle = ax_evt.scatter(start_num, y, marker="D", s=28, color=color, zorder=5)
        else:
            height = 0.68 if event["kind"] == "state" else 0.28
            alpha = 0.55 if event["kind"] == "state" else 0.95
            handle = ax_evt.barh(y, width_days, left=start_num, height=height, color=color, alpha=alpha)[0]

        legend_handles.setdefault(event["event"], handle)

    ax_evt.set_yticks(range(len(animal_order)))
    ax_evt.set_yticklabels(animal_order)
    ax_evt.invert_yaxis()
    ax_evt.set_ylabel("Caterpillar")
    ax_evt.set_xlabel(f"Local time ({local_tz.key})")
    ax_evt.grid(axis="x", alpha=0.25)
    ax_evt.set_xlim(start_all, end_all)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=14)
    formatter = mdates.ConciseDateFormatter(locator, tz=local_tz)
    ax_evt.xaxis.set_major_locator(locator)
    ax_evt.xaxis.set_major_formatter(formatter)

    if legend_handles:
        ax_evt.legend(
            legend_handles.values(),
            legend_handles.keys(),
            title="Manual annotations",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
        )

    fig.suptitle("Caterpillar recording + behavior timeline", y=0.995, fontsize=14)
    fig.subplots_adjust(right=0.84, top=0.93, bottom=0.10)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Root directory containing recording/session folders")
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("behavior_events.csv"),
        help="Manual behavior CSV (default: behavior_events.csv)",
    )
    parser.add_argument(
        "--animals",
        nargs="*",
        default=[],
        help="Display order, e.g. --animals C01 C02 C03 C04 C05 C06 C07 C08",
    )
    parser.add_argument(
        "--timezone",
        default="America/New_York",
        help="IANA timezone (default: America/New_York for Woods Hole)",
    )
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        default=Path("recording_coverage.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recording_behavior_timeline.png"),
    )
    parser.add_argument(
        "--annotate-clips",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Annotate clip durations. Default: on for <=60 clips, off for larger datasets.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    local_tz = ZoneInfo(args.timezone)
    clips = discover_clips(root, local_tz)
    if not clips:
        raise SystemExit(f"No timestamp files found under {root}")

    write_coverage_csv(clips, args.coverage_csv)
    ensure_event_csv(args.events)
    events = read_events(args.events, local_tz)

    annotate = args.annotate_clips if args.annotate_clips is not None else len(clips) <= 60
    plot_timeline(clips, events, args.animals, local_tz, args.output, annotate)

    print(f"Found {len(clips)} clips")
    print(f"Wrote: {args.coverage_csv}")
    print(f"Manual annotations: {args.events}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
