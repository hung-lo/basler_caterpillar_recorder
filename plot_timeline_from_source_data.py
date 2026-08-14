#!/usr/bin/env python3
"""
Recreate the monarch caterpillar recording + behavior timeline from a frozen
portable source_data directory.

This wrapper deliberately reads recording_coverage.csv directly, so the
original MP4 recordings and *.timestamps.csv(.gz) files are not required for
figure reproduction.

The actual plotting implementation is imported from the archived
plot_recording_timeline.py sitting beside this file. That preserves the exact
plot style/semantics from the repository snapshot archived with the data.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import plot_recording_timeline as timeline


def _optional_int(value: str | None) -> int | None:
    text = str(value or "").strip()
    return int(text) if text else None


def load_frozen_recording_coverage(
    path: Path,
) -> list[timeline.RecordingClip]:
    """Reconstruct RecordingClip objects from frozen recording_coverage.csv."""
    clips: list[timeline.RecordingClip] = []

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            try:
                clip_id = str(row.get("clip_id") or "").strip()
                if not clip_id:
                    raise ValueError("missing clip_id")

                timestamp_text = str(row.get("timestamp_file") or "").strip()
                video_text = str(row.get("video_file") or "").strip()

                timestamp_file = Path(timestamp_text) if timestamp_text else Path(
                    f"{clip_id}.timestamps.csv.gz"
                )
                video_file = Path(video_text) if video_text else Path(f"{clip_id}.mp4")

                if timestamp_text:
                    camera_label = timeline.strip_timestamp_suffix(timestamp_file.name)
                else:
                    camera_label = Path(clip_id).name

                start_utc = timeline.parse_utc_value(
                    str(row.get("start_utc") or "").strip()
                )
                end_utc = timeline.parse_utc_value(
                    str(row.get("end_utc") or "").strip()
                )
                if end_utc <= start_utc:
                    raise ValueError("end_utc must be after start_utc")

                duration_text = str(row.get("duration_s") or "").strip()
                duration_s = (
                    float(duration_text)
                    if duration_text
                    else (end_utc - start_utc).total_seconds()
                )

                frames_text = str(row.get("frames") or "").strip()
                frames = int(frames_text) if frames_text else 0

                clips.append(
                    timeline.RecordingClip(
                        clip_id=clip_id,
                        timestamp_file=timestamp_file,
                        video_file=video_file,
                        camera_label=camera_label,
                        start_utc=start_utc,
                        end_utc=end_utc,
                        duration_s=duration_s,
                        frames=frames,
                        timestamp_size_bytes=_optional_int(
                            row.get("timestamp_size_bytes")
                        ),
                        timestamp_mtime_ns=_optional_int(
                            row.get("timestamp_mtime_ns")
                        ),
                    )
                )
            except Exception as exc:
                raise ValueError(
                    f"Malformed recording coverage row {row_number} in {path}: {exc}"
                ) from exc

    clips.sort(key=lambda clip: (clip.start_utc, clip.clip_id))
    return clips


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate the frozen monarch caterpillar timeline from files archived "
            "in a source_data directory."
        )
    )
    parser.add_argument(
        "source_data",
        nargs="?",
        default=".",
        help="Portable source_data directory (default: current directory).",
    )
    parser.add_argument(
        "--timezone",
        default=timeline.DEFAULT_TIMEZONE,
        help=f"Display timezone (default: {timeline.DEFAULT_TIMEZONE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output PNG. Defaults to "
            "<source_data>/recording_behavior_timeline.png."
        ),
    )
    parser.add_argument(
        "--annotate-clips",
        dest="annotate_clips",
        action="store_true",
        help="Enable clip labels using the archived clip metadata.",
    )
    parser.add_argument(
        "--no-annotate-clips",
        dest="annotate_clips",
        action="store_false",
        help="Disable clip labels.",
    )
    parser.set_defaults(annotate_clips=True)
    return parser


def require_file(root: Path, name: str) -> Path:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(f"Required source-data file is missing: {path}")
    return path


def main() -> int:
    args = build_parser().parse_args()

    root = Path(args.source_data).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"source_data must be an existing directory: {root}")

    coverage_path = require_file(root, "recording_coverage.csv")
    behavior_path = require_file(root, "behavior_events_used.csv")
    feeding_path = require_file(root, "feeding_events.csv")
    motion_path = require_file(root, "motion_states.csv")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "recording_behavior_timeline.png"
    )

    local_tz = timeline.load_timezone(args.timezone)

    # Frozen recording coverage.
    clips = load_frozen_recording_coverage(coverage_path)

    # Frozen manual/global behavior annotations.
    events, global_events = timeline.load_event_tables(behavior_path, local_tz)

    # Frozen automatic feeding bouts use the same event-table schema as the
    # normal plot_recording_timeline.py command.
    feeding_events, feeding_global_events = timeline.load_event_tables(
        feeding_path, local_tz
    )
    events.extend(feeding_events)
    global_events.extend(feeding_global_events)

    # Frozen motion-derived categorical states.
    motion_states = timeline.load_motion_states(motion_path)

    # Match the current command: no optional quantitative motion-energy panel.
    motion_energy_samples: list[timeline.MotionEnergySample] = []

    # Match current main() behavior after feeding events have been added.
    animals = timeline.infer_animals(events) or list(timeline.DEFAULT_ANIMALS)

    timeline.plot_recording_timeline(
        clips,
        events,
        motion_states,
        animals,
        global_events=global_events,
        motion_energy_samples=motion_energy_samples,
        motion_plot_bin_minutes=1,
        motion_plot_stat="p90",
        timezone=local_tz,
        output_path=output_path,
        annotate_clips=args.annotate_clips,
        profile_timing=False,
    )

    print(f"Wrote recreated timeline: {output_path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
