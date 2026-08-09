#!/usr/bin/env python3

from __future__ import annotations

import csv
import datetime as dt
import gzip
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import plot_recording_timeline as timeline


class TimelinePlotTests(unittest.TestCase):
    def capture_timeline_figure(
        self,
        *,
        clips: list[timeline.RecordingClip],
        events: list[timeline.BehaviorEvent],
        animals: list[str] | None = None,
    ):
        captured: dict[str, object] = {}

        def fake_savefig(fig, *args, **kwargs):
            captured["fig"] = fig

        with mock.patch("matplotlib.figure.Figure.savefig", new=fake_savefig):
            with mock.patch("matplotlib.pyplot.close"):
                timeline.plot_recording_timeline(
                    clips=clips,
                    events=events,
                    animals=animals or timeline.ANIMAL_ORDER,
                    timezone=dt.timezone(dt.timedelta(hours=-4)),
                    output_path=Path("ignored.png"),
                    annotate_clips=False,
                )

        fig = captured["fig"]
        self.addCleanup(timeline.plt.close, fig)
        return fig

    def write_timestamp_clip(
        self,
        clip_dir: Path,
        *,
        name: str = "camera1",
        timestamps: list[dt.datetime],
        gzip_output: bool = True,
        malformed: bool = False,
    ) -> tuple[Path, Path]:
        clip_dir.mkdir(parents=True, exist_ok=True)
        video_path = clip_dir / f"{name}.mp4"
        video_path.write_bytes(b"fake-mp4")
        suffix = ".timestamps.csv.gz" if gzip_output else ".timestamps.csv"
        timestamp_path = clip_dir / f"{name}{suffix}"
        opener = gzip.open if gzip_output else open
        with opener(timestamp_path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if malformed:
                writer.writerow(["frame_index", "bad_column"])
                writer.writerow([0, "oops"])
            else:
                writer.writerow(
                    [
                        "frame_index",
                        "host_utc_ns",
                        "host_utc_iso",
                        "host_monotonic_ns",
                        "camera_timestamp",
                        "block_id",
                        "skipped_images",
                    ]
                )
                for index, timestamp in enumerate(timestamps):
                    ns = int(timestamp.timestamp() * 1e9)
                    writer.writerow(
                        [
                            index,
                            ns,
                            timestamp.isoformat().replace("+00:00", "Z"),
                            1000 + index,
                            "",
                            "",
                            0,
                        ]
                    )
        return timestamp_path, video_path

    def test_extracts_timestamp_stats_from_gzip_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_dir = root / "session_a" / "clip_0000"
            timestamps = [
                dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2026, 8, 9, 19, 0, 1, tzinfo=dt.timezone.utc),
                dt.datetime(2026, 8, 9, 19, 0, 2, tzinfo=dt.timezone.utc),
            ]
            timestamp_path, video_path = self.write_timestamp_clip(clip_dir, timestamps=timestamps)

            clip = timeline.read_timestamp_clip(timestamp_path, root)

            self.assertEqual(clip.frames, 3)
            self.assertEqual(clip.start_utc, timestamps[0])
            self.assertEqual(clip.end_utc, timestamps[-1])
            self.assertAlmostEqual(clip.duration_s, 2.0)
            self.assertEqual(clip.video_file, video_path)

    def test_discovers_timestamp_files_and_ignores_cropped_by_caterpillar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_path, _ = self.write_timestamp_clip(
                root / "session_a" / "clip_0000",
                timestamps=[dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc)],
            )
            self.write_timestamp_clip(
                root / "cropped_by_caterpillar" / "session_b" / "clip_0001",
                timestamps=[dt.datetime(2026, 8, 9, 19, 5, 0, tzinfo=dt.timezone.utc)],
            )

            discovered = timeline.discover_timestamp_files(root)

            self.assertEqual(discovered, [valid_path])

    def test_timezone_conversion_and_label_are_safe(self) -> None:
        local_tz = timeline.load_timezone("America/New_York")
        utc_value = dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc)
        local_value = utc_value.astimezone(local_tz)

        self.assertEqual(local_value.isoformat(sep=" "), "2026-08-09 15:00:00-04:00")
        self.assertEqual(timeline.timezone_label(dt.timezone(dt.timedelta(hours=-4))), "UTC-04:00")

    def test_bad_timestamp_file_is_skipped_but_valid_clip_survives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_timestamp_clip(
                root / "session_a" / "clip_0000",
                timestamps=[dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc)],
            )
            self.write_timestamp_clip(
                root / "session_a" / "clip_0001",
                timestamps=[],
                malformed=True,
            )

            clips, warnings = timeline.collect_recording_clips(root)

            self.assertEqual(len(clips), 1)
            self.assertEqual(len(warnings), 1)
            self.assertEqual(clips[0].camera_label, "camera1")

    def test_main_generates_png_and_coverage_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_timestamp_clip(
                root / "session_a" / "clip_0000",
                timestamps=[
                    dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 8, 9, 19, 0, 1, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 8, 9, 19, 0, 2, tzinfo=dt.timezone.utc),
                ],
            )
            events_path = root / "behavior_events.csv"
            with events_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["animal_id", "start_local", "end_local", "event", "kind", "notes"])
                writer.writerow(["C01", "2026-08-09 15:00:00", "2026-08-09 15:05:00", "feeding", "event", ""])

            coverage_csv = root / "recording_coverage.csv"
            output_png = root / "recording_behavior_timeline.png"

            rc = timeline.main(
                [
                    str(root),
                    "--events",
                    str(events_path),
                    "--coverage-csv",
                    str(coverage_csv),
                    "--output",
                    str(output_png),
                    "--no-annotate-clips",
                    "--animals",
                    "C01",
                    "C02",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue(coverage_csv.exists())
            self.assertTrue(output_png.exists())
            self.assertGreater(output_png.stat().st_size, 0)

            with coverage_csv.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["frames"], "3")
            self.assertEqual(rows[0]["start_utc"], "2026-08-09T19:00:00.000000Z")

    def test_blank_end_local_becomes_point_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "behavior_events.csv"
            with events_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["animal_id", "start_local", "end_local", "event", "kind", "notes"])
                writer.writerow(["C01", "2026-08-09 15:00:00", "", "electrical_stimulation", "event", ""])
                writer.writerow(["", "", "2026-08-09 15:10:00", "shed", "event", "unknown start"])

            events = timeline.load_behavior_events(events_path, dt.timezone(dt.timedelta(hours=-4)))

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event, "electrical_stimulation")
            self.assertEqual(
                (events[0].end_local - events[0].start_local).total_seconds(),
                float(timeline.POINT_EVENT_DURATION_SECONDS),
            )

    def test_missing_start_local_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "behavior_events.csv"
            with events_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["animal_id", "start_local", "end_local", "event", "kind", "notes"])
                writer.writerow(["C06", "", "", "shed", "development", "missing start"])

            events = timeline.load_behavior_events(
                events_path,
                dt.timezone(dt.timedelta(hours=-4)),
            )

            self.assertEqual(events, [])

    def test_bad_explicit_interval_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "behavior_events.csv"
            with events_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["animal_id", "start_local", "end_local", "event", "kind", "notes"])
                writer.writerow(
                    [
                        "C06",
                        "2026-08-07T14:32:00-04:00",
                        "2026-08-07T14:32:00-04:00",
                        "shed",
                        "development",
                        "bad interval",
                    ]
                )

            events = timeline.load_behavior_events(
                events_path,
                dt.timezone(dt.timedelta(hours=-4)),
            )

            self.assertEqual(events, [])

    def test_short_events_draw_a_marker_without_changing_the_bar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_png = root / "timeline.png"
            events = [
                timeline.BehaviorEvent(
                    animal_id="C01",
                    start_local=dt.datetime(2026, 8, 9, 15, 0, 0),
                    end_local=dt.datetime(2026, 8, 9, 15, 0, 1),
                    event="electrical_stimulation",
                    kind="event",
                    notes="",
                ),
                timeline.BehaviorEvent(
                    animal_id="C01",
                    start_local=dt.datetime(2026, 8, 9, 15, 5, 0),
                    end_local=dt.datetime(2026, 8, 9, 15, 15, 0),
                    event="feeding",
                    kind="event",
                    notes="",
                ),
            ]

            with mock.patch("matplotlib.axes.Axes.scatter") as mock_scatter:
                timeline.plot_recording_timeline(
                    clips=[],
                    events=events,
                    animals=timeline.ANIMAL_ORDER,
                    timezone=dt.timezone(dt.timedelta(hours=-4)),
                    output_path=output_png,
                    annotate_clips=False,
                )

            self.assertTrue(output_png.exists())
            self.assertEqual(mock_scatter.call_count, 1)
            scatter_args, scatter_kwargs = mock_scatter.call_args
            self.assertEqual(scatter_kwargs["marker"], "*")
            self.assertEqual(scatter_kwargs["s"], timeline.STIM_MARKER_SIZE)
            self.assertEqual(scatter_kwargs["edgecolors"], "black")
            self.assertEqual(scatter_kwargs["linewidths"], 0.6)

    def test_behavior_axis_uses_canonical_animal_order(self) -> None:
        events = [
            timeline.BehaviorEvent(
                animal_id="C08",
                start_local=dt.datetime(2026, 8, 9, 15, 0, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 0, 1),
                event="shed",
                kind="event",
                notes="",
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 5, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 5, 1),
                event="death",
                kind="event",
                notes="",
            ),
            timeline.BehaviorEvent(
                animal_id="C06",
                start_local=dt.datetime(2026, 8, 9, 15, 10, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 15, 0),
                event="feeding",
                kind="event",
                notes="",
            ),
        ]

        fig = self.capture_timeline_figure(clips=[], events=events, animals=["C08", "C01", "C06"])
        ax_beh = fig.axes[1]

        self.assertEqual(
            [tick.get_text() for tick in ax_beh.get_yticklabels()],
            timeline.ANIMAL_ORDER,
        )

    def test_empty_animal_rows_remain_visible(self) -> None:
        events = [
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 0, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 20, 0),
                event="feeding",
                kind="event",
                notes="",
            ),
            timeline.BehaviorEvent(
                animal_id="C08",
                start_local=dt.datetime(2026, 8, 9, 16, 0, 0),
                end_local=dt.datetime(2026, 8, 9, 16, 0, 1),
                event="shed",
                kind="event",
                notes="",
            ),
        ]

        fig = self.capture_timeline_figure(clips=[], events=events)
        ax_beh = fig.axes[1]

        self.assertEqual(
            [tick.get_text() for tick in ax_beh.get_yticklabels()],
            timeline.ANIMAL_ORDER,
        )

    def test_shed_and_molt_use_triangle_marker(self) -> None:
        self.assertEqual(timeline.get_point_event_style("shed")[0], "^")
        self.assertEqual(timeline.get_point_event_style("molt")[0], "^")

    def test_electrical_stimulation_uses_large_star_marker(self) -> None:
        marker, size, _color = timeline.get_point_event_style("electrical_stimulation")

        self.assertEqual(marker, "*")
        self.assertEqual(size, timeline.STIM_MARKER_SIZE)

    def test_death_uses_x_marker(self) -> None:
        marker, _size, _color = timeline.get_point_event_style("death")

        self.assertEqual(marker, "X")

    def test_unknown_point_event_uses_generic_circle(self) -> None:
        marker, size, _color = timeline.get_point_event_style("feeding_resumed")

        self.assertEqual(marker, "o")
        self.assertEqual(size, timeline.DEFAULT_POINT_MARKER_SIZE)

    def test_long_interval_does_not_create_point_marker(self) -> None:
        event = timeline.BehaviorEvent(
            animal_id="C01",
            start_local=dt.datetime(2026, 8, 9, 15, 0, 0),
            end_local=dt.datetime(2026, 8, 9, 16, 0, 0),
            event="feeding",
            kind="event",
            notes="",
        )

        with mock.patch("matplotlib.axes.Axes.scatter") as mock_scatter:
            with mock.patch("matplotlib.axes.Axes.broken_barh") as mock_broken_barh:
                timeline.plot_recording_timeline(
                    clips=[],
                    events=[event],
                    animals=timeline.ANIMAL_ORDER,
                    timezone=dt.timezone(dt.timedelta(hours=-4)),
                    output_path=Path("ignored.png"),
                    annotate_clips=False,
                )

        self.assertEqual(mock_scatter.call_count, 0)
        self.assertGreaterEqual(mock_broken_barh.call_count, 1)


if __name__ == "__main__":
    unittest.main()
