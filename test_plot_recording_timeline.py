#!/usr/bin/env python3

from __future__ import annotations

import csv
import datetime as dt
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from urllib import error as urllib_error
from unittest import mock

import plot_recording_timeline as timeline


class TimelinePlotTests(unittest.TestCase):
    class FakeHTTPResponse:
        def __init__(self, data: bytes):
            self._data = data

        def read(self) -> bytes:
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def capture_timeline_figure(
        self,
        *,
        clips: list[timeline.RecordingClip],
        events: list[timeline.BehaviorEvent],
        motion_states: list[timeline.MotionState] | None = None,
        motion_energy_samples: list[timeline.MotionEnergySample] | None = None,
        global_events: list[timeline.GlobalEvent] | None = None,
        animals: list[str] | None = None,
        timezone: dt.tzinfo | None = None,
    ):
        captured: dict[str, object] = {}

        def fake_savefig(fig, *args, **kwargs):
            captured["fig"] = fig

        with mock.patch("matplotlib.figure.Figure.savefig", new=fake_savefig):
            with mock.patch("matplotlib.pyplot.close"):
                timeline.plot_recording_timeline(
                    clips=clips,
                    events=events,
                    motion_states=motion_states or [],
                    animals=animals or timeline.ANIMAL_ORDER,
                    global_events=global_events or [],
                    motion_energy_samples=motion_energy_samples or [],
                    timezone=timezone or dt.timezone(dt.timedelta(hours=-4)),
                    output_path=Path("ignored.png"),
                    annotate_clips=False,
                )

        fig = captured["fig"]
        self.addCleanup(timeline.plt.close, fig)
        return fig

    def behavior_axis(self, fig):
        return fig.axes[-1]

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

    def write_motion_states_csv(self, path: Path, rows: list[list[object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
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
            )
            for row in rows:
                writer.writerow(row)

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
            self.assertIn("timestamp_size_bytes", rows[0])
            self.assertIn("timestamp_mtime_ns", rows[0])
            self.assertTrue(rows[0]["timestamp_size_bytes"])
            self.assertTrue(rows[0]["timestamp_mtime_ns"])

    def test_collect_recording_clips_reuses_coverage_cache_when_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_timestamp_clip(
                root / "session_a" / "clip_0000",
                timestamps=[
                    dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 8, 9, 19, 0, 1, tzinfo=dt.timezone.utc),
                ],
            )
            self.write_timestamp_clip(
                root / "session_a" / "clip_0001",
                timestamps=[
                    dt.datetime(2026, 8, 9, 19, 5, 0, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 8, 9, 19, 5, 1, tzinfo=dt.timezone.utc),
                ],
            )

            initial_clips, _warnings = timeline.collect_recording_clips(root)
            coverage_csv = root / "recording_coverage.csv"
            timeline.write_coverage_csv(initial_clips, coverage_csv, dt.timezone(dt.timedelta(hours=-4)))

            with mock.patch.object(
                timeline,
                "read_timestamp_clip",
                side_effect=AssertionError("cache hit should not reparse timestamp files"),
            ):
                cached_clips, warnings = timeline.collect_recording_clips(root, coverage_cache=coverage_csv)

            self.assertEqual(warnings, [])
            self.assertEqual([clip.clip_id for clip in cached_clips], [clip.clip_id for clip in initial_clips])
            self.assertEqual([clip.frames for clip in cached_clips], [2, 2])

    def test_collect_recording_clips_reparses_only_changed_timestamp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_a_dir = root / "session_a" / "clip_0000"
            clip_b_dir = root / "session_a" / "clip_0001"
            self.write_timestamp_clip(
                clip_a_dir,
                timestamps=[
                    dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 8, 9, 19, 0, 1, tzinfo=dt.timezone.utc),
                ],
            )
            self.write_timestamp_clip(
                clip_b_dir,
                timestamps=[
                    dt.datetime(2026, 8, 9, 19, 5, 0, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 8, 9, 19, 5, 1, tzinfo=dt.timezone.utc),
                ],
            )

            initial_clips, _warnings = timeline.collect_recording_clips(root)
            coverage_csv = root / "recording_coverage.csv"
            timeline.write_coverage_csv(initial_clips, coverage_csv, dt.timezone(dt.timedelta(hours=-4)))

            self.write_timestamp_clip(
                clip_b_dir,
                timestamps=[
                    dt.datetime(2026, 8, 9, 19, 5, 0, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 8, 9, 19, 5, 1, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 8, 9, 19, 5, 2, tzinfo=dt.timezone.utc),
                ],
            )

            with mock.patch.object(
                timeline,
                "read_timestamp_clip",
                wraps=timeline.read_timestamp_clip,
            ) as mock_read:
                clips, warnings = timeline.collect_recording_clips(root, coverage_cache=coverage_csv)

            self.assertEqual(warnings, [])
            self.assertEqual(mock_read.call_count, 1)
            self.assertEqual(sorted(clip.frames for clip in clips), [2, 3])

    def test_collect_recording_clips_upgrades_old_coverage_cache_schema(self) -> None:
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
            coverage_csv = root / "recording_coverage.csv"
            with coverage_csv.open("w", newline="", encoding="utf-8") as handle:
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
                writer.writerow(
                    {
                        "clip_id": "session_a/camera1",
                        "timestamp_file": str(root / "session_a" / "clip_0000" / "camera1.timestamps.csv.gz"),
                        "video_file": str(root / "session_a" / "clip_0000" / "camera1.mp4"),
                        "start_utc": "2026-08-09T19:00:00.000000Z",
                        "end_utc": "2026-08-09T19:00:02.000000Z",
                        "start_local": "2026-08-09 15:00:00-04:00",
                        "end_local": "2026-08-09 15:00:02-04:00",
                        "duration_s": "2.000000",
                        "frames": "3",
                    }
                )

            with mock.patch.object(
                timeline,
                "read_timestamp_clip",
                wraps=timeline.read_timestamp_clip,
            ) as mock_read:
                clips, warnings = timeline.collect_recording_clips(root, coverage_cache=coverage_csv)

            self.assertEqual(warnings, [])
            self.assertEqual(mock_read.call_count, 1)
            timeline.write_coverage_csv(clips, coverage_csv, dt.timezone(dt.timedelta(hours=-4)))
            with coverage_csv.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn("timestamp_size_bytes", rows[0])
            self.assertTrue(rows[0]["timestamp_size_bytes"])
            self.assertTrue(rows[0]["timestamp_mtime_ns"])

    def test_parser_preserves_point_vs_duration_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "behavior_events.csv"
            with events_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["animal_id", "start_local", "end_local", "event", "kind", "notes"])
                writer.writerow(["C01", "2026-08-10 12:00:00-04:00", "", "observation", "observation", "test"])
                writer.writerow(["C01", "2026-08-10 13:00:00-04:00", "2026-08-10 13:00:30-04:00", "observation", "observation", "test interval"])

            events = timeline.load_behavior_events(events_path, dt.timezone(dt.timedelta(hours=-4)))

            self.assertEqual(len(events), 2)
            self.assertTrue(events[0].is_point)
            self.assertFalse(events[1].is_point)
            self.assertEqual(
                (events[0].end_local - events[0].start_local).total_seconds(),
                float(timeline.POINT_EVENT_DURATION_SECONDS),
            )
            self.assertEqual(
                (events[1].end_local - events[1].start_local).total_seconds(),
                30.0,
            )

    def test_global_event_aliases_are_recognized_case_insensitively(self) -> None:
        csv_text = (
            "animal_id,start_local,end_local,event,kind,approximate,notes\n"
            "All,2026-08-09 15:00:00,2026-08-09 15:10:00,video_quality_low,video_quality,TRUE,first\n"
            "ALL,2026-08-09 15:12:00,2026-08-09 15:14:00,video_quality_low,video_quality,FALSE,third\n"
            "global,2026-08-09 15:20:00,2026-08-09 15:30:00,bad_lighting,video_quality,FALSE,second\n"
            "GLOBAL,2026-08-09 15:32:00,2026-08-09 15:34:00,bad_lighting,video_quality,FALSE,fourth\n"
            "*,2026-08-09 15:40:00,2026-08-09 15:50:00,lights_back_on,event,,fifth\n"
        )

        events, global_events = timeline.load_event_tables_from_text(
            csv_text,
            dt.timezone(dt.timedelta(hours=-4)),
            source_name="inline.csv",
        )

        self.assertEqual(events, [])
        self.assertEqual(len(global_events), 5)
        self.assertTrue(global_events[0].approximate)
        self.assertFalse(global_events[1].approximate)

    def test_is_global_event_alias_handles_all_supported_values(self) -> None:
        for alias in ["All", "ALL", "all", "Global", "GLOBAL", "global", "*"]:
            self.assertTrue(timeline.is_global_event_alias(alias))
        self.assertFalse(timeline.is_global_event_alias("C01"))

    def test_global_interval_requires_end_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "behavior_events.csv"
            with events_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["animal_id", "start_local", "end_local", "event", "kind", "notes"])
                writer.writerow(["All", "2026-08-09 15:00:00", "", "video_quality_low", "video_quality", ""])
                writer.writerow(["C01", "2026-08-09 15:10:00", "", "shed", "event", ""])

            events, global_events = timeline.load_event_tables(
                events_path,
                dt.timezone(dt.timedelta(hours=-4)),
            )

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].animal_id, "C01")
            self.assertEqual(global_events, [])

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

    def test_supported_point_events_render_and_generic_points_do_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_png = root / "timeline.png"
            events = [
                timeline.BehaviorEvent(
                    animal_id="C01",
                    start_local=dt.datetime(2026, 8, 9, 15, 0, 0),
                    end_local=dt.datetime(2026, 8, 9, 15, 0, 1),
                    event="manual_note",
                    kind="electrical_stimulation",
                    notes="",
                    is_point=True,
                ),
                timeline.BehaviorEvent(
                    animal_id="C01",
                    start_local=dt.datetime(2026, 8, 9, 15, 2, 0),
                    end_local=dt.datetime(2026, 8, 9, 15, 2, 1),
                    event="shed",
                    kind="event",
                    notes="",
                    is_point=True,
                ),
                timeline.BehaviorEvent(
                    animal_id="C02",
                    start_local=dt.datetime(2026, 8, 9, 15, 2, 30),
                    end_local=dt.datetime(2026, 8, 9, 15, 2, 31),
                    event="J_hang",
                    kind="status",
                    notes="",
                    is_point=True,
                ),
                timeline.BehaviorEvent(
                    animal_id="C03",
                    start_local=dt.datetime(2026, 8, 9, 15, 2, 45),
                    end_local=dt.datetime(2026, 8, 9, 15, 2, 46),
                    event="Pupation",
                    kind="status",
                    notes="",
                    is_point=True,
                ),
                timeline.BehaviorEvent(
                    animal_id="C04",
                    start_local=dt.datetime(2026, 8, 9, 15, 3, 0),
                    end_local=dt.datetime(2026, 8, 9, 15, 3, 1),
                    event="dead",
                    kind="event",
                    notes="",
                    is_point=True,
                ),
                timeline.BehaviorEvent(
                    animal_id="C01",
                    start_local=dt.datetime(2026, 8, 9, 15, 4, 0),
                    end_local=dt.datetime(2026, 8, 9, 15, 4, 1),
                    event="observation",
                    kind="observation",
                    notes="",
                    is_point=True,
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
                with mock.patch("matplotlib.axes.Axes.text") as mock_text:
                    timeline.plot_recording_timeline(
                        clips=[],
                        events=events,
                        motion_states=[],
                        animals=timeline.ANIMAL_ORDER,
                        timezone=dt.timezone(dt.timedelta(hours=-4)),
                        output_path=output_png,
                        annotate_clips=False,
                    )

            self.assertTrue(output_png.exists())
            self.assertEqual(mock_scatter.call_count, 4)
            scatter_markers = [call.kwargs["marker"] for call in mock_scatter.call_args_list]
            self.assertIn("^", scatter_markers)
            self.assertIn("v", scatter_markers)
            self.assertIn("D", scatter_markers)
            self.assertIn("X", scatter_markers)
            self.assertNotIn("o", scatter_markers)
            self.assertTrue(any(call.args[2] == "\u26a1" for call in mock_text.call_args_list if len(call.args) >= 3))
            _scatter_args, scatter_kwargs = mock_scatter.call_args_list[0]
            self.assertEqual(scatter_kwargs["marker"], "^")
            self.assertEqual(scatter_kwargs["s"], timeline.SHED_MARKER_SIZE)
            self.assertEqual(scatter_kwargs["edgecolors"], "black")
            self.assertEqual(scatter_kwargs["linewidths"], 0.6)

    def test_explicit_short_duration_events_remain_bars(self) -> None:
        event = timeline.BehaviorEvent(
            animal_id="C01",
            start_local=dt.datetime(2026, 8, 9, 15, 0, 0),
            end_local=dt.datetime(2026, 8, 9, 15, 0, 30),
            event="observation",
            kind="observation",
            notes="",
        )

        with mock.patch("matplotlib.axes.Axes.scatter") as mock_scatter:
            with mock.patch("matplotlib.axes.Axes.broken_barh") as mock_broken_barh:
                with mock.patch("matplotlib.axes.Axes.text") as mock_text:
                    timeline.plot_recording_timeline(
                        clips=[],
                        events=[event],
                        motion_states=[],
                        animals=timeline.ANIMAL_ORDER,
                        timezone=dt.timezone(dt.timedelta(hours=-4)),
                        output_path=Path("ignored.png"),
                        annotate_clips=False,
                    )

        self.assertEqual(mock_scatter.call_count, 0)
        self.assertGreaterEqual(mock_broken_barh.call_count, 1)
        self.assertFalse(
            any(len(call.args) >= 3 and call.args[2] == "\u26a1" for call in mock_text.call_args_list)
        )

    def test_explicit_duration_stimulation_renders_bar_and_marker(self) -> None:
        event = timeline.BehaviorEvent(
            animal_id="C01",
            start_local=dt.datetime(2026, 8, 9, 13, 37, 0),
            end_local=dt.datetime(2026, 8, 9, 13, 37, 10),
            event="electrical_stimulation",
            kind="stimulus",
            notes="50 uA pulse train at 100 Hz for 10 seconds",
            is_point=False,
        )

        with mock.patch("matplotlib.axes.Axes.broken_barh") as mock_broken_barh:
            with mock.patch("matplotlib.axes.Axes.text") as mock_text:
                timeline.plot_recording_timeline(
                    clips=[],
                    events=[event],
                    motion_states=[],
                    animals=timeline.ANIMAL_ORDER,
                    timezone=dt.timezone(dt.timedelta(hours=-4)),
                    output_path=Path("ignored.png"),
                    annotate_clips=False,
                )

        stim_bars = [call for call in mock_broken_barh.call_args_list if call.kwargs.get("facecolors") == timeline.STIM_COLOR]
        self.assertEqual(len(stim_bars), 1)
        lightning_calls = [call for call in mock_text.call_args_list if len(call.args) >= 3 and call.args[2] == "\u26a1"]
        self.assertEqual(len(lightning_calls), 1)
        self.assertEqual(lightning_calls[0].kwargs["zorder"], timeline.STIM_MARKER_ZORDER)
        self.assertFalse(lightning_calls[0].kwargs.get("clip_on", True))
        self.assertIn("path_effects", lightning_calls[0].kwargs)
        self.assertGreaterEqual(len(lightning_calls[0].kwargs["path_effects"]), 1)
        self.assertAlmostEqual(
            lightning_calls[0].args[0],
            timeline.mdates.date2num(dt.datetime(2026, 8, 9, 13, 37, 0)),
        )

    def test_long_duration_stimulation_still_renders_marker(self) -> None:
        event = timeline.BehaviorEvent(
            animal_id="C01",
            start_local=dt.datetime(2026, 8, 9, 13, 40, 0),
            end_local=dt.datetime(2026, 8, 9, 13, 50, 0),
            event="shock",
            kind="stimulus",
            notes="long stimulation",
            is_point=False,
        )

        with mock.patch("matplotlib.axes.Axes.broken_barh") as mock_broken_barh:
            with mock.patch("matplotlib.axes.Axes.text") as mock_text:
                timeline.plot_recording_timeline(
                    clips=[],
                    events=[event],
                    motion_states=[],
                    animals=timeline.ANIMAL_ORDER,
                    timezone=dt.timezone(dt.timedelta(hours=-4)),
                    output_path=Path("ignored.png"),
                    annotate_clips=False,
                )

        stim_bars = [call for call in mock_broken_barh.call_args_list if call.kwargs.get("facecolors") == timeline.STIM_COLOR]
        self.assertEqual(len(stim_bars), 1)
        lightning_calls = [call for call in mock_text.call_args_list if len(call.args) >= 3 and call.args[2] == "\u26a1"]
        self.assertEqual(len(lightning_calls), 1)
        self.assertEqual(lightning_calls[0].kwargs["zorder"], timeline.STIM_MARKER_ZORDER)
        self.assertFalse(lightning_calls[0].kwargs.get("clip_on", True))
        self.assertIn("path_effects", lightning_calls[0].kwargs)
        self.assertGreaterEqual(len(lightning_calls[0].kwargs["path_effects"]), 1)
        self.assertAlmostEqual(
            lightning_calls[0].args[0],
            timeline.mdates.date2num(dt.datetime(2026, 8, 9, 13, 40, 0)),
        )

    def test_real_stimulation_rows_render_in_final_overlay_with_diagnostics(self) -> None:
        csv_text = (
            "animal_id,start_local,end_local,event,kind,approximate,notes\n"
            "C01,2026-08-09T13:37:00-04:00,2026-08-09T13:37:10-04:00,electrical_stimulation,stimulus,FALSE,50 uA pulse train at 100 Hz for 10 seconds\n"
            "C08,2026-08-09T13:40:00-04:00,2026-08-09T13:40:10-04:00,electrical_stimulation,stimulus,FALSE,50 uA pulse train at 100 Hz for 10 seconds\n"
        )
        events, _global_events = timeline.load_event_tables_from_text(
            csv_text,
            dt.timezone(dt.timedelta(hours=-4)),
            source_name="inline.csv",
        )

        self.assertEqual(len(events), 2)
        self.assertTrue(all(timeline.is_stimulation_event(event) for event in events))

        with self.assertLogs("plot_recording_timeline", level="DEBUG") as logs:
            with mock.patch("matplotlib.axes.Axes.broken_barh") as mock_broken_barh:
                with mock.patch("matplotlib.axes.Axes.text") as mock_text:
                    timeline.plot_recording_timeline(
                        clips=[],
                        events=events,
                        motion_states=[],
                        animals=timeline.ANIMAL_ORDER,
                        timezone=dt.timezone(dt.timedelta(hours=-4)),
                        output_path=Path("ignored.png"),
                        annotate_clips=False,
                    )

        lightning_calls = [call for call in mock_text.call_args_list if len(call.args) >= 3 and call.args[2] == "\u26a1"]
        self.assertEqual(len(lightning_calls), 2)
        self.assertEqual(
            {call.kwargs["zorder"] for call in lightning_calls},
            {timeline.STIM_MARKER_ZORDER},
        )
        self.assertTrue(all(not call.kwargs.get("clip_on", True) for call in lightning_calls))
        self.assertTrue(all("path_effects" in call.kwargs for call in lightning_calls))
        self.assertGreaterEqual(
            sum(1 for call in mock_broken_barh.call_args_list if call.kwargs.get("facecolors") == timeline.STIM_COLOR),
            2,
        )
        self.assertTrue(any("Electrical stimulation events available: 2" in line for line in logs.output))
        self.assertTrue(any("Electrical stimulation markers rendered: 2" in line for line in logs.output))
        self.assertTrue(any("stimulation C01" in line for line in logs.output))
        self.assertTrue(any("stimulation C08" in line for line in logs.output))

    def test_behavior_axis_uses_canonical_animal_order(self) -> None:
        events = [
            timeline.BehaviorEvent(
                animal_id="C08",
                start_local=dt.datetime(2026, 8, 9, 15, 0, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 0, 1),
                event="shed",
                kind="event",
                notes="",
                is_point=True,
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 5, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 5, 1),
                event="death",
                kind="event",
                notes="",
                is_point=True,
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
        ax_beh = self.behavior_axis(fig)

        self.assertEqual(
            [tick.get_text() for tick in ax_beh.get_yticklabels()],
            timeline.ANIMAL_ORDER,
        )

    def test_global_events_do_not_create_a_ninth_animal_row(self) -> None:
        global_events = [
            timeline.GlobalEvent(
                start_local=dt.datetime(2026, 8, 9, 15, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
                end_local=dt.datetime(2026, 8, 9, 15, 30, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
                event="video_quality_low",
                kind="video_quality",
                notes="",
            )
        ]

        fig = self.capture_timeline_figure(clips=[], events=[], global_events=global_events)
        ax_beh = self.behavior_axis(fig)
        labels = [tick.get_text() for tick in ax_beh.get_yticklabels()]

        self.assertEqual(labels, timeline.ANIMAL_ORDER)
        self.assertEqual(len(labels), 8)
        self.assertNotIn("All", labels)

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
                is_point=True,
            ),
        ]

        fig = self.capture_timeline_figure(clips=[], events=events)
        ax_beh = self.behavior_axis(fig)

        self.assertEqual(
            [tick.get_text() for tick in ax_beh.get_yticklabels()],
            timeline.ANIMAL_ORDER,
        )

    def test_shed_and_molt_use_triangle_marker(self) -> None:
        self.assertEqual(timeline.get_point_event_style("shed")[0], "^")
        self.assertEqual(timeline.get_point_event_style("molt")[0], "^")
        self.assertEqual(timeline.get_point_event_style("J_hang")[0], "v")
        self.assertEqual(timeline.get_point_event_style("Pupation")[0], "D")

    def test_electrical_stimulation_is_recognized(self) -> None:
        self.assertTrue(timeline.is_stimulation_event_name("electrical_stimulation"))
        self.assertTrue(timeline.is_stimulation_event_name("electric_shock"))
        self.assertTrue(timeline.is_stimulation_event_name("shock"))

    def test_death_uses_x_marker(self) -> None:
        marker, _size, _color = timeline.get_point_event_style("death")

        self.assertEqual(marker, "X")

    def test_food_unavailable_behavior_interval_uses_dedicated_color_and_not_manual_interval(self) -> None:
        event = timeline.BehaviorEvent(
            animal_id="C01",
            start_local=dt.datetime(2026, 8, 9, 15, 50, 0),
            end_local=dt.datetime(2026, 8, 9, 15, 55, 0),
            event="food_unavailable",
            kind="event",
            notes="behavior interval",
        )

        self.assertFalse(timeline.is_manual_interval_event(event))
        self.assertEqual(timeline.event_bar_color(event), timeline.FOOD_UNAVAILABLE_COLOR)

        legend_handle = timeline.food_unavailable_legend_handle([event], [])
        self.assertIsNotNone(legend_handle)
        self.assertEqual(legend_handle.get_alpha(), 0.74)

        labels = [handle.get_label() for handle in timeline.timeline_legend_handles(
            events=[event],
            motion_states=[],
            global_events=[],
        )]

        self.assertIn("Food unavailable", labels)
        self.assertNotIn("Manual interval", labels)

    def test_food_unavailable_global_event_uses_lighter_legend_alpha(self) -> None:
        global_event = timeline.GlobalEvent(
            start_local=dt.datetime(2026, 8, 9, 15, 30, 0, tzinfo=dt.timezone.utc),
            end_local=dt.datetime(2026, 8, 9, 15, 45, 0, tzinfo=dt.timezone.utc),
            event="food_unavailable",
            kind="event",
            notes="global interval",
        )

        handle = timeline.food_unavailable_legend_handle([], [global_event])

        self.assertIsNotNone(handle)
        self.assertEqual(handle.get_alpha(), timeline.FOOD_UNAVAILABLE_ALPHA)

    def test_low_mobility_interval_is_hidden_and_not_manual_interval(self) -> None:
        low_mobility_event = timeline.BehaviorEvent(
            animal_id="C03",
            start_local=dt.datetime(2026, 8, 9, 15, 40, 0),
            end_local=dt.datetime(2026, 8, 9, 16, 0, 0),
            event="low_mobility",
            kind="event",
            notes="manual low mobility",
        )
        manual_interval_event = timeline.BehaviorEvent(
            animal_id="C03",
            start_local=dt.datetime(2026, 8, 9, 16, 10, 0),
            end_local=dt.datetime(2026, 8, 9, 16, 30, 0),
            event="observation",
            kind="event",
            notes="manual interval",
        )

        self.assertTrue(timeline.is_low_mobility_event(low_mobility_event))
        self.assertFalse(timeline.is_manual_interval_event(low_mobility_event))
        self.assertFalse(timeline.behavior_event_is_visible(low_mobility_event))
        self.assertTrue(timeline.is_manual_interval_event(manual_interval_event))

        with mock.patch("matplotlib.axes.Axes.broken_barh") as mock_broken_barh:
            timeline.plot_recording_timeline(
                clips=[],
                events=[low_mobility_event, manual_interval_event],
                motion_states=[],
                animals=timeline.ANIMAL_ORDER,
                timezone=dt.timezone(dt.timedelta(hours=-4)),
                output_path=Path("ignored.png"),
                annotate_clips=False,
            )

        self.assertEqual(mock_broken_barh.call_count, 1)
        self.assertEqual(mock_broken_barh.call_args.kwargs["facecolors"], timeline.INTERVAL_BAR_COLOR)

    def test_display_time_bounds_adds_small_symmetric_padding(self) -> None:
        plot_start = dt.datetime(2026, 8, 9, 15, 0, 0)
        plot_end = dt.datetime(2026, 8, 9, 15, 30, 0)

        display_start, display_end = timeline.display_time_bounds(plot_start, plot_end)

        self.assertLess(display_start, plot_start)
        self.assertGreater(display_end, plot_end)
        self.assertGreaterEqual(display_end - plot_end, dt.timedelta(minutes=10))
        self.assertGreaterEqual(plot_start - display_start, dt.timedelta(minutes=10))

        fig = self.capture_timeline_figure(
            clips=[
                timeline.RecordingClip(
                    clip_id="session_a/camera1",
                    timestamp_file=Path("camera1.timestamps.csv.gz"),
                    video_file=Path("camera1.mp4"),
                    camera_label="camera1",
                    start_utc=dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                    end_utc=dt.datetime(2026, 8, 9, 19, 30, 0, tzinfo=dt.timezone.utc),
                    duration_s=1800.0,
                    frames=9000,
                    timestamp_size_bytes=123,
                    timestamp_mtime_ns=456,
                )
            ],
            events=[
                timeline.BehaviorEvent(
                    animal_id="C01",
                    start_local=dt.datetime(2026, 8, 9, 15, 30, 0),
                    end_local=dt.datetime(2026, 8, 9, 15, 30, 1),
                    event="Pupation",
                    kind="status",
                    notes="",
                    is_point=True,
                )
            ],
        )

        coverage_xlim = fig.axes[0].get_xlim()
        behavior_xlim = fig.axes[-1].get_xlim()
        self.assertEqual(coverage_xlim, behavior_xlim)
        scientific_start = timeline.mdates.date2num(plot_start)
        scientific_end = timeline.mdates.date2num(plot_end)
        self.assertLess(coverage_xlim[0], scientific_start)
        self.assertGreater(coverage_xlim[1], scientific_end)

    def test_death_cutoffs_use_earliest_death_per_animal(self) -> None:
        events = [
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 20, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 20, 1),
                event="death",
                kind="event",
                notes="",
                is_point=True,
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 10, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 10, 1),
                event="dead",
                kind="event",
                notes="",
                is_point=True,
            ),
            timeline.BehaviorEvent(
                animal_id="C02",
                start_local=dt.datetime(2026, 8, 9, 16, 0, 0),
                end_local=dt.datetime(2026, 8, 9, 16, 0, 1),
                event="feeding",
                kind="event",
                notes="",
            ),
        ]

        self.assertEqual(
            timeline.death_cutoffs_local(events),
            {"C01": dt.datetime(2026, 8, 9, 15, 10, 0)},
        )

    def test_unknown_point_events_are_hidden_from_the_plot(self) -> None:
        event = timeline.BehaviorEvent(
            animal_id="C01",
            start_local=dt.datetime(2026, 8, 9, 15, 0, 0),
            end_local=dt.datetime(2026, 8, 9, 15, 0, 1),
            event="feeding_resumed",
            kind="observation",
            notes="",
            is_point=True,
        )

        with mock.patch("matplotlib.axes.Axes.scatter") as mock_scatter:
            with mock.patch("matplotlib.axes.Axes.text") as mock_text:
                timeline.plot_recording_timeline(
                    clips=[],
                    events=[event],
                    motion_states=[],
                    animals=timeline.ANIMAL_ORDER,
                    timezone=dt.timezone(dt.timedelta(hours=-4)),
                    output_path=Path("ignored.png"),
                    annotate_clips=False,
                )

        self.assertEqual(mock_scatter.call_count, 0)
        self.assertFalse(
            any(len(call.args) >= 3 and call.args[2] == "\u26a1" for call in mock_text.call_args_list)
        )

    def test_manual_annotation_legend_uses_expected_labels(self) -> None:
        labels = [handle.get_label() for handle in timeline.manual_annotation_legend_handles()]

        self.assertIn("Manual interval", labels)
        self.assertIn("Shed / molt", labels)
        self.assertIn("J-hang", labels)
        self.assertIn("Pupation", labels)
        self.assertIn("\u26a1 Electrical stimulation", labels)
        self.assertIn("Death", labels)
        self.assertNotIn("Duration / state", labels)

    def test_timeline_legend_handles_are_ordered_and_named(self) -> None:
        motion_states = [
            timeline.MotionState(
                animal_id="C01",
                clip_key="clip_1",
                start_utc=dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 19, 5, 0, tzinfo=dt.timezone.utc),
                state="mobile",
                threshold=4.0,
                threshold_source="manual",
                mean_motion_energy=5.0,
                peak_motion_energy=7.0,
                n_windows=300,
            )
        ]
        events = [
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 0, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 30, 0),
                event="feeding",
                kind="event",
                notes="",
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 40, 0),
                end_local=dt.datetime(2026, 8, 9, 16, 0, 0),
                event="observation",
                kind="event",
                notes="manual interval",
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 50, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 55, 0),
                event="food_unavailable",
                kind="event",
                notes="behavior interval",
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 16, 5, 0),
                end_local=dt.datetime(2026, 8, 9, 16, 5, 1),
                event="shed",
                kind="event",
                notes="",
                is_point=True,
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 16, 6, 0),
                end_local=dt.datetime(2026, 8, 9, 16, 6, 1),
                event="J_hang",
                kind="status",
                notes="",
                is_point=True,
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 16, 7, 0),
                end_local=dt.datetime(2026, 8, 9, 16, 7, 1),
                event="Pupation",
                kind="status",
                notes="",
                is_point=True,
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 16, 10, 0),
                end_local=dt.datetime(2026, 8, 9, 16, 10, 1),
                event="electrical_stimulation",
                kind="stimulus",
                notes="",
                is_point=True,
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 16, 20, 0),
                end_local=dt.datetime(2026, 8, 9, 16, 20, 1),
                event="dead",
                kind="event",
                notes="",
                is_point=True,
            ),
        ]
        global_events = [
            timeline.GlobalEvent(
                start_local=dt.datetime(2026, 8, 9, 15, 0, 0, tzinfo=dt.timezone.utc),
                end_local=dt.datetime(2026, 8, 9, 15, 20, 0, tzinfo=dt.timezone.utc),
                event="video_quality_low",
                kind="video_quality",
                notes="",
            ),
            timeline.GlobalEvent(
                start_local=dt.datetime(2026, 8, 9, 15, 30, 0, tzinfo=dt.timezone.utc),
                end_local=dt.datetime(2026, 8, 9, 15, 45, 0, tzinfo=dt.timezone.utc),
                event="food_unavailable",
                kind="event",
                notes="",
            ),
        ]

        labels = [handle.get_label() for handle in timeline.timeline_legend_handles(
            events=events,
            motion_states=motion_states,
            global_events=global_events,
        )]

        self.assertEqual(
            labels,
            [
                "Shed / molt",
                "J-hang",
                "Pupation",
                "\u26a1 Electrical stimulation",
                "Death",
                "Automatic feeding bouts",
                "Motion-derived immobile",
                "Motion-derived mobile",
                "Low video quality",
                "Food unavailable",
                "Manual interval",
            ],
        )

    def test_motion_alpha_constants_are_shared_between_bars_and_legend(self) -> None:
        motion_states = [
            timeline.MotionState(
                animal_id="C01",
                clip_key="clip_1",
                start_utc=dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 19, 5, 0, tzinfo=dt.timezone.utc),
                state="immobile",
                threshold=4.0,
                threshold_source="manual",
                mean_motion_energy=1.0,
                peak_motion_energy=2.0,
                n_windows=300,
            ),
            timeline.MotionState(
                animal_id="C02",
                clip_key="clip_1",
                start_utc=dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 19, 5, 0, tzinfo=dt.timezone.utc),
                state="mobile",
                threshold=4.0,
                threshold_source="manual",
                mean_motion_energy=5.0,
                peak_motion_energy=7.0,
                n_windows=300,
            ),
        ]

        with mock.patch("matplotlib.axes.Axes.broken_barh") as mock_broken_barh:
            timeline.plot_recording_timeline(
                clips=[],
                events=[],
                motion_states=motion_states,
                animals=timeline.ANIMAL_ORDER,
                timezone=dt.timezone(dt.timedelta(hours=-4)),
                output_path=Path("ignored.png"),
                annotate_clips=False,
            )

        motion_alphas = sorted(
            call.kwargs["alpha"]
            for call in mock_broken_barh.call_args_list
            if call.kwargs.get("alpha") in {timeline.MOTION_IMMOBILE_ALPHA, timeline.MOTION_MOBILE_ALPHA}
        )
        self.assertEqual(motion_alphas, [timeline.MOTION_IMMOBILE_ALPHA, timeline.MOTION_MOBILE_ALPHA])
        self.assertLess(timeline.MOTION_IMMOBILE_ALPHA, timeline.MOTION_MOBILE_ALPHA)

        legend_alphas = [handle.get_alpha() for handle in timeline.motion_legend_handles()]
        self.assertEqual(legend_alphas, [timeline.MOTION_IMMOBILE_ALPHA, timeline.MOTION_MOBILE_ALPHA])

    def test_draw_timeline_legend_uses_single_row_and_stable_order(self) -> None:
        motion_states = [
            timeline.MotionState(
                animal_id="C01",
                clip_key="clip_1",
                start_utc=dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 19, 5, 0, tzinfo=dt.timezone.utc),
                state="immobile",
                threshold=4.0,
                threshold_source="manual",
                mean_motion_energy=1.0,
                peak_motion_energy=2.0,
                n_windows=300,
            ),
            timeline.MotionState(
                animal_id="C02",
                clip_key="clip_1",
                start_utc=dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 19, 5, 0, tzinfo=dt.timezone.utc),
                state="mobile",
                threshold=4.0,
                threshold_source="manual",
                mean_motion_energy=5.0,
                peak_motion_energy=7.0,
                n_windows=300,
            ),
        ]
        events = [
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 0, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 30, 0),
                event="feeding",
                kind="event",
                notes="",
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 40, 0),
                end_local=dt.datetime(2026, 8, 9, 16, 0, 0),
                event="observation",
                kind="event",
                notes="manual interval",
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 16, 5, 0),
                end_local=dt.datetime(2026, 8, 9, 16, 5, 1),
                event="shed",
                kind="event",
                notes="",
                is_point=True,
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 16, 10, 0),
                end_local=dt.datetime(2026, 8, 9, 16, 10, 1),
                event="electrical_stimulation",
                kind="stimulus",
                notes="",
                is_point=True,
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 16, 20, 0),
                end_local=dt.datetime(2026, 8, 9, 16, 20, 1),
                event="dead",
                kind="event",
                notes="",
                is_point=True,
            ),
        ]
        global_events = [
            timeline.GlobalEvent(
                start_local=dt.datetime(2026, 8, 9, 15, 0, 0, tzinfo=dt.timezone.utc),
                end_local=dt.datetime(2026, 8, 9, 15, 20, 0, tzinfo=dt.timezone.utc),
                event="video_quality_low",
                kind="video_quality",
                notes="",
            ),
            timeline.GlobalEvent(
                start_local=dt.datetime(2026, 8, 9, 15, 30, 0, tzinfo=dt.timezone.utc),
                end_local=dt.datetime(2026, 8, 9, 15, 45, 0, tzinfo=dt.timezone.utc),
                event="food_unavailable",
                kind="event",
                notes="",
            ),
        ]

        legend_handle_map = timeline.timeline_legend_handle_map(
            events=events,
            motion_states=motion_states,
            global_events=global_events,
        )
        axis = mock.Mock()

        legend = timeline.draw_timeline_legend(axis, legend_handle_map)

        self.assertIsNotNone(legend)
        axis.axis.assert_called_once_with("off")
        axis.legend.assert_called_once()
        legend_kwargs = axis.legend.call_args.kwargs
        self.assertNotIn("mode", legend_kwargs)
        self.assertEqual(
            legend_kwargs["labels"],
            [
                "Shed / molt",
                "\u26a1 Electrical stimulation",
                "Death",
                "Automatic feeding bouts",
                "Motion-derived immobile",
                "Motion-derived mobile",
                "Low video quality",
                "Food unavailable",
                "Manual interval",
            ],
        )
        self.assertEqual(legend_kwargs["ncol"], 9)
        self.assertEqual(
            legend_kwargs["bbox_to_anchor"],
            (0.0, 0.52),
        )

    def test_draw_timeline_legend_skips_missing_categories(self) -> None:
        axis = mock.Mock()
        legend = timeline.draw_timeline_legend(axis, {"Manual interval": timeline.manual_interval_legend_handle()})

        self.assertIsNotNone(legend)
        axis.axis.assert_called_once_with("off")
        axis.legend.assert_called_once()
        self.assertEqual(axis.legend.call_args.kwargs["labels"], ["Manual interval"])
        self.assertEqual(axis.legend.call_args.kwargs["ncol"], 1)

    def test_alternating_row_backgrounds_use_subtle_stripes(self) -> None:
        axis = mock.Mock()

        timeline.alternating_row_backgrounds(axis, 4)

        self.assertEqual(
            axis.axhspan.call_args_list,
            [
                mock.call(-0.5, 0.5, facecolor=timeline.ROW_BACKGROUND_COLOR, alpha=1.0, zorder=0.02),
                mock.call(1.5, 2.5, facecolor=timeline.ROW_BACKGROUND_COLOR, alpha=1.0, zorder=0.02),
            ],
        )

    def test_video_quality_and_food_unavailable_global_styles_are_rendered(self) -> None:
        events = [
            timeline.GlobalEvent(
                start_local=dt.datetime(2026, 8, 9, 15, 0, 0, tzinfo=dt.timezone.utc),
                end_local=dt.datetime(2026, 8, 9, 15, 1, 0, tzinfo=dt.timezone.utc),
                event="video_quality_low",
                kind="event",
                notes="",
            ),
            timeline.GlobalEvent(
                start_local=dt.datetime(2026, 8, 9, 15, 2, 0, tzinfo=dt.timezone.utc),
                end_local=dt.datetime(2026, 8, 9, 15, 3, 0, tzinfo=dt.timezone.utc),
                event="bad_lighting",
                kind="event",
                notes="",
            ),
            timeline.GlobalEvent(
                start_local=dt.datetime(2026, 8, 9, 15, 4, 0, tzinfo=dt.timezone.utc),
                end_local=dt.datetime(2026, 8, 9, 15, 5, 0, tzinfo=dt.timezone.utc),
                event="custom_interval",
                kind="video_quality",
                notes="",
            ),
            timeline.GlobalEvent(
                start_local=dt.datetime(2026, 8, 9, 15, 6, 0, tzinfo=dt.timezone.utc),
                end_local=dt.datetime(2026, 8, 9, 15, 7, 0, tzinfo=dt.timezone.utc),
                event="food_unavailable",
                kind="event",
                notes="",
            ),
            timeline.GlobalEvent(
                start_local=dt.datetime(2026, 8, 9, 15, 8, 0, tzinfo=dt.timezone.utc),
                end_local=dt.datetime(2026, 8, 9, 15, 9, 0, tzinfo=dt.timezone.utc),
                event="lights_repositioned",
                kind="event",
                notes="",
            ),
        ]

        styles = [timeline.global_event_band_style(event) for event in events]
        self.assertEqual(
            styles,
            [
                (timeline.VIDEO_QUALITY_LOW_COLOR, timeline.VIDEO_QUALITY_LOW_ALPHA),
                (timeline.VIDEO_QUALITY_LOW_COLOR, timeline.VIDEO_QUALITY_LOW_ALPHA),
                (timeline.VIDEO_QUALITY_LOW_COLOR, timeline.VIDEO_QUALITY_LOW_ALPHA),
                (timeline.FOOD_UNAVAILABLE_COLOR, timeline.FOOD_UNAVAILABLE_ALPHA),
                None,
            ],
        )

    def test_low_quality_label_is_placed_above_recording_band(self) -> None:
        clips = [
            timeline.RecordingClip(
                clip_id="session_a/camera1",
                timestamp_file=Path("camera1.timestamps.csv.gz"),
                video_file=Path("camera1.mp4"),
                camera_label="camera1",
                start_utc=dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 19, 30, 0, tzinfo=dt.timezone.utc),
                duration_s=1800.0,
                frames=9000,
                timestamp_size_bytes=123,
                timestamp_mtime_ns=456,
            )
        ]
        global_events = [
            timeline.GlobalEvent(
                start_local=dt.datetime(2026, 8, 9, 15, 5, 0, tzinfo=dt.timezone.utc),
                end_local=dt.datetime(2026, 8, 9, 15, 20, 0, tzinfo=dt.timezone.utc),
                event="video_quality_low",
                kind="video_quality",
                notes="",
            )
        ]

        with mock.patch("matplotlib.axes.Axes.text") as mock_text:
            timeline.plot_recording_timeline(
                clips=clips,
                events=[],
                motion_states=[],
                animals=timeline.ANIMAL_ORDER,
                global_events=global_events,
                timezone=dt.timezone(dt.timedelta(hours=-4)),
                output_path=Path("ignored.png"),
                annotate_clips=False,
            )

        label_calls = [
            call for call in mock_text.call_args_list if len(call.args) >= 3 and call.args[2] == "Low video quality"
        ]
        self.assertEqual(len(label_calls), 1)
        self.assertGreater(label_calls[0].args[1], 0.8)
        self.assertIn("transform", label_calls[0].kwargs)

    def test_plot_uses_clean_title_subtitle_and_timezone_axis_label(self) -> None:
        clips = [
            timeline.RecordingClip(
                clip_id="session_a/camera1",
                timestamp_file=Path("camera1.timestamps.csv.gz"),
                video_file=Path("camera1.mp4"),
                camera_label="camera1",
                start_utc=dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 19, 30, 0, tzinfo=dt.timezone.utc),
                duration_s=1800.0,
                frames=9000,
                timestamp_size_bytes=123,
                timestamp_mtime_ns=456,
            )
        ]
        fig = self.capture_timeline_figure(
            clips=clips,
            events=[],
            motion_states=[],
            global_events=[],
            timezone=timeline.load_timezone("America/New_York"),
        )

        self.assertEqual(fig._suptitle.get_text(), "Continuous behavioral monitoring of monarch caterpillars")
        subtitle_texts = [text.get_text() for text in fig.texts if text.get_text().startswith("Elapsed ")]
        self.assertEqual(len(subtitle_texts), 1)
        self.assertIn("recorded", subtitle_texts[0])
        self.assertIn("coverage", subtitle_texts[0])
        self.assertNotIn("Woods Hole local time", subtitle_texts[0])
        self.assertIn("Local time - America/New_York", self.behavior_axis(fig).get_xlabel())
        self.assertIn("EDT", self.behavior_axis(fig).get_xlabel())

    def test_subtract_intervals_no_overlap_returns_original_interval(self) -> None:
        start = dt.datetime(2026, 8, 9, 10, 0, 0, tzinfo=dt.timezone.utc)
        end = dt.datetime(2026, 8, 9, 11, 0, 0, tzinfo=dt.timezone.utc)
        exclusions = [(dt.datetime(2026, 8, 9, 12, 0, 0, tzinfo=dt.timezone.utc), dt.datetime(2026, 8, 9, 13, 0, 0, tzinfo=dt.timezone.utc))]

        self.assertEqual(timeline.subtract_intervals(start, end, exclusions), [(start, end)])

    def test_subtract_intervals_full_coverage_returns_empty(self) -> None:
        start = dt.datetime(2026, 8, 9, 10, 0, 0, tzinfo=dt.timezone.utc)
        end = dt.datetime(2026, 8, 9, 11, 0, 0, tzinfo=dt.timezone.utc)
        exclusions = [(dt.datetime(2026, 8, 9, 9, 0, 0, tzinfo=dt.timezone.utc), dt.datetime(2026, 8, 9, 12, 0, 0, tzinfo=dt.timezone.utc))]

        self.assertEqual(timeline.subtract_intervals(start, end, exclusions), [])

    def test_subtract_intervals_clips_overlap_at_start(self) -> None:
        start = dt.datetime(2026, 8, 9, 10, 0, 0, tzinfo=dt.timezone.utc)
        end = dt.datetime(2026, 8, 9, 11, 0, 0, tzinfo=dt.timezone.utc)
        exclusions = [(dt.datetime(2026, 8, 9, 9, 30, 0, tzinfo=dt.timezone.utc), dt.datetime(2026, 8, 9, 10, 15, 0, tzinfo=dt.timezone.utc))]

        self.assertEqual(
            timeline.subtract_intervals(start, end, exclusions),
            [(dt.datetime(2026, 8, 9, 10, 15, 0, tzinfo=dt.timezone.utc), end)],
        )

    def test_subtract_intervals_clips_overlap_at_end(self) -> None:
        start = dt.datetime(2026, 8, 9, 10, 0, 0, tzinfo=dt.timezone.utc)
        end = dt.datetime(2026, 8, 9, 11, 0, 0, tzinfo=dt.timezone.utc)
        exclusions = [(dt.datetime(2026, 8, 9, 10, 45, 0, tzinfo=dt.timezone.utc), dt.datetime(2026, 8, 9, 12, 0, 0, tzinfo=dt.timezone.utc))]

        self.assertEqual(
            timeline.subtract_intervals(start, end, exclusions),
            [(start, dt.datetime(2026, 8, 9, 10, 45, 0, tzinfo=dt.timezone.utc))],
        )

    def test_subtract_intervals_splits_middle_overlap(self) -> None:
        start = dt.datetime(2026, 8, 9, 10, 0, 0, tzinfo=dt.timezone.utc)
        end = dt.datetime(2026, 8, 9, 11, 0, 0, tzinfo=dt.timezone.utc)
        exclusions = [(dt.datetime(2026, 8, 9, 10, 20, 0, tzinfo=dt.timezone.utc), dt.datetime(2026, 8, 9, 10, 40, 0, tzinfo=dt.timezone.utc))]

        self.assertEqual(
            timeline.subtract_intervals(start, end, exclusions),
            [
                (start, dt.datetime(2026, 8, 9, 10, 20, 0, tzinfo=dt.timezone.utc)),
                (dt.datetime(2026, 8, 9, 10, 40, 0, tzinfo=dt.timezone.utc), end),
            ],
        )

    def test_subtract_intervals_handles_multiple_masks(self) -> None:
        start = dt.datetime(2026, 8, 9, 10, 0, 0, tzinfo=dt.timezone.utc)
        end = dt.datetime(2026, 8, 9, 12, 0, 0, tzinfo=dt.timezone.utc)
        exclusions = [
            (dt.datetime(2026, 8, 9, 10, 15, 0, tzinfo=dt.timezone.utc), dt.datetime(2026, 8, 9, 10, 30, 0, tzinfo=dt.timezone.utc)),
            (dt.datetime(2026, 8, 9, 11, 0, 0, tzinfo=dt.timezone.utc), dt.datetime(2026, 8, 9, 11, 15, 0, tzinfo=dt.timezone.utc)),
        ]

        self.assertEqual(
            timeline.subtract_intervals(start, end, exclusions),
            [
                (start, dt.datetime(2026, 8, 9, 10, 15, 0, tzinfo=dt.timezone.utc)),
                (dt.datetime(2026, 8, 9, 10, 30, 0, tzinfo=dt.timezone.utc), dt.datetime(2026, 8, 9, 11, 0, 0, tzinfo=dt.timezone.utc)),
                (dt.datetime(2026, 8, 9, 11, 15, 0, tzinfo=dt.timezone.utc), end),
            ],
        )

    def test_subtract_intervals_touching_boundary_does_not_clip(self) -> None:
        start = dt.datetime(2026, 8, 9, 10, 0, 0, tzinfo=dt.timezone.utc)
        end = dt.datetime(2026, 8, 9, 11, 0, 0, tzinfo=dt.timezone.utc)
        exclusions = [(dt.datetime(2026, 8, 9, 11, 0, 0, tzinfo=dt.timezone.utc), dt.datetime(2026, 8, 9, 12, 0, 0, tzinfo=dt.timezone.utc))]

        self.assertEqual(timeline.subtract_intervals(start, end, exclusions), [(start, end)])

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
                    motion_states=[],
                    animals=timeline.ANIMAL_ORDER,
                    timezone=dt.timezone(dt.timedelta(hours=-4)),
                    output_path=Path("ignored.png"),
                    annotate_clips=False,
                )

        self.assertEqual(mock_scatter.call_count, 0)
        self.assertGreaterEqual(mock_broken_barh.call_count, 1)

    def test_global_spans_are_drawn_on_coverage_and_behavior_axes(self) -> None:
        global_events = [
            timeline.GlobalEvent(
                start_local=dt.datetime(2026, 8, 9, 15, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
                end_local=dt.datetime(2026, 8, 9, 15, 20, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
                event="poor_video_quality",
                kind="event",
                notes="",
            ),
            timeline.GlobalEvent(
                start_local=dt.datetime(2026, 8, 9, 15, 30, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
                end_local=dt.datetime(2026, 8, 9, 15, 45, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
                event="food_unavailable",
                kind="event",
                notes="",
            ),
            timeline.GlobalEvent(
                start_local=dt.datetime(2026, 8, 9, 16, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
                end_local=dt.datetime(2026, 8, 9, 16, 15, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
                event="lights_repositioned",
                kind="event",
                notes="",
            ),
        ]

        with mock.patch("matplotlib.axes.Axes.axvspan") as mock_axvspan:
            timeline.plot_recording_timeline(
                clips=[],
                events=[],
                motion_states=[],
                animals=timeline.ANIMAL_ORDER,
                global_events=global_events,
                timezone=dt.timezone(dt.timedelta(hours=-4)),
                output_path=Path("ignored.png"),
                annotate_clips=False,
            )

        self.assertEqual(mock_axvspan.call_count, 4)
        facecolors = [call.kwargs["facecolor"] for call in mock_axvspan.call_args_list]
        self.assertEqual(facecolors.count(timeline.VIDEO_QUALITY_LOW_COLOR), 2)
        self.assertEqual(facecolors.count(timeline.FOOD_UNAVAILABLE_COLOR), 2)
        self.assertNotIn(timeline.GENERIC_GLOBAL_EVENT_COLOR, facecolors)

    def test_hidden_global_events_do_not_expand_the_timeline_bounds(self) -> None:
        visible_event = timeline.BehaviorEvent(
            animal_id="C01",
            start_local=dt.datetime(2026, 8, 9, 15, 0, 0),
            end_local=dt.datetime(2026, 8, 9, 15, 10, 0),
            event="feeding",
            kind="event",
            notes="",
        )
        hidden_global = timeline.GlobalEvent(
            start_local=dt.datetime(2026, 8, 9, 18, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
            end_local=dt.datetime(2026, 8, 9, 18, 30, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
            event="lights_repositioned",
            kind="event",
            notes="",
        )

        fig = self.capture_timeline_figure(
            clips=[],
            events=[visible_event],
            global_events=[hidden_global],
            animals=timeline.ANIMAL_ORDER,
        )
        ax_beh = self.behavior_axis(fig)

        xmin, xmax = ax_beh.get_xlim()
        expected_xmin, expected_xmax = timeline.display_time_bounds(
            dt.datetime(2026, 8, 9, 15, 0, 0),
            dt.datetime(2026, 8, 9, 15, 10, 0),
        )
        self.assertAlmostEqual(xmin, timeline.mdates.date2num(expected_xmin))
        self.assertAlmostEqual(xmax, timeline.mdates.date2num(expected_xmax))

    def test_manual_markers_survive_global_quality_spans(self) -> None:
        global_events = [
            timeline.GlobalEvent(
                start_local=dt.datetime(2026, 8, 9, 15, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
                end_local=dt.datetime(2026, 8, 9, 15, 30, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
                event="video_quality_low",
                kind="video_quality",
                notes="",
            )
        ]
        events = [
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 5, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 5, 1),
                event="shed",
                kind="event",
                notes="",
                is_point=True,
            )
        ]

        with mock.patch("matplotlib.axes.Axes.scatter") as mock_scatter:
            timeline.plot_recording_timeline(
                clips=[],
                events=events,
                motion_states=[],
                animals=timeline.ANIMAL_ORDER,
                global_events=global_events,
                timezone=dt.timezone(dt.timedelta(hours=-4)),
                output_path=Path("ignored.png"),
                annotate_clips=False,
            )

        self.assertEqual(mock_scatter.call_count, 1)

    def test_motion_states_are_masked_inside_global_quality_spans(self) -> None:
        global_events = [
            timeline.GlobalEvent(
                start_local=dt.datetime(2026, 8, 9, 15, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
                end_local=dt.datetime(2026, 8, 9, 15, 30, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
                event="bad_lighting",
                kind="event",
                notes="",
            )
        ]
        motion_states = [
            timeline.MotionState(
                animal_id="C01",
                clip_key="clip_1",
                start_utc=dt.datetime(2026, 8, 9, 18, 55, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 19, 5, 0, tzinfo=dt.timezone.utc),
                state="mobile",
                threshold=4.0,
                threshold_source="manual",
                mean_motion_energy=5.0,
                peak_motion_energy=7.0,
                n_windows=300,
            ),
            timeline.MotionState(
                animal_id="C02",
                clip_key="clip_2",
                start_utc=dt.datetime(2026, 8, 9, 19, 10, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 19, 20, 0, tzinfo=dt.timezone.utc),
                state="immobile",
                threshold=4.0,
                threshold_source="manual",
                mean_motion_energy=1.0,
                peak_motion_energy=1.5,
                n_windows=300,
            )
        ]

        with mock.patch("matplotlib.axes.Axes.broken_barh") as mock_broken_barh:
            timeline.plot_recording_timeline(
                clips=[],
                events=[],
                motion_states=motion_states,
                animals=timeline.ANIMAL_ORDER,
                global_events=global_events,
                timezone=dt.timezone(dt.timedelta(hours=-4)),
                output_path=Path("ignored.png"),
                annotate_clips=False,
            )

        state_calls = [
            call
            for call in mock_broken_barh.call_args_list
            if call.kwargs.get("facecolors") in {timeline.MOTION_MOBILE_COLOR, timeline.MOTION_IMMOBILE_COLOR}
        ]
        self.assertEqual(len(state_calls), 1)
        segments = state_calls[0].args[0]
        self.assertEqual(len(segments), 1)
        rendered_left, rendered_width = segments[0]
        expected_left = timeline.mdates.date2num(dt.datetime(2026, 8, 9, 14, 55, 0))
        expected_width = 5 * 60 / 86400.0
        self.assertAlmostEqual(rendered_left, expected_left)
        self.assertAlmostEqual(rendered_width, expected_width)

    def test_manual_duration_events_are_not_masked_by_global_quality_spans(self) -> None:
        global_events = [
            timeline.GlobalEvent(
                start_local=dt.datetime(2026, 8, 9, 15, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
                end_local=dt.datetime(2026, 8, 9, 15, 30, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
                event="video_quality_low",
                kind="video_quality",
                notes="",
            )
        ]
        events = [
            timeline.BehaviorEvent(
                animal_id="C03",
                start_local=dt.datetime(2026, 8, 9, 15, 5, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 20, 0),
                event="feeding",
                kind="event",
                notes="",
            )
        ]

        with mock.patch("matplotlib.axes.Axes.broken_barh") as mock_broken_barh:
            timeline.plot_recording_timeline(
                clips=[],
                events=events,
                motion_states=[],
                animals=timeline.ANIMAL_ORDER,
                global_events=global_events,
                timezone=dt.timezone(dt.timedelta(hours=-4)),
                output_path=Path("ignored.png"),
                annotate_clips=False,
            )

        manual_calls = [
            call
            for call in mock_broken_barh.call_args_list
            if call.kwargs.get("facecolors") in {timeline.INTERVAL_BAR_COLOR, timeline.FEEDING_COLOR}
        ]
        self.assertEqual(len(manual_calls), 1)
        rendered_left, rendered_width = manual_calls[0].args[0][0]
        expected_left = timeline.mdates.date2num(dt.datetime(2026, 8, 9, 15, 5, 0))
        expected_width = 15 * 60 / 86400.0
        self.assertAlmostEqual(rendered_left, expected_left)
        self.assertAlmostEqual(rendered_width, expected_width)

    def test_motion_states_are_batched_by_animal_and_state(self) -> None:
        motion_states = [
            timeline.MotionState(
                animal_id="C01",
                clip_key="clip_a",
                start_utc=dt.datetime(2026, 8, 9, 18, 50, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 18, 55, 0, tzinfo=dt.timezone.utc),
                state="mobile",
                threshold=4.0,
                threshold_source="manual",
                mean_motion_energy=5.0,
                peak_motion_energy=7.0,
                n_windows=300,
            ),
            timeline.MotionState(
                animal_id="C01",
                clip_key="clip_b",
                start_utc=dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 19, 5, 0, tzinfo=dt.timezone.utc),
                state="mobile",
                threshold=4.0,
                threshold_source="manual",
                mean_motion_energy=5.0,
                peak_motion_energy=7.0,
                n_windows=300,
            ),
            timeline.MotionState(
                animal_id="C01",
                clip_key="clip_c",
                start_utc=dt.datetime(2026, 8, 9, 19, 10, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 19, 15, 0, tzinfo=dt.timezone.utc),
                state="immobile",
                threshold=4.0,
                threshold_source="manual",
                mean_motion_energy=1.0,
                peak_motion_energy=1.5,
                n_windows=300,
            ),
        ]

        with mock.patch("matplotlib.axes.Axes.broken_barh") as mock_broken_barh:
            timeline.plot_recording_timeline(
                clips=[],
                events=[],
                motion_states=motion_states,
                animals=timeline.ANIMAL_ORDER,
                timezone=dt.timezone(dt.timedelta(hours=-4)),
                output_path=Path("ignored.png"),
                annotate_clips=False,
            )

        motion_calls = [
            call
            for call in mock_broken_barh.call_args_list
            if call.kwargs.get("facecolors") in {timeline.MOTION_MOBILE_COLOR, timeline.MOTION_IMMOBILE_COLOR}
        ]
        self.assertEqual(len(motion_calls), 2)
        mobile_call = next(call for call in motion_calls if call.kwargs["facecolors"] == timeline.MOTION_MOBILE_COLOR)
        immobile_call = next(call for call in motion_calls if call.kwargs["facecolors"] == timeline.MOTION_IMMOBILE_COLOR)
        self.assertEqual(len(mobile_call.args[0]), 2)
        self.assertEqual(len(immobile_call.args[0]), 1)
        self.assertAlmostEqual(mobile_call.args[0][0][1], 5 * 60 / 86400.0)
        self.assertAlmostEqual(mobile_call.args[0][1][1], 5 * 60 / 86400.0)
        self.assertAlmostEqual(immobile_call.args[0][0][1], 5 * 60 / 86400.0)

    def test_automatic_feeding_events_are_batched_by_animal(self) -> None:
        events = [
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 0, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 5, 0),
                event="feeding",
                kind="event",
                notes="",
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 10, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 15, 0),
                event="feeding",
                kind="event",
                notes="",
            ),
            timeline.BehaviorEvent(
                animal_id="C02",
                start_local=dt.datetime(2026, 8, 9, 15, 20, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 27, 0),
                event="feeding",
                kind="event",
                notes="",
            ),
            timeline.BehaviorEvent(
                animal_id="C03",
                start_local=dt.datetime(2026, 8, 9, 15, 30, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 45, 0),
                event="observation",
                kind="event",
                notes="manual duration",
            ),
        ]

        with mock.patch("matplotlib.axes.Axes.broken_barh") as mock_broken_barh:
            timeline.plot_recording_timeline(
                clips=[],
                events=events,
                motion_states=[],
                animals=timeline.ANIMAL_ORDER,
                timezone=dt.timezone(dt.timedelta(hours=-4)),
                output_path=Path("ignored.png"),
                annotate_clips=False,
            )

        feeding_calls = [
            call
            for call in mock_broken_barh.call_args_list
            if call.kwargs.get("facecolors") == timeline.FEEDING_COLOR
        ]
        manual_calls = [
            call
            for call in mock_broken_barh.call_args_list
            if call.kwargs.get("facecolors") == timeline.INTERVAL_BAR_COLOR
        ]
        self.assertEqual(len(feeding_calls), 2)
        self.assertEqual(len(manual_calls), 1)
        self.assertEqual(len(feeding_calls[0].args[0]) + len(feeding_calls[1].args[0]), 3)
        self.assertEqual(sorted(len(call.args[0]) for call in feeding_calls), [1, 2])

    def test_motion_states_stop_at_death_time(self) -> None:
        motion_states = [
            timeline.MotionState(
                animal_id="C01",
                clip_key="clip_1",
                start_utc=dt.datetime(2026, 8, 9, 18, 50, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 19, 20, 0, tzinfo=dt.timezone.utc),
                state="mobile",
                threshold=4.0,
                threshold_source="manual",
                mean_motion_energy=5.0,
                peak_motion_energy=7.0,
                n_windows=300,
            ),
            timeline.MotionState(
                animal_id="C01",
                clip_key="clip_2",
                start_utc=dt.datetime(2026, 8, 9, 19, 21, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 19, 30, 0, tzinfo=dt.timezone.utc),
                state="immobile",
                threshold=4.0,
                threshold_source="manual",
                mean_motion_energy=1.0,
                peak_motion_energy=1.5,
                n_windows=300,
            ),
        ]
        events = [
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 0, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 0, 1),
                event="death",
                kind="event",
                notes="",
                is_point=True,
            )
        ]

        with mock.patch("matplotlib.axes.Axes.broken_barh") as mock_broken_barh:
            timeline.plot_recording_timeline(
                clips=[],
                events=events,
                motion_states=motion_states,
                animals=timeline.ANIMAL_ORDER,
                timezone=dt.timezone(dt.timedelta(hours=-4)),
                output_path=Path("ignored.png"),
                annotate_clips=False,
            )

        state_calls = [
            call
            for call in mock_broken_barh.call_args_list
            if call.kwargs.get("facecolors") in {timeline.MOTION_MOBILE_COLOR, timeline.MOTION_IMMOBILE_COLOR}
        ]
        self.assertEqual(len(state_calls), 1)
        rendered_left, rendered_width = state_calls[0].args[0][0]
        expected_left = timeline.mdates.date2num(dt.datetime(2026, 8, 9, 14, 50, 0))
        expected_width = 10 * 60 / 86400.0
        self.assertAlmostEqual(rendered_left, expected_left)
        self.assertAlmostEqual(rendered_width, expected_width)

    def test_terminal_activity_cutoff_prefers_earliest_terminal_event(self) -> None:
        events = [
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 10, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 10, 1),
                event="J_hang",
                kind="event",
                notes="",
                is_point=True,
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 20, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 20, 1),
                event="Pupation",
                kind="event",
                notes="",
                is_point=True,
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 30, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 30, 1),
                event="death",
                kind="event",
                notes="",
                is_point=True,
            ),
            timeline.BehaviorEvent(
                animal_id="C02",
                start_local=dt.datetime(2026, 8, 9, 15, 12, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 12, 1),
                event="Pupation",
                kind="event",
                notes="",
                is_point=True,
            ),
            timeline.BehaviorEvent(
                animal_id="C03",
                start_local=dt.datetime(2026, 8, 9, 15, 25, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 25, 1),
                event="death",
                kind="event",
                notes="",
                is_point=True,
            ),
        ]

        cutoffs = timeline.terminal_activity_cutoff_by_animal(events)

        self.assertEqual(cutoffs["C01"], dt.datetime(2026, 8, 9, 15, 10, 0))
        self.assertEqual(cutoffs["C02"], dt.datetime(2026, 8, 9, 15, 12, 0))
        self.assertEqual(cutoffs["C03"], dt.datetime(2026, 8, 9, 15, 25, 0))
        self.assertNotIn("C04", cutoffs)

    def test_terminal_activity_cutoff_clips_derived_layers_and_keeps_marker(self) -> None:
        events = [
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 0, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 0, 1),
                event="J_hang",
                kind="event",
                notes="",
                is_point=True,
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 14, 58, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 2, 0),
                event="feeding",
                kind="event",
                notes="before cutoff",
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 5, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 10, 0),
                event="feeding",
                kind="event",
                notes="after cutoff",
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 6, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 20, 0),
                event="observation",
                kind="event",
                notes="manual after cutoff",
            ),
        ]
        motion_states = [
            timeline.MotionState(
                animal_id="C01",
                clip_key="clip_a",
                start_utc=dt.datetime(2026, 8, 9, 18, 55, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 19, 5, 0, tzinfo=dt.timezone.utc),
                state="mobile",
                threshold=4.0,
                threshold_source="manual",
                mean_motion_energy=5.0,
                peak_motion_energy=7.0,
                n_windows=300,
            ),
            timeline.MotionState(
                animal_id="C01",
                clip_key="clip_b",
                start_utc=dt.datetime(2026, 8, 9, 19, 5, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 19, 15, 0, tzinfo=dt.timezone.utc),
                state="immobile",
                threshold=4.0,
                threshold_source="manual",
                mean_motion_energy=1.0,
                peak_motion_energy=1.5,
                n_windows=300,
            ),
        ]
        motion_energy_samples = [
            timeline.MotionEnergySample(
                animal_id="C01",
                clip_key="clip_a",
                timestamp_utc=dt.datetime(2026, 8, 9, 18, 59, 0, tzinfo=dt.timezone.utc),
                motion_energy=5.0,
            ),
            timeline.MotionEnergySample(
                animal_id="C01",
                clip_key="clip_a",
                timestamp_utc=dt.datetime(2026, 8, 9, 20, 0, 0, tzinfo=dt.timezone.utc),
                motion_energy=50.0,
            ),
        ]

        with mock.patch("matplotlib.axes.Axes.broken_barh") as mock_broken_barh:
            with mock.patch("matplotlib.axes.Axes.scatter") as mock_scatter:
                fig = self.capture_timeline_figure(
                    clips=[],
                    events=events,
                    motion_states=motion_states,
                    motion_energy_samples=motion_energy_samples,
                    global_events=[],
                    animals=timeline.ANIMAL_ORDER,
                    timezone=dt.timezone(dt.timedelta(hours=-4)),
                )

        motion_calls = [
            call
            for call in mock_broken_barh.call_args_list
            if call.kwargs.get("facecolors") in {timeline.MOTION_MOBILE_COLOR, timeline.MOTION_IMMOBILE_COLOR}
        ]
        self.assertEqual(len(motion_calls), 1)
        motion_segments = motion_calls[0].args[0]
        self.assertEqual(len(motion_segments), 1)
        rendered_left, rendered_width = motion_segments[0]
        self.assertAlmostEqual(rendered_left, timeline.mdates.date2num(dt.datetime(2026, 8, 9, 14, 55, 0)))
        self.assertAlmostEqual(rendered_width, 5 * 60 / 86400.0)

        feeding_calls = [call for call in mock_broken_barh.call_args_list if call.kwargs.get("facecolors") == timeline.FEEDING_COLOR]
        self.assertEqual(len(feeding_calls), 1)
        feeding_segments = feeding_calls[0].args[0]
        self.assertEqual(len(feeding_segments), 1)
        feeding_left, feeding_width = feeding_segments[0]
        self.assertAlmostEqual(feeding_left, timeline.mdates.date2num(dt.datetime(2026, 8, 9, 14, 58, 0)))
        self.assertAlmostEqual(feeding_width, 2 * 60 / 86400.0)

        scatter_markers = [call.kwargs["marker"] for call in mock_scatter.call_args_list]
        self.assertIn("v", scatter_markers)
        self.assertNotIn("D", scatter_markers)
        self.assertNotIn("X", scatter_markers)
        self.assertTrue(fig.axes)
        ax_motion = fig.axes[1]
        line_xmax = max(
            timeline.mdates.date2num(max(line.get_xdata()))
            for line in ax_motion.get_lines()
            if len(line.get_xdata()) > 0
        )
        self.assertLessEqual(line_xmax, timeline.mdates.date2num(dt.datetime(2026, 8, 9, 15, 0, 0)))

    def test_events_after_death_are_not_plotted_but_death_marker_is_kept(self) -> None:
        events = [
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 14, 55, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 10, 0),
                event="feeding",
                kind="event",
                notes="",
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 0, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 0, 1),
                event="death",
                kind="event",
                notes="",
                is_point=True,
            ),
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 5, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 20, 0),
                event="feeding_resumed",
                kind="event",
                notes="",
            ),
        ]

        with mock.patch("matplotlib.axes.Axes.broken_barh") as mock_broken_barh:
            with mock.patch("matplotlib.axes.Axes.scatter") as mock_scatter:
                timeline.plot_recording_timeline(
                    clips=[],
                    events=events,
                    motion_states=[],
                    animals=timeline.ANIMAL_ORDER,
                    timezone=dt.timezone(dt.timedelta(hours=-4)),
                    output_path=Path("ignored.png"),
                    annotate_clips=False,
                )

        manual_calls = [
            call
            for call in mock_broken_barh.call_args_list
            if call.kwargs.get("facecolors") in {timeline.INTERVAL_BAR_COLOR, timeline.FEEDING_COLOR}
        ]
        self.assertEqual(len(manual_calls), 1)

        feeding_call = manual_calls[0]
        feeding_left, feeding_width = feeding_call.args[0][0]
        expected_left = timeline.mdates.date2num(dt.datetime(2026, 8, 9, 14, 55, 0))
        expected_width = 5 * 60 / 86400.0
        self.assertAlmostEqual(feeding_left, expected_left)
        self.assertAlmostEqual(feeding_width, expected_width)

        scatter_markers = [call.kwargs["marker"] for call in mock_scatter.call_args_list]
        self.assertIn("X", scatter_markers)
        self.assertNotIn("o", scatter_markers)

    def test_load_motion_states_parses_mobile_and_immobile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion_path = root / "motion_states.csv"
            self.write_motion_states_csv(
                motion_path,
                [
                    [
                        "C01",
                        "clip_a",
                        "2026-08-09T19:00:00Z",
                        "2026-08-09T19:00:10Z",
                        "2026-08-09 15:00:00-04:00",
                        "2026-08-09 15:00:10-04:00",
                        "mobile",
                        4.5,
                        "manual",
                        5.1,
                        8.3,
                        10,
                    ],
                    [
                        "C02",
                        "clip_b",
                        "2026-08-09T19:10:00Z",
                        "2026-08-09T19:10:10Z",
                        "2026-08-09 15:10:00-04:00",
                        "2026-08-09 15:10:10-04:00",
                        "immobile",
                        4.5,
                        "manual",
                        1.1,
                        1.8,
                        10,
                    ],
                ],
            )

            states = timeline.load_motion_states(motion_path)

            self.assertEqual([state.state for state in states], ["mobile", "immobile"])
            self.assertEqual([state.animal_id for state in states], ["C01", "C02"])

    def test_motion_states_keep_canonical_row_order(self) -> None:
        motion_states = [
            timeline.MotionState(
                animal_id="C08",
                clip_key="clip_8",
                start_utc=dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 19, 1, 0, tzinfo=dt.timezone.utc),
                state="mobile",
                threshold=4.0,
                threshold_source="manual",
                mean_motion_energy=5.0,
                peak_motion_energy=7.0,
                n_windows=60,
            ),
            timeline.MotionState(
                animal_id="C01",
                clip_key="clip_1",
                start_utc=dt.datetime(2026, 8, 9, 18, 0, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 18, 1, 0, tzinfo=dt.timezone.utc),
                state="immobile",
                threshold=4.0,
                threshold_source="manual",
                mean_motion_energy=1.0,
                peak_motion_energy=1.5,
                n_windows=60,
            ),
        ]

        fig = self.capture_timeline_figure(clips=[], events=[], motion_states=motion_states)
        ax_beh = self.behavior_axis(fig)

        self.assertEqual(
            [tick.get_text() for tick in ax_beh.get_yticklabels()],
            timeline.ANIMAL_ORDER,
        )

    def test_manual_event_overlay_survives_motion_state_background(self) -> None:
        motion_states = [
            timeline.MotionState(
                animal_id="C01",
                clip_key="clip_1",
                start_utc=dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                end_utc=dt.datetime(2026, 8, 9, 19, 5, 0, tzinfo=dt.timezone.utc),
                state="mobile",
                threshold=4.0,
                threshold_source="manual",
                mean_motion_energy=5.0,
                peak_motion_energy=7.0,
                n_windows=300,
            )
        ]
        events = [
            timeline.BehaviorEvent(
                animal_id="C01",
                start_local=dt.datetime(2026, 8, 9, 15, 1, 0),
                end_local=dt.datetime(2026, 8, 9, 15, 1, 1),
                event="shed",
                kind="event",
                notes="",
                is_point=True,
            )
        ]

        with mock.patch("matplotlib.axes.Axes.scatter") as mock_scatter:
            timeline.plot_recording_timeline(
                clips=[],
                events=events,
                motion_states=motion_states,
                animals=timeline.ANIMAL_ORDER,
                timezone=dt.timezone(dt.timedelta(hours=-4)),
                output_path=Path("ignored.png"),
                annotate_clips=False,
            )

        self.assertEqual(mock_scatter.call_count, 1)

    def test_plot_uses_dedicated_legend_axis(self) -> None:
        fig = self.capture_timeline_figure(clips=[], events=[], motion_states=[])

        self.assertEqual(len(fig.axes), 3)

    def test_major_tick_interval_prefers_twelve_hours_for_four_day_span(self) -> None:
        start = dt.datetime(2026, 8, 7, 0, 0, 0)
        end = dt.datetime(2026, 8, 11, 0, 0, 0)

        self.assertEqual(timeline.major_tick_interval_hours(start, end), 12)

    def test_configure_time_axis_uses_day_locator_for_multi_day_span(self) -> None:
        fig, ax = timeline.plt.subplots()
        try:
            timeline.configure_time_axis(
                ax,
                dt.datetime(2026, 8, 7, 0, 0, 0),
                dt.datetime(2026, 8, 11, 0, 0, 0),
            )

            self.assertIsInstance(ax.xaxis.get_major_locator(), timeline.mdates.DayLocator)
            self.assertEqual(ax.xaxis.get_major_formatter().fmt, "%b %d")
        finally:
            timeline.plt.close(fig)

    def test_parse_google_sheet_url_extracts_sheet_id_and_gid(self) -> None:
        source = timeline.parse_google_sheet_url(
            "https://docs.google.com/spreadsheets/d/1yqe8VII3YNzX2EmOlbBckgTCJKXD47PLZEfHeO2yQ2w/edit?gid=1696022641#gid=1696022641"
        )

        self.assertEqual(source.sheet_id, "1yqe8VII3YNzX2EmOlbBckgTCJKXD47PLZEfHeO2yQ2w")
        self.assertEqual(source.gid, "1696022641")
        self.assertEqual(
            source.export_url,
            "https://docs.google.com/spreadsheets/d/1yqe8VII3YNzX2EmOlbBckgTCJKXD47PLZEfHeO2yQ2w/export?format=csv&gid=1696022641",
        )

    def test_parse_google_sheet_url_uses_fragment_gid_when_needed(self) -> None:
        source = timeline.parse_google_sheet_url(
            "https://docs.google.com/spreadsheets/d/1yqe8VII3YNzX2EmOlbBckgTCJKXD47PLZEfHeO2yQ2w/edit#gid=1696022641"
        )

        self.assertEqual(source.gid, "1696022641")

    def test_parse_google_sheet_export_url_is_supported(self) -> None:
        source = timeline.parse_google_sheet_url(
            "https://docs.google.com/spreadsheets/d/1yqe8VII3YNzX2EmOlbBckgTCJKXD47PLZEfHeO2yQ2w/export?format=csv&gid=1696022641"
        )

        self.assertEqual(source.gid, "1696022641")

    def test_non_google_remote_event_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with self.assertRaises(ValueError):
                timeline.load_behavior_events_source(
                    "https://example.com/events.csv",
                    root=root,
                    tz=dt.timezone(dt.timedelta(hours=-4)),
                )

    def test_google_sheet_source_is_fetched_and_snapshotted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_timestamp_clip(
                root / "session_a" / "clip_0000",
                timestamps=[
                    dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 8, 9, 19, 0, 1, tzinfo=dt.timezone.utc),
                ],
            )
            url = "https://docs.google.com/spreadsheets/d/1yqe8VII3YNzX2EmOlbBckgTCJKXD47PLZEfHeO2yQ2w/edit?gid=1696022641#gid=1696022641"
            csv_bytes = (
                "animal_id,start_local,end_local,event,kind,approximate,notes\n"
                "C06,2026-08-07T14:32:00-04:00,,shed,development,FALSE,Shed observed\n"
                "C01,2026-08-09T13:37:00-04:00,2026-08-09T13:37:10-04:00,electrical_stimulation,stimulus,FALSE,50 uA\n"
            ).encode("utf-8")

            with mock.patch(
                "plot_recording_timeline.urllib_request.urlopen",
                return_value=self.FakeHTTPResponse(csv_bytes),
            ):
                with mock.patch.object(timeline, "plot_recording_timeline") as mock_plot:
                    rc = timeline.main([str(root), "--events", url])

            self.assertEqual(rc, 0)
            self.assertEqual(len(mock_plot.call_args.args[1]), 2)
            self.assertEqual((root / "behavior_events_used.csv").read_bytes(), csv_bytes)
            metadata = json.loads((root / "behavior_events_source.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_type"], "google_sheet")
            self.assertEqual(metadata["gid"], "1696022641")
            self.assertEqual(metadata["rows"], 2)

    def test_google_sheet_global_rows_become_global_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            url = "https://docs.google.com/spreadsheets/d/1yqe8VII3YNzX2EmOlbBckgTCJKXD47PLZEfHeO2yQ2w/edit?gid=1696022641#gid=1696022641"
            csv_bytes = (
                "animal_id,start_local,end_local,event,kind,notes\n"
                "All,2026-08-09T15:00:00-04:00,2026-08-09T15:30:00-04:00,video_quality_low,video_quality,dim light\n"
                "C02,2026-08-09T15:05:00-04:00,,shed,event,\n"
            ).encode("utf-8")

            with mock.patch(
                "plot_recording_timeline.urllib_request.urlopen",
                return_value=self.FakeHTTPResponse(csv_bytes),
            ):
                events, global_events = timeline.load_event_tables_source(
                    url,
                    root=root,
                    tz=dt.timezone(dt.timedelta(hours=-4)),
                )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].animal_id, "C02")
        self.assertEqual(len(global_events), 1)
        self.assertTrue(timeline.is_video_quality_global_event(global_events[0]))

    def test_resolve_behavior_event_source_prefers_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "animal_event_log.csv").write_text("", encoding="utf-8")
            (root / "behavior_events.csv").write_text("", encoding="utf-8")
            (root / "behavior_events_source.json").write_text(
                json.dumps(
                    {
                        "source_type": "google_sheet",
                        "source_url": "https://docs.google.com/spreadsheets/d/1yqe8VII3YNzX2EmOlbBckgTCJKXD47PLZEfHeO2yQ2w/edit?gid=1696022641#gid=1696022641",
                    }
                ),
                encoding="utf-8",
            )

            source, reason = timeline.resolve_behavior_event_source(root, "https://docs.google.com/spreadsheets/d/example/edit?gid=1")

        self.assertEqual(source, "https://docs.google.com/spreadsheets/d/example/edit?gid=1")
        self.assertEqual(reason, "explicit")

    def test_resolve_behavior_event_source_uses_saved_google_sheet_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_url = "https://docs.google.com/spreadsheets/d/1yqe8VII3YNzX2EmOlbBckgTCJKXD47PLZEfHeO2yQ2w/edit?gid=1696022641#gid=1696022641"
            (root / "behavior_events_source.json").write_text(
                json.dumps({"source_type": "google_sheet", "source_url": source_url}),
                encoding="utf-8",
            )

            source, reason = timeline.resolve_behavior_event_source(root, None)

        self.assertEqual(source, source_url)
        self.assertEqual(reason, "saved_google_sheet_source")

    def test_resolve_behavior_event_source_falls_back_to_animal_event_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            animal_log = root / "animal_event_log.csv"
            animal_log.write_text("", encoding="utf-8")

            source, reason = timeline.resolve_behavior_event_source(root, None)

        self.assertEqual(source, str(animal_log.resolve()))
        self.assertEqual(reason, "animal_event_log.csv")

    def test_resolve_behavior_event_source_falls_back_to_behavior_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            behavior_events = root / "behavior_events.csv"
            behavior_events.write_text("", encoding="utf-8")

            source, reason = timeline.resolve_behavior_event_source(root, None)

        self.assertEqual(source, str(behavior_events.resolve()))
        self.assertEqual(reason, "behavior_events.csv")

    def test_resolve_behavior_event_source_returns_none_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            source, reason = timeline.resolve_behavior_event_source(root, None)

        self.assertIsNone(source)
        self.assertEqual(reason, "none")

    def test_event_loader_prefers_utc_fields_when_present(self) -> None:
        csv_text = (
            "animal_id,start_local,end_local,start_utc,end_utc,event,kind,notes\n"
            "C01,2026-08-09 00:00:00,2026-08-09 00:01:00,2026-08-09T19:00:00Z,2026-08-09T19:05:00Z,feeding,feeding,\n"
        )

        events, _global_events = timeline.load_event_tables_from_text(
            csv_text,
            dt.timezone(dt.timedelta(hours=-4)),
            source_name="inline.csv",
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].start_local.isoformat(sep=" "), "2026-08-09 15:00:00-04:00")
        self.assertEqual(events[0].end_local.isoformat(sep=" "), "2026-08-09 15:05:00-04:00")

    def test_feeding_intervals_do_not_get_point_markers(self) -> None:
        event = timeline.BehaviorEvent(
            animal_id="C01",
            start_local=dt.datetime(2026, 8, 9, 15, 0, 0),
            end_local=dt.datetime(2026, 8, 9, 15, 2, 0),
            event="feeding",
            kind="feeding",
            notes="",
        )

        with mock.patch("matplotlib.axes.Axes.scatter") as mock_scatter:
            timeline.plot_recording_timeline(
                clips=[],
                events=[event],
                motion_states=[],
                animals=timeline.ANIMAL_ORDER,
                timezone=dt.timezone(dt.timedelta(hours=-4)),
                output_path=Path("ignored.png"),
                annotate_clips=False,
            )

        self.assertEqual(mock_scatter.call_count, 0)

    def test_feeding_events_use_dedicated_color(self) -> None:
        event = timeline.BehaviorEvent(
            animal_id="C01",
            start_local=dt.datetime(2026, 8, 9, 15, 0, 0),
            end_local=dt.datetime(2026, 8, 9, 15, 2, 0),
            event="feeding",
            kind="feeding",
            notes="",
        )

        self.assertEqual(timeline.event_bar_color(event), timeline.FEEDING_COLOR)

    def test_quantitative_motion_panel_is_added_only_when_samples_are_supplied(self) -> None:
        samples = [
            timeline.MotionEnergySample(
                animal_id="C01",
                clip_key="clip_a",
                timestamp_utc=dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                motion_energy=1.0,
            ),
            timeline.MotionEnergySample(
                animal_id="C01",
                clip_key="clip_a",
                timestamp_utc=dt.datetime(2026, 8, 9, 19, 0, 30, tzinfo=dt.timezone.utc),
                motion_energy=3.0,
            ),
        ]

        fig = self.capture_timeline_figure(clips=[], events=[], global_events=[], motion_states=[], animals=timeline.ANIMAL_ORDER)
        self.assertEqual(len(fig.axes), 3)

        captured: dict[str, object] = {}

        def fake_savefig(fig_obj, *args, **kwargs):
            captured["fig"] = fig_obj

        with mock.patch("matplotlib.figure.Figure.savefig", new=fake_savefig):
            with mock.patch("matplotlib.pyplot.close"):
                timeline.plot_recording_timeline(
                    clips=[],
                    events=[],
                    motion_states=[],
                    animals=timeline.ANIMAL_ORDER,
                    motion_energy_samples=samples,
                    timezone=dt.timezone(dt.timedelta(hours=-4)),
                    output_path=Path("ignored.png"),
                    annotate_clips=False,
                )

        fig_with_motion = captured["fig"]
        self.addCleanup(timeline.plt.close, fig_with_motion)
        self.assertEqual(len(fig_with_motion.axes), 4)

    def test_main_auto_discovers_feeding_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_timestamp_clip(
                root / "session_a" / "clip_0000",
                timestamps=[
                    dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 8, 9, 19, 0, 1, tzinfo=dt.timezone.utc),
                ],
            )
            feeding_path = root / "cropped_by_caterpillar" / "leaf_feeding" / "feeding_events.csv"
            feeding_path.parent.mkdir(parents=True, exist_ok=True)
            with feeding_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["animal_id", "start_utc", "end_utc", "start_local", "end_local", "event", "kind", "notes"])
                writer.writerow(["C01", "2026-08-09T19:00:00Z", "2026-08-09T19:05:00Z", "", "", "feeding", "feeding", "automatic"])

            with mock.patch.object(timeline, "plot_recording_timeline") as mock_plot:
                timeline.main([str(root)])

            events = mock_plot.call_args.args[1]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event, "feeding")

    def test_main_explicit_feeding_events_override_auto_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_timestamp_clip(
                root / "session_a" / "clip_0000",
                timestamps=[
                    dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 8, 9, 19, 0, 1, tzinfo=dt.timezone.utc),
                ],
            )
            auto_path = root / "cropped_by_caterpillar" / "leaf_feeding" / "feeding_events.csv"
            auto_path.parent.mkdir(parents=True, exist_ok=True)
            with auto_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["animal_id", "start_utc", "end_utc", "start_local", "end_local", "event", "kind", "notes"])
                writer.writerow(["C01", "2026-08-09T19:00:00Z", "2026-08-09T19:05:00Z", "", "", "feeding", "feeding", "auto"])
            explicit_path = root / "custom_feeding.csv"
            with explicit_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["animal_id", "start_utc", "end_utc", "start_local", "end_local", "event", "kind", "notes"])
                writer.writerow(["C02", "2026-08-09T19:10:00Z", "2026-08-09T19:15:00Z", "", "", "feeding", "feeding", "explicit"])

            with mock.patch.object(timeline, "plot_recording_timeline") as mock_plot:
                timeline.main([str(root), "--feeding-events", str(explicit_path)])

            events = mock_plot.call_args.args[1]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].animal_id, "C02")

    def test_main_no_feeding_events_ignores_auto_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_timestamp_clip(
                root / "session_a" / "clip_0000",
                timestamps=[
                    dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 8, 9, 19, 0, 1, tzinfo=dt.timezone.utc),
                ],
            )
            feeding_path = root / "cropped_by_caterpillar" / "leaf_feeding" / "feeding_events.csv"
            feeding_path.parent.mkdir(parents=True, exist_ok=True)
            with feeding_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["animal_id", "start_utc", "end_utc", "start_local", "end_local", "event", "kind", "notes"])
                writer.writerow(["C01", "2026-08-09T19:00:00Z", "2026-08-09T19:05:00Z", "", "", "feeding", "feeding", "automatic"])

            with mock.patch.object(timeline, "plot_recording_timeline") as mock_plot:
                timeline.main([str(root), "--no-feeding-events"])

            events = mock_plot.call_args.args[1]
            self.assertEqual(events, [])

    def test_google_sheet_html_response_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            url = "https://docs.google.com/spreadsheets/d/1yqe8VII3YNzX2EmOlbBckgTCJKXD47PLZEfHeO2yQ2w/edit?gid=1696022641#gid=1696022641"

            with mock.patch(
                "plot_recording_timeline.urllib_request.urlopen",
                return_value=self.FakeHTTPResponse(b"<!DOCTYPE html><html><body>login</body></html>"),
            ):
                with self.assertRaises(RuntimeError):
                    timeline.load_behavior_events_source(
                        url,
                        root=root,
                        tz=dt.timezone(dt.timedelta(hours=-4)),
                    )

    def test_google_sheet_network_failure_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_timestamp_clip(
                root / "session_a" / "clip_0000",
                timestamps=[
                    dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 8, 9, 19, 0, 1, tzinfo=dt.timezone.utc),
                ],
            )
            url = "https://docs.google.com/spreadsheets/d/1yqe8VII3YNzX2EmOlbBckgTCJKXD47PLZEfHeO2yQ2w/edit?gid=1696022641#gid=1696022641"

            with mock.patch(
                "plot_recording_timeline.urllib_request.urlopen",
                side_effect=urllib_error.URLError("offline"),
            ):
                rc = timeline.main([str(root), "--events", url])

            self.assertEqual(rc, 1)

    def test_main_auto_detects_motion_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_timestamp_clip(
                root / "session_a" / "clip_0000",
                timestamps=[
                    dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 8, 9, 19, 0, 1, tzinfo=dt.timezone.utc),
                ],
            )
            motion_path = root / "cropped_by_caterpillar" / "motion_energy" / "motion_states.csv"
            self.write_motion_states_csv(
                motion_path,
                [
                    [
                        "C01",
                        "clip_a",
                        "2026-08-09T19:00:00Z",
                        "2026-08-09T19:00:05Z",
                        "2026-08-09 15:00:00-04:00",
                        "2026-08-09 15:00:05-04:00",
                        "mobile",
                        4.5,
                        "manual",
                        5.1,
                        8.3,
                        5,
                    ]
                ],
            )

            with mock.patch.object(timeline, "plot_recording_timeline") as mock_plot:
                timeline.main([str(root)])

            self.assertEqual(len(mock_plot.call_args.args[2]), 1)

    def test_no_motion_states_disables_auto_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_timestamp_clip(
                root / "session_a" / "clip_0000",
                timestamps=[
                    dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 8, 9, 19, 0, 1, tzinfo=dt.timezone.utc),
                ],
            )
            motion_path = root / "cropped_by_caterpillar" / "motion_energy" / "motion_states.csv"
            self.write_motion_states_csv(
                motion_path,
                [
                    [
                        "C01",
                        "clip_a",
                        "2026-08-09T19:00:00Z",
                        "2026-08-09T19:00:05Z",
                        "2026-08-09 15:00:00-04:00",
                        "2026-08-09 15:00:05-04:00",
                        "mobile",
                        4.5,
                        "manual",
                        5.1,
                        8.3,
                        5,
                    ]
                ],
            )

            with mock.patch.object(timeline, "plot_recording_timeline") as mock_plot:
                timeline.main([str(root), "--no-motion-states"])

            self.assertEqual(mock_plot.call_args.args[2], [])


if __name__ == "__main__":
    unittest.main()
