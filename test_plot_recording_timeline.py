#!/usr/bin/env python3

from __future__ import annotations

import csv
import datetime as dt
import gzip
import tempfile
import unittest
from pathlib import Path

import plot_recording_timeline as timeline


class TimelinePlotTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
