#!/usr/bin/env python3

from __future__ import annotations

import datetime as dt
import io
import unittest
from pathlib import Path

import numpy as np

import analyze_leaf_feeding as leaf


class LeafFeedingLogicTests(unittest.TestCase):
    def test_format_progress_bar_reports_percent(self) -> None:
        bar = leaf.format_progress_bar(50, 100)

        self.assertIn("50.0%", bar)
        self.assertIn("[", bar)
        self.assertIn("]", bar)

    def test_build_progress_line_contains_clip_and_status_fields(self) -> None:
        entry = leaf.ManifestEntry("C03", "clip_0012", Path("/tmp/video.mp4"), Path("/tmp/timestamps.csv"))
        line = leaf.build_progress_line(
            clip_index=2,
            total_clips=10,
            entry=entry,
            decoded_frames=4500,
            total_frames=9000,
            minute_bins=5,
            status="analyzing",
        )

        self.assertIn("clip 2/10", line)
        self.assertIn("C03", line)
        self.assertIn("clip_0012", line)
        self.assertIn("4500/9000 frames", line)
        self.assertIn("5 bins", line)
        self.assertIn("analyzing", line)

    def test_progress_reporter_disabled_writes_no_interactive_output(self) -> None:
        stream = io.StringIO()
        reporter = leaf.ProgressReporter(total_clips=3, enabled=False, stream=stream)
        entry = leaf.ManifestEntry("C01", "clip_a", Path("/tmp/video.mp4"), Path("/tmp/timestamps.csv"))

        reporter.start_clip(1, entry, 100)
        reporter.update(decoded_frames=50, minute_bins=2, status="analyzing")
        reporter.finish_clip(decoded_frames=100, minute_bins=3, status="done")

        self.assertEqual(stream.getvalue(), "")

    def test_progress_reporter_finish_adds_newline(self) -> None:
        stream = io.StringIO()
        reporter = leaf.ProgressReporter(total_clips=3, enabled=True, stream=stream, update_interval_s=0.0)
        entry = leaf.ManifestEntry("C01", "clip_a", Path("/tmp/video.mp4"), Path("/tmp/timestamps.csv"))

        reporter.start_clip(1, entry, 100)
        reporter.finish_clip(decoded_frames=100, minute_bins=3, status="done")

        self.assertTrue(stream.getvalue().endswith("\n"))

    def test_stable_leaf_is_not_feeding(self) -> None:
        losses = [None, -0.2, 0.4, -0.3]

        flags = leaf.classify_feeding_hysteresis(
            losses,
            start_loss_pct=2.0,
            continue_loss_pct=1.0,
        )

        self.assertEqual(flags, [False, False, False, False])

    def test_clear_feeding_stays_active(self) -> None:
        losses = [None, 4.0, 4.2, 4.3]

        flags = leaf.classify_feeding_hysteresis(
            losses,
            start_loss_pct=2.0,
            continue_loss_pct=1.0,
        )

        self.assertEqual(flags, [False, True, True, True])

    def test_hysteresis_starts_and_stops_at_expected_thresholds(self) -> None:
        losses = [None, 2.1, 1.5, 0.8, 2.2]

        flags = leaf.classify_feeding_hysteresis(
            losses,
            start_loss_pct=2.0,
            continue_loss_pct=1.0,
        )

        self.assertEqual(flags, [False, True, True, False, True])

    def test_short_gap_merge_and_short_bout_cleanup(self) -> None:
        merged = leaf.merge_short_gaps([True, False, True], max_gap_minutes=2)
        cleaned = leaf.remove_short_bouts([False, True, False], min_bout_minutes=2)

        self.assertEqual(merged, [True, True, True])
        self.assertEqual(cleaned, [False, False, False])

    def test_segment_leaf_area_sums_multiple_components(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[10:30, 10:30] = (0, 255, 0)
        frame[50:70, 50:70] = (0, 200, 0)

        area, _mask = leaf.segment_leaf_area(
            frame,
            hue_low=25,
            hue_high=95,
            sat_min=35,
            value_min=25,
            min_component_px=20,
            morph_kernel=1,
        )

        self.assertEqual(area, 800.0)

    def test_leaf_reset_starts_new_epoch(self) -> None:
        base = dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc)
        estimates = [
            leaf.MinuteLeafEstimate("C01", "clip", base + dt.timedelta(minutes=1), 10000, 6, 6),
            leaf.MinuteLeafEstimate("C01", "clip", base + dt.timedelta(minutes=2), 9500, 6, 6),
            leaf.MinuteLeafEstimate("C01", "clip", base + dt.timedelta(minutes=3), 9000, 6, 6),
            leaf.MinuteLeafEstimate("C01", "clip", base + dt.timedelta(minutes=4), 15000, 6, 6),
            leaf.MinuteLeafEstimate("C01", "clip", base + dt.timedelta(minutes=5), 14950, 6, 6),
        ]

        rows = leaf.finalize_leaf_rows(
            estimates,
            timezone=dt.timezone.utc,
            allowed_gap_minutes=2,
            start_loss_pct=2.0,
            continue_loss_pct=1.0,
            merge_gap_minutes=2,
            min_bout_minutes=2,
            leaf_reset_increase_pct=20.0,
        )

        self.assertEqual([row.leaf_epoch for row in rows], [1, 1, 1, 2, 2])
        self.assertIsNone(rows[3].loss_prev_min_pct)

    def test_feeding_events_single_positive_minute_use_minute_start_to_end(self) -> None:
        rows = [
            leaf.LeafAreaRow(
                animal_id="C01",
                clip_key="clip",
                timestamp_utc=dt.datetime(2026, 8, 9, 10, 1, 0, tzinfo=dt.timezone.utc),
                leaf_epoch=1,
                leaf_area_proxy_px=100.0,
                relative_leaf_area=1.0,
                loss_prev_min_pct=2.5,
                feeding_raw=True,
                feeding_final=True,
                n_sampled_frames=6,
                n_valid_frames=6,
            )
        ]

        events = leaf.feeding_event_dicts(rows, dt.timezone.utc, sample_minutes=1)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start_utc"], "2026-08-09T10:00:00.000000Z")
        self.assertEqual(events[0]["end_utc"], "2026-08-09T10:01:00.000000Z")

    def test_feeding_events_three_positive_minutes_do_not_shift_late(self) -> None:
        base = dt.datetime(2026, 8, 9, 10, 0, 0, tzinfo=dt.timezone.utc)
        rows = [
            leaf.LeafAreaRow("C01", "clip", base + dt.timedelta(minutes=1), 1, 100, 1.0, 2.5, True, True, 6, 6),
            leaf.LeafAreaRow("C01", "clip", base + dt.timedelta(minutes=2), 1, 99, 0.99, 2.0, True, True, 6, 6),
            leaf.LeafAreaRow("C01", "clip", base + dt.timedelta(minutes=3), 1, 98, 0.98, 2.0, True, True, 6, 6),
        ]

        events = leaf.feeding_event_dicts(rows, dt.timezone.utc, sample_minutes=1)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start_utc"], "2026-08-09T10:00:00.000000Z")
        self.assertEqual(events[0]["end_utc"], "2026-08-09T10:03:00.000000Z")

    def test_feeding_events_start_after_initial_nonfeeding_minute(self) -> None:
        base = dt.datetime(2026, 8, 9, 10, 0, 0, tzinfo=dt.timezone.utc)
        rows = [
            leaf.LeafAreaRow("C01", "clip", base + dt.timedelta(minutes=1), 1, 100, 1.0, None, False, False, 6, 6),
            leaf.LeafAreaRow("C01", "clip", base + dt.timedelta(minutes=2), 1, 99, 0.99, 2.5, True, True, 6, 6),
            leaf.LeafAreaRow("C01", "clip", base + dt.timedelta(minutes=3), 1, 98, 0.98, 2.0, True, True, 6, 6),
        ]

        events = leaf.feeding_event_dicts(rows, dt.timezone.utc, sample_minutes=1)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start_utc"], "2026-08-09T10:01:00.000000Z")
        self.assertEqual(events[0]["end_utc"], "2026-08-09T10:03:00.000000Z")

    def test_feeding_events_final_boundary_does_not_extend_extra_minute(self) -> None:
        rows = [
            leaf.LeafAreaRow(
                animal_id="C01",
                clip_key="clip",
                timestamp_utc=dt.datetime(2026, 8, 9, 14, 5, 0, tzinfo=dt.timezone.utc),
                leaf_epoch=1,
                leaf_area_proxy_px=100.0,
                relative_leaf_area=1.0,
                loss_prev_min_pct=2.5,
                feeding_raw=True,
                feeding_final=True,
                n_sampled_frames=6,
                n_valid_frames=6,
            )
        ]

        events = leaf.feeding_event_dicts(rows, dt.timezone.utc, sample_minutes=1)

        self.assertEqual(events[0]["end_utc"], "2026-08-09T14:05:00.000000Z")


if __name__ == "__main__":
    unittest.main()
