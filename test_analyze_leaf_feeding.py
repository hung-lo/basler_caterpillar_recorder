#!/usr/bin/env python3

from __future__ import annotations

import datetime as dt
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import analyze_leaf_feeding as leaf


class FakeCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        frame_count: Optional[int] = None,
        seek_returns: bool = True,
        bad_position_index: Optional[int] = None,
    ) -> None:
        self.frames = frames
        self.frame_count = len(frames) if frame_count is None else frame_count
        self.seek_returns = seek_returns
        self.bad_position_index = bad_position_index
        self.current_index = 0
        self.last_pos_after_read = 0.0
        self.set_calls: list[float] = []
        self.read_calls = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop_id: float) -> float:
        if prop_id == leaf.cv2.CAP_PROP_FRAME_COUNT:
            return float(self.frame_count)
        if prop_id == leaf.cv2.CAP_PROP_POS_FRAMES:
            return self.last_pos_after_read
        return 0.0

    def set(self, prop_id: float, value: float) -> bool:
        if prop_id == leaf.cv2.CAP_PROP_POS_FRAMES:
            self.set_calls.append(value)
            if not self.seek_returns:
                return False
            self.current_index = int(value)
            return True
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_calls += 1
        if self.current_index < 0 or self.current_index >= len(self.frames):
            return False, None
        frame = self.frames[self.current_index]
        if self.bad_position_index is not None and self.current_index == self.bad_position_index:
            self.last_pos_after_read = float(self.current_index + 3)
        else:
            self.last_pos_after_read = float(self.current_index + 1)
        self.current_index += 1
        return True, frame

    def release(self) -> None:
        return None


class LeafFeedingLogicTests(unittest.TestCase):
    def test_format_progress_bar_reports_percent(self) -> None:
        bar = leaf.format_progress_bar(18, 36)

        self.assertIn("50.0%", bar)
        self.assertIn("[", bar)
        self.assertIn("]", bar)

    def test_build_progress_line_contains_sparse_status_fields(self) -> None:
        entry = leaf.ManifestEntry("C03", "clip_0012", Path("/tmp/video.mp4"), Path("/tmp/timestamps.csv"))
        line = leaf.build_progress_line(
            clip_index=2,
            total_clips=10,
            entry=entry,
            decoded_leaf_frames=18,
            total_leaf_frames=36,
            completed_estimates=3,
            total_estimates=6,
            status="analyzing",
        )

        self.assertIn("clip 2/10", line)
        self.assertIn("C03", line)
        self.assertIn("clip_0012", line)
        self.assertIn("samples 18/36", line)
        self.assertIn("estimates 3/6", line)
        self.assertIn("analyzing", line)

    def test_progress_reporter_disabled_writes_no_interactive_output(self) -> None:
        stream = io.StringIO()
        reporter = leaf.ProgressReporter(total_clips=3, enabled=False, stream=stream)
        entry = leaf.ManifestEntry("C01", "clip_a", Path("/tmp/video.mp4"), Path("/tmp/timestamps.csv"))

        reporter.start_clip(1, entry, source_frames=9000, selected_leaf_frames=36, total_estimates=6)
        reporter.update(decoded_leaf_frames=18, completed_estimates=3, status="analyzing")
        reporter.finish_clip(decoded_leaf_frames=36, completed_estimates=6, status="done")

        self.assertEqual(stream.getvalue(), "")

    def test_progress_reporter_finish_adds_newline(self) -> None:
        stream = io.StringIO()
        reporter = leaf.ProgressReporter(total_clips=3, enabled=True, stream=stream, update_interval_s=0.0)
        entry = leaf.ManifestEntry("C01", "clip_a", Path("/tmp/video.mp4"), Path("/tmp/timestamps.csv"))

        reporter.start_clip(1, entry, source_frames=9000, selected_leaf_frames=36, total_estimates=6)
        reporter.finish_clip(decoded_leaf_frames=36, completed_estimates=6, status="done")

        self.assertTrue(stream.getvalue().endswith("\n"))

    def test_select_leaf_sample_targets_uses_five_minute_grid(self) -> None:
        start = dt.datetime(2026, 8, 9, 10, 0, 0, tzinfo=dt.timezone.utc)
        timestamps = [start + dt.timedelta(seconds=10 * index) for index in range(121)]

        targets = leaf.select_leaf_sample_targets(
            timestamps,
            estimate_interval_minutes=5,
            burst_duration_seconds=60,
            burst_step_seconds=10,
        )

        buckets = sorted({target.estimate_bucket_utc for target in targets})
        self.assertEqual(
            buckets,
            [
                dt.datetime(2026, 8, 9, 10, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2026, 8, 9, 10, 5, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2026, 8, 9, 10, 10, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2026, 8, 9, 10, 15, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2026, 8, 9, 10, 20, 0, tzinfo=dt.timezone.utc),
            ],
        )

    def test_select_leaf_sample_targets_uses_burst_offsets(self) -> None:
        start = dt.datetime(2026, 8, 9, 10, 0, 0, tzinfo=dt.timezone.utc)
        timestamps = [start + dt.timedelta(seconds=10 * index) for index in range(40)]

        targets = leaf.select_leaf_sample_targets(
            timestamps,
            estimate_interval_minutes=5,
            burst_duration_seconds=60,
            burst_step_seconds=10,
        )

        first_bucket_targets = [target for target in targets if target.estimate_bucket_utc == start]
        self.assertEqual([target.frame_index for target in first_bucket_targets], [0, 1, 2, 3, 4, 5])

    def test_select_leaf_sample_targets_rejects_non_monotonic_timestamps(self) -> None:
        timestamps = [
            dt.datetime(2026, 8, 9, 10, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 8, 9, 10, 0, 10, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 8, 9, 10, 0, 5, tzinfo=dt.timezone.utc),
        ]

        with self.assertRaisesRegex(ValueError, "not monotonic"):
            leaf.select_leaf_sample_targets(timestamps)

    def test_hysteresis_uses_absolute_loss_thresholds(self) -> None:
        losses = [None, 800.0, 400.0, 250.0, 800.0, 700.0]

        flags = leaf.classify_feeding_hysteresis(
            losses,
            start_loss_px2=750.0,
            continue_loss_px2=300.0,
        )

        self.assertEqual(flags, [False, True, True, False, True, True])

    def test_short_gap_merge_and_short_bout_cleanup_use_five_minute_steps(self) -> None:
        merged = leaf.merge_short_gaps([True, False, True], step_minutes=5, max_gap_minutes=5)
        cleaned = leaf.remove_short_bouts([False, True, False], step_minutes=5, min_bout_minutes=10)

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
            leaf.LeafAreaEstimate("C01", "clip", base, 10000, (10000,), 6, 6),
            leaf.LeafAreaEstimate("C01", "clip", base + dt.timedelta(minutes=5), 9500, (9500,), 6, 6),
            leaf.LeafAreaEstimate("C01", "clip", base + dt.timedelta(minutes=10), 9000, (9000,), 6, 6),
            leaf.LeafAreaEstimate("C01", "clip", base + dt.timedelta(minutes=15), 15000, (15000,), 6, 6),
            leaf.LeafAreaEstimate("C01", "clip", base + dt.timedelta(minutes=20), 14950, (14950,), 6, 6),
        ]

        rows = leaf.finalize_leaf_rows(
            estimates,
            timezone=dt.timezone.utc,
            leaf_area_percentile=95.0,
            estimate_interval_minutes=5,
            allowed_gap_minutes=6,
            start_loss_px2=750.0,
            continue_loss_px2=300.0,
            merge_gap_minutes=5,
            min_bout_minutes=5,
            leaf_reset_increase_pct=20.0,
        )

        self.assertEqual([row.leaf_epoch for row in rows], [1, 1, 1, 2, 2])
        self.assertIsNone(rows[3].delta_area_5min_px2)

    def test_cross_clip_continuity_does_not_force_new_leaf_epoch(self) -> None:
        base = dt.datetime(2026, 8, 9, 12, 0, 0, tzinfo=dt.timezone.utc)
        estimates = [
            leaf.LeafAreaEstimate("C01", "clip_0010", base, 10000, (10000,), 6, 6),
            leaf.LeafAreaEstimate("C01", "clip_0010", base + dt.timedelta(minutes=5), 9700, (9700,), 6, 6),
            leaf.LeafAreaEstimate("C01", "clip_0011", base + dt.timedelta(minutes=10), 9400, (9400,), 6, 6),
            leaf.LeafAreaEstimate("C01", "clip_0011", base + dt.timedelta(minutes=15), 9100, (9100,), 6, 6),
        ]

        rows = leaf.finalize_leaf_rows(
            estimates,
            timezone=dt.timezone.utc,
            leaf_area_percentile=95.0,
            estimate_interval_minutes=5,
            allowed_gap_minutes=6,
            start_loss_px2=750.0,
            continue_loss_px2=300.0,
            merge_gap_minutes=5,
            min_bout_minutes=5,
            leaf_reset_increase_pct=20.0,
        )

        self.assertEqual([row.leaf_epoch for row in rows], [1, 1, 1, 1])

    def test_bad_video_endpoint_invalidates_loss(self) -> None:
        base = dt.datetime(2026, 8, 9, 12, 0, 0, tzinfo=dt.timezone.utc)
        estimates = [
            leaf.LeafAreaEstimate("C01", "clip", base, 10000, (10000,), 6, 6),
            leaf.LeafAreaEstimate("C01", "clip", base + dt.timedelta(minutes=5), 9200, (9200,), 6, 6, video_quality_excluded=True),
            leaf.LeafAreaEstimate("C01", "clip", base + dt.timedelta(minutes=10), 8400, (8400,), 6, 6),
        ]

        rows = leaf.finalize_leaf_rows(
            estimates,
            timezone=dt.timezone.utc,
            leaf_area_percentile=95.0,
            estimate_interval_minutes=5,
            allowed_gap_minutes=6,
            start_loss_px2=750.0,
            continue_loss_px2=300.0,
            merge_gap_minutes=5,
            min_bout_minutes=5,
            leaf_reset_increase_pct=20.0,
        )

        self.assertFalse(rows[1].feeding_valid)
        self.assertIsNone(rows[1].delta_area_5min_px2)
        self.assertIsNone(rows[2].delta_area_5min_px2)

    def test_bad_video_between_bursts_invalidates_whole_transition(self) -> None:
        base = dt.datetime(2026, 8, 9, 10, 0, 0, tzinfo=dt.timezone.utc)
        estimates = [
            leaf.LeafAreaEstimate("C01", "clip", base, 10000, (10000,), 6, 6),
            leaf.LeafAreaEstimate("C01", "clip", base + dt.timedelta(minutes=5), 9000, (9000,), 6, 6),
        ]

        rows = leaf.finalize_leaf_rows(
            estimates,
            timezone=dt.timezone.utc,
            leaf_area_percentile=95.0,
            estimate_interval_minutes=5,
            allowed_gap_minutes=6,
            start_loss_px2=750.0,
            continue_loss_px2=300.0,
            merge_gap_minutes=5,
            min_bout_minutes=5,
            leaf_reset_increase_pct=20.0,
            video_quality_intervals_utc=[
                (
                    dt.datetime(2026, 8, 9, 10, 2, 0, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 8, 9, 10, 3, 0, tzinfo=dt.timezone.utc),
                )
            ],
        )

        self.assertFalse(rows[1].feeding_valid)
        self.assertIsNone(rows[1].delta_area_5min_px2)

    def test_consolidate_leaf_estimates_combines_duplicate_bucket_samples(self) -> None:
        timestamp = dt.datetime(2026, 8, 9, 12, 30, 0, tzinfo=dt.timezone.utc)
        estimates = [
            leaf.LeafAreaEstimate("C01", "clip_0010", timestamp, 9500, (9400, 9500), 3, 2),
            leaf.LeafAreaEstimate("C01", "clip_0011", timestamp, 9600, (9600, 9650), 3, 2),
        ]

        consolidated = leaf.consolidate_leaf_estimates(estimates, leaf_area_percentile=95.0)

        self.assertEqual(len(consolidated), 1)
        self.assertEqual(consolidated[0].n_sampled_frames, 6)
        self.assertEqual(consolidated[0].n_valid_frames, 4)
        self.assertEqual(consolidated[0].clip_key, "clip_0010+clip_0011")

    def test_feeding_events_start_at_estimate_time_and_extend_forward(self) -> None:
        rows = [
            leaf.LeafAreaRow("C01", "clip", dt.datetime(2026, 8, 9, 10, 0, 0, tzinfo=dt.timezone.utc), 1, 10000, 1.0, None, None, True, False, False, False, 6, 6),
            leaf.LeafAreaRow("C01", "clip", dt.datetime(2026, 8, 9, 10, 5, 0, tzinfo=dt.timezone.utc), 1, 9200, 0.92, -8.0, 800.0, True, True, True, False, 6, 6),
            leaf.LeafAreaRow("C01", "clip", dt.datetime(2026, 8, 9, 10, 10, 0, tzinfo=dt.timezone.utc), 1, 8800, 0.88, -4.0, 400.0, True, True, True, False, 6, 6),
        ]

        events = leaf.feeding_event_dicts(
            rows,
            dt.timezone.utc,
            estimate_interval_minutes=5,
            start_loss_px2=750.0,
            continue_loss_px2=300.0,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start_utc"], "2026-08-09T10:05:00.000000Z")
        self.assertEqual(events[0]["end_utc"], "2026-08-09T10:15:00.000000Z")

    def test_frame_mismatch_is_reported_before_sampling(self) -> None:
        entry = leaf.ManifestEntry("C01", "clip_a", Path("/tmp/video.mp4"), Path("/tmp/timestamps.csv"))
        timestamps = [dt.datetime(2026, 8, 9, 12, 0, 0, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=index) for index in range(9000)]
        fake_capture = FakeCapture([np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(5)], frame_count=5890)

        with mock.patch.object(leaf, "load_timestamp_series", return_value=timestamps):
            with mock.patch.object(leaf.cv2, "VideoCapture", return_value=fake_capture):
                result = leaf.extract_clip_leaf_estimates(
                    entry,
                    clip_index=1,
                    estimate_interval_minutes=5,
                    burst_duration_seconds=60,
                    burst_step_seconds=10,
                    leaf_area_percentile=95.0,
                    min_valid_frames=3,
                    hue_low=25,
                    hue_high=95,
                    sat_min=35,
                    value_min=25,
                    min_component_px=20,
                    morph_kernel=1,
                    frame_access="sparse",
                    video_quality_intervals_utc=[],
                )

        self.assertEqual(result.status, "frame_mismatch")
        self.assertEqual(result.estimates, [])

    def test_invalid_timestamps_are_reported(self) -> None:
        entry = leaf.ManifestEntry("C01", "clip_a", Path("/tmp/video.mp4"), Path("/tmp/timestamps.csv"))
        timestamps = [
            dt.datetime(2026, 8, 9, 12, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 8, 9, 12, 0, 10, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 8, 9, 12, 0, 5, tzinfo=dt.timezone.utc),
        ]

        with mock.patch.object(leaf, "load_timestamp_series", return_value=timestamps):
            result = leaf.extract_clip_leaf_estimates(
                entry,
                clip_index=1,
                estimate_interval_minutes=5,
                burst_duration_seconds=60,
                burst_step_seconds=10,
                leaf_area_percentile=95.0,
                min_valid_frames=3,
                hue_low=25,
                hue_high=95,
                sat_min=35,
                value_min=25,
                min_component_px=20,
                morph_kernel=1,
                frame_access="sparse",
                video_quality_intervals_utc=[],
            )

        self.assertEqual(result.status, "invalid_timestamps")
        self.assertIn("not monotonic", result.error)

    def test_sparse_sampling_seeks_only_selected_frames(self) -> None:
        entry = leaf.ManifestEntry("C01", "clip_a", Path("/tmp/video.mp4"), Path("/tmp/timestamps.csv"))
        start = dt.datetime(2026, 8, 9, 12, 0, 0, tzinfo=dt.timezone.utc)
        timestamps = [start + dt.timedelta(milliseconds=200 * index) for index in range(9000)]
        frames = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(9000)]
        frames[0][:] = (0, 255, 0)
        fake_capture = FakeCapture(frames, frame_count=9000)

        with mock.patch.object(leaf, "load_timestamp_series", return_value=timestamps):
            with mock.patch.object(leaf.cv2, "VideoCapture", return_value=fake_capture):
                with mock.patch.object(leaf, "segment_leaf_area", return_value=(1000.0, np.zeros((20, 20), dtype=np.uint8))):
                    result = leaf.extract_clip_leaf_estimates(
                        entry,
                        clip_index=1,
                        estimate_interval_minutes=5,
                        burst_duration_seconds=60,
                        burst_step_seconds=10,
                        leaf_area_percentile=95.0,
                        min_valid_frames=3,
                        hue_low=25,
                        hue_high=95,
                        sat_min=35,
                        value_min=25,
                        min_component_px=20,
                        morph_kernel=1,
                        frame_access="sparse",
                        video_quality_intervals_utc=[],
                    )

        self.assertEqual(result.status, "computed")
        self.assertEqual(result.decoded_leaf_frames, 36)
        self.assertEqual(fake_capture.read_calls, 36)
        self.assertEqual(len(fake_capture.set_calls), 36)
        self.assertEqual(len(result.estimates), 6)

    def test_seek_mismatch_is_reported(self) -> None:
        entry = leaf.ManifestEntry("C01", "clip_a", Path("/tmp/video.mp4"), Path("/tmp/timestamps.csv"))
        start = dt.datetime(2026, 8, 9, 12, 0, 0, tzinfo=dt.timezone.utc)
        timestamps = [start + dt.timedelta(seconds=10 * index) for index in range(40)]
        frames = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(40)]
        fake_capture = FakeCapture(frames, frame_count=40, bad_position_index=0)

        with mock.patch.object(leaf, "load_timestamp_series", return_value=timestamps):
            with mock.patch.object(leaf.cv2, "VideoCapture", return_value=fake_capture):
                result = leaf.extract_clip_leaf_estimates(
                    entry,
                    clip_index=1,
                    estimate_interval_minutes=5,
                    burst_duration_seconds=60,
                    burst_step_seconds=10,
                    leaf_area_percentile=95.0,
                    min_valid_frames=3,
                    hue_low=25,
                    hue_high=95,
                    sat_min=35,
                    value_min=25,
                    min_component_px=20,
                    morph_kernel=1,
                    frame_access="sparse",
                    video_quality_intervals_utc=[],
                )

        self.assertEqual(result.status, "seek_mismatch")

    def test_load_analysis_events_supports_google_sheet_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            url = "https://docs.google.com/spreadsheets/d/1yqe8VII3YNzX2EmOlbBckgTCJKXD47PLZEfHeO2yQ2w/edit?gid=1696022641#gid=1696022641"
            csv_bytes = (
                "animal_id,start_local,end_local,event,kind,notes\n"
                "All,2026-08-07T18:49:00-04:00,2026-08-07T20:59:00-04:00,video_quality_low,video,White balance is off\n"
            ).encode("utf-8")

            with mock.patch(
                "plot_recording_timeline.urllib_request.urlopen",
                return_value=mock.Mock(
                    __enter__=mock.Mock(return_value=mock.Mock(read=mock.Mock(return_value=csv_bytes))),
                    __exit__=mock.Mock(return_value=False),
                ),
            ):
                resets, intervals = leaf.load_analysis_events(
                    root=root,
                    source=url,
                    timezone=dt.timezone(dt.timedelta(hours=-4)),
                )
            self.assertEqual(resets, {})
            self.assertEqual(len(intervals), 1)
            self.assertEqual(intervals[0][0], dt.datetime(2026, 8, 7, 22, 49, 0, tzinfo=dt.timezone.utc))
            self.assertTrue((root / "behavior_events_used.csv").exists())
            self.assertTrue((root / "behavior_events_source.json").exists())

    def test_main_wires_progress_reporter_into_clip_extraction(self) -> None:
        repo_root = Path.cwd()
        entry_a = leaf.ManifestEntry("C01", "clip_a", repo_root / "a.mp4", repo_root / "a.timestamps.csv")
        entry_b = leaf.ManifestEntry("C02", "clip_b", repo_root / "b.mp4", repo_root / "b.timestamps.csv")

        with mock.patch.object(leaf, "load_manifest_entries", return_value=[entry_a, entry_b]):
            with mock.patch.object(leaf, "load_analysis_events", return_value=({}, [])):
                with mock.patch.object(leaf, "finalize_leaf_rows", return_value=[]):
                    with mock.patch.object(leaf, "feeding_event_dicts", return_value=[]):
                        with mock.patch.object(leaf, "leaf_rows_to_dicts", return_value=[]):
                            with mock.patch.object(leaf, "write_csv"):
                                with mock.patch.object(leaf, "load_timezone", return_value=dt.timezone.utc):
                                    with mock.patch.object(
                                        leaf,
                                        "extract_clip_leaf_estimates",
                                        side_effect=[
                                            leaf.ClipFeedingResult([], "computed", 9000, 9000, 36, 36),
                                            leaf.ClipFeedingResult([], "computed", 9000, 9000, 36, 36),
                                        ],
                                    ) as mock_extract:
                                        rc = leaf.main(["."])

        self.assertEqual(rc, 0)
        first_call = mock_extract.call_args_list[0]
        second_call = mock_extract.call_args_list[1]
        self.assertEqual(first_call.kwargs["clip_index"], 1)
        self.assertEqual(second_call.kwargs["clip_index"], 2)
        self.assertIs(first_call.kwargs["progress_reporter"], second_call.kwargs["progress_reporter"])


if __name__ == "__main__":
    unittest.main()
