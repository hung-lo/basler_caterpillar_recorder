#!/usr/bin/env python3

from __future__ import annotations

import datetime as dt
import unittest

import numpy as np

import analyze_leaf_feeding as leaf


class LeafFeedingLogicTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
