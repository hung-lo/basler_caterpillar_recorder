#!/usr/bin/env python3

from __future__ import annotations

import datetime as dt
import unittest

import analysis_timing as timing


class AnalysisTimingTests(unittest.TestCase):
    def test_parse_timestamp_row_from_ns_matches_expected_utc(self) -> None:
        value = dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc)
        row = {"host_utc_ns": str(int(value.timestamp() * 1e9))}

        parsed = timing.parse_timestamp_row(row)

        self.assertEqual(parsed, value)

    def test_wwoods_hole_display_conversion_is_preserved(self) -> None:
        tz = timing.load_timezone("America/New_York")
        utc_value = dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc)

        local_value = utc_value.astimezone(tz)

        self.assertEqual(local_value.isoformat(sep=" "), "2026-08-09 15:00:00-04:00")

    def test_naive_utc_timestamp_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            timing.parse_utc_value("2026-08-09T19:00:00")


if __name__ == "__main__":
    unittest.main()
