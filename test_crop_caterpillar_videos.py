#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import PureWindowsPath

import crop_caterpillar_videos as cropper
import rename_existing_crops_with_date as renamer


class CropperTimestampTests(unittest.TestCase):
    def test_current_same_day_local(self) -> None:
        source = PureWindowsPath(
            "D:/data/20260809_134217-0400/clip_0000_134218-0400/camera1.mp4"
        )

        metadata = cropper.source_metadata(source)
        naming = cropper.resolve_source_naming(source)

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.stem, "20260809_134218-0400_clip_0000")
        self.assertEqual(metadata.source_format, "current")
        self.assertEqual(naming.stem, "20260809_134218-0400_clip_0000")

    def test_current_local_midnight_rollover(self) -> None:
        source = PureWindowsPath(
            "D:/data/20260809_235500-0400/clip_0007_001000-0400/camera1.mp4"
        )

        metadata = cropper.source_metadata(source)

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.stem, "20260810_001000-0400_clip_0007")
        self.assertEqual(metadata.source_format, "current")

    def test_current_clip_inherits_session_offset(self) -> None:
        source = PureWindowsPath(
            "D:/data/20260809_134217-0400/clip_0002_140000/camera1.mp4"
        )

        metadata = cropper.source_metadata(source)

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.stem, "20260809_140000-0400_clip_0002")
        self.assertEqual(metadata.source_format, "current")

    def test_explicit_oldcommit_utc_conversion(self) -> None:
        source = PureWindowsPath(
            "D:/data/20260806_133944_oldcommit/clip_0000_133946/camera1.mp4"
        )

        metadata = cropper.source_metadata(source)

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.stem, "20260806_093946-0400_clip_0000")
        self.assertEqual(metadata.source_format, "oldcommit")

    def test_oldcommit_conversion_goes_to_previous_local_date(self) -> None:
        source = PureWindowsPath(
            "D:/data/20260806_010000_oldcommit/clip_0000_010100/camera1.mp4"
        )

        metadata = cropper.source_metadata(source)

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.stem, "20260805_210100-0400_clip_0000")
        self.assertEqual(metadata.source_format, "oldcommit")

    def test_oldcommit_utc_midnight_rollover_before_conversion(self) -> None:
        source = PureWindowsPath(
            "D:/data/20260806_235500_oldcommit/clip_0001_001000/camera1.mp4"
        )

        metadata = cropper.source_metadata(source)

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.stem, "20260806_201000-0400_clip_0001")
        self.assertEqual(metadata.source_format, "oldcommit")

    def test_legacy_no_offset_utc_conversion(self) -> None:
        source = PureWindowsPath(
            "D:/data/20260804_145301/clip_0000_145302/camera1.mp4"
        )

        metadata = cropper.source_metadata(source)

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.stem, "20260804_105302-0400_clip_0000")
        self.assertEqual(metadata.source_format, "legacy_utc")

    def test_legacy_no_offset_utc_midnight_rollover(self) -> None:
        source = PureWindowsPath(
            "D:/data/20260804_235500/clip_0001_001000/camera1.mp4"
        )

        metadata = cropper.source_metadata(source)

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.stem, "20260804_201000-0400_clip_0001")
        self.assertEqual(metadata.source_format, "legacy_utc")

    def test_invalid_nested_session_fails_closed(self) -> None:
        source = PureWindowsPath(
            "D:/data/some_unknown_session/clip_0000_120000/camera1.mp4"
        )

        self.assertIsNone(cropper.source_metadata(source))
        naming = cropper.resolve_source_naming(source)
        self.assertIsNone(naming.stem)
        self.assertIsNotNone(naming.error)

    def test_malformed_clock_fails_closed(self) -> None:
        source = PureWindowsPath(
            "D:/data/20260809_134217-0400/clip_0000_996099-0400/camera1.mp4"
        )

        self.assertIsNone(cropper.source_metadata(source))
        naming = cropper.resolve_source_naming(source)
        self.assertIsNone(naming.stem)
        self.assertIsNotNone(naming.error)

    def test_flat_legacy_input_remains_supported(self) -> None:
        source = PureWindowsPath("D:/data/clip_0000_152652.mp4")

        naming = cropper.resolve_source_naming(source)

        self.assertEqual(naming.stem, "clip_0000_152652")
        self.assertEqual(naming.layout, "flat_legacy")

    def test_renamer_uses_the_same_canonical_stem(self) -> None:
        samples = [
            PureWindowsPath(
                "D:/data/20260809_134217-0400/clip_0000_134218-0400/camera1.mp4"
            ),
            PureWindowsPath(
                "D:/data/20260806_133944_oldcommit/clip_0000_133946/camera1.mp4"
            ),
            PureWindowsPath("D:/data/20260804_145301/clip_0000_145302/camera1.mp4"),
        ]

        for source in samples:
            with self.subTest(source=source):
                crop_name = cropper.resolve_source_naming(source).stem
                rename_name = renamer.resolve_source_naming(source).stem
                self.assertEqual(crop_name, rename_name)


if __name__ == "__main__":
    unittest.main()
