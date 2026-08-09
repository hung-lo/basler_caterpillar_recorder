#!/usr/bin/env python3

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path, PureWindowsPath
from unittest import mock

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
        with self.assertRaises(cropper.ClipParseError):
            cropper.parse_clip_metadata(source)

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

    def test_missing_one_caterpillar_output_only_returns_that_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "cropped_by_caterpillar"
            output_dir.mkdir()
            source = root / "20260809_134217-0400" / "clip_0000_134218-0400" / "camera1.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"fake")
            resolved = cropper.resolve_source_naming(source)

            for cid in ("C01", "C02", "C04", "C05", "C06", "C07", "C08"):
                (output_dir / f"{cid}_{resolved.stem}.mp4").write_bytes(b"x")

            missing = cropper.missing_ids(output_dir, resolved)

            self.assertEqual(missing, ["C03"])

    def test_dry_run_uses_canonical_output_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "20260809_134217-0400" / "clip_0000_134218-0400" / "camera1.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"fake")
            old_mtime = source.stat().st_mtime - 600
            os.utime(source, (old_mtime, old_mtime))

            with mock.patch.object(cropper, "probe_resolution", return_value=(1200, 1920)):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    count = cropper.scan_once(root, min_age=0, dry_run=True)

            self.assertEqual(count, 1)
            self.assertIn(
                "C01_20260809_134218-0400_clip_0000.mp4",
                buf.getvalue(),
            )

    def test_unparseable_source_is_skipped_without_generating_fallback_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "mystery_session" / "clip_0000_202612-0400" / "camera1.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"fake")
            old_mtime = source.stat().st_mtime - 600
            os.utime(source, (old_mtime, old_mtime))

            buf = io.StringIO()
            with redirect_stdout(buf):
                count = cropper.scan_once(root, min_age=0, dry_run=True)

            self.assertEqual(count, 0)
            self.assertIn("SKIP unrecognized recorder timestamp layout:", buf.getvalue())
            self.assertNotIn("C01_clip_", buf.getvalue())

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

    def test_renamer_rejects_ambiguous_legacy_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "cropped_by_caterpillar"
            output_dir.mkdir()

            session_a = root / "20260806_133944_oldcommit" / "clip_0000_133946" / "camera1.mp4"
            session_b = root / "20260808_133944_oldcommit" / "clip_0000_133946" / "camera1.mp4"
            session_a.parent.mkdir(parents=True)
            session_b.parent.mkdir(parents=True)
            session_a.write_bytes(b"a")
            session_b.write_bytes(b"b")

            (output_dir / "C01_clip_0000_133946.mp4").write_bytes(b"x")

            with mock.patch.object(
                renamer,
                "discover_sources",
                return_value=[session_a, session_b],
            ):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = renamer.main(["--root", str(root)])

            self.assertEqual(rc, 0)
            self.assertIn("AMBIGUOUS:", buf.getvalue())
            self.assertTrue((output_dir / "C01_clip_0000_133946.mp4").exists())


if __name__ == "__main__":
    unittest.main()
