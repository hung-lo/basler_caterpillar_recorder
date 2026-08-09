#!/usr/bin/env python3

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path, PureWindowsPath
from unittest import mock

import caterpillar_clip_naming as naming
import crop_caterpillar_videos as cropper
import rename_existing_crops_with_date as renamer


class CropNamingTests(unittest.TestCase):
    def test_current_recorder_path_uses_session_date_and_clip_time(self) -> None:
        source = PureWindowsPath(
            "D:/data/20260809_134217-0400/clip_0000_134218-0400/camera1.mp4"
        )

        metadata = naming.parse_clip_metadata(source)

        self.assertEqual(metadata.stem, "20260809_134218-0400_clip_0000")
        self.assertEqual(
            metadata.local_datetime.isoformat(sep=" "),
            "2026-08-09 13:42:18-04:00",
        )

    def test_oldcommit_path_converts_utc_to_woods_hole(self) -> None:
        source = PureWindowsPath(
            "D:/data/20260806_133944_oldcommit/clip_0000_133946/camera1.mp4"
        )

        metadata = naming.parse_clip_metadata(source)

        self.assertEqual(metadata.stem, "20260806_093946-0400_clip_0000")
        self.assertEqual(
            metadata.local_datetime.isoformat(sep=" "),
            "2026-08-06 09:39:46-04:00",
        )

    def test_oldcommit_midnight_rollover_is_handled(self) -> None:
        source = PureWindowsPath(
            "D:/data/20260807_235900_oldcommit/clip_0004_000200/camera1.mp4"
        )

        metadata = naming.parse_clip_metadata(source)

        self.assertEqual(metadata.stem, "20260807_200200-0400_clip_0004")
        self.assertEqual(
            metadata.local_datetime.isoformat(sep=" "),
            "2026-08-07 20:02:00-04:00",
        )

    def test_unknown_session_folder_is_rejected(self) -> None:
        source = PureWindowsPath(
            "D:/data/mystery_session/clip_0000_202612-0400/camera1.mp4"
        )

        with self.assertRaises(naming.ClipParseError):
            naming.parse_clip_metadata(source)

    def test_missing_one_caterpillar_output_only_returns_that_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "cropped_by_caterpillar"
            output_dir.mkdir()
            source = root / "20260809_134217-0400" / "clip_0000_134218-0400" / "camera1.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"fake")
            resolved = naming.resolve_source_naming(source)

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

            for cid in ("C01",):
                (output_dir / f"{cid}_clip_0000_133946.mp4").write_bytes(b"x")

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
