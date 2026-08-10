#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

import crop_caterpillar_videos as cropper
import prepare_cropped_timestamps as prep


class PrepareCroppedTimestampsTests(unittest.TestCase):
    def write_timestamp_sidecar(self, path: Path, *, rows: int = 3) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_index", "host_utc_ns", "host_utc_iso"])
            for index in range(rows):
                writer.writerow(
                    [
                        index,
                        1_754_760_000_000_000_000 + index * 1_000_000_000,
                        f"2025-08-09T19:00:0{index}Z",
                    ]
                )

    def create_current_source(self, root: Path) -> Path:
        source = root / "20260809_134217-0400" / "clip_0000_134218-0400" / "camera1.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"fake-mp4")
        return source

    def create_crops(self, root: Path, source: Path, animal_ids: list[str]) -> str:
        naming = cropper.resolve_source_naming(source)
        self.assertIsNotNone(naming.stem)
        cropped_dir = root / "cropped_by_caterpillar"
        cropped_dir.mkdir(parents=True, exist_ok=True)
        for animal_id in animal_ids:
            cropper.output_path(cropped_dir, animal_id, naming).write_bytes(b"fake-crop")
        return naming.stem or ""

    def read_manifest_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_canonical_mapping_copies_one_shared_timestamp_for_multiple_crops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.create_current_source(root)
            self.write_timestamp_sidecar(source.with_name("camera1.timestamps.csv.gz"))
            stem = self.create_crops(root, source, ["C01", "C02"])

            rows, warnings, errors = prep.prepare_manifest(root, force=False)
            manifest_path = root / "cropped_by_caterpillar" / "crop_manifest.csv"
            prep.write_manifest(rows, manifest_path)

            self.assertEqual(warnings, [])
            self.assertEqual(errors, [])
            copied = root / "cropped_by_caterpillar" / "timestamps" / f"{stem}.timestamps.csv.gz"
            self.assertTrue(copied.exists())

            manifest_rows = self.read_manifest_rows(manifest_path)
            self.assertEqual(len(manifest_rows), 2)
            self.assertEqual(
                {row["copied_timestamp_file"] for row in manifest_rows},
                {f"cropped_by_caterpillar/timestamps/{stem}.timestamps.csv.gz"},
            )

    def test_all_eight_crops_share_one_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.create_current_source(root)
            self.write_timestamp_sidecar(source.with_name("camera1.timestamps.csv.gz"))
            stem = self.create_crops(root, source, sorted(cropper.CROPS))

            rows, _warnings, errors = prep.prepare_manifest(root, force=False)

            self.assertEqual(errors, [])
            self.assertEqual(len(rows), 8)
            copied_files = list((root / "cropped_by_caterpillar" / "timestamps").glob("*"))
            self.assertEqual([path.name for path in copied_files], [f"{stem}.timestamps.csv.gz"])

    def test_missing_adjacent_timestamp_marks_rows_unresolved_without_guessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.create_current_source(root)
            self.create_crops(root, source, ["C01"])
            unrelated = root / "elsewhere" / "camera1.timestamps.csv.gz"
            self.write_timestamp_sidecar(unrelated)

            rows, warnings, errors = prep.prepare_manifest(root, force=False)

            self.assertEqual(errors, [])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].frame_count_status, "missing_timestamp")
            self.assertEqual(rows[0].source_timestamp_file, "")
            self.assertEqual(rows[0].copied_timestamp_file, "")
            self.assertEqual(len(warnings), 1)

    def test_running_twice_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.create_current_source(root)
            self.write_timestamp_sidecar(source.with_name("camera1.timestamps.csv.gz"))
            stem = self.create_crops(root, source, ["C01", "C02"])

            first_rows, _warnings, first_errors = prep.prepare_manifest(root, force=False)
            second_rows, _warnings_2, second_errors = prep.prepare_manifest(root, force=False)

            self.assertEqual(first_errors, [])
            self.assertEqual(second_errors, [])
            self.assertEqual([row.as_dict() for row in first_rows], [row.as_dict() for row in second_rows])
            copied_files = list((root / "cropped_by_caterpillar" / "timestamps").glob("*"))
            self.assertEqual([path.name for path in copied_files], [f"{stem}.timestamps.csv.gz"])

    def test_conflicting_destination_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.create_current_source(root)
            self.write_timestamp_sidecar(source.with_name("camera1.timestamps.csv.gz"))
            stem = self.create_crops(root, source, ["C01"])

            prep.prepare_manifest(root, force=False)
            conflicting_copy = root / "cropped_by_caterpillar" / "timestamps" / f"{stem}.timestamps.csv.gz"
            conflicting_copy.write_bytes(b"not-the-same")

            rows, _warnings, errors = prep.prepare_manifest(root, force=False)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].frame_count_status, "timestamp_conflict")
            self.assertEqual(len(errors), 1)
            self.assertEqual(conflicting_copy.read_bytes(), b"not-the-same")

    def test_flat_cropped_output_path_remains_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "clip_0000_152652.mp4"
            source.write_bytes(b"legacy-flat")
            self.write_timestamp_sidecar(root / "clip_0000_152652.timestamps.csv.gz")
            cropped_dir = root / "cropped_by_caterpillar"
            cropped_dir.mkdir()
            crop_path = cropped_dir / "C01_clip_0000_152652.mp4"
            crop_path.write_bytes(b"crop")

            rows, _warnings, errors = prep.prepare_manifest(root, force=False)

            self.assertEqual(errors, [])
            self.assertEqual(len(rows), 1)
            self.assertTrue(crop_path.exists())
            self.assertEqual(rows[0].cropped_video, "cropped_by_caterpillar/C01_clip_0000_152652.mp4")


if __name__ == "__main__":
    unittest.main()
