#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import csv
import datetime as dt
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path

import yaml

import validate_session


def write_timestamp_sidecar(path: Path) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_index", "host_utc_ns", "host_utc_iso", "host_monotonic_ns"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "frame_index": 0,
                "host_utc_ns": 1,
                "host_utc_iso": "1970-01-01T00:00:00.000Z",
                "host_monotonic_ns": 1,
            }
        )


def write_minimal_session(
    session_dir: Path,
    *,
    review_rows: list[dict[str, str]],
    create_all_jpegs: bool = True,
    create_review_csv: bool = True,
    review_temp_files: list[str] | None = None,
    review_metadata_overrides: dict[str, object] | None = None,
) -> Path:
    config = {
        "project": "test",
        "subject": "subject",
        "schedule": {
            "number_of_clips": 1,
        },
        "cameras": [
            {
                "label": "camera1",
                "fps": 5.0,
            }
        ],
        "archive": {"enabled": False},
    }
    (session_dir / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    manifest = {
        "created_utc": "2026-08-11T12:00:00+00:00",
        "created_local": "2026-08-11T08:00:00-04:00",
        "session_start_utc": "2026-08-11T12:00:00+00:00",
        "session_start_local": "2026-08-11T08:00:00-04:00",
        "naming": {"version": 3},
        "recording_plan": {
            "schedule_start_utc": "2026-08-11T12:00:00+00:00",
            "schedule_start_local": "2026-08-11T08:00:00-04:00",
            "planned_finish_utc": "2026-08-11T12:10:00+00:00",
            "planned_finish_local": "2026-08-11T08:10:00-04:00",
        },
    }
    (session_dir / "session_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    summary = {
        "completed_clips": 1,
        "requested_clips": 1,
        "stopped_by_signal": False,
        "any_failure": False,
        "exit_status": "success",
        "finished_utc": "2026-08-11T12:10:00+00:00",
        "finished_local": "2026-08-11T08:10:00-04:00",
        "unexpected_exception": "",
    }
    (session_dir / "session_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    clip_dir = session_dir / "clip_0000_120000+0000"
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "camera1.mp4").write_bytes(b"mp4")
    write_timestamp_sidecar(clip_dir / "camera1.timestamps.csv.gz")

    review_dir = clip_dir / "review_snapshots"
    review_dir.mkdir(exist_ok=True)
    for row in review_rows:
        filename = row["filename"]
        if create_all_jpegs:
            (review_dir / filename).write_bytes(b"jpg")

    if review_temp_files:
        for temp_name in review_temp_files:
            (review_dir / temp_name).write_bytes(b"tmp")

    if create_review_csv:
        with (review_dir / "camera1_snapshots.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "snapshot_index",
                    "filename",
                    "target_elapsed_s",
                    "actual_clip_elapsed_s",
                    "target_error_s",
                    "video_time_s",
                    "frame_index",
                    "host_utc_ns",
                    "host_utc_iso",
                    "host_local_iso",
                    "host_monotonic_ns",
                ],
            )
            writer.writeheader()
            for row in review_rows:
                writer.writerow(row)

    metadata = {
        "label": "camera1",
        "requested_settings": {
            "label": "camera1",
            "fps": 5.0,
        },
        "success": True,
        "grab_failures": 0,
        "mp4_remux_succeeded": True,
        "review_snapshots": {
            "enabled": True,
            "operational": True,
            "writer_started": True,
            "writer_finalized": True,
            "index_csv_written": True,
            "directory": "review_snapshots",
            "index_csv": "review_snapshots/camera1_snapshots.csv",
            "requested_per_full_clip": 10,
            "saved": len(review_rows),
            "failed": 0,
            "dropped_queue_full": 0,
            "missed_due_acquisition_gap": 0,
            "unreached_targets": 10 - len(review_rows),
            "error": None,
        },
    }
    if review_metadata_overrides:
        metadata["review_snapshots"].update(review_metadata_overrides)
    (clip_dir / "camera1.json").write_text(json.dumps(metadata), encoding="utf-8")
    return session_dir


class ReviewSnapshotValidationTests(unittest.TestCase):
    def _make_row(self, snapshot_index: int, frame_index: int) -> dict[str, str]:
        elapsed = 30.0 + snapshot_index * 60.0
        return {
            "snapshot_index": str(snapshot_index),
            "filename": f"camera1_review_snapshot_{snapshot_index:04d}.jpg",
            "target_elapsed_s": f"{elapsed:.6f}",
            "actual_clip_elapsed_s": f"{elapsed + 0.250000:.6f}",
            "target_error_s": "0.250000",
            "video_time_s": f"{elapsed - 0.250000:.6f}",
            "frame_index": str(frame_index),
            "host_utc_ns": str(1 + snapshot_index),
            "host_utc_iso": "1970-01-01T00:00:00.000Z",
            "host_local_iso": "1970-01-01T00:00:00.000+00:00",
            "host_monotonic_ns": str(1 + snapshot_index),
        }

    def _write_session_with_rows(
        self,
        tmp: str,
        rows: list[dict[str, str]],
        *,
        create_all_jpegs: bool = True,
        create_review_csv: bool = True,
        review_temp_files: list[str] | None = None,
        review_metadata_overrides: dict[str, object] | None = None,
    ) -> Path:
        session_dir = Path(tmp) / "20260811_120000+0000"
        session_dir.mkdir()
        write_minimal_session(
            session_dir,
            review_rows=rows,
            create_all_jpegs=create_all_jpegs,
            create_review_csv=create_review_csv,
            review_temp_files=review_temp_files,
            review_metadata_overrides=review_metadata_overrides,
        )
        return session_dir

    def test_validate_review_snapshots_complete_set_has_no_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = [self._make_row(index, 40 + index) for index in range(10)]
            session_dir = self._write_session_with_rows(tmp, rows)
            clip_dir = session_dir / "clip_0000_120000+0000"
            metadata = json.loads((clip_dir / "camera1.json").read_text(encoding="utf-8"))

            warnings = validate_session.validate_review_snapshots(
                clip_dir=clip_dir,
                camera_json_path=clip_dir / "camera1.json",
                camera_metadata=metadata,
            )

        self.assertEqual(warnings, [])

    def test_validator_warning_does_not_fail_core_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "20260811_120000+0000"
            session_dir.mkdir()
            write_minimal_session(
                session_dir,
                review_rows=[
                    {
                        "snapshot_index": 0,
                        "filename": "camera1_review_snapshot_0000.jpg",
                        "target_elapsed_s": "30.000000",
                        "actual_clip_elapsed_s": "30.250000",
                        "target_error_s": "0.250000",
                        "video_time_s": "29.750000",
                        "frame_index": "42",
                        "host_utc_ns": "1",
                        "host_utc_iso": "1970-01-01T00:00:00.000Z",
                        "host_local_iso": "1970-01-01T00:00:00.000+00:00",
                        "host_monotonic_ns": "1",
                    }
                ],
                create_all_jpegs=False,
            )

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                rc = validate_session.validate_session(session_dir)

        self.assertEqual(rc, 0)
        output = buffer.getvalue()
        self.assertIn("Warnings:", output)
        self.assertIn("review snapshot JPEG is missing", output)
        self.assertNotIn("FAIL:", output)

    def test_validate_review_snapshots_detects_duplicate_snapshot_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip_dir = Path(tmp) / "clip_0000_120000+0000"
            clip_dir.mkdir()
            review_dir = clip_dir / "review_snapshots"
            review_dir.mkdir()
            for filename in ["camera1_a.jpg", "camera1_b.jpg"]:
                (review_dir / filename).write_bytes(b"jpg")
            rows = [
                {
                    "snapshot_index": "0",
                    "filename": "camera1_a.jpg",
                    "target_elapsed_s": "30.000000",
                    "actual_clip_elapsed_s": "30.250000",
                    "target_error_s": "0.250000",
                    "video_time_s": "29.750000",
                    "frame_index": "42",
                    "host_utc_ns": "1",
                    "host_utc_iso": "1970-01-01T00:00:00.000Z",
                    "host_local_iso": "1970-01-01T00:00:00.000+00:00",
                    "host_monotonic_ns": "1",
                },
                {
                    "snapshot_index": "0",
                    "filename": "camera1_b.jpg",
                    "target_elapsed_s": "90.000000",
                    "actual_clip_elapsed_s": "90.250000",
                    "target_error_s": "0.250000",
                    "video_time_s": "89.750000",
                    "frame_index": "43",
                    "host_utc_ns": "2",
                    "host_utc_iso": "1970-01-01T00:00:00.000Z",
                    "host_local_iso": "1970-01-01T00:00:00.000+00:00",
                    "host_monotonic_ns": "2",
                },
            ]
            with (review_dir / "camera1_snapshots.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "snapshot_index",
                        "filename",
                        "target_elapsed_s",
                        "actual_clip_elapsed_s",
                        "target_error_s",
                        "video_time_s",
                        "frame_index",
                        "host_utc_ns",
                        "host_utc_iso",
                        "host_local_iso",
                        "host_monotonic_ns",
                    ],
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            metadata = {
                "label": "camera1",
                "success": True,
                "grab_failures": 0,
                "mp4_remux_succeeded": True,
                "review_snapshots": {
                    "enabled": True,
                    "operational": True,
                    "writer_started": True,
                    "writer_finalized": True,
                    "index_csv_written": True,
                    "directory": "review_snapshots",
                    "index_csv": "review_snapshots/camera1_snapshots.csv",
                    "requested_per_full_clip": 10,
                    "saved": 2,
                    "failed": 0,
                    "dropped_queue_full": 0,
                    "missed_due_acquisition_gap": 0,
                    "unreached_targets": 8,
                    "error": None,
                },
            }
            warnings = validate_session.validate_review_snapshots(
                clip_dir=clip_dir,
                camera_json_path=clip_dir / "camera1.json",
                camera_metadata=metadata,
            )

        self.assertTrue(any("snapshot index 0 is duplicated" in warning for warning in warnings), warnings)

    def test_validate_review_snapshots_detects_duplicate_frame_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip_dir = Path(tmp) / "clip_0000_120000+0000"
            clip_dir.mkdir()
            review_dir = clip_dir / "review_snapshots"
            review_dir.mkdir()
            for filename in ["camera1_a.jpg", "camera1_b.jpg"]:
                (review_dir / filename).write_bytes(b"jpg")

            with (review_dir / "camera1_snapshots.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "snapshot_index",
                        "filename",
                        "target_elapsed_s",
                        "actual_clip_elapsed_s",
                        "target_error_s",
                        "video_time_s",
                        "frame_index",
                        "host_utc_ns",
                        "host_utc_iso",
                        "host_local_iso",
                        "host_monotonic_ns",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "snapshot_index": "0",
                        "filename": "camera1_a.jpg",
                        "target_elapsed_s": "30.000000",
                        "actual_clip_elapsed_s": "30.250000",
                        "target_error_s": "0.250000",
                        "video_time_s": "29.750000",
                        "frame_index": "42",
                        "host_utc_ns": "1",
                        "host_utc_iso": "1970-01-01T00:00:00.000Z",
                        "host_local_iso": "1970-01-01T00:00:00.000+00:00",
                        "host_monotonic_ns": "1",
                    }
                )
                writer.writerow(
                    {
                        "snapshot_index": "1",
                        "filename": "camera1_b.jpg",
                        "target_elapsed_s": "90.000000",
                        "actual_clip_elapsed_s": "90.250000",
                        "target_error_s": "0.250000",
                        "video_time_s": "89.750000",
                        "frame_index": "42",
                        "host_utc_ns": "2",
                        "host_utc_iso": "1970-01-01T00:00:00.000Z",
                        "host_local_iso": "1970-01-01T00:00:00.000+00:00",
                        "host_monotonic_ns": "2",
                    }
                )

            metadata = json.loads(
                json.dumps(
                    {
                        "label": "camera1",
                        "success": True,
                        "grab_failures": 0,
                        "mp4_remux_succeeded": True,
                        "review_snapshots": {
                            "enabled": True,
                            "operational": True,
                            "writer_started": True,
                            "writer_finalized": True,
                            "index_csv_written": True,
                            "directory": "review_snapshots",
                            "index_csv": "review_snapshots/camera1_snapshots.csv",
                            "requested_per_full_clip": 10,
                            "saved": 2,
                            "failed": 0,
                            "dropped_queue_full": 0,
                            "missed_due_acquisition_gap": 0,
                            "unreached_targets": 8,
                            "error": None,
                        },
                    }
                )
            )
            (clip_dir / "camera1.json").write_text(json.dumps(metadata), encoding="utf-8")

            warnings = validate_session.validate_review_snapshots(
                clip_dir=clip_dir,
                camera_json_path=clip_dir / "camera1.json",
                camera_metadata=metadata,
            )

        self.assertTrue(any("frame index 42 is duplicated" in warning for warning in warnings), warnings)

    def test_validate_review_snapshots_detects_row_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = [self._make_row(index, 40 + index) for index in range(9)]
            session_dir = self._write_session_with_rows(
                tmp,
                rows,
                review_metadata_overrides={
                    "saved": 10,
                    "unreached_targets": 0,
                },
            )
            clip_dir = session_dir / "clip_0000_120000+0000"
            metadata = json.loads((clip_dir / "camera1.json").read_text(encoding="utf-8"))

            warnings = validate_session.validate_review_snapshots(
                clip_dir=clip_dir,
                camera_json_path=clip_dir / "camera1.json",
                camera_metadata=metadata,
            )

        self.assertTrue(
            any("metadata saved=10 but index contains 9 rows" in warning for warning in warnings),
            warnings,
        )

    def test_validate_review_snapshots_warns_when_index_csv_not_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = [self._make_row(index, 40 + index) for index in range(4)]
            session_dir = self._write_session_with_rows(
                tmp,
                rows,
                create_review_csv=False,
                review_metadata_overrides={
                    "writer_started": True,
                    "writer_finalized": True,
                    "index_csv_written": False,
                    "saved": 0,
                    "unreached_targets": 10,
                },
            )
            clip_dir = session_dir / "clip_0000_120000+0000"
            metadata = json.loads((clip_dir / "camera1.json").read_text(encoding="utf-8"))

            warnings = validate_session.validate_review_snapshots(
                clip_dir=clip_dir,
                camera_json_path=clip_dir / "camera1.json",
                camera_metadata=metadata,
            )

        self.assertTrue(
            any("review snapshot index CSV was not finalized" in warning for warning in warnings),
            warnings,
        )

    def test_validate_review_snapshots_warns_when_writer_did_not_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = [self._make_row(index, 40 + index) for index in range(2)]
            session_dir = self._write_session_with_rows(
                tmp,
                rows,
                review_metadata_overrides={
                    "writer_started": True,
                    "writer_finalized": False,
                    "index_csv_written": False,
                },
            )
            clip_dir = session_dir / "clip_0000_120000+0000"
            metadata = json.loads((clip_dir / "camera1.json").read_text(encoding="utf-8"))

            warnings = validate_session.validate_review_snapshots(
                clip_dir=clip_dir,
                camera_json_path=clip_dir / "camera1.json",
                camera_metadata=metadata,
            )

        self.assertTrue(
            any("review snapshot writer did not finalize safely" in warning for warning in warnings),
            warnings,
        )

    def test_validate_review_snapshots_warns_on_leftover_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = [self._make_row(index, 40 + index) for index in range(2)]
            session_dir = self._write_session_with_rows(
                tmp,
                rows,
                review_temp_files=[".camera1_review_snapshot_0000.jpg.tmp"],
            )
            clip_dir = session_dir / "clip_0000_120000+0000"
            metadata = json.loads((clip_dir / "camera1.json").read_text(encoding="utf-8"))

            warnings = validate_session.validate_review_snapshots(
                clip_dir=clip_dir,
                camera_json_path=clip_dir / "camera1.json",
                camera_metadata=metadata,
            )

        self.assertTrue(
            any("review snapshot temporary files remain" in warning for warning in warnings),
            warnings,
        )

    def test_validate_review_snapshots_accepts_partial_ctrl_c_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = [self._make_row(index, 40 + index) for index in range(4)]
            session_dir = self._write_session_with_rows(
                tmp,
                rows,
                review_metadata_overrides={
                    "saved": 4,
                    "failed": 0,
                    "dropped_queue_full": 0,
                    "missed_due_acquisition_gap": 0,
                    "unreached_targets": 6,
                },
            )
            clip_dir = session_dir / "clip_0000_120000+0000"
            metadata = json.loads((clip_dir / "camera1.json").read_text(encoding="utf-8"))

            warnings = validate_session.validate_review_snapshots(
                clip_dir=clip_dir,
                camera_json_path=clip_dir / "camera1.json",
                camera_metadata=metadata,
            )

        self.assertFalse(any("saved=4" in warning and "rows" in warning for warning in warnings), warnings)
        self.assertFalse(any("counts do not add up" in warning for warning in warnings), warnings)

    def test_validate_review_snapshots_writer_never_started_emits_auxiliary_warning_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "20260811_120000+0000"
            session_dir.mkdir()
            write_minimal_session(
                session_dir,
                review_rows=[],
                create_review_csv=False,
                review_metadata_overrides={
                    "operational": False,
                    "writer_started": False,
                    "writer_finalized": True,
                    "index_csv_written": False,
                    "saved": 0,
                    "unreached_targets": 10,
                },
            )
            clip_dir = session_dir / "clip_0000_120000+0000"
            metadata = json.loads((clip_dir / "camera1.json").read_text(encoding="utf-8"))

            warnings = validate_session.validate_review_snapshots(
                clip_dir=clip_dir,
                camera_json_path=clip_dir / "camera1.json",
                camera_metadata=metadata,
            )

        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn("review snapshot subsystem was requested but could not start", warnings[0])


if __name__ == "__main__":
    unittest.main()
