#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import tempfile
import threading
import unittest
from pathlib import Path, PureWindowsPath
from unittest import mock

import numpy as np

import record_basler


class ArchiveBackendTests(unittest.TestCase):
    def test_resolve_archive_backend_auto_uses_robocopy_on_windows(self) -> None:
        settings = dataclasses.replace(record_basler.ArchiveSettings(), backend="auto")
        with mock.patch.object(record_basler.os, "name", "nt"):
            with mock.patch.object(
                record_basler,
                "resolve_executable",
                return_value="C:/Windows/System32/robocopy.exe",
            ):
                backend, executable = record_basler.resolve_archive_backend(settings)
        self.assertEqual(backend, "robocopy")
        self.assertEqual(executable, "C:/Windows/System32/robocopy.exe")

    def test_resolve_archive_backend_auto_uses_rsync_off_windows(self) -> None:
        settings = dataclasses.replace(record_basler.ArchiveSettings(), backend="auto")
        with mock.patch.object(record_basler.os, "name", "posix"):
            with mock.patch.object(
                record_basler,
                "resolve_executable",
                return_value="/usr/bin/rsync",
            ):
                backend, executable = record_basler.resolve_archive_backend(settings)
        self.assertEqual(backend, "rsync")
        self.assertEqual(executable, "/usr/bin/rsync")

    def test_archive_copy_succeeded_accepts_robocopy_codes_below_8(self) -> None:
        self.assertTrue(
            record_basler.archive_copy_succeeded(
                record_basler.ArchiveCopyRun("robocopy", 7, "")
            )
        )
        self.assertFalse(
            record_basler.archive_copy_succeeded(
                record_basler.ArchiveCopyRun("robocopy", 8, "")
            )
        )
        self.assertTrue(
            record_basler.archive_copy_succeeded(
                record_basler.ArchiveCopyRun("rsync", 0, "")
            )
        )

    def test_path_is_within(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = root / "a" / "b"
            self.assertTrue(record_basler.path_is_within(child, root))
            self.assertFalse(record_basler.path_is_within(root, child))

    def test_verify_file_trees_reports_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "camera1.mp4").write_bytes(b"abc123")
            (source / "camera1.json").write_text("{}", encoding="utf-8")
            (destination / "camera1.mp4").write_bytes(b"abc123")
            (destination / "camera1.json").write_text("{}", encoding="utf-8")
            (destination / "unexpected.txt").write_text("x", encoding="utf-8")

            summary = record_basler.verify_file_trees(source, destination)

        self.assertFalse(summary.success)
        self.assertEqual(summary.error, "unexpected destination files: ['unexpected.txt']")

    def test_robocopy_guard_rejects_non_partial_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_session_dir = root / "local"
            local_session_dir.mkdir()
            archive_session_dir = root / "archive"
            incoming_dir = archive_session_dir / ".incoming"
            incoming_dir.mkdir(parents=True)
            settings = dataclasses.replace(record_basler.ArchiveSettings(), backend="robocopy")
            preflight = record_basler.ArchivePreflightResult(
                enabled=True,
                ok=True,
                errors=[],
                platform="nt",
                copy_backend="robocopy",
                copy_executable_path="robocopy",
                required_mount_point=str(root),
                required_mount_is_mount=True,
                destination_root=str(root),
                destination_created=True,
                destination_writable=True,
                local_free_gb=100.0,
                destination_free_gb=100.0,
                path_conflict=False,
                archive_session_dir=str(archive_session_dir),
            )
            manager = record_basler.ArchiveManager(
                settings=settings,
                local_session_dir=local_session_dir,
                project="project",
                subject="subject",
                session_name="session",
                archive_failure_event=threading.Event(),
                preflight=preflight,
            )

            with self.assertRaisesRegex(RuntimeError, "non-partial directory"):
                manager._ensure_robocopy_partial_destination(incoming_dir / "clip_0000")

    def test_robocopy_guard_rejects_destination_outside_incoming(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_session_dir = root / "local"
            local_session_dir.mkdir()
            archive_session_dir = root / "archive"
            incoming_dir = archive_session_dir / ".incoming"
            incoming_dir.mkdir(parents=True)
            settings = dataclasses.replace(record_basler.ArchiveSettings(), backend="robocopy")
            preflight = record_basler.ArchivePreflightResult(
                enabled=True,
                ok=True,
                errors=[],
                platform="nt",
                copy_backend="robocopy",
                copy_executable_path="robocopy",
                required_mount_point=str(root),
                required_mount_is_mount=True,
                destination_root=str(root),
                destination_created=True,
                destination_writable=True,
                local_free_gb=100.0,
                destination_free_gb=100.0,
                path_conflict=False,
                archive_session_dir=str(archive_session_dir),
            )
            manager = record_basler.ArchiveManager(
                settings=settings,
                local_session_dir=local_session_dir,
                project="project",
                subject="subject",
                session_name="session",
                archive_failure_event=threading.Event(),
                preflight=preflight,
            )

            with self.assertRaisesRegex(RuntimeError, "outside archive incoming directory"):
                manager._ensure_robocopy_partial_destination(
                    archive_session_dir / "clip_0000.partial"
                )


class JsonSerializationTests(unittest.TestCase):
    def test_windows_path_serialization(self) -> None:
        payload = {
            "destination_root": PureWindowsPath("D:/Hung_MBL"),
            "required_mount_point": PureWindowsPath("D:/"),
        }

        encoded = json.dumps(payload, default=record_basler.json_default)
        decoded = json.loads(encoded)

        self.assertEqual(
            PureWindowsPath(decoded["destination_root"]),
            PureWindowsPath("D:/Hung_MBL"),
        )
        self.assertEqual(
            PureWindowsPath(decoded["required_mount_point"]),
            PureWindowsPath("D:/"),
        )

    def test_datetime_serialization(self) -> None:
        value = dt.datetime(2026, 8, 4, 22, 0, 9, tzinfo=dt.timezone.utc)

        encoded = json.dumps({"time": value}, default=record_basler.json_default)
        decoded = json.loads(encoded)

        self.assertEqual(decoded["time"], value.isoformat())

    def test_unsupported_type_still_fails(self) -> None:
        class Unsupported:
            pass

        with self.assertRaises(TypeError):
            json.dumps({"value": Unsupported()}, default=record_basler.json_default)

    def test_write_json_handles_nested_windows_paths(self) -> None:
        payload = {
            "archive": {
                "destination_root": PureWindowsPath("D:/Hung_MBL"),
                "required_mount_point": PureWindowsPath("D:/"),
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session_manifest.json"

            record_basler.write_json(path, payload)

            with path.open("r", encoding="utf-8") as handle:
                decoded = json.load(handle)

            self.assertEqual(
                PureWindowsPath(decoded["archive"]["destination_root"]),
                PureWindowsPath("D:/Hung_MBL"),
            )
            self.assertFalse((path.parent / f".{path.name}.tmp").exists())


class PreviewResizeTests(unittest.TestCase):
    def test_portrait_frame_fits_width_and_height(self) -> None:
        frame = np.zeros((1920, 1200, 3), dtype=np.uint8)

        resized = record_basler.resize_to_fit(
            frame,
            max_width=600,
            max_height=640,
        )

        self.assertEqual(resized.shape[:2], (640, 400))

    def test_landscape_frame_fits_width(self) -> None:
        frame = np.zeros((1200, 1920, 3), dtype=np.uint8)

        resized = record_basler.resize_to_fit(
            frame,
            max_width=600,
            max_height=640,
        )

        self.assertEqual(resized.shape[:2], (375, 600))


class RecordingPreviewTests(unittest.TestCase):
    def test_card_panel_adds_panel_and_preserves_camera_frame(self) -> None:
        frame = np.zeros((375, 600, 3), dtype=np.uint8)
        packet = record_basler.PreviewPacket(
            label="camera1",
            clip_index=0,
            total_clips=3,
            frame_index=42,
            frame=frame,
            host_monotonic_ns=123456789,
            elapsed_s=12.5,
            planned_duration_s=60.0,
            session_elapsed_s=12.5,
            planned_session_duration_s=180.0,
            planned_finish_utc=dt.datetime(2026, 8, 5, 14, 0, tzinfo=dt.timezone.utc),
            measured_receive_fps=5.0,
        )

        preview = record_basler.draw_recording_preview(
            packet,
            record_basler.RecordingPreviewSettings(),
        )

        self.assertEqual(preview.shape, (417, 816, 3))
        np.testing.assert_array_equal(preview[:375, :600], frame)

    def test_legacy_overlay_keeps_original_shape(self) -> None:
        frame = np.zeros((120, 220, 3), dtype=np.uint8)
        packet = record_basler.PreviewPacket(
            label="camera1",
            clip_index=0,
            total_clips=1,
            frame_index=0,
            frame=frame,
            host_monotonic_ns=123456789,
            elapsed_s=0.0,
            planned_duration_s=10.0,
            session_elapsed_s=0.0,
            planned_session_duration_s=10.0,
            planned_finish_utc=dt.datetime(2026, 8, 5, 14, 0, tzinfo=dt.timezone.utc),
            measured_receive_fps=None,
        )

        preview = record_basler.draw_recording_preview(
            packet,
            record_basler.RecordingPreviewSettings(layout="legacy_overlay"),
        )

        self.assertEqual(preview.shape, frame.shape)
        self.assertFalse(np.array_equal(preview, frame))


class RecordingPlanTests(unittest.TestCase):
    def test_expected_clip_count_uses_number_of_clips(self) -> None:
        self.assertEqual(
            record_basler.expected_clip_count(
                interval_s=600,
                total_duration_s=None,
                number_of_clips=30,
            ),
            30,
        )

    def test_expected_clip_count_uses_total_duration(self) -> None:
        self.assertEqual(
            record_basler.expected_clip_count(
                interval_s=600,
                total_duration_s=5 * 3600,
                number_of_clips=None,
            ),
            30,
        )

    def test_expected_clip_count_uses_earlier_limit(self) -> None:
        self.assertEqual(
            record_basler.expected_clip_count(
                interval_s=600,
                total_duration_s=5 * 3600,
                number_of_clips=20,
            ),
            20,
        )

    def test_planned_session_span(self) -> None:
        self.assertEqual(
            record_basler.planned_session_span_s(
                clip_count=30,
                clip_duration_s=600,
                interval_s=600,
            ),
            18000.0,
        )

    def test_format_local_finish_time_same_day(self) -> None:
        now = dt.datetime(2026, 8, 5, 12, 0, tzinfo=dt.timezone.utc)
        finish = dt.datetime(2026, 8, 5, 17, 16, tzinfo=dt.timezone.utc)
        self.assertEqual(
            record_basler.format_local_finish_time(finish, now_utc=now),
            finish.astimezone().strftime("%H:%M"),
        )


if __name__ == "__main__":
    unittest.main()
