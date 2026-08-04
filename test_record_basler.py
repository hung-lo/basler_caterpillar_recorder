#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
