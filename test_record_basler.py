#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import tempfile
import threading
import unittest
from pathlib import Path, PureWindowsPath
from unittest import mock

import numpy as np

import record_basler
import validate_session


def make_preview_packet(
    frame: np.ndarray,
    *,
    label: str = "camera1",
    clip_index: int = 0,
    total_clips: int = 3,
    frame_index: int = 42,
    elapsed_s: float = 12.5,
    planned_duration_s: float = 60.0,
    session_elapsed_s: float = 12.5,
    planned_session_duration_s: float = 180.0,
    planned_finish_utc: dt.datetime | None = None,
    measured_receive_fps: float | None = 5.0,
) -> record_basler.PreviewPacket:
    return record_basler.PreviewPacket(
        label=label,
        clip_index=clip_index,
        total_clips=total_clips,
        frame_index=frame_index,
        frame=frame,
        host_monotonic_ns=123456789,
        elapsed_s=elapsed_s,
        planned_duration_s=planned_duration_s,
        session_elapsed_s=session_elapsed_s,
        planned_session_duration_s=planned_session_duration_s,
        planned_finish_utc=planned_finish_utc
        or dt.datetime(2026, 8, 5, 14, 0, tzinfo=dt.timezone.utc),
        measured_receive_fps=measured_receive_fps,
    )


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


class TimeFormattingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixed_eastern = dt.timezone(dt.timedelta(hours=-4))
        self.utc_value = dt.datetime(2026, 8, 4, 14, 53, 1, 254000, tzinfo=dt.timezone.utc)

    def test_isoformat_utc_uses_trailing_z(self) -> None:
        self.assertEqual(
            record_basler.isoformat_utc(self.utc_value),
            "2026-08-04T14:53:01.254Z",
        )

    def test_isoformat_local_uses_numeric_offset(self) -> None:
        with mock.patch.object(
            record_basler,
            "to_local",
            side_effect=lambda value: value.astimezone(self.fixed_eastern),
        ):
            self.assertEqual(
                record_basler.isoformat_local(self.utc_value),
                "2026-08-04T10:53:01.254-04:00",
            )

    def test_filename_local_timestamp_is_filename_safe(self) -> None:
        with mock.patch.object(
            record_basler,
            "to_local",
            side_effect=lambda value: value.astimezone(self.fixed_eastern),
        ):
            self.assertEqual(
                record_basler.filename_local_timestamp(self.utc_value),
                "20260804_105301-0400",
            )

    def test_timestamp_pair_round_trips_same_instant(self) -> None:
        with mock.patch.object(
            record_basler,
            "to_local",
            side_effect=lambda value: value.astimezone(self.fixed_eastern),
        ):
            pair = record_basler.timestamp_pair(self.utc_value)

        self.assertEqual(pair["utc"], "2026-08-04T14:53:01.254Z")
        self.assertEqual(pair["local"], "2026-08-04T10:53:01.254-04:00")
        self.assertEqual(
            validate_session.parse_iso_datetime(pair["utc"]).astimezone(dt.timezone.utc),
            validate_session.parse_iso_datetime(pair["local"]).astimezone(dt.timezone.utc),
        )

    def test_local_iso_formatter_uses_numeric_offset(self) -> None:
        formatter = record_basler.LocalIsoFormatter("%(asctime)s | %(levelname)s | %(message)s")
        record = logging.LogRecord(
            name="basler_recorder",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        record.created = self.utc_value.timestamp()
        with mock.patch.object(
            record_basler,
            "to_local",
            side_effect=lambda value: value.astimezone(self.fixed_eastern),
        ):
            self.assertEqual(
                formatter.formatTime(record),
                "2026-08-04T10:53:01.254-04:00",
            )


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


class SetupPreviewFpsTests(unittest.TestCase):
    def test_setup_preview_fps_uses_a_window_and_recovers_after_pause(self) -> None:
        displayed_fps, start, frames = record_basler.update_setup_preview_fps(
            now=100.0,
            fps_window_start=None,
            fps_window_frames=0,
            displayed_fps=None,
        )
        self.assertIsNone(displayed_fps)
        self.assertEqual(start, 100.0)
        self.assertEqual(frames, 1)

        displayed_fps, start, frames = record_basler.update_setup_preview_fps(
            now=100.4,
            fps_window_start=start,
            fps_window_frames=frames,
            displayed_fps=displayed_fps,
        )
        self.assertIsNone(displayed_fps)
        self.assertEqual(start, 100.0)
        self.assertEqual(frames, 2)

        displayed_fps, start, frames = record_basler.update_setup_preview_fps(
            now=101.2,
            fps_window_start=start,
            fps_window_frames=frames,
            displayed_fps=displayed_fps,
        )
        self.assertAlmostEqual(displayed_fps or 0.0, 1.7, places=1)
        self.assertEqual(start, 101.2)
        self.assertEqual(frames, 1)

        displayed_fps, start, frames = record_basler.update_setup_preview_fps(
            now=107.0,
            fps_window_start=start,
            fps_window_frames=frames,
            displayed_fps=displayed_fps,
        )
        self.assertIsNone(displayed_fps)
        self.assertEqual(start, 107.0)
        self.assertEqual(frames, 1)


class RecordingPreviewSettingsTests(unittest.TestCase):
    def test_default_layout_is_card_panel(self) -> None:
        settings = record_basler.parse_recording_preview_settings({})
        self.assertEqual(settings.layout, "card_panel")

    def test_legacy_layout_is_accepted(self) -> None:
        settings = record_basler.parse_recording_preview_settings(
            {"recording_preview": {"layout": "legacy_overlay"}}
        )
        self.assertEqual(settings.layout, "legacy_overlay")

    def test_layout_is_case_normalized(self) -> None:
        settings = record_basler.parse_recording_preview_settings(
            {"recording_preview": {"layout": "CARD_PANEL"}}
        )
        self.assertEqual(settings.layout, "card_panel")

    def test_invalid_layout_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "recording_preview.layout"):
            record_basler.parse_recording_preview_settings(
                {"recording_preview": {"layout": "side_panel"}}
            )


class RecordingPreviewTests(unittest.TestCase):
    def test_card_panel_layout_stays_out_of_footer_and_minimal_mode_fits_short_frame(self) -> None:
        layout = record_basler._calculate_card_panel_layout(260, 180, 42)

        self.assertEqual(layout.footer_y, 180)
        self.assertEqual(layout.mode, "minimal")
        self.assertLessEqual(layout.recording_card[1] + layout.recording_card[3], 180 - 12)
        self.assertLessEqual(layout.session_card[1] + layout.session_card[3], 180 - 12)

    def test_full_mode_selected_for_tall_portrait_frame(self) -> None:
        layout = record_basler._calculate_card_panel_layout(450, 720, 42)
        self.assertEqual(layout.mode, "full")

    def test_compact_mode_selected_for_medium_frame(self) -> None:
        layout = record_basler._calculate_card_panel_layout(450, 360, 42)
        self.assertEqual(layout.mode, "compact")

    def test_minimal_mode_selected_for_small_frame(self) -> None:
        layout = record_basler._calculate_card_panel_layout(260, 180, 42)
        self.assertEqual(layout.mode, "minimal")

    def test_recording_card_and_session_card_bounds_are_inside_layout(self) -> None:
        for image_h, image_w in [(720, 450), (375, 600), (180, 260)]:
            layout = record_basler._calculate_card_panel_layout(image_w, image_h, 42)
            recording_x, recording_y, recording_w, recording_h = layout.recording_card
            session_x, session_y, session_w, session_h = layout.session_card

            self.assertEqual(recording_x, image_w + 12)
            self.assertEqual(session_x, image_w + 12)
            self.assertLessEqual(recording_y + recording_h, layout.footer_y - 12)
            self.assertLessEqual(session_y + session_h, layout.footer_y - 12)
            self.assertLess(recording_y + recording_h, layout.footer_y)
            self.assertLess(session_y + session_h, layout.footer_y)

            recording_content_bottom = recording_y + record_basler._measure_recording_card_height(
                recording_w,
                mode=layout.recording_mode,
            )
            session_content_bottom = session_y + record_basler._measure_session_card_height(
                session_w,
                mode=layout.mode,
            )
            self.assertLessEqual(recording_content_bottom, recording_y + recording_h)
            self.assertLessEqual(session_content_bottom, session_y + session_h)
            self.assertLessEqual(session_content_bottom, layout.footer_y - 12)

    def test_card_panel_adds_panel_and_preserves_camera_frame(self) -> None:
        frame = np.zeros((720, 450, 3), dtype=np.uint8)
        original = frame.copy()
        packet = make_preview_packet(frame)

        preview = record_basler.draw_recording_preview(
            packet,
            record_basler.RecordingPreviewSettings(),
        )

        self.assertGreater(preview.shape[0], frame.shape[0])
        self.assertGreater(preview.shape[1], frame.shape[1])
        np.testing.assert_array_equal(preview[: frame.shape[0], : frame.shape[1]], frame)
        np.testing.assert_array_equal(frame, original)

    def test_card_panel_keeps_landscape_source_unchanged(self) -> None:
        frame = np.zeros((375, 600, 3), dtype=np.uint8)
        original = frame.copy()
        packet = make_preview_packet(frame, label="arena_B_M05-M07")

        preview = record_basler.draw_recording_preview(packet, record_basler.RecordingPreviewSettings())

        self.assertGreater(preview.shape[0], frame.shape[0])
        self.assertGreater(preview.shape[1], frame.shape[1])
        np.testing.assert_array_equal(preview[: frame.shape[0], : frame.shape[1]], frame)
        np.testing.assert_array_equal(frame, original)

    def test_card_panel_handles_small_frame_without_error(self) -> None:
        frame = np.zeros((180, 260, 3), dtype=np.uint8)
        packet = make_preview_packet(
            frame,
            label="arena_B_M05-M07",
            elapsed_s=-5.0,
            planned_duration_s=10.0,
            session_elapsed_s=-2.0,
            planned_session_duration_s=20.0,
            measured_receive_fps=None,
        )

        preview = record_basler.draw_recording_preview(packet, record_basler.RecordingPreviewSettings())
        self.assertGreater(preview.shape[0], frame.shape[0])
        self.assertGreater(preview.shape[1], frame.shape[1])
        np.testing.assert_array_equal(preview[: frame.shape[0], : frame.shape[1]], frame)

    def test_card_panel_handles_very_short_frame_without_error(self) -> None:
        frame = np.zeros((150, 220, 3), dtype=np.uint8)
        layout = record_basler._calculate_card_panel_layout(frame.shape[1], frame.shape[0], 38)
        self.assertEqual(layout.mode, "minimal")
        packet = make_preview_packet(frame, label="very_long_camera_label_for_testing_overflow")

        preview = record_basler.draw_recording_preview(packet, record_basler.RecordingPreviewSettings())
        np.testing.assert_array_equal(preview[: frame.shape[0], : frame.shape[1]], frame)

    def test_show_status_false_returns_original_dimensions(self) -> None:
        frame = np.zeros((260, 450, 3), dtype=np.uint8)
        packet = make_preview_packet(frame)

        preview = record_basler.draw_recording_preview(
            packet,
            record_basler.RecordingPreviewSettings(show_status=False),
        )

        self.assertEqual(preview.shape, frame.shape)
        np.testing.assert_array_equal(preview, frame)

    def test_legacy_overlay_keeps_original_shape(self) -> None:
        frame = np.zeros((120, 220, 3), dtype=np.uint8)
        packet = make_preview_packet(frame, measured_receive_fps=None, planned_duration_s=10.0, session_elapsed_s=0.0, planned_session_duration_s=10.0)

        preview = record_basler.draw_recording_preview(
            packet,
            record_basler.RecordingPreviewSettings(layout="legacy_overlay"),
        )

        self.assertEqual(preview.shape, frame.shape)
        self.assertFalse(np.array_equal(preview, frame))

    def test_negative_and_overrun_progress_render_without_error(self) -> None:
        frame = np.zeros((375, 600, 3), dtype=np.uint8)
        cases = [
            make_preview_packet(frame, elapsed_s=-2.0, session_elapsed_s=-4.0, measured_receive_fps=None),
            make_preview_packet(frame, elapsed_s=999.0, planned_duration_s=12.0, session_elapsed_s=999.0, planned_session_duration_s=12.0),
        ]

        for packet in cases:
            preview = record_basler.draw_recording_preview(packet, record_basler.RecordingPreviewSettings())
            np.testing.assert_array_equal(preview[: frame.shape[0], : frame.shape[1]], frame)

    def test_long_camera_label_renders_without_error(self) -> None:
        frame = np.zeros((375, 600, 3), dtype=np.uint8)
        packet = make_preview_packet(frame, label="arena_B_M05-M07_very_long_camera_label_that_should_be_ellipsized")

        preview = record_basler.draw_recording_preview(packet, record_basler.RecordingPreviewSettings())
        np.testing.assert_array_equal(preview[: frame.shape[0], : frame.shape[1]], frame)
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


class ValidatorTimestampTests(unittest.TestCase):
    def test_validator_accepts_legacy_and_local_session_names(self) -> None:
        self.assertEqual(
            validate_session.detect_session_timestamp_naming("20260804_145301"),
            "legacy",
        )
        self.assertEqual(
            validate_session.detect_session_timestamp_naming("20260804_105301-0400"),
            "local_with_offset",
        )

    def test_validator_accepts_legacy_and_local_clip_names(self) -> None:
        self.assertEqual(
            validate_session.detect_clip_timestamp_naming("clip_0000_145302"),
            "legacy",
        )
        self.assertEqual(
            validate_session.detect_clip_timestamp_naming("clip_0000_105302-0400"),
            "local_with_offset",
        )

    def test_validator_rejects_invalid_timestamp_names(self) -> None:
        self.assertIsNone(
            validate_session.detect_session_timestamp_naming("20260804_105301_EDT")
        )
        self.assertIsNone(
            validate_session.detect_clip_timestamp_naming("clip_0000_10:53:02-0400")
        )

    def test_parse_iso_datetime_requires_timezone(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone offset"):
            validate_session.parse_iso_datetime("2026-08-04T10:53:01.254")


if __name__ == "__main__":
    unittest.main()
