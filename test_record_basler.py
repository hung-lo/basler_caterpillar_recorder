#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import datetime as dt
import gzip
import json
import logging
import tempfile
import threading
import unittest
from pathlib import Path, PureWindowsPath
from unittest import mock

import numpy as np
import yaml

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
    exposure_us: float | None = None,
    auto_exposure: bool = False,
    auto_exposure_upper_us: float | None = None,
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
        exposure_us=exposure_us,
        auto_exposure=auto_exposure,
        auto_exposure_upper_us=auto_exposure_upper_us,
    )


class FakeNode:
    def __init__(
        self,
        name: str,
        value: object = None,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        increment: float | None = None,
        log: list[tuple[str, object]] | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self.minimum = minimum
        self.maximum = maximum
        self.increment = increment
        self.log = log

    def TrySetValue(self, value: object) -> bool:
        self.SetValue(value)
        return True

    def SetValue(self, value: object) -> None:
        self.value = value
        if self.log is not None:
            self.log.append((self.name, value))

    def GetValue(self) -> object:
        return self.value

    def GetMin(self) -> float:
        if self.minimum is None:
            raise AttributeError("no minimum")
        return self.minimum

    def GetMax(self) -> float:
        if self.maximum is None:
            raise AttributeError("no maximum")
        return self.maximum

    def GetInc(self) -> float:
        if self.increment is None:
            raise AttributeError("no increment")
        return self.increment


class FakeDeviceInfo:
    def GetModelName(self) -> str:
        return "a2A1920-160ucBAS"

    def GetSerialNumber(self) -> str:
        return "40604036"

    def GetUserDefinedName(self) -> str:
        return ""

    def GetDeviceClass(self) -> str:
        return "BaslerUsb"

    def GetFriendlyName(self) -> str:
        return "fake"

    def GetFullName(self) -> str:
        return "fake-full"


class FakeCamera:
    def __init__(self, nodes: dict[str, FakeNode]) -> None:
        self._device_info = FakeDeviceInfo()
        for name, node in nodes.items():
            setattr(self, name, node)

    def Open(self) -> None:
        return None

    def GetDeviceInfo(self) -> FakeDeviceInfo:
        return self._device_info


def make_fake_camera(modern: bool) -> tuple[FakeCamera, list[tuple[str, object]]]:
    log: list[tuple[str, object]] = []
    nodes = {
        "AcquisitionMode": FakeNode("AcquisitionMode", "SingleFrame", log=log),
        "ExposureMode": FakeNode("ExposureMode", "TriggerWidth", log=log),
        "TriggerSelector": FakeNode("TriggerSelector", "FrameStart", log=log),
        "TriggerMode": FakeNode("TriggerMode", "On", log=log),
        "Width": FakeNode("Width", 1920, minimum=16, maximum=1920, increment=1, log=log),
        "Height": FakeNode("Height", 1200, minimum=16, maximum=1200, increment=1, log=log),
        "OffsetX": FakeNode("OffsetX", 0, minimum=0, maximum=1919, increment=1, log=log),
        "OffsetY": FakeNode("OffsetY", 0, minimum=0, maximum=1199, increment=1, log=log),
        "PixelFormat": FakeNode("PixelFormat", "BayerRG8", log=log),
        "ExposureAuto": FakeNode("ExposureAuto", "Off", log=log),
        "ExposureTime": FakeNode("ExposureTime", 5000.0, minimum=50, maximum=500000, increment=1, log=log),
        "GainAuto": FakeNode("GainAuto", "Off", log=log),
        "Gain": FakeNode("Gain", 0.0, minimum=0, maximum=24, increment=1, log=log),
        "AcquisitionFrameRateEnable": FakeNode("AcquisitionFrameRateEnable", False, log=log),
        "AcquisitionFrameRate": FakeNode(
            "AcquisitionFrameRate",
            5.0,
            minimum=1,
            maximum=60,
            increment=0.01,
            log=log,
        ),
        "MaxNumBuffer": FakeNode("MaxNumBuffer", 20, minimum=1, maximum=100, increment=1, log=log),
    }
    if modern:
        nodes.update(
            {
                "AutoExposureTimeLowerLimit": FakeNode(
                    "AutoExposureTimeLowerLimit",
                    6000.0,
                    minimum=50,
                    maximum=500000,
                    increment=1,
                    log=log,
                ),
                "AutoExposureTimeUpperLimit": FakeNode(
                    "AutoExposureTimeUpperLimit",
                    180000.0,
                    minimum=50,
                    maximum=500000,
                    increment=1,
                    log=log,
                ),
                "AutoTargetBrightness": FakeNode(
                    "AutoTargetBrightness",
                    0.59,
                    minimum=0.0,
                    maximum=1.0,
                    increment=0.001,
                    log=log,
                ),
                "AutoFunctionROISelector": FakeNode("AutoFunctionROISelector", "ROI1", log=log),
                "AutoFunctionROIOffsetX": FakeNode("AutoFunctionROIOffsetX", 0, log=log),
                "AutoFunctionROIOffsetY": FakeNode("AutoFunctionROIOffsetY", 0, log=log),
                "AutoFunctionROIWidth": FakeNode("AutoFunctionROIWidth", 1920, log=log),
                "AutoFunctionROIHeight": FakeNode("AutoFunctionROIHeight", 1200, log=log),
                "AutoFunctionROIUseBrightness": FakeNode("AutoFunctionROIUseBrightness", False, log=log),
            }
        )
    else:
        nodes.update(
            {
                "AutoExposureTimeLowerLimitRaw": FakeNode(
                    "AutoExposureTimeLowerLimitRaw",
                    6000,
                    minimum=50,
                    maximum=500000,
                    increment=1,
                    log=log,
                ),
                "AutoExposureTimeUpperLimitRaw": FakeNode(
                    "AutoExposureTimeUpperLimitRaw",
                    180000,
                    minimum=50,
                    maximum=500000,
                    increment=1,
                    log=log,
                ),
                "AutoTargetValue": FakeNode(
                    "AutoTargetValue",
                    150,
                    minimum=0,
                    maximum=255,
                    increment=1,
                    log=log,
                ),
                "AutoFunctionAOISelector": FakeNode("AutoFunctionAOISelector", "AOI1", log=log),
                "AutoFunctionAOIOffsetX": FakeNode("AutoFunctionAOIOffsetX", 0, log=log),
                "AutoFunctionAOIOffsetY": FakeNode("AutoFunctionAOIOffsetY", 0, log=log),
                "AutoFunctionAOIWidth": FakeNode("AutoFunctionAOIWidth", 1920, log=log),
                "AutoFunctionAOIHeight": FakeNode("AutoFunctionAOIHeight", 1200, log=log),
                "AutoFunctionAOIUsageIntensity": FakeNode("AutoFunctionAOIUsageIntensity", False, log=log),
            }
        )
    return FakeCamera(nodes), log


def configure_fake_camera(camera_cfg: dict[str, object], *, modern: bool) -> tuple[record_basler.CameraBinding, list[tuple[str, object]]]:
    camera, log = make_fake_camera(modern)

    class FakeFactory:
        def CreateDevice(self, device_info: object) -> object:
            return device_info

    class FakeTlFactory:
        @staticmethod
        def GetInstance() -> FakeFactory:
            return FakeFactory()

    class FakeConverter:
        def __init__(self) -> None:
            self.OutputPixelFormat = None
            self.OutputBitAlignment = None

    fake_pylon = mock.Mock()
    fake_pylon.TlFactory = FakeTlFactory
    fake_pylon.InstantCamera = mock.Mock(return_value=camera)
    fake_pylon.ImageFormatConverter = FakeConverter
    fake_pylon.PixelType_BGR8packed = "bgr8"
    fake_pylon.OutputBitAlignment_MsbAligned = "msb"

    with mock.patch.object(record_basler, "pylon", fake_pylon):
        binding = record_basler.configure_camera(dict(camera_cfg), FakeDeviceInfo())
    return binding, log


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

    def test_valid_interrupted_clip_is_ready_for_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip_dir = Path(tmp) / "clip_0000_105302-0400"
            clip_dir.mkdir()
            metadata_path = clip_dir / "camera1.json"
            (clip_dir / "camera1.mp4").write_bytes(b"mp4")
            with gzip.open(
                clip_dir / "camera1.timestamps.csv.gz",
                "wt",
                encoding="utf-8",
                newline="",
            ) as handle:
                handle.write("frame_index,host_utc_ns,host_utc_iso,host_monotonic_ns,camera_timestamp,block_id,skipped_images\n")
                handle.write("0,1,1970-01-01T00:00:00.000Z,1,1,1,0\n")
            record_basler.write_json(
                metadata_path,
                {
                    "success": True,
                    "planned_clip_complete": False,
                    "interrupted_by_user": True,
                    "stop_reason": "user_interrupt",
                    "grab_failures": 0,
                    "mp4_remux_succeeded": True,
                },
            )

            ready, issues, _total_bytes = record_basler.clip_directory_ready_for_archive(
                clip_dir,
                expected_camera_count=1,
                max_clip_size_bytes=10_000_000,
            )

        self.assertTrue(ready)
        self.assertEqual(issues, [])

    def test_summarize_clip_results_treats_valid_user_interrupt_as_non_failure(self) -> None:
        results = [
            record_basler.ClipResult(
                label="camera1",
                success=True,
                planned_complete=False,
                interrupted_by_user=True,
                metadata_path=Path("camera1.json"),
                video_path=Path("camera1.mp4"),
                error=None,
            )
        ]

        summary = record_basler.summarize_clip_results(results, expected_camera_count=1)

        self.assertEqual(summary, (True, True, False, True))

    def test_summarize_clip_results_keeps_internal_stop_as_failure(self) -> None:
        results = [
            record_basler.ClipResult(
                label="camera1",
                success=False,
                planned_complete=False,
                interrupted_by_user=False,
                metadata_path=Path("camera1.json"),
                video_path=Path("camera1.mp4"),
                error="Recording stopped early by a non-user stop request",
            )
        ]

        summary = record_basler.summarize_clip_results(results, expected_camera_count=1)

        self.assertEqual(summary, (True, False, False, False))


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


class AutoExposureSettingsTests(unittest.TestCase):
    def test_invalid_auto_exposure_limits_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than auto_exposure_lower_us"):
            record_basler.parse_auto_exposure_settings(
                {
                    "fps": 5,
                    "auto_exposure_lower_us": 6000,
                    "auto_exposure_upper_us": 6000,
                    "auto_target_brightness": 0.59,
                    "exposure_us": 6000,
                },
                label="camera1",
            )

    def test_invalid_auto_exposure_target_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            record_basler.parse_auto_exposure_settings(
                {
                    "fps": 5,
                    "auto_exposure_lower_us": 6000,
                    "auto_exposure_upper_us": 180000,
                    "auto_target_brightness": 1.2,
                    "exposure_us": 30000,
                },
                label="camera1",
            )

    def test_upper_limit_at_frame_period_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "below the nominal frame period"):
            record_basler.parse_auto_exposure_settings(
                {
                    "fps": 5,
                    "auto_exposure_lower_us": 6000,
                    "auto_exposure_upper_us": 200000,
                    "auto_target_brightness": 0.59,
                    "exposure_us": 30000,
                },
                label="camera1",
            )

    def test_continuous_mode_maps_to_camera_enum(self) -> None:
        settings = record_basler.parse_auto_exposure_settings(
            {
                "fps": 5,
                "auto_exposure_mode": "continuous",
                "auto_exposure_lower_us": 6000,
                "auto_exposure_upper_us": 180000,
                "auto_target_brightness": 0.59,
                "exposure_us": 30000,
            },
            label="camera1",
        )
        self.assertEqual(settings.mode_value, "Continuous")


class PreviewExposureFormattingTests(unittest.TestCase):
    def test_formats_manual_exposure_ms(self) -> None:
        text, near_limit = record_basler.format_preview_exposure(
            6000,
            auto_exposure=False,
            upper_us=None,
        )
        self.assertEqual(text, "EXP 6.0 ms")
        self.assertFalse(near_limit)

    def test_formats_auto_exposure_ms(self) -> None:
        text, near_limit = record_basler.format_preview_exposure(
            63400,
            auto_exposure=True,
            upper_us=180000,
        )
        self.assertEqual(text, "AUTO EXP 63.4 ms")
        self.assertFalse(near_limit)

    def test_marks_max_at_upper_limit(self) -> None:
        text, near_limit = record_basler.format_preview_exposure(
            180000,
            auto_exposure=True,
            upper_us=180000,
        )
        self.assertEqual(text, "AUTO EXP 180.0 ms  MAX")
        self.assertTrue(near_limit)

    def test_marks_max_at_ninety_five_percent(self) -> None:
        text, near_limit = record_basler.format_preview_exposure(
            171000,
            auto_exposure=True,
            upper_us=180000,
        )
        self.assertEqual(text, "AUTO EXP 171.0 ms  MAX")
        self.assertTrue(near_limit)

    def test_does_not_mark_max_below_threshold(self) -> None:
        text, near_limit = record_basler.format_preview_exposure(
            160000,
            auto_exposure=True,
            upper_us=180000,
        )
        self.assertEqual(text, "AUTO EXP 160.0 ms")
        self.assertFalse(near_limit)

    def test_formats_missing_exposure_without_crashing(self) -> None:
        text, near_limit = record_basler.format_preview_exposure(
            None,
            auto_exposure=True,
            upper_us=180000,
        )
        self.assertEqual(text, "AUTO EXP --")
        self.assertFalse(near_limit)


class ConfigureCameraAutoExposureTests(unittest.TestCase):
    def test_modern_auto_exposure_nodes_are_configured_in_order(self) -> None:
        binding, log = configure_fake_camera(
            {
                "label": "camera1",
                "fps": 5,
                "width": 1920,
                "height": 1200,
                "offset_x": 0,
                "offset_y": 0,
                "pixel_format": "auto",
                "exposure_us": 30000,
                "gain": 0,
                "auto_exposure": True,
                "auto_exposure_mode": "continuous",
                "auto_exposure_lower_us": 6000,
                "auto_exposure_upper_us": 180000,
                "auto_target_brightness": 0.59,
                "auto_exposure_roi": "full",
                "auto_gain": False,
                "max_num_buffer": 30,
            },
            modern=True,
        )

        relevant = [
            item
            for item in log
            if item[0]
            in {
                "ExposureAuto",
                "GainAuto",
                "Gain",
                "ExposureTime",
                "AutoExposureTimeLowerLimit",
                "AutoExposureTimeUpperLimit",
                "AutoTargetBrightness",
                "AutoFunctionROISelector",
                "AutoFunctionROIOffsetX",
                "AutoFunctionROIOffsetY",
                "AutoFunctionROIWidth",
                "AutoFunctionROIHeight",
                "AutoFunctionROIUseBrightness",
            }
        ]
        self.assertEqual(
            relevant,
            [
                ("ExposureAuto", "Off"),
                ("GainAuto", "Off"),
                ("Gain", 0.0),
                ("ExposureTime", 30000.0),
                ("AutoExposureTimeLowerLimit", 6000.0),
                ("AutoExposureTimeUpperLimit", 180000.0),
                ("AutoTargetBrightness", 0.59),
                ("AutoFunctionROISelector", "ROI1"),
                ("AutoFunctionROIOffsetX", 0),
                ("AutoFunctionROIOffsetY", 0),
                ("AutoFunctionROIWidth", 1920),
                ("AutoFunctionROIHeight", 1200),
                ("AutoFunctionROIUseBrightness", True),
                ("ExposureAuto", "Continuous"),
            ],
        )
        self.assertEqual(binding.actual_settings["ExposureAuto"], "Continuous")
        self.assertEqual(binding.actual_settings["GainAuto"], "Off")
        self.assertEqual(binding.actual_settings["Gain"], 0.0)

    def test_legacy_auto_exposure_uses_raw_and_aoi_nodes(self) -> None:
        binding, log = configure_fake_camera(
            {
                "label": "camera1",
                "fps": 5,
                "width": 1920,
                "height": 1200,
                "offset_x": 0,
                "offset_y": 0,
                "pixel_format": "auto",
                "exposure_us": 30000,
                "gain": 0,
                "auto_exposure": True,
                "auto_exposure_mode": "continuous",
                "auto_exposure_lower_us": 6000,
                "auto_exposure_upper_us": 180000,
                "auto_target_brightness": 0.59,
                "auto_exposure_roi": "full",
                "auto_gain": False,
            },
            modern=False,
        )

        self.assertIn(("AutoExposureTimeLowerLimitRaw", 6000.0), log)
        self.assertIn(("AutoExposureTimeUpperLimitRaw", 180000.0), log)
        self.assertIn(("AutoTargetValue", 150), log)
        self.assertIn(("AutoFunctionAOISelector", "AOI1"), log)
        self.assertIn(("AutoFunctionAOIUsageIntensity", True), log)
        self.assertEqual(binding.actual_settings["ExposureAuto"], "Continuous")

    def test_manual_exposure_behavior_is_unchanged(self) -> None:
        binding, log = configure_fake_camera(
            {
                "label": "camera1",
                "fps": 5,
                "width": 1920,
                "height": 1200,
                "offset_x": 0,
                "offset_y": 0,
                "pixel_format": "auto",
                "exposure_us": 30000,
                "gain": 0,
                "auto_exposure": False,
                "auto_gain": False,
            },
            modern=True,
        )

        relevant = [item for item in log if item[0] in {"ExposureAuto", "ExposureTime", "GainAuto", "Gain"}]
        self.assertEqual(
            relevant,
            [
                ("ExposureAuto", "Off"),
                ("GainAuto", "Off"),
                ("Gain", 0.0),
                ("ExposureTime", 30000.0),
            ],
        )
        self.assertNotIn(("AutoExposureTimeLowerLimit", 6000.0), log)
        self.assertEqual(binding.actual_settings["ExposureAuto"], "Off")
        self.assertEqual(binding.actual_settings["GainAuto"], "Off")


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

    def test_card_panel_handles_missing_exposure_without_error(self) -> None:
        frame = np.zeros((720, 450, 3), dtype=np.uint8)
        packet = make_preview_packet(
            frame,
            exposure_us=None,
            auto_exposure=True,
            auto_exposure_upper_us=180000.0,
        )

        preview = record_basler.draw_recording_preview(packet, record_basler.RecordingPreviewSettings())
        np.testing.assert_array_equal(preview[: frame.shape[0], : frame.shape[1]], frame)


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


class ConfigTemplateConsistencyTests(unittest.TestCase):
    def test_all_tracked_config_templates_use_standard_auto_exposure(self) -> None:
        root = Path(__file__).resolve().parent
        config_paths = sorted(
            path for path in root.glob("config_*.yaml") if not path.name.startswith("config_local")
        )

        self.assertTrue(config_paths, "expected at least one config_*.yaml template")

        for config_path in config_paths:
            with config_path.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)

            self.assertIsInstance(config, dict, f"{config_path.name}: YAML root must be a mapping")
            cameras = config.get("cameras")
            self.assertIsInstance(cameras, list, f"{config_path.name}: cameras must be a list")

            for camera in cameras:
                self.assertIsInstance(camera, dict, f"{config_path.name}: each camera must be a mapping")
                label = str(camera.get("label") or "").strip() or "<missing label>"
                prefix = f"{config_path.name} {label}:"

                self.assertIs(
                    camera.get("auto_exposure"),
                    True,
                    f"{prefix} auto_exposure must be true",
                )
                self.assertEqual(
                    camera.get("auto_exposure_mode"),
                    "continuous",
                    f"{prefix} auto_exposure_mode must be continuous",
                )
                self.assertEqual(
                    float(camera.get("auto_exposure_lower_us")),
                    6000.0,
                    f"{prefix} auto_exposure_lower_us must be 6000",
                )
                self.assertEqual(
                    float(camera.get("auto_exposure_upper_us")),
                    180000.0,
                    f"{prefix} auto_exposure_upper_us must be 180000",
                )
                self.assertEqual(
                    float(camera.get("auto_target_brightness")),
                    0.70,
                    f"{prefix} auto_target_brightness must be 0.70",
                )
                self.assertEqual(
                    camera.get("auto_exposure_roi"),
                    "full",
                    f"{prefix} auto_exposure_roi must be full",
                )
                self.assertEqual(
                    float(camera.get("gain")),
                    0.0,
                    f"{prefix} gain must be 0",
                )
                self.assertIs(
                    camera.get("auto_gain"),
                    False,
                    f"{prefix} auto_gain must be false",
                )

                lower = float(camera["auto_exposure_lower_us"])
                upper = float(camera["auto_exposure_upper_us"])
                seed = float(camera["exposure_us"])
                fps = float(camera["fps"])

                self.assertLessEqual(
                    lower,
                    seed,
                    f"{prefix} exposure_us must be >= auto_exposure_lower_us",
                )
                self.assertLessEqual(
                    seed,
                    upper,
                    f"{prefix} exposure_us must be <= auto_exposure_upper_us",
                )
                self.assertLess(
                    upper,
                    1_000_000.0 / fps,
                    f"{prefix} auto_exposure_upper_us must stay below the frame period",
                )


if __name__ == "__main__":
    unittest.main()
