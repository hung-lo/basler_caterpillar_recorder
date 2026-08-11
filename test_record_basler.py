#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import datetime as dt
import gzip
import json
import logging
import queue
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
        camera: "FakeCamera" | None = None,
        selector_family: str | None = None,
        selector_states: dict[str, object] | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self.minimum = minimum
        self.maximum = maximum
        self.increment = increment
        self.log = log
        self.camera = camera
        self.selector_family = selector_family
        self.selector_states = selector_states

    def TrySetValue(self, value: object) -> bool:
        self.SetValue(value)
        return True

    def SetValue(self, value: object) -> None:
        if self.name == "AutoFunctionROISelector" and self.camera is not None:
            self.camera.current_roi_selector = str(value)
        elif self.name == "AutoFunctionAOISelector" and self.camera is not None:
            self.camera.current_aoi_selector = str(value)

        if self.selector_states is not None and self.camera is not None:
            if self.selector_family == "ROI":
                selector = self.camera.current_roi_selector
            elif self.selector_family == "AOI":
                selector = self.camera.current_aoi_selector
            else:
                selector = None
            if selector is not None:
                self.selector_states[selector] = value
            else:
                self.value = value
        else:
            self.value = value
        if self.log is not None:
            self.log.append((self.name, value))

    def GetValue(self) -> object:
        if self.selector_states is not None and self.camera is not None:
            if self.selector_family == "ROI":
                selector = self.camera.current_roi_selector
            elif self.selector_family == "AOI":
                selector = self.camera.current_aoi_selector
            else:
                selector = None
            if selector is not None and selector in self.selector_states:
                return self.selector_states[selector]
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
    def __init__(self) -> None:
        self._device_info = FakeDeviceInfo()
        self.current_roi_selector = "ROI1"
        self.current_aoi_selector = "AOI1"

    def install_nodes(self, nodes: dict[str, FakeNode]) -> None:
        for name, node in nodes.items():
            node.camera = self
            setattr(self, name, node)

    def Open(self) -> None:
        return None

    def GetDeviceInfo(self) -> FakeDeviceInfo:
        return self._device_info


def make_fake_camera(
    modern: bool,
    *,
    include_white_balance_usage: bool = True,
    roi_usage_state: dict[str, dict[str, object]] | None = None,
    aoi_usage_state: dict[str, dict[str, object]] | None = None,
) -> tuple[FakeCamera, list[tuple[str, object]]]:
    log: list[tuple[str, object]] = []
    camera = FakeCamera()
    roi_usage_state = roi_usage_state or {}
    aoi_usage_state = aoi_usage_state or {}
    roi_brightness_states = {"ROI1": False, "ROI2": False}
    roi_brightness_states.update({str(key): value for key, value in roi_usage_state.get("brightness", {}).items()})
    roi_white_balance_states = {"ROI1": False, "ROI2": False}
    roi_white_balance_states.update(
        {str(key): value for key, value in roi_usage_state.get("white_balance", {}).items()}
    )
    aoi_intensity_states = {"AOI1": False, "AOI2": False}
    aoi_intensity_states.update({str(key): value for key, value in aoi_usage_state.get("brightness", {}).items()})
    aoi_white_balance_states = {"AOI1": False, "AOI2": False}
    aoi_white_balance_states.update(
        {str(key): value for key, value in aoi_usage_state.get("white_balance", {}).items()}
    )
    nodes = {
        "AcquisitionMode": FakeNode("AcquisitionMode", "SingleFrame", log=log, camera=camera),
        "ExposureMode": FakeNode("ExposureMode", "TriggerWidth", log=log, camera=camera),
        "TriggerSelector": FakeNode("TriggerSelector", "FrameStart", log=log, camera=camera),
        "TriggerMode": FakeNode("TriggerMode", "On", log=log, camera=camera),
        "Width": FakeNode("Width", 1920, minimum=16, maximum=1920, increment=1, log=log, camera=camera),
        "Height": FakeNode("Height", 1200, minimum=16, maximum=1200, increment=1, log=log, camera=camera),
        "OffsetX": FakeNode("OffsetX", 0, minimum=0, maximum=1919, increment=1, log=log, camera=camera),
        "OffsetY": FakeNode("OffsetY", 0, minimum=0, maximum=1199, increment=1, log=log, camera=camera),
        "PixelFormat": FakeNode("PixelFormat", "BayerRG8", log=log, camera=camera),
        "ExposureAuto": FakeNode("ExposureAuto", "Off", log=log, camera=camera),
        "ExposureTime": FakeNode("ExposureTime", 5000.0, minimum=50, maximum=500000, increment=1, log=log, camera=camera),
        "GainAuto": FakeNode("GainAuto", "Off", log=log, camera=camera),
        "Gain": FakeNode("Gain", 0.0, minimum=0, maximum=24, increment=1, log=log, camera=camera),
        "BalanceWhiteAuto": FakeNode("BalanceWhiteAuto", "Off", log=log, camera=camera),
        "AcquisitionFrameRateEnable": FakeNode("AcquisitionFrameRateEnable", False, log=log, camera=camera),
        "AcquisitionFrameRate": FakeNode(
            "AcquisitionFrameRate",
            5.0,
            minimum=1,
            maximum=60,
            increment=0.01,
            log=log,
            camera=camera,
        ),
        "MaxNumBuffer": FakeNode("MaxNumBuffer", 20, minimum=1, maximum=100, increment=1, log=log, camera=camera),
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
                    camera=camera,
                ),
                "AutoExposureTimeUpperLimit": FakeNode(
                    "AutoExposureTimeUpperLimit",
                    180000.0,
                    minimum=50,
                    maximum=500000,
                    increment=1,
                    log=log,
                    camera=camera,
                ),
                "AutoTargetBrightness": FakeNode(
                    "AutoTargetBrightness",
                    0.59,
                    minimum=0.0,
                    maximum=1.0,
                    increment=0.001,
                    log=log,
                    camera=camera,
                ),
                "AutoFunctionROISelector": FakeNode("AutoFunctionROISelector", "ROI1", log=log, camera=camera),
                "AutoFunctionROIOffsetX": FakeNode("AutoFunctionROIOffsetX", 0, log=log, camera=camera),
                "AutoFunctionROIOffsetY": FakeNode("AutoFunctionROIOffsetY", 0, log=log, camera=camera),
                "AutoFunctionROIWidth": FakeNode("AutoFunctionROIWidth", 1920, log=log, camera=camera),
                "AutoFunctionROIHeight": FakeNode("AutoFunctionROIHeight", 1200, log=log, camera=camera),
                "AutoFunctionROIUseBrightness": FakeNode(
                    "AutoFunctionROIUseBrightness",
                    roi_brightness_states["ROI1"],
                    log=log,
                    camera=camera,
                    selector_family="ROI",
                    selector_states=roi_brightness_states,
                ),
                **(
                    {
                        "AutoFunctionROIUseWhiteBalance": FakeNode(
                            "AutoFunctionROIUseWhiteBalance",
                            roi_white_balance_states["ROI1"],
                            log=log,
                            camera=camera,
                            selector_family="ROI",
                            selector_states=roi_white_balance_states,
                        )
                    }
                    if include_white_balance_usage
                    else {}
                ),
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
                    camera=camera,
                ),
                "AutoExposureTimeUpperLimitRaw": FakeNode(
                    "AutoExposureTimeUpperLimitRaw",
                    180000,
                    minimum=50,
                    maximum=500000,
                    increment=1,
                    log=log,
                    camera=camera,
                ),
                "AutoTargetValue": FakeNode(
                    "AutoTargetValue",
                    150,
                    minimum=0,
                    maximum=255,
                    increment=1,
                    log=log,
                    camera=camera,
                ),
                "AutoFunctionAOISelector": FakeNode("AutoFunctionAOISelector", "AOI1", log=log, camera=camera),
                "AutoFunctionAOIOffsetX": FakeNode("AutoFunctionAOIOffsetX", 0, log=log, camera=camera),
                "AutoFunctionAOIOffsetY": FakeNode("AutoFunctionAOIOffsetY", 0, log=log, camera=camera),
                "AutoFunctionAOIWidth": FakeNode("AutoFunctionAOIWidth", 1920, log=log, camera=camera),
                "AutoFunctionAOIHeight": FakeNode("AutoFunctionAOIHeight", 1200, log=log, camera=camera),
                "AutoFunctionAOIUsageIntensity": FakeNode(
                    "AutoFunctionAOIUsageIntensity",
                    aoi_intensity_states["AOI1"],
                    log=log,
                    camera=camera,
                    selector_family="AOI",
                    selector_states=aoi_intensity_states,
                ),
                **(
                    {
                        "AutoFunctionAOIUsageWhiteBalance": FakeNode(
                            "AutoFunctionAOIUsageWhiteBalance",
                            aoi_white_balance_states["AOI1"],
                            log=log,
                            camera=camera,
                            selector_family="AOI",
                            selector_states=aoi_white_balance_states,
                        )
                    }
                    if include_white_balance_usage
                    else {}
                ),
            }
        )
    camera.install_nodes(nodes)
    return camera, log


def configure_fake_camera(
    camera_cfg: dict[str, object],
    *,
    modern: bool,
    include_white_balance_usage: bool = True,
    roi_usage_state: dict[str, dict[str, object]] | None = None,
    aoi_usage_state: dict[str, dict[str, object]] | None = None,
) -> tuple[record_basler.CameraBinding, list[tuple[str, object]]]:
    camera, log = make_fake_camera(
        modern,
        include_white_balance_usage=include_white_balance_usage,
        roi_usage_state=roi_usage_state,
        aoi_usage_state=aoi_usage_state,
    )

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

    def test_clip_directory_ready_for_archive_allows_complete_review_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip_dir = Path(tmp)
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
                    "planned_clip_complete": True,
                    "interrupted_by_user": False,
                    "stop_reason": "planned_end",
                    "grab_failures": 0,
                    "mp4_remux_succeeded": True,
                    "review_snapshots": {
                        "enabled": True,
                        "operational": True,
                        "writer_started": True,
                        "writer_finalized": True,
                        "index_csv_written": True,
                        "directory": "review_snapshots",
                        "saved": 10,
                        "failed": 0,
                    },
                },
            )

            ready, issues, _total_bytes = record_basler.clip_directory_ready_for_archive(
                clip_dir,
                expected_camera_count=1,
                max_clip_size_bytes=10_000_000,
            )

        self.assertTrue(ready)
        self.assertEqual(issues, [])

    def test_clip_directory_ready_for_archive_allows_review_snapshots_with_one_failed_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip_dir = Path(tmp)
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
                    "planned_clip_complete": True,
                    "interrupted_by_user": False,
                    "stop_reason": "planned_end",
                    "grab_failures": 0,
                    "mp4_remux_succeeded": True,
                    "review_snapshots": {
                        "enabled": True,
                        "operational": True,
                        "writer_started": True,
                        "writer_finalized": True,
                        "index_csv_written": True,
                        "directory": "review_snapshots",
                        "saved": 9,
                        "failed": 1,
                    },
                },
            )

            ready, issues, _total_bytes = record_basler.clip_directory_ready_for_archive(
                clip_dir,
                expected_camera_count=1,
                max_clip_size_bytes=10_000_000,
            )

        self.assertTrue(ready)
        self.assertEqual(issues, [])

    def test_clip_directory_ready_for_archive_rejects_unfinalized_review_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip_dir = Path(tmp)
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
            review_dir = clip_dir / "review_snapshots"
            review_dir.mkdir()
            (review_dir / ".camera1_review_snapshot_0000.jpg.tmp").write_bytes(b"tmp")
            record_basler.write_json(
                metadata_path,
                {
                    "success": True,
                    "planned_clip_complete": True,
                    "interrupted_by_user": False,
                    "stop_reason": "planned_end",
                    "grab_failures": 0,
                    "mp4_remux_succeeded": True,
                    "review_snapshots": {
                        "enabled": True,
                        "operational": True,
                        "writer_started": True,
                        "writer_finalized": False,
                        "index_csv_written": False,
                        "directory": "review_snapshots",
                    },
                },
            )

            ready, issues, _total_bytes = record_basler.clip_directory_ready_for_archive(
                clip_dir,
                expected_camera_count=1,
                max_clip_size_bytes=10_000_000,
            )

        self.assertFalse(ready)
        self.assertTrue(
            any("review_snapshots.writer_finalized" in issue for issue in issues),
            issues,
        )

    def test_clip_directory_ready_for_archive_allows_review_snapshots_when_csv_write_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip_dir = Path(tmp)
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
                    "planned_clip_complete": True,
                    "interrupted_by_user": False,
                    "stop_reason": "planned_end",
                    "grab_failures": 0,
                    "mp4_remux_succeeded": True,
                    "review_snapshots": {
                        "enabled": True,
                        "operational": True,
                        "writer_started": True,
                        "writer_finalized": True,
                        "index_csv_written": False,
                        "directory": "review_snapshots",
                        "saved": 10,
                        "failed": 0,
                    },
                },
            )

            ready, issues, _total_bytes = record_basler.clip_directory_ready_for_archive(
                clip_dir,
                expected_camera_count=1,
                max_clip_size_bytes=10_000_000,
            )

        self.assertTrue(ready)
        self.assertEqual(issues, [])

    def test_clip_directory_ready_for_archive_allows_review_snapshots_when_writer_never_started(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip_dir = Path(tmp)
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
                    "planned_clip_complete": True,
                    "interrupted_by_user": False,
                    "stop_reason": "planned_end",
                    "grab_failures": 0,
                    "mp4_remux_succeeded": True,
                    "review_snapshots": {
                        "enabled": True,
                        "operational": False,
                        "writer_started": False,
                        "writer_finalized": True,
                        "index_csv_written": False,
                        "directory": "review_snapshots",
                        "requested_per_full_clip": 10,
                        "saved": 0,
                        "failed": 0,
                        "dropped_queue_full": 0,
                        "missed_due_acquisition_gap": 0,
                        "unreached_targets": 10,
                    },
                },
            )

            ready, issues, _total_bytes = record_basler.clip_directory_ready_for_archive(
                clip_dir,
                expected_camera_count=1,
                max_clip_size_bytes=10_000_000,
            )

        self.assertTrue(ready)
        self.assertEqual(issues, [])

    def test_write_review_snapshot_index_csv_propagates_write_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review_snapshots" / "camera1_snapshots.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            result = record_basler.ReviewSnapshotResult(
                snapshot_index=0,
                target_elapsed_s=1.0,
                frame_index=3,
                filename="camera1_review_snapshot_0001.jpg",
                host_utc_ns=1,
                host_monotonic_ns=2,
                actual_clip_elapsed_s=1.25,
                video_time_s=0.75,
                success=True,
                error=None,
            )

            with mock.patch.object(record_basler.os, "replace", side_effect=OSError("boom")):
                with self.assertRaisesRegex(OSError, "boom"):
                    record_basler.write_review_snapshot_index_csv(path, [result])

            self.assertFalse(path.exists())
            self.assertFalse(path.with_name(f".{path.name}.tmp").exists())

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

    def test_classify_clip_stop_clean_user_interrupt(self) -> None:
        classification = record_basler.classify_clip_stop(
            error_message=None,
            planned_complete=False,
            stop_event_set=True,
            operator_stop_requested=True,
            stop_reason="unknown",
        )

        self.assertIsNone(classification.error_message)
        self.assertFalse(classification.planned_complete)
        self.assertTrue(classification.operator_stop_requested)
        self.assertTrue(classification.interrupted_by_user)
        self.assertEqual(classification.stop_reason, "user_interrupt")

    def test_classify_clip_stop_error_takes_precedence_over_user_interrupt(self) -> None:
        classification = record_basler.classify_clip_stop(
            error_message="FFmpeg remux failed",
            planned_complete=False,
            stop_event_set=True,
            operator_stop_requested=True,
            stop_reason="user_interrupt",
        )

        self.assertEqual(classification.error_message, "FFmpeg remux failed")
        self.assertTrue(classification.operator_stop_requested)
        self.assertFalse(classification.interrupted_by_user)
        self.assertEqual(classification.stop_reason, "failure")

    def test_classify_clip_stop_normal_completion_remains_planned_end(self) -> None:
        classification = record_basler.classify_clip_stop(
            error_message=None,
            planned_complete=True,
            stop_event_set=False,
            operator_stop_requested=False,
            stop_reason="unknown",
        )

        self.assertIsNone(classification.error_message)
        self.assertFalse(classification.interrupted_by_user)
        self.assertEqual(classification.stop_reason, "planned_end")

    def test_describe_operator_stop_completion_uses_real_archive_state(self) -> None:
        self.assertEqual(
            record_basler.describe_operator_stop_completion(
                clip_finalized_successfully=True,
                clip_queued_for_archive=True,
                archive_enabled=True,
            ),
            "Operator stop complete; active clip finalized and queued for archive. No additional clips will be started.",
        )
        self.assertEqual(
            record_basler.describe_operator_stop_completion(
                clip_finalized_successfully=True,
                clip_queued_for_archive=False,
                archive_enabled=False,
            ),
            "Operator stop complete; active clip finalized locally. Archiving is disabled. No additional clips will be started.",
        )


class FFmpegWriterTests(unittest.TestCase):
    def make_writer(self, root: Path) -> record_basler.FFmpegWriter:
        return record_basler.FFmpegWriter(
            ffmpeg="ffmpeg",
            temp_path=root / "camera1.capture.mkv",
            final_path=root / "camera1.mp4",
            width=1920,
            height=1200,
            fps=5.0,
            encoding_cfg={"codec": "libx264", "preset": "veryfast", "crf": 23},
        )

    def test_start_uses_new_process_group_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = self.make_writer(Path(tmp))
            process = mock.Mock()
            with mock.patch.object(record_basler.os, "name", "nt"):
                with mock.patch.object(
                    record_basler.subprocess,
                    "CREATE_NEW_PROCESS_GROUP",
                    0x200,
                    create=True,
                ):
                    with mock.patch.object(
                        record_basler.subprocess,
                        "Popen",
                        return_value=process,
                    ) as popen_mock:
                        writer.start()

            self.assertIs(writer.process, process)
            self.assertEqual(popen_mock.call_args.kwargs["creationflags"], 0x200)
            if writer.stderr_handle is not None:
                writer.stderr_handle.close()

    def test_start_uses_zero_creationflags_off_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = self.make_writer(Path(tmp))
            process = mock.Mock()
            with mock.patch.object(record_basler.os, "name", "posix"):
                with mock.patch.object(
                    record_basler.subprocess,
                    "Popen",
                    return_value=process,
                ) as popen_mock:
                    writer.start()

            self.assertIs(writer.process, process)
            self.assertEqual(popen_mock.call_args.kwargs["creationflags"], 0)
            if writer.stderr_handle is not None:
                writer.stderr_handle.close()

    def test_close_and_remux_uses_new_process_group_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = self.make_writer(Path(tmp))
            writer.temp_path.write_bytes(b"mkv")
            writer.process = mock.Mock()
            writer.process.stdin = mock.Mock()
            writer.process.wait.return_value = 0
            writer.stderr_handle = writer.stderr_path.open("wb")

            with mock.patch.object(record_basler.os, "name", "nt"):
                with mock.patch.object(
                    record_basler.subprocess,
                    "CREATE_NEW_PROCESS_GROUP",
                    0x200,
                    create=True,
                ):
                    with mock.patch.object(
                        record_basler.subprocess,
                        "run",
                        return_value=mock.Mock(returncode=0),
                    ) as run_mock:
                        return_code, remuxed = writer.close_and_remux(keep_temp=True)

            self.assertEqual((return_code, remuxed), (0, True))
            self.assertEqual(run_mock.call_args.kwargs["creationflags"], 0x200)

    def test_close_and_remux_success_on_zero_capture_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = self.make_writer(Path(tmp))
            writer.temp_path.write_bytes(b"mkv")
            writer.process = mock.Mock()
            writer.process.stdin = mock.Mock()
            writer.process.wait.return_value = 0
            writer.stderr_handle = writer.stderr_path.open("wb")

            with mock.patch.object(
                record_basler.subprocess,
                "run",
                return_value=mock.Mock(returncode=0),
            ) as run_mock:
                return_code, remuxed = writer.close_and_remux(keep_temp=True)

            self.assertEqual((return_code, remuxed), (0, True))
            self.assertTrue(remuxed)
            run_mock.assert_called_once()

    def test_close_and_remux_recovers_nonzero_capture_exit_for_user_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = self.make_writer(Path(tmp))
            writer.temp_path.write_bytes(b"mkv")
            writer.process = mock.Mock()
            writer.process.stdin = mock.Mock()
            writer.process.wait.return_value = 255
            writer.stderr_handle = writer.stderr_path.open("wb")

            with mock.patch.object(
                record_basler.subprocess,
                "run",
                return_value=mock.Mock(returncode=0),
            ) as run_mock:
                return_code, remuxed = writer.close_and_remux(
                    keep_temp=True,
                    allow_nonzero_capture_recovery=True,
                )

            self.assertEqual((return_code, remuxed), (255, True))
            run_mock.assert_called_once()

    def test_close_and_remux_does_not_recover_nonzero_capture_exit_without_allowance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = self.make_writer(Path(tmp))
            writer.process = mock.Mock()
            writer.process.stdin = mock.Mock()
            writer.process.wait.return_value = 255
            writer.stderr_handle = writer.stderr_path.open("wb")

            with mock.patch.object(record_basler.subprocess, "run") as run_mock:
                return_code, remuxed = writer.close_and_remux(
                    keep_temp=True,
                    allow_nonzero_capture_recovery=False,
                )

            self.assertEqual((return_code, remuxed), (255, False))
            run_mock.assert_not_called()

    def test_close_and_remux_failed_recovery_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = self.make_writer(Path(tmp))
            writer.temp_path.write_bytes(b"mkv")
            writer.process = mock.Mock()
            writer.process.stdin = mock.Mock()
            writer.process.wait.return_value = 255
            writer.stderr_handle = writer.stderr_path.open("wb")

            with mock.patch.object(
                record_basler.subprocess,
                "run",
                return_value=mock.Mock(returncode=1),
            ):
                return_code, remuxed = writer.close_and_remux(
                    keep_temp=True,
                    allow_nonzero_capture_recovery=True,
                )

            self.assertEqual((return_code, remuxed), (255, False))


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
        self.assertNotIn(("AutoFunctionROIUseWhiteBalance", True), log)
        self.assertEqual(binding.actual_settings["ExposureAuto"], "Continuous")
        self.assertEqual(binding.actual_settings["GainAuto"], "Off")
        self.assertEqual(binding.actual_settings["Gain"], 0.0)

    def test_white_balance_maps_boolean_and_is_saved_in_metadata(self) -> None:
        on_binding, on_log = configure_fake_camera(
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
                "balance_white_auto": True,
            },
            modern=True,
        )
        self.assertIn(("BalanceWhiteAuto", "Continuous"), on_log)
        self.assertEqual(on_binding.actual_settings["BalanceWhiteAuto"], "Continuous")

        off_binding, off_log = configure_fake_camera(
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
                "balance_white_auto": False,
            },
            modern=True,
        )
        self.assertIn(("BalanceWhiteAuto", "Off"), off_log)
        self.assertEqual(off_binding.actual_settings["BalanceWhiteAuto"], "Off")

    def test_modern_white_balance_normalizes_stale_roi_state(self) -> None:
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
                "balance_white_auto": True,
            },
            modern=True,
            roi_usage_state={
                "brightness": {"ROI1": False, "ROI2": True},
                "white_balance": {"ROI1": True, "ROI2": False},
            },
        )

        self.assertIn(("AutoFunctionROISelector", "ROI2"), log)
        self.assertIn(("AutoFunctionROIUseWhiteBalance", True), log)
        self.assertIn(("AutoFunctionROIUseBrightness", False), log)

        camera = binding.camera
        record_basler.try_set(camera, "AutoFunctionROISelector", "ROI1", required=True)
        self.assertTrue(record_basler.read_setting(camera, "AutoFunctionROIUseBrightness"))
        self.assertFalse(record_basler.read_setting(camera, "AutoFunctionROIUseWhiteBalance"))

        record_basler.try_set(camera, "AutoFunctionROISelector", "ROI2", required=True)
        self.assertFalse(record_basler.read_setting(camera, "AutoFunctionROIUseBrightness"))
        self.assertTrue(record_basler.read_setting(camera, "AutoFunctionROIUseWhiteBalance"))

        self.assertEqual(binding.actual_settings["AutoFunctionROISelector"], "ROI2")
        self.assertTrue(binding.actual_settings["AutoFunctionROIUseWhiteBalance"])
        self.assertEqual(binding.actual_settings["BalanceWhiteAuto"], "Continuous")

    def test_legacy_white_balance_normalizes_stale_aoi_state(self) -> None:
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
                "balance_white_auto": True,
            },
            modern=False,
            aoi_usage_state={
                "brightness": {"AOI1": False, "AOI2": True},
                "white_balance": {"AOI1": True, "AOI2": False},
            },
        )

        self.assertIn(("AutoFunctionAOISelector", "AOI2"), log)
        self.assertIn(("AutoFunctionAOIUsageWhiteBalance", True), log)
        self.assertIn(("AutoFunctionAOIUsageIntensity", False), log)

        camera = binding.camera
        record_basler.try_set(camera, "AutoFunctionAOISelector", "AOI1", required=True)
        self.assertTrue(record_basler.read_setting(camera, "AutoFunctionAOIUsageIntensity"))
        self.assertFalse(record_basler.read_setting(camera, "AutoFunctionAOIUsageWhiteBalance"))

        record_basler.try_set(camera, "AutoFunctionAOISelector", "AOI2", required=True)
        self.assertFalse(record_basler.read_setting(camera, "AutoFunctionAOIUsageIntensity"))
        self.assertTrue(record_basler.read_setting(camera, "AutoFunctionAOIUsageWhiteBalance"))

        self.assertEqual(binding.actual_settings["AutoFunctionAOISelector"], "AOI2")
        self.assertTrue(binding.actual_settings["AutoFunctionAOIUsageWhiteBalance"])
        self.assertEqual(binding.actual_settings["BalanceWhiteAuto"], "Continuous")

    def test_white_balance_roi_unsupported_warns_and_continues(self) -> None:
        with self.assertLogs(record_basler.LOG.name, level="WARNING") as captured:
            binding, _ = configure_fake_camera(
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
                    "balance_white_auto": True,
                },
                modern=True,
                include_white_balance_usage=False,
            )

        self.assertTrue(
            any(
                "could not explicitly configure a white-balance auto-function ROI" in line
                for line in captured.output
            )
        )
        self.assertEqual(binding.actual_settings["BalanceWhiteAuto"], "Continuous")

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


class ReviewSnapshotSettingsTests(unittest.TestCase):
    def test_default_is_disabled(self) -> None:
        settings = record_basler.parse_review_snapshot_settings({})
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.count_per_clip, 10)
        self.assertEqual(settings.jpeg_quality, 95)

    def test_equal_bin_centers_match_expected_ten_minute_targets(self) -> None:
        self.assertEqual(
            record_basler.review_snapshot_targets(600.0, 10),
            [30.0, 90.0, 150.0, 210.0, 270.0, 330.0, 390.0, 450.0, 510.0, 570.0],
        )

    def test_unreached_targets_count_remaining_slots(self) -> None:
        self.assertEqual(record_basler.compute_review_snapshot_unreached_targets(10, 0), 10)
        self.assertEqual(record_basler.compute_review_snapshot_unreached_targets(10, 4), 6)

    def test_invalid_quality_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "review_snapshots.jpeg_quality"):
            record_basler.parse_review_snapshot_settings(
                {"review_snapshots": {"enabled": True, "jpeg_quality": 0}}
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


class StartSchedulingTests(unittest.TestCase):
    def _fixed_now_utc(self) -> dt.datetime:
        return dt.datetime(2026, 8, 7, 4, 0, tzinfo=dt.timezone.utc)

    def _scheduled_local_text(
        self,
        *,
        delta: dt.timedelta,
        include_seconds: bool = False,
    ) -> tuple[str, dt.datetime]:
        target_local = self._fixed_now_utc().astimezone() + delta
        fmt = "%Y-%m-%d %H:%M:%S" if include_seconds else "%Y-%m-%d %H:%M"
        return target_local.strftime(fmt), target_local.astimezone(dt.timezone.utc)

    def _advancing_utc_now(self, start: dt.datetime) -> object:
        state = {"count": -1}

        def fake_utc_now() -> dt.datetime:
            state["count"] += 1
            return start + dt.timedelta(seconds=state["count"])

        return fake_utc_now

    def _make_config(
        self,
        output_root: Path,
        *,
        start_at_local: str | None | object = ...,
        number_of_clips: int = 1,
    ) -> dict[str, object]:
        schedule: dict[str, object] = {
            "clip_duration_s": 1,
            "interval_s": 1,
            "number_of_clips": number_of_clips,
        }
        if start_at_local is not ...:
            schedule["start_at_local"] = start_at_local
        return {
            "project": "project",
            "subject": "subject",
            "output_root": str(output_root),
            "schedule": schedule,
            "archive": {"enabled": False},
            "status": {"terminal_interval_s": 0},
            "system": {"prevent_sleep_during_recording": False},
            "cameras": [{"label": "camera1", "fps": 5}],
        }

    def _make_binding(self, label: str = "camera1") -> record_basler.CameraBinding:
        camera = mock.Mock()
        camera.IsGrabbing.return_value = False
        camera.IsOpen.return_value = False
        camera.StopGrabbing.return_value = None
        camera.Close.return_value = None
        camera.DestroyDevice.return_value = None
        return record_basler.CameraBinding(
            label=label,
            requested={"label": label, "fps": 5},
            camera=camera,
            info={"model": "fake", "serial": "40604036"},
            actual_settings={"AcquisitionFrameRate": 5.0},
            converter=mock.Mock(),
        )

    def _fake_record_one_camera(self, **kwargs: object) -> None:
        binding = kwargs["binding"]
        clip_dir = kwargs["clip_dir"]
        ready_barrier = kwargs["ready_barrier"]
        result_queue = kwargs["result_queue"]
        assert isinstance(binding, record_basler.CameraBinding)
        assert isinstance(clip_dir, Path)
        assert isinstance(ready_barrier, threading.Barrier)
        assert isinstance(result_queue, queue.Queue)

        ready_barrier.wait(timeout=5)

        file_stem = record_basler.sanitize_token(binding.label)
        video_path = clip_dir / f"{file_stem}.mp4"
        metadata_path = clip_dir / f"{file_stem}.json"
        timestamps_path = clip_dir / f"{file_stem}.timestamps.csv.gz"
        video_path.write_bytes(b"mp4")
        with gzip.open(timestamps_path, "wt", encoding="utf-8", newline="") as handle:
            handle.write(
                "frame_index,host_utc_ns,host_utc_iso,host_monotonic_ns,camera_timestamp,block_id,skipped_images\n"
            )
            handle.write("0,1,1970-01-01T00:00:00.000Z,1,1,1,0\n")
        record_basler.write_json(
            metadata_path,
            {
                "success": True,
                "planned_clip_complete": True,
                "interrupted_by_user": False,
                "stop_reason": "planned_end",
                "grab_failures": 0,
                "mp4_remux_succeeded": True,
                "actual_elapsed_s": 1.0,
            },
        )
        result_queue.put(
            record_basler.ClipResult(
                label=binding.label,
                success=True,
                planned_complete=True,
                interrupted_by_user=False,
                metadata_path=metadata_path,
                video_path=video_path,
                error=None,
            )
        )

    def test_reset_recording_preview_for_clip_reenables_preview_when_configured(self) -> None:
        preview_settings = record_basler.RecordingPreviewSettings(enabled=True)
        preview_active_event = threading.Event()

        record_basler.reset_recording_preview_for_clip(
            preview_settings=preview_settings,
            preview_active_event=preview_active_event,
            clip_index=1,
            total_clips=3,
        )

        self.assertTrue(preview_active_event.is_set())

    def test_reset_recording_preview_for_clip_keeps_preview_disabled_when_config_disabled(self) -> None:
        preview_settings = record_basler.RecordingPreviewSettings(enabled=False)
        preview_active_event = threading.Event()
        preview_active_event.set()

        record_basler.reset_recording_preview_for_clip(
            preview_settings=preview_settings,
            preview_active_event=preview_active_event,
            clip_index=1,
            total_clips=3,
        )

        self.assertFalse(preview_active_event.is_set())

    def test_parse_start_at_local_absent_returns_none(self) -> None:
        self.assertIsNone(record_basler.parse_start_at_local({}))

    def test_parse_start_at_local_null_returns_none(self) -> None:
        self.assertIsNone(record_basler.parse_start_at_local({"start_at_local": None}))

    def test_parse_start_at_local_future_minute_format(self) -> None:
        schedule_text, expected_utc = self._scheduled_local_text(delta=dt.timedelta(hours=5))
        target = record_basler.parse_start_at_local(
            {"start_at_local": schedule_text},
            now_utc=self._fixed_now_utc(),
        )

        self.assertEqual(record_basler.isoformat_utc(target), record_basler.isoformat_utc(expected_utc))

    def test_parse_start_at_local_seconds_format(self) -> None:
        schedule_text, expected_utc = self._scheduled_local_text(
            delta=dt.timedelta(hours=5, seconds=30),
            include_seconds=True,
        )
        target = record_basler.parse_start_at_local(
            {"start_at_local": schedule_text},
            now_utc=self._fixed_now_utc(),
        )

        self.assertEqual(record_basler.isoformat_utc(target), record_basler.isoformat_utc(expected_utc))

    def test_parse_start_at_local_rejects_bad_formats(self) -> None:
        for raw in ("05:00", "tomorrow 5am", "2026/08/08 05:00", ""):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "schedule.start_at_local"):
                    record_basler.parse_start_at_local({"start_at_local": raw})

    def test_parse_start_at_local_rejects_past_time(self) -> None:
        schedule_text, _expected_utc = self._scheduled_local_text(delta=dt.timedelta(hours=-1))
        with self.assertRaisesRegex(ValueError, "already in the past"):
            record_basler.parse_start_at_local(
                {"start_at_local": schedule_text},
                now_utc=self._fixed_now_utc(),
            )

    def test_wait_until_scheduled_start_uses_short_sleeps(self) -> None:
        base = dt.datetime(2026, 8, 8, 8, 0, tzinfo=dt.timezone.utc)
        state = {"utc": base, "mono": 0.0}
        sleeps: list[float] = []

        def now_utc_fn() -> dt.datetime:
            return state["utc"]

        def monotonic_fn() -> float:
            state["mono"] += 0.5
            return state["mono"]

        def sleep_fn(duration: float) -> None:
            sleeps.append(duration)
            state["utc"] += dt.timedelta(seconds=duration)

        reached = record_basler.wait_until_scheduled_start(
            target_utc=base + dt.timedelta(seconds=2.5),
            terminal_interval_s=60.0,
            now_utc_fn=now_utc_fn,
            monotonic_fn=monotonic_fn,
            sleep_fn=sleep_fn,
        )

        self.assertTrue(reached)
        self.assertTrue(sleeps)
        self.assertLessEqual(max(sleeps), 1.0)

    def test_wait_until_scheduled_start_can_cancel_cleanly(self) -> None:
        base = dt.datetime(2026, 8, 8, 8, 0, tzinfo=dt.timezone.utc)
        state = {"utc": base, "mono": 0.0}
        cancel_event = threading.Event()

        def now_utc_fn() -> dt.datetime:
            return state["utc"]

        def monotonic_fn() -> float:
            state["mono"] += 1.0
            return state["mono"]

        def sleep_fn(duration: float) -> None:
            state["utc"] += dt.timedelta(seconds=duration)
            cancel_event.set()

        reached = record_basler.wait_until_scheduled_start(
            target_utc=base + dt.timedelta(minutes=5),
            terminal_interval_s=60.0,
            cancel_event=cancel_event,
            now_utc_fn=now_utc_fn,
            monotonic_fn=monotonic_fn,
            sleep_fn=sleep_fn,
        )

        self.assertFalse(reached)

    def test_dry_run_absolute_start_writes_requested_start_metadata_without_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "recordings"
            config_path = Path(tmp) / "config.yaml"
            schedule_text, expected_utc = self._scheduled_local_text(delta=dt.timedelta(hours=5))
            config = {
                "project": "project",
                "subject": "subject",
                "output_root": str(output_root),
                "schedule": {
                    "start_at_local": schedule_text,
                    "clip_duration_s": 600,
                    "interval_s": 600,
                    "number_of_clips": 2,
                },
                "archive": {"enabled": False},
                "cameras": [{"label": "camera1"}],
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with mock.patch.object(
                record_basler,
                "wait_until_scheduled_start",
                side_effect=AssertionError("wait should not run during dry-run"),
            ):
                with mock.patch.object(
                    record_basler,
                    "utc_now",
                    side_effect=self._advancing_utc_now(self._fixed_now_utc()),
                ):
                    exit_code = record_basler.run_recording(
                        config_path,
                        config,
                        verbose=False,
                        dry_run=True,
                    )

            self.assertEqual(exit_code, 0)
            dry_run_paths = list(output_root.glob("project/subject/*/dry_run.json"))
            self.assertEqual(len(dry_run_paths), 1)
            payload = json.loads(dry_run_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["recording_plan"]["start_mode"], "absolute_local")
            self.assertEqual(
                payload["recording_plan"]["requested_start_local"],
                record_basler.isoformat_local(expected_utc),
            )
            self.assertEqual(
                payload["recording_plan"]["requested_start_utc"],
                record_basler.isoformat_utc(expected_utc),
            )

    def test_immediate_mode_does_not_enter_scheduled_wait_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "recordings"
            config_path = Path(tmp) / "config.yaml"
            config = self._make_config(output_root)
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with mock.patch.object(
                record_basler,
                "wait_until_scheduled_start",
                side_effect=AssertionError("immediate mode should not wait"),
            ):
                with mock.patch.object(record_basler, "find_ffmpeg", return_value="ffmpeg"):
                    with mock.patch.object(
                        record_basler,
                        "run_recording_session",
                        return_value=0,
                    ) as session_mock:
                        exit_code = record_basler.run_recording(
                            config_path,
                            config,
                            verbose=False,
                            dry_run=False,
                        )

            self.assertEqual(exit_code, 0)
            session_mock.assert_called_once()
            self.assertEqual(
                session_mock.call_args.kwargs["recording_plan"]["start_mode"],
                "immediate",
            )

    def test_null_start_at_local_does_not_enter_scheduled_wait_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "recordings"
            config_path = Path(tmp) / "config.yaml"
            config = self._make_config(output_root, start_at_local=None)
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with mock.patch.object(
                record_basler,
                "wait_until_scheduled_start",
                side_effect=AssertionError("null start_at_local should not wait"),
            ):
                with mock.patch.object(record_basler, "find_ffmpeg", return_value="ffmpeg"):
                    with mock.patch.object(
                        record_basler,
                        "run_recording_session",
                        return_value=0,
                    ) as session_mock:
                        exit_code = record_basler.run_recording(
                            config_path,
                            config,
                            verbose=False,
                            dry_run=False,
                        )

            self.assertEqual(exit_code, 0)
            session_mock.assert_called_once()
            self.assertEqual(
                session_mock.call_args.kwargs["recording_plan"]["start_mode"],
                "immediate",
            )

    def test_run_recording_session_initializes_without_scheduler_only_variables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "recordings"
            config_path = Path(tmp) / "config.yaml"
            config = self._make_config(output_root, number_of_clips=0)
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            binding = self._make_binding()

            exit_code = None
            with mock.patch.object(record_basler, "match_configured_devices", return_value=[object()]):
                with mock.patch.object(record_basler, "configure_camera", return_value=binding):
                    with mock.patch.object(record_basler.signal, "signal"):
                        exit_code = record_basler.run_recording_session(
                            config_path=config_path,
                            config=config,
                            verbose=False,
                            ffmpeg="ffmpeg",
                            clip_duration_s=1.0,
                            interval_s=1.0,
                            total_duration_s=None,
                            number_of_clips=0,
                            total_clips=0,
                            planned_session_duration_s=0.0,
                            planned_finish_utc=dt.datetime(2026, 8, 8, 9, 0, tzinfo=dt.timezone.utc),
                            system_settings=record_basler.SystemSettings(False),
                            status_settings=record_basler.StatusSettings(0.0),
                            preview_settings=record_basler.RecordingPreviewSettings(enabled=False),
                            archive_settings=record_basler.ArchiveSettings(enabled=False),
                            review_snapshot_settings=record_basler.ReviewSnapshotSettings(
                                enabled=False
                            ),
                            recording_plan={
                                "start_mode": "immediate",
                                "requested_start_utc": None,
                                "requested_start_local": None,
                            },
                            output_root=output_root,
                            project="project",
                            subject="subject",
                            camera_cfgs=[{"label": "camera1", "fps": 5}],
                            manage_sleep=False,
                        )

            self.assertEqual(exit_code, 0)
            manifest_paths = list(output_root.glob("project/subject/*/session_manifest.json"))
            self.assertEqual(len(manifest_paths), 1)
            manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["recording_plan"]["start_mode"], "immediate")

    def test_run_recording_immediate_initialization_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "recordings"
            config_path = Path(tmp) / "config.yaml"
            config = self._make_config(output_root)
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with mock.patch.object(record_basler, "find_ffmpeg", return_value="ffmpeg"):
                with mock.patch.object(record_basler, "match_configured_devices", return_value=[object()]):
                    with mock.patch.object(
                        record_basler,
                        "configure_camera",
                        side_effect=lambda camera_cfg, device: self._make_binding(str(camera_cfg["label"])),
                    ):
                        with mock.patch.object(record_basler, "record_one_camera", side_effect=self._fake_record_one_camera):
                            with mock.patch.object(record_basler.signal, "signal"):
                                exit_code = record_basler.run_recording(
                                    config_path,
                                    config,
                                    verbose=False,
                                    dry_run=False,
                                )

            self.assertEqual(exit_code, 0)
            manifest_paths = list(output_root.glob("project/subject/*/session_manifest.json"))
            self.assertEqual(len(manifest_paths), 1)
            manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["recording_plan"]["start_mode"], "immediate")

    def test_run_recording_scheduled_initialization_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "recordings"
            config_path = Path(tmp) / "config.yaml"
            schedule_text, expected_utc = self._scheduled_local_text(delta=dt.timedelta(hours=5))
            config = self._make_config(output_root, start_at_local=schedule_text)
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with mock.patch.object(record_basler, "find_ffmpeg", return_value="ffmpeg"):
                with mock.patch.object(record_basler, "match_configured_devices", return_value=[object()]):
                    with mock.patch.object(
                        record_basler,
                        "configure_camera",
                        side_effect=lambda camera_cfg, device: self._make_binding(str(camera_cfg["label"])),
                    ):
                        with mock.patch.object(record_basler, "wait_until_scheduled_start", return_value=True) as wait_mock:
                            with mock.patch.object(record_basler, "record_one_camera", side_effect=self._fake_record_one_camera):
                                with mock.patch.object(record_basler.signal, "signal"):
                                    with mock.patch.object(
                                        record_basler,
                                        "utc_now",
                                        side_effect=self._advancing_utc_now(self._fixed_now_utc()),
                                    ):
                                        exit_code = record_basler.run_recording(
                                            config_path,
                                            config,
                                            verbose=False,
                                            dry_run=False,
                                        )

            self.assertEqual(exit_code, 0)
            wait_mock.assert_called_once()
            manifest_paths = list(output_root.glob("project/subject/*/session_manifest.json"))
            self.assertEqual(len(manifest_paths), 1)
            manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["recording_plan"]["start_mode"], "absolute_local")
            self.assertEqual(
                manifest["recording_plan"]["requested_start_local"],
                record_basler.isoformat_local(expected_utc),
            )
            self.assertEqual(
                manifest["recording_plan"]["requested_start_utc"],
                record_basler.isoformat_utc(expected_utc),
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
