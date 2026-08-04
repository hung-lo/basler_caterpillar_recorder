#!/usr/bin/env python3
"""Cross-platform scheduled recorder for one or more Basler USB cameras.

The recorder:
- discovers cameras by serial number or exact model name;
- applies low-level camera settings through pypylon;
- records simultaneous clips from multiple cameras;
- writes H.264 video through FFmpeg;
- optionally shows a non-blocking low-resolution preview while recording;
- saves compressed per-frame timestamps and JSON metadata;
- supports continuous recording via back-to-back finite clips or duty-cycled recording.

Tested syntactically without cameras. Camera-node availability differs by model and
firmware, so every requested setting is queried and logged rather than assumed.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import gzip
import json
import logging
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import yaml

try:
    from pypylon import pylon
except ImportError:  # Allows --help and syntax checks before pypylon is installed.
    pylon = None  # type: ignore[assignment]


LOG = logging.getLogger("basler_recorder")


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def session_stamp_utc(value: dt.datetime) -> str:
    """UTC session directory timestamp, sortable to one-second precision."""

    return value.astimezone(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def clip_clock_utc(value: dt.datetime) -> str:
    """UTC clip clock time; the session directory already stores the date."""

    return value.astimezone(dt.timezone.utc).strftime("%H%M%S")


def snapshot_stamp_utc(value: dt.datetime) -> str:
    """Compact UTC timestamp for optional monitoring snapshots."""

    return value.astimezone(dt.timezone.utc).strftime("%H%M%S")


def iso_utc_from_ns(value_ns: int) -> str:
    return dt.datetime.fromtimestamp(value_ns / 1e9, tz=dt.timezone.utc).isoformat()


def format_clock_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def sanitize_token(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    cleaned = "".join(ch if ch in allowed else "-" for ch in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-_") or "unnamed"


def choose_unique_directory(path: Path) -> Path:
    if not path.exists():
        return path

    for suffix in range(2, 1000):
        candidate = path.with_name(f"{path.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not choose a unique directory for {path}")


def ensure_even(value: int) -> int:
    return value if value % 2 == 0 else value - 1


def require_pypylon() -> None:
    if pylon is None:
        raise RuntimeError(
            "pypylon is not installed. Install the Basler pylon Software Suite, "
            "then run: python -m pip install pypylon"
        )


def find_ffmpeg(configured: Optional[str]) -> str:
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.exists():
            return str(candidate)
        located = shutil.which(configured)
        if located:
            return located
        raise FileNotFoundError(f"Configured FFmpeg executable was not found: {configured}")
    located = shutil.which("ffmpeg")
    if not located:
        raise FileNotFoundError(
            "FFmpeg was not found on PATH. Install FFmpeg and confirm that "
            "`ffmpeg -version` works in the same terminal."
        )
    return located


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping/object.")
    return config


def setup_logging(session_dir: Optional[Path], verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if session_dir is not None:
        handlers.append(logging.FileHandler(session_dir / "recorder.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def camera_info_dict(device_info: Any) -> dict[str, str]:
    def safe_call(name: str) -> str:
        try:
            return str(getattr(device_info, name)())
        except Exception:
            return ""

    return {
        "model": safe_call("GetModelName"),
        "serial": safe_call("GetSerialNumber"),
        "user_defined_name": safe_call("GetUserDefinedName"),
        "device_class": safe_call("GetDeviceClass"),
        "friendly_name": safe_call("GetFriendlyName"),
        "full_name": safe_call("GetFullName"),
    }


def enumerate_devices() -> list[Any]:
    require_pypylon()
    factory = pylon.TlFactory.GetInstance()
    return list(factory.EnumerateDevices())


def list_cameras() -> int:
    devices = enumerate_devices()
    if not devices:
        print("No Basler cameras were detected.")
        return 1
    print(f"Detected {len(devices)} Basler camera(s):")
    for index, info in enumerate(devices):
        data = camera_info_dict(info)
        print(
            f"[{index}] model={data['model']!r} serial={data['serial']!r} "
            f"name={data['user_defined_name']!r} class={data['device_class']!r}"
        )
    return 0


def get_node(camera: Any, name: str) -> Any | None:
    try:
        return getattr(camera, name)
    except Exception:
        return None


def node_value(node: Any) -> Any:
    if node is None:
        return None
    for accessor in ("GetValue",):
        try:
            return getattr(node, accessor)()
        except Exception:
            pass
    try:
        return node.Value
    except Exception:
        return None


def node_bounds(node: Any) -> tuple[Optional[float], Optional[float], Optional[float]]:
    values: list[Optional[float]] = []
    for accessor in ("GetMin", "GetMax", "GetInc"):
        try:
            values.append(float(getattr(node, accessor)()))
        except Exception:
            values.append(None)
    return values[0], values[1], values[2]


def align_numeric(value: float, minimum: Optional[float], maximum: Optional[float], increment: Optional[float]) -> float:
    result = value
    if minimum is not None:
        result = max(result, minimum)
    if maximum is not None:
        result = min(result, maximum)
    if increment is not None and increment > 0 and minimum is not None:
        result = minimum + round((result - minimum) / increment) * increment
        if maximum is not None:
            result = min(result, maximum)
    return result


def try_set(camera: Any, name: str, value: Any, *, required: bool = False) -> tuple[bool, Any]:
    node = get_node(camera, name)
    if node is None:
        message = f"Node {name} is unavailable"
        if required:
            raise RuntimeError(message)
        LOG.debug(message)
        return False, None

    requested = value
    try:
        if isinstance(value, bool):
            try:
                success = bool(node.TrySetValue(value))
                if not success:
                    node.SetValue(value)
            except Exception:
                node.SetValue(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            minimum, maximum, increment = node_bounds(node)
            aligned = align_numeric(float(value), minimum, maximum, increment)
            if isinstance(value, int) and (increment is None or float(increment).is_integer()):
                aligned = int(round(aligned))
            node.SetValue(aligned)
        else:
            try:
                success = bool(node.TrySetValue(value))
                if not success:
                    node.SetValue(value)
            except Exception:
                node.SetValue(value)
        actual = node_value(node)
        LOG.debug("Set %s requested=%r actual=%r", name, requested, actual)
        return True, actual
    except Exception as exc:
        if required:
            raise RuntimeError(f"Failed to set required node {name}={value!r}: {exc}") from exc
        LOG.warning("Could not set %s=%r: %s", name, value, exc)
        return False, node_value(node)


def first_settable_enum(camera: Any, name: str, candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        ok, actual = try_set(camera, name, candidate)
        if ok:
            return str(actual)
    return None


def read_setting(camera: Any, name: str) -> Any:
    return node_value(get_node(camera, name))


def enable_chunk_timestamp(camera: Any) -> bool:
    ok, _ = try_set(camera, "ChunkModeActive", True)
    if not ok:
        return False
    ok, _ = try_set(camera, "ChunkSelector", "Timestamp")
    if not ok:
        return False
    ok, _ = try_set(camera, "ChunkEnable", True)
    return ok


def get_grab_value(grab_result: Any, names: Iterable[str]) -> Any:
    for name in names:
        try:
            value = getattr(grab_result, name)
            if callable(value):
                value = value()
            if hasattr(value, "Value"):
                value = value.Value
            if value is not None:
                return value
        except Exception:
            continue
    return None


def apply_frame_transform(frame: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    rotate = int(cfg.get("rotate", 0) or 0) % 360
    if rotate == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotate == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate != 0:
        raise ValueError("rotate must be one of 0, 90, 180, or 270")

    if bool(cfg.get("flip_horizontal", False)):
        frame = cv2.flip(frame, 1)
    if bool(cfg.get("flip_vertical", False)):
        frame = cv2.flip(frame, 0)

    output_width = cfg.get("output_width")
    output_height = cfg.get("output_height")
    if output_width or output_height:
        h, w = frame.shape[:2]
        if output_width and output_height:
            target_w = int(output_width)
            target_h = int(output_height)
        elif output_width:
            target_w = int(output_width)
            target_h = round(h * target_w / w)
        else:
            target_h = int(output_height)
            target_w = round(w * target_h / h)
        target_w = max(2, ensure_even(target_w))
        target_h = max(2, ensure_even(target_h))
        if (target_w, target_h) != (w, h):
            interpolation = cv2.INTER_AREA if target_w < w or target_h < h else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (target_w, target_h), interpolation=interpolation)

    h, w = frame.shape[:2]
    if w % 2 or h % 2:
        frame = frame[: ensure_even(h), : ensure_even(w)]
    return np.ascontiguousarray(frame)


@dataclasses.dataclass(frozen=True)
class RecordingPreviewSettings:
    """Low-overhead monitor preview shown while video is being recorded."""

    enabled: bool = False
    fps: float = 1.0
    max_width: int = 640
    max_height: int = 720
    show_status: bool = True


@dataclasses.dataclass(frozen=True)
class ArchiveSettings:
    enabled: bool = False
    destination_root: Path = dataclasses.field(
        default_factory=lambda: Path("/Volumes/Dr. Rose/Hung_MBL")
    )
    required_mount_point: Path = dataclasses.field(
        default_factory=lambda: Path("/Volumes/Dr. Rose")
    )
    rsync_executable: str = "/usr/bin/rsync"
    transfer_after_each_clip: bool = True
    background_transfer: bool = True
    delete_local_clip_after_verified_transfer: bool = True
    retain_local_session_metadata: bool = True
    verification: str = "checksum"
    max_clip_size_gb: float = 50.0
    min_local_free_gb_before_clip: float = 120.0
    min_external_free_gb_before_transfer: float = 60.0
    max_unarchived_clips: int = 2
    stop_before_next_clip_on_transfer_failure: bool = True


@dataclasses.dataclass(frozen=True)
class ArchivePreflightResult:
    enabled: bool
    ok: bool
    errors: list[str]
    rsync_path: Optional[str]
    required_mount_point: str
    required_mount_is_mount: bool
    destination_root: str
    destination_created: bool
    destination_writable: bool
    local_free_gb: Optional[float]
    destination_free_gb: Optional[float]
    path_conflict: bool
    archive_session_dir: str


@dataclasses.dataclass
class PreviewPacket:
    label: str
    clip_index: int
    frame_index: int
    frame: np.ndarray
    host_monotonic_ns: int
    elapsed_s: float
    planned_duration_s: float
    measured_receive_fps: Optional[float]


def parse_recording_preview_settings(config: dict[str, Any]) -> RecordingPreviewSettings:
    raw = config.get("recording_preview") or {}
    if not isinstance(raw, dict):
        raise ValueError("recording_preview must be a mapping/object")

    settings = RecordingPreviewSettings(
        enabled=bool(raw.get("enabled", False)),
        fps=float(raw.get("fps", 1.0)),
        max_width=int(raw.get("max_width", 640)),
        max_height=int(raw.get("max_height", 720)),
        show_status=bool(raw.get("show_status", True)),
    )
    if settings.fps <= 0:
        raise ValueError("recording_preview.fps must be positive")
    if settings.max_width < 64 or settings.max_height < 64:
        raise ValueError("recording_preview.max_width and max_height must be at least 64")
    return settings


def parse_archive_settings(config: dict[str, Any]) -> ArchiveSettings:
    raw = config.get("archive") or {}
    if not isinstance(raw, dict):
        raise ValueError("archive must be a mapping/object")

    destination_root = Path(str(raw.get("destination_root", "/Volumes/Dr. Rose/Hung_MBL"))).expanduser()
    required_mount_point = Path(str(raw.get("required_mount_point", "/Volumes/Dr. Rose"))).expanduser()
    rsync_executable = str(raw.get("rsync_executable", "/usr/bin/rsync"))
    verification = str(raw.get("verification", "checksum")).strip().lower() or "checksum"

    return ArchiveSettings(
        enabled=bool(raw.get("enabled", False)),
        destination_root=destination_root,
        required_mount_point=required_mount_point,
        rsync_executable=rsync_executable,
        transfer_after_each_clip=bool(raw.get("transfer_after_each_clip", True)),
        background_transfer=bool(raw.get("background_transfer", True)),
        delete_local_clip_after_verified_transfer=bool(
            raw.get("delete_local_clip_after_verified_transfer", True)
        ),
        retain_local_session_metadata=bool(raw.get("retain_local_session_metadata", True)),
        verification=verification,
        max_clip_size_gb=float(raw.get("max_clip_size_gb", 50.0)),
        min_local_free_gb_before_clip=float(raw.get("min_local_free_gb_before_clip", 120.0)),
        min_external_free_gb_before_transfer=float(raw.get("min_external_free_gb_before_transfer", 60.0)),
        max_unarchived_clips=int(raw.get("max_unarchived_clips", 2)),
        stop_before_next_clip_on_transfer_failure=bool(
            raw.get("stop_before_next_clip_on_transfer_failure", True)
        ),
    )


def resize_frame_to_fit(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Return a display copy that fits inside the requested bounding box."""

    height, width = frame.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale < 1.0:
        target_width = max(2, round(width * scale))
        target_height = max(2, round(height * scale))
        resized = cv2.resize(
            frame,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = frame

    # The pylon-converted array may refer to reusable grab-buffer memory. The
    # queue must therefore own an independent copy before the grab is released.
    return np.ascontiguousarray(resized).copy()


def put_latest_preview(preview_queue: queue.Queue[PreviewPacket], packet: PreviewPacket) -> None:
    """Publish without ever blocking acquisition or building a backlog."""

    try:
        preview_queue.put_nowait(packet)
        return
    except queue.Full:
        pass

    try:
        preview_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        preview_queue.put_nowait(packet)
    except queue.Full:
        # A consumer/producer race can refill the one-item queue. Dropping this
        # monitor frame is always preferable to delaying scientific acquisition.
        pass


@dataclasses.dataclass
class CameraBinding:
    label: str
    requested: dict[str, Any]
    camera: Any
    info: dict[str, str]
    actual_settings: dict[str, Any]
    converter: Any

    @property
    def fps(self) -> float:
        value = self.actual_settings.get("AcquisitionFrameRate")
        if value is None:
            value = self.requested.get("fps", 5.0)
        return float(value)


def match_device(camera_cfg: dict[str, Any], devices: list[Any], used_serials: set[str]) -> Any:
    serial = str(camera_cfg.get("serial") or "").strip()
    model = str(camera_cfg.get("model") or "").strip()
    user_name = str(camera_cfg.get("user_defined_name") or "").strip()

    matches = []
    for device in devices:
        info = camera_info_dict(device)
        if info["serial"] in used_serials:
            continue
        if serial and info["serial"] != serial:
            continue
        if model and info["model"] != model:
            continue
        if user_name and info["user_defined_name"] != user_name:
            continue
        matches.append(device)

    if not matches:
        raise RuntimeError(
            f"No unused camera matched serial={serial!r}, model={model!r}, "
            f"user_defined_name={user_name!r}. Run --list-cameras."
        )
    if len(matches) > 1:
        descriptions = [camera_info_dict(item) for item in matches]
        raise RuntimeError(
            "Camera match was ambiguous. Add a serial number to the YAML. "
            f"Matches: {descriptions}"
        )
    return matches[0]


def configure_camera(camera_cfg: dict[str, Any], device_info: Any) -> CameraBinding:
    require_pypylon()
    factory = pylon.TlFactory.GetInstance()
    camera = pylon.InstantCamera(factory.CreateDevice(device_info))
    camera.Open()
    info = camera_info_dict(camera.GetDeviceInfo())
    label = sanitize_token(str(camera_cfg.get("label") or info["serial"] or info["model"]))
    LOG.info("Configuring %s: model=%s serial=%s", label, info["model"], info["serial"])

    # Return to a known state if requested. Off by default to avoid unexpectedly
    # overwriting a user set stored in the camera.
    if bool(camera_cfg.get("load_default_user_set", False)):
        try_set(camera, "UserSetSelector", "Default")
        node = get_node(camera, "UserSetLoad")
        if node is not None:
            try:
                node.Execute()
            except Exception as exc:
                LOG.warning("Could not load default user set on %s: %s", label, exc)

    # Force free-running continuous acquisition unless a future configuration
    # explicitly implements hardware triggering. This prevents a stored camera
    # user set from silently leaving FrameStart triggering enabled.
    try_set(camera, "AcquisitionMode", "Continuous")
    try_set(camera, "ExposureMode", "Timed")
    for trigger_selector in ("FrameStart", "AcquisitionStart"):
        selected, _ = try_set(camera, "TriggerSelector", trigger_selector)
        if selected:
            try_set(camera, "TriggerMode", "Off")
            break

    # Binning changes full-sensor spatial sampling while preserving field of view.
    # Unsupported nodes are logged and skipped.
    binning = int(camera_cfg.get("binning", 1) or 1)
    if binning < 1:
        raise ValueError("camera binning must be >= 1")
    if binning > 1:
        horizontal_ok, _ = try_set(camera, "BinningHorizontal", binning)
        vertical_ok, _ = try_set(camera, "BinningVertical", binning)
        if not horizontal_ok or not vertical_ok:
            LOG.warning(
                "%s does not support the requested %dx%d camera binning; "
                "recording will continue at the camera ROI resolution",
                label,
                binning,
                binning,
            )
        if camera_cfg.get("binning_mode"):
            try_set(camera, "BinningHorizontalMode", str(camera_cfg["binning_mode"]))
            try_set(camera, "BinningVerticalMode", str(camera_cfg["binning_mode"]))

    # Reset offsets before enlarging the ROI, then apply requested dimensions and
    # final offsets. Width/height values are aligned to each node's valid increment.
    if camera_cfg.get("width") is not None or camera_cfg.get("height") is not None:
        try_set(camera, "OffsetX", 0)
        try_set(camera, "OffsetY", 0)
    for name, key in (("Width", "width"), ("Height", "height")):
        if camera_cfg.get(key) is not None:
            try_set(camera, name, int(camera_cfg[key]), required=True)
    for name, key in (("OffsetX", "offset_x"), ("OffsetY", "offset_y")):
        if camera_cfg.get(key) is not None:
            try_set(camera, name, int(camera_cfg[key]))

    pixel_format = str(camera_cfg.get("pixel_format") or "auto")
    if pixel_format.lower() == "auto":
        selected = first_settable_enum(
            camera,
            "PixelFormat",
            ("BayerRG8", "BayerBG8", "BayerGR8", "BayerGB8", "RGB8", "BGR8"),
        )
        if selected is None:
            LOG.warning("No preferred 8-bit color pixel format could be selected for %s", label)
    else:
        try_set(camera, "PixelFormat", pixel_format, required=True)

    # Stable brightness is preferable for behavioral segmentation. Auto modes are
    # available for setup, but manual exposure/gain should be used for the real run.
    auto_exposure = bool(camera_cfg.get("auto_exposure", False))
    if auto_exposure:
        first_settable_enum(camera, "ExposureAuto", ("Continuous", "Once"))
    else:
        try_set(camera, "ExposureAuto", "Off")
        if camera_cfg.get("exposure_us") is not None:
            exposure_ok, _ = try_set(camera, "ExposureTime", float(camera_cfg["exposure_us"]))
            if not exposure_ok:
                try_set(camera, "ExposureTimeAbs", float(camera_cfg["exposure_us"]))

    auto_gain = bool(camera_cfg.get("auto_gain", False))
    if auto_gain:
        first_settable_enum(camera, "GainAuto", ("Continuous", "Once"))
    else:
        try_set(camera, "GainAuto", "Off")
        if camera_cfg.get("gain") is not None:
            gain_ok, _ = try_set(camera, "Gain", float(camera_cfg["gain"]))
            if not gain_ok:
                try_set(camera, "GainRaw", int(camera_cfg["gain"]))

    if camera_cfg.get("balance_white_auto") is not None:
        mode = "Continuous" if bool(camera_cfg["balance_white_auto"]) else "Off"
        try_set(camera, "BalanceWhiteAuto", mode)

    fps = float(camera_cfg.get("fps", 5.0))
    try_set(camera, "AcquisitionFrameRateEnable", True)
    frame_rate_ok, actual_fps = try_set(camera, "AcquisitionFrameRate", fps)
    if not frame_rate_ok:
        frame_rate_ok, actual_fps = try_set(camera, "AcquisitionFrameRateAbs", fps)
    if not frame_rate_ok:
        LOG.warning("Could not set an acquisition frame rate on %s; using camera default", label)

    if camera_cfg.get("device_link_throughput_limit") is not None:
        try_set(
            camera,
            "DeviceLinkThroughputLimit",
            int(camera_cfg["device_link_throughput_limit"]),
        )

    max_buffers = int(camera_cfg.get("max_num_buffer", 20) or 20)
    try_set(camera, "MaxNumBuffer", max_buffers)

    chunk_enabled = enable_chunk_timestamp(camera)

    actual_names = (
        "Width",
        "Height",
        "OffsetX",
        "OffsetY",
        "BinningHorizontal",
        "BinningVertical",
        "PixelFormat",
        "ExposureAuto",
        "ExposureTime",
        "ExposureTimeAbs",
        "GainAuto",
        "Gain",
        "GainRaw",
        "AcquisitionFrameRate",
        "AcquisitionFrameRateAbs",
        "ResultingFrameRate",
        "ResultingFrameRateAbs",
        "DeviceLinkThroughputLimit",
        "TimestampTickFrequency",
        "GevTimestampTickFrequency",
        "TriggerMode",
        "AcquisitionMode",
    )
    actual = {name: read_setting(camera, name) for name in actual_names}
    if actual.get("AcquisitionFrameRate") is None:
        actual["AcquisitionFrameRate"] = actual.get("AcquisitionFrameRateAbs") or actual_fps or fps
    actual["chunk_timestamp_enabled"] = chunk_enabled

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    return CameraBinding(
        label=label,
        requested=dict(camera_cfg),
        camera=camera,
        info=info,
        actual_settings=actual,
        converter=converter,
    )


def close_bindings(bindings: Iterable[CameraBinding]) -> None:
    for binding in bindings:
        try:
            if binding.camera.IsGrabbing():
                binding.camera.StopGrabbing()
        except Exception:
            pass
        try:
            if binding.camera.IsOpen():
                binding.camera.Close()
        except Exception:
            pass
        try:
            binding.camera.DestroyDevice()
        except Exception:
            pass


class FFmpegWriter:
    def __init__(
        self,
        ffmpeg: str,
        temp_path: Path,
        final_path: Path,
        width: int,
        height: int,
        fps: float,
        encoding_cfg: dict[str, Any],
    ) -> None:
        self.ffmpeg = ffmpeg
        self.temp_path = temp_path
        self.final_path = final_path
        self.width = width
        self.height = height
        self.fps = fps
        self.encoding_cfg = encoding_cfg
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.stderr_path = final_path.with_name(f"{final_path.stem}.ffmpeg.log")
        self.stderr_handle: Any = None

    def start(self) -> None:
        codec = str(self.encoding_cfg.get("codec", "libx264"))
        preset = str(self.encoding_cfg.get("preset", "veryfast"))
        crf = int(self.encoding_cfg.get("crf", 23))
        gop_seconds = float(self.encoding_cfg.get("gop_seconds", 2.0))
        gop = max(1, round(self.fps * gop_seconds))

        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            f"{self.fps:.9g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            codec,
        ]
        if codec in {"libx264", "libx265"}:
            command += ["-preset", preset, "-crf", str(crf)]
        elif self.encoding_cfg.get("bitrate"):
            command += ["-b:v", str(self.encoding_cfg["bitrate"])]
        command += [
            "-g",
            str(gop),
            "-pix_fmt",
            "yuv420p",
            str(self.temp_path),
        ]

        self.stderr_handle = self.stderr_path.open("wb")
        LOG.debug("Starting FFmpeg: %s", " ".join(command))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self.stderr_handle,
            bufsize=0,
        )

    def write(self, frame: np.ndarray) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("FFmpeg writer was not started")
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Frame size changed from {self.width}x{self.height} to "
                f"{frame.shape[1]}x{frame.shape[0]}"
            )
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(f"FFmpeg stopped while writing {self.temp_path}") from exc

    def close_and_remux(self, *, keep_temp: bool = False) -> tuple[int, bool]:
        if self.process is None:
            return -1, False
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except Exception:
                pass
        return_code = self.process.wait()
        if self.stderr_handle is not None:
            self.stderr_handle.close()
        if return_code != 0:
            LOG.error("FFmpeg encoding failed for %s; see %s", self.temp_path, self.stderr_path)
            return return_code, False

        # MKV is resilient during capture. Remuxing is fast and does not recompress.
        remux_log = self.final_path.with_name(f"{self.final_path.stem}.remux.log")
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(self.temp_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(self.final_path),
        ]
        with remux_log.open("wb") as log_handle:
            remux = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=log_handle, check=False)
        if remux.returncode == 0:
            if not keep_temp:
                self.temp_path.unlink(missing_ok=True)
            if remux_log.stat().st_size == 0:
                remux_log.unlink(missing_ok=True)
            if self.stderr_path.exists() and self.stderr_path.stat().st_size == 0:
                self.stderr_path.unlink(missing_ok=True)
            return return_code, True

        LOG.error(
            "MP4 remux failed for %s. The recoverable MKV was kept at %s; see %s",
            self.final_path,
            self.temp_path,
            remux_log,
        )
        return return_code, False


@dataclasses.dataclass
class ClipResult:
    label: str
    success: bool
    metadata_path: Path
    video_path: Optional[Path]
    error: Optional[str] = None


def bytes_to_gib(value: int) -> float:
    return value / float(1024**3)


def tree_stats(path: Path) -> tuple[int, int]:
    """Return recursive file count and byte total for a directory tree."""

    if not path.exists():
        return 0, 0
    file_count = 0
    total_bytes = 0
    for candidate in path.rglob("*"):
        if candidate.is_file():
            file_count += 1
            try:
                total_bytes += candidate.stat().st_size
            except FileNotFoundError:
                continue
    return file_count, total_bytes


def paths_overlap(a: Path, b: Path) -> bool:
    try:
        resolved_a = a.resolve()
        resolved_b = b.resolve()
    except Exception:
        return False
    return (
        resolved_a == resolved_b
        or resolved_a in resolved_b.parents
        or resolved_b in resolved_a.parents
    )


def read_json_mapping(path: Path) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        return None, f"could not read JSON: {exc}"
    if not isinstance(data, dict):
        return None, "JSON root is not an object"
    return data, None


def atomic_write_json(path: Path, payload: Any) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def copy_file_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def flush_logging_handlers() -> None:
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def remove_capture_files(clip_dir: Path) -> list[Path]:
    removed: list[Path] = []
    for candidate in sorted(clip_dir.glob("*.capture.mkv")):
        try:
            candidate.unlink()
            removed.append(candidate)
        except FileNotFoundError:
            continue
    return removed


def clip_directory_ready_for_archive(
    clip_dir: Path,
    expected_camera_count: int,
    max_clip_size_bytes: int,
) -> tuple[bool, list[str], int]:
    issues: list[str] = []
    json_files = sorted(clip_dir.glob("*.json"))
    mp4_files = sorted(clip_dir.glob("*.mp4"))
    timestamp_files = sorted(clip_dir.glob("*.timestamps.csv.gz"))
    temp_files = sorted(clip_dir.glob("*.capture.mkv"))

    if len(json_files) != expected_camera_count:
        issues.append(
            f"expected {expected_camera_count} JSON sidecars, found {len(json_files)}"
        )
    if len(mp4_files) != expected_camera_count:
        issues.append(f"expected {expected_camera_count} MP4 files, found {len(mp4_files)}")
    if len(timestamp_files) != expected_camera_count:
        issues.append(
            f"expected {expected_camera_count} timestamp sidecars, found {len(timestamp_files)}"
        )
    if temp_files:
        issues.append(f"temporary capture MKV files remain: {len(temp_files)}")

    total_files, total_bytes = tree_stats(clip_dir)
    if total_bytes > max_clip_size_bytes:
        issues.append(
            f"clip directory size {bytes_to_gib(total_bytes):.1f} GiB exceeds the cap "
            f"of {bytes_to_gib(max_clip_size_bytes):.1f} GiB"
        )

    for json_path in json_files:
        metadata, error = read_json_mapping(json_path)
        if error is not None:
            issues.append(f"{json_path.name}: {error}")
            continue

        mp4_path = json_path.with_suffix(".mp4")
        timestamps_path = json_path.with_suffix(".timestamps.csv.gz")
        if not mp4_path.exists():
            issues.append(f"{json_path.name}: missing MP4 {mp4_path.name}")
        elif mp4_path.stat().st_size <= 0:
            issues.append(f"{json_path.name}: MP4 file is empty")
        if not timestamps_path.exists():
            issues.append(f"{json_path.name}: missing timestamps {timestamps_path.name}")

        if metadata.get("success") is not True:
            issues.append(f"{json_path.name}: success is not true")
        if metadata.get("grab_failures") not in (0, "0"):
            issues.append(f"{json_path.name}: grab_failures is {metadata.get('grab_failures')!r}")
        if metadata.get("mp4_remux_succeeded") is not True:
            issues.append(
                f"{json_path.name}: mp4_remux_succeeded is {metadata.get('mp4_remux_succeeded')!r}"
            )

    return not issues, issues, total_bytes


def resolve_executable(configured: str) -> Optional[str]:
    candidate = Path(configured).expanduser()
    if candidate.exists():
        return str(candidate)
    return shutil.which(configured)


def preflight_archive_settings(
    settings: ArchiveSettings,
    *,
    local_output_root: Path,
    session_dir: Path,
    project: str,
    subject: str,
    session_name: str,
) -> ArchivePreflightResult:
    errors: list[str] = []
    destination_root = settings.destination_root.expanduser()
    required_mount_point = settings.required_mount_point.expanduser()
    archive_session_dir = destination_root / project / subject / session_name

    rsync_path = resolve_executable(settings.rsync_executable)
    if rsync_path is None:
        errors.append(f"rsync executable was not found: {settings.rsync_executable}")

    required_mount_is_mount = required_mount_point.exists() and os.path.ismount(required_mount_point)
    if not required_mount_point.exists():
        errors.append(f"required mount point does not exist: {required_mount_point}")
    elif not required_mount_is_mount:
        errors.append(f"required mount point is not a mounted filesystem: {required_mount_point}")

    destination_created = False
    destination_writable = False
    destination_free_gb: Optional[float] = None
    local_free_gb: Optional[float] = None
    path_conflict = paths_overlap(destination_root, local_output_root)

    if path_conflict:
        errors.append(
            f"archive destination {destination_root} conflicts with local output root {local_output_root}"
        )

    if required_mount_is_mount and not path_conflict:
        try:
            destination_root.mkdir(parents=True, exist_ok=True)
            destination_created = True
        except Exception as exc:
            errors.append(f"could not create archive destination root {destination_root}: {exc}")

    if destination_root.exists():
        try:
            destination_free_gb = bytes_to_gib(shutil.disk_usage(destination_root).free)
            if destination_free_gb < settings.min_external_free_gb_before_transfer:
                errors.append(
                    "archive destination free space "
                    f"{destination_free_gb:.1f} GiB is below the minimum "
                    f"{settings.min_external_free_gb_before_transfer:.1f} GiB"
                )
        except Exception as exc:
            errors.append(f"could not read archive destination disk usage: {exc}")

        probe_path = destination_root / f".archive_write_test_{os.getpid()}_{int(time.time())}"
        try:
            with probe_path.open("w", encoding="utf-8") as handle:
                handle.write("ok\n")
            probe_path.unlink()
            destination_writable = True
        except Exception as exc:
            errors.append(f"archive destination is not writable: {exc}")

    try:
        local_free_gb = bytes_to_gib(shutil.disk_usage(session_dir).free)
        if local_free_gb < settings.min_local_free_gb_before_clip:
            errors.append(
                f"local free space {local_free_gb:.1f} GiB is below the minimum "
                f"{settings.min_local_free_gb_before_clip:.1f} GiB"
            )
    except Exception as exc:
        errors.append(f"could not read local disk usage: {exc}")

    if destination_root.exists() and local_output_root.exists():
        try:
            if paths_overlap(destination_root, local_output_root):
                if destination_root.resolve() == local_output_root.resolve():
                    errors.append("archive destination root and local output root are the same path")
        except Exception:
            pass

    return ArchivePreflightResult(
        enabled=settings.enabled,
        ok=not errors,
        errors=errors,
        rsync_path=rsync_path,
        required_mount_point=str(required_mount_point),
        required_mount_is_mount=required_mount_is_mount,
        destination_root=str(destination_root),
        destination_created=destination_created,
        destination_writable=destination_writable,
        local_free_gb=local_free_gb,
        destination_free_gb=destination_free_gb,
        path_conflict=path_conflict,
        archive_session_dir=str(archive_session_dir),
    )


@dataclasses.dataclass
class ArchiveResult:
    clip_name: str
    source_path: str
    destination_path: str
    success: bool
    bytes_transferred: Optional[int]
    started_utc: str
    completed_utc: Optional[str]
    rsync_return_code: Optional[int]
    verification_succeeded: bool
    local_deleted: bool
    error: Optional[str]


@dataclasses.dataclass
class _ArchiveRequest:
    clip_dir: Path
    done: threading.Event = dataclasses.field(default_factory=threading.Event)
    result: Optional[ArchiveResult] = None


class ArchiveManager:
    def __init__(
        self,
        settings: ArchiveSettings,
        local_session_dir: Path,
        project: str,
        subject: str,
        session_name: str,
        stop_event: threading.Event,
        preflight: ArchivePreflightResult,
    ) -> None:
        self.settings = settings
        self.local_session_dir = local_session_dir
        self.project = project
        self.subject = subject
        self.session_name = session_name
        self.stop_event = stop_event
        self.preflight = preflight
        self.archive_session_dir = Path(settings.destination_root) / project / subject / session_name
        self.incoming_dir = self.archive_session_dir / ".incoming"
        self.rsync_path = preflight.rsync_path or resolve_executable(settings.rsync_executable)
        self.local_transfers_path = local_session_dir / "archive_transfers.jsonl"
        self.local_summary_path = local_session_dir / "archive_summary.json"
        self.archive_transfers_path = self.archive_session_dir / "archive_transfers.jsonl"
        self.archive_summary_path = self.archive_session_dir / "archive_summary.json"
        self._queue: queue.Queue[Optional[_ArchiveRequest]] = queue.Queue()
        self._worker = threading.Thread(target=self._worker_main, name="archive-worker", daemon=False)
        self._lock = threading.Lock()
        self._started = False
        self._closed = False
        self._failure = False
        self._pending_local_clips = 0
        self._enqueued = 0
        self._processed = 0
        self._successful = 0
        self._failed = 0
        self._already_complete = 0
        self._bytes_transferred = 0
        self._results: list[ArchiveResult] = []

    def start(self) -> None:
        if not self.settings.enabled or self._started:
            return
        if not self.preflight.ok:
            raise RuntimeError("Archive preflight failed; cannot start archive manager")

        self.archive_session_dir.mkdir(parents=True, exist_ok=True)
        self.incoming_dir.mkdir(parents=True, exist_ok=True)

        for filename in ("config_used.yaml", "session_manifest.json"):
            source = self.local_session_dir / filename
            destination = self.archive_session_dir / filename
            if not source.exists():
                raise FileNotFoundError(f"Missing required session file for archive: {source}")
            shutil.copy2(source, destination)

        self._worker.start()
        self._started = True
        LOG.info("Archive session directory: %s", self.archive_session_dir)

    def enqueue_clip(self, clip_dir: Path) -> None:
        if not self.settings.enabled:
            return
        if not self._started:
            raise RuntimeError("ArchiveManager.start() must be called before enqueue_clip()")
        if self._closed:
            raise RuntimeError("ArchiveManager is closed")
        request = _ArchiveRequest(clip_dir=clip_dir)
        with self._lock:
            self._pending_local_clips += 1
            self._enqueued += 1
        self._queue.put(request)
        if not self.settings.background_transfer:
            request.done.wait()
            if request.result is not None and not request.result.success:
                self._mark_failure(request.result.error or "archive transfer failed")

    def failure_detected(self) -> bool:
        return self._failure

    def mark_failure(self, error: str) -> None:
        self._mark_failure(error)

    def unarchived_count(self) -> int:
        with self._lock:
            return self._pending_local_clips

    def wait_until_idle(self) -> None:
        if self.settings.enabled and self._started:
            self._queue.join()

    def close(self) -> None:
        if not self.settings.enabled or self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._queue.join()
        if self._worker.is_alive():
            self._worker.join(timeout=30)

        summary = self._build_summary()
        atomic_write_json(self.local_summary_path, summary)
        flush_logging_handlers()
        self._copy_final_metadata()

    def _copy_final_metadata(self) -> None:
        for filename in (
            "config_used.yaml",
            "session_manifest.json",
            "session_summary.json",
            "recorder.log",
            "archive_transfers.jsonl",
            "archive_summary.json",
        ):
            copy_file_if_exists(self.local_session_dir / filename, self.archive_session_dir / filename)

    def _build_summary(self) -> dict[str, Any]:
        return {
            "enabled": self.settings.enabled,
            "preflight": dataclasses.asdict(self.preflight),
            "archive_session_dir": str(self.archive_session_dir),
            "queued_clips": self._enqueued,
            "processed_clips": self._processed,
            "successful_clips": self._successful,
            "failed_clips": self._failed,
            "already_complete_clips": self._already_complete,
            "bytes_transferred": self._bytes_transferred,
            "failure_detected": self._failure,
            "results": [dataclasses.asdict(result) for result in self._results],
            "completed_utc": utc_now().isoformat(),
        }

    def _append_transfer_record(self, result: ArchiveResult) -> None:
        self.local_transfers_path.parent.mkdir(parents=True, exist_ok=True)
        with self.local_transfers_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dataclasses.asdict(result), default=str))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _mark_failure(self, error: str) -> None:
        if not self._failure:
            LOG.error("Archive failure detected: %s", error)
        self._failure = True
        if self.settings.stop_before_next_clip_on_transfer_failure:
            self.stop_event.set()

    def _verify_tree(self, source: Path, destination: Path) -> tuple[bool, str]:
        source_files, source_bytes = tree_stats(source)
        dest_files, dest_bytes = tree_stats(destination)
        if source_files != dest_files:
            return False, f"file count mismatch: source={source_files} destination={dest_files}"
        if source_bytes != dest_bytes:
            return False, f"byte count mismatch: source={source_bytes} destination={dest_bytes}"

        verify_command = [
            self.rsync_path or "rsync",
            "-a",
            "--checksum",
            "--dry-run",
            "--itemize-changes",
            f"{source}/",
            f"{destination}/",
        ]
        verify = subprocess.run(
            verify_command,
            capture_output=True,
            text=True,
            check=False,
        )
        if verify.returncode != 0:
            return False, f"verification rsync exited with code {verify.returncode}"
        if verify.stdout.strip():
            return False, "verification rsync reported differences"
        return True, ""

    def _transfer_clip(self, clip_dir: Path) -> ArchiveResult:
        started_utc = utc_now().isoformat()
        clip_name = clip_dir.name
        final_destination = self.archive_session_dir / clip_name
        partial_destination = self.incoming_dir / f"{clip_name}.partial"
        bytes_transferred: Optional[int] = None
        completed_utc: Optional[str] = None
        rsync_return_code: Optional[int] = None
        verification_succeeded = False
        local_deleted = False
        error: Optional[str] = None
        success = False

        try:
            if not clip_dir.exists():
                raise FileNotFoundError(f"Source clip directory does not exist: {clip_dir}")

            dest_free_gb = bytes_to_gib(shutil.disk_usage(self.archive_session_dir).free)
            if dest_free_gb < self.settings.min_external_free_gb_before_transfer:
                raise RuntimeError(
                    "archive destination free space "
                    f"{dest_free_gb:.1f} GiB is below the minimum "
                    f"{self.settings.min_external_free_gb_before_transfer:.1f} GiB"
                )

            source_files, source_bytes = tree_stats(clip_dir)
            bytes_transferred = source_bytes

            if final_destination.exists():
                verified, reason = self._verify_tree(clip_dir, final_destination)
                if not verified:
                    raise RuntimeError(
                        f"destination already exists but does not match source: {reason}"
                    )
                verification_succeeded = True
                if partial_destination.exists():
                    LOG.warning("Removing stale partial archive directory %s", partial_destination)
                    shutil.rmtree(partial_destination, ignore_errors=True)
                success = True
                self._already_complete += 1
                self._successful += 1
                self._bytes_transferred += bytes_transferred or 0
            else:
                partial_destination.parent.mkdir(parents=True, exist_ok=True)
                if partial_destination.exists():
                    LOG.warning("Resuming or replacing stale partial archive directory %s", partial_destination)
                copy_command = [
                    self.rsync_path or "rsync",
                    "-a",
                    "--partial",
                    f"{clip_dir}/",
                    f"{partial_destination}/",
                ]
                copy = subprocess.run(copy_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
                rsync_return_code = copy.returncode
                if copy.returncode != 0:
                    raise RuntimeError(
                        f"rsync returned {copy.returncode}: {copy.stderr.strip() or 'no stderr output'}"
                    )

                verified, reason = self._verify_tree(clip_dir, partial_destination)
                if not verified:
                    raise RuntimeError(f"verification failed: {reason}")
                verification_succeeded = True

                if final_destination.exists():
                    raise RuntimeError(
                        f"final archive destination appeared during transfer: {final_destination}"
                    )
                partial_destination.rename(final_destination)
                success = True
                self._successful += 1
                self._bytes_transferred += bytes_transferred or 0

            if success and self.settings.delete_local_clip_after_verified_transfer:
                shutil.rmtree(clip_dir)
                local_deleted = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._failed += 1
            self._mark_failure(error)
            LOG.exception("Archive failed for %s", clip_dir)
        else:
            completed_utc = utc_now().isoformat()
        finally:
            if success and verification_succeeded and not local_deleted and self.settings.delete_local_clip_after_verified_transfer:
                try:
                    shutil.rmtree(clip_dir)
                    local_deleted = True
                except Exception as exc:
                    error = f"Could not delete local clip directory after archive verification: {exc}"
                    success = False
                    self._failed += 1
                    self._mark_failure(error)
                    LOG.exception("Archive cleanup failed for %s", clip_dir)
            if success and final_destination.exists() and partial_destination.exists():
                shutil.rmtree(partial_destination, ignore_errors=True)
            self._processed += 1
            if success:
                self._pending_local_clips = max(0, self._pending_local_clips - 1)
            result = ArchiveResult(
                clip_name=clip_name,
                source_path=str(clip_dir),
                destination_path=str(final_destination),
                success=success,
                bytes_transferred=bytes_transferred,
                started_utc=started_utc,
                completed_utc=completed_utc,
                rsync_return_code=rsync_return_code,
                verification_succeeded=verification_succeeded,
                local_deleted=local_deleted,
                error=error,
            )
            self._results.append(result)
            self._append_transfer_record(result)
        return result

    def _worker_main(self) -> None:
        while True:
            request = self._queue.get()
            try:
                if request is None:
                    return
                request.result = self._transfer_clip(request.clip_dir)
            except Exception as exc:
                result = ArchiveResult(
                    clip_name=request.clip_dir.name if request is not None else "unknown",
                    source_path=str(request.clip_dir) if request is not None else "",
                    destination_path=str(self.archive_session_dir / request.clip_dir.name) if request is not None else "",
                    success=False,
                    bytes_transferred=None,
                    started_utc=utc_now().isoformat(),
                    completed_utc=None,
                    rsync_return_code=None,
                    verification_succeeded=False,
                    local_deleted=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if request is not None:
                    request.result = result
                self._failed += 1
                self._mark_failure(result.error or "archive worker failed")
                self._results.append(result)
                self._append_transfer_record(result)
            finally:
                if request is not None:
                    request.done.set()
                self._queue.task_done()


def draw_recording_preview(packet: PreviewPacket, settings: RecordingPreviewSettings) -> np.ndarray:
    """Create a display-only copy with a compact recording status overlay."""

    display = packet.frame.copy()
    if not settings.show_status:
        return display

    font = cv2.FONT_HERSHEY_SIMPLEX
    status_font_scale = 0.48
    controls_font_scale = 0.34
    text_thickness = 1
    controls_text = "q hide  |  s snapshot"
    fps_text = (
        f"{packet.measured_receive_fps:.2f} fps"
        if packet.measured_receive_fps is not None
        else "starting"
    )
    elapsed_text = format_clock_duration(packet.elapsed_s)
    total_text = format_clock_duration(packet.planned_duration_s)
    progress = 0.0
    if packet.planned_duration_s > 0:
        progress = max(0.0, min(1.0, packet.elapsed_s / packet.planned_duration_s))
    lines = [
        f"REC | {packet.label} | clip {packet.clip_index:04d}",
        f"{elapsed_text} / {total_text} | frame {packet.frame_index + 1} | {fps_text}",
    ]
    (controls_width, _), _ = cv2.getTextSize(
        controls_text,
        font,
        controls_font_scale,
        text_thickness,
    )

    overlay = display.copy()
    panel_height = 72
    cv2.rectangle(
        overlay,
        (0, 0),
        (display.shape[1], panel_height),
        (24, 24, 24),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

    first_line_y = 24
    line_spacing = 22
    y = first_line_y
    for line in lines:
        cv2.putText(
            display,
            line,
            (12, y),
            font,
            status_font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
        y += line_spacing

    controls_x = max(12, display.shape[1] - controls_width - 12)
    cv2.putText(
        display,
        controls_text,
        (controls_x, 24),
        font,
        controls_font_scale,
        (235, 235, 235),
        text_thickness,
        cv2.LINE_AA,
    )

    bar_left = 12
    bar_top = 57
    bar_width = max(60, display.shape[1] - 24)
    bar_height = 7
    cv2.rectangle(
        display,
        (bar_left, bar_top),
        (bar_left + bar_width, bar_top + bar_height),
        (82, 82, 82),
        -1,
    )
    filled_width = int(round(bar_width * progress))
    if filled_width > 0:
        cv2.rectangle(
            display,
            (bar_left, bar_top),
            (bar_left + filled_width, bar_top + bar_height),
            (90, 190, 255),
            -1,
        )
    return display


def destroy_preview_windows(window_names: Iterable[str]) -> None:
    for window_name in window_names:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass
    try:
        cv2.waitKey(1)
    except cv2.error:
        pass


def monitor_recording_threads(
    threads: list[threading.Thread],
    preview_queues: dict[str, queue.Queue[PreviewPacket]],
    preview_settings: RecordingPreviewSettings,
    preview_active_event: threading.Event,
    clip_dir: Path,
) -> None:
    """Keep recording windows responsive while camera workers acquire frames."""

    if not preview_active_event.is_set() or not preview_queues:
        for thread in threads:
            thread.join()
        return

    latest: dict[str, PreviewPacket] = {}
    window_names: set[str] = set()
    snapshot_count = 0

    try:
        while any(thread.is_alive() for thread in threads):
            updated_labels: set[str] = set()
            for label, preview_queue in preview_queues.items():
                newest: Optional[PreviewPacket] = None
                while True:
                    try:
                        newest = preview_queue.get_nowait()
                    except queue.Empty:
                        break
                if newest is not None:
                    latest[label] = newest
                    updated_labels.add(label)

            key = -1
            if preview_active_event.is_set() and (latest or window_names):
                try:
                    for label in updated_labels:
                        packet = latest[label]
                        window_name = f"Recording preview - {label}"
                        window_names.add(window_name)
                        cv2.imshow(window_name, draw_recording_preview(packet, preview_settings))
                    if window_names:
                        key = cv2.waitKey(20) & 0xFF
                    else:
                        time.sleep(0.02)
                except cv2.error as exc:
                    LOG.warning(
                        "Recording preview was disabled because OpenCV could not "
                        "open or update a window: %s",
                        exc,
                    )
                    preview_active_event.clear()
                    destroy_preview_windows(window_names)
            else:
                time.sleep(0.02)

            if key in (ord("q"), 27):
                LOG.info("Recording preview hidden; acquisition continues")
                preview_active_event.clear()
                destroy_preview_windows(window_names)
            elif key == ord("s") and latest:
                snapshot_count += 1
                snapshot_dir = clip_dir / "monitor_snapshots"
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                for packet in latest.values():
                    snapshot_path = snapshot_dir / (
                        f"{sanitize_token(packet.label)}_"
                        f"frame{packet.frame_index:08d}_"
                        f"{snapshot_stamp_utc(utc_now())}_{snapshot_count:03d}.png"
                    )
                    if cv2.imwrite(str(snapshot_path), packet.frame):
                        LOG.info("Saved monitor snapshot %s", snapshot_path)
                    else:
                        LOG.warning("Could not save monitor snapshot %s", snapshot_path)

        for thread in threads:
            thread.join()
    finally:
        destroy_preview_windows(window_names)


def create_preview_frame(binding: CameraBinding, timeout_ms: int = 3000) -> np.ndarray:
    camera = binding.camera
    camera.StartGrabbingMax(1)
    try:
        grab = camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
        try:
            if not grab.GrabSucceeded():
                raise RuntimeError(f"Grab failed: {grab.ErrorCode} {grab.ErrorDescription}")
            image = binding.converter.Convert(grab)
            frame = image.GetArray()
            return apply_frame_transform(frame, binding.requested)
        finally:
            grab.Release()
    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()


def record_one_camera(
    binding: CameraBinding,
    clip_index: int,
    clip_start_utc: dt.datetime,
    planned_start_mono_ns: int,
    planned_stop_mono_ns: int,
    clip_dir: Path,
    ffmpeg: str,
    encoding_cfg: dict[str, Any],
    ready_barrier: threading.Barrier,
    stop_event: threading.Event,
    storage_root: Path,
    clip_stop_threshold_bytes: Optional[int],
    result_queue: queue.Queue[ClipResult],
    preview_queue: Optional[queue.Queue[PreviewPacket]],
    preview_settings: RecordingPreviewSettings,
    preview_active_event: threading.Event,
) -> None:
    label = binding.label
    camera = binding.camera
    requested_fps = float(binding.requested.get("fps", binding.fps))
    actual_fps = float(binding.actual_settings.get("AcquisitionFrameRate") or requested_fps)
    file_stem = sanitize_token(label)
    final_path = clip_dir / f"{file_stem}.mp4"
    temp_path = clip_dir / f"{file_stem}.capture.mkv"
    timestamps_path = clip_dir / f"{file_stem}.timestamps.csv.gz"
    metadata_path = clip_dir / f"{file_stem}.json"

    writer: Optional[FFmpegWriter] = None
    timestamp_handle: Any = None
    csv_writer: Any = None
    frame_count = 0
    grab_failures = 0
    first_host_utc_ns: Optional[int] = None
    last_host_utc_ns: Optional[int] = None
    first_host_mono_ns: Optional[int] = None
    last_host_mono_ns: Optional[int] = None
    first_camera_timestamp: Any = None
    last_camera_timestamp: Any = None
    error_message: Optional[str] = None
    remuxed = False
    ffmpeg_return_code: Optional[int] = None
    output_width: Optional[int] = None
    output_height: Optional[int] = None
    preview_interval_ns = max(1, round(1e9 / preview_settings.fps))
    next_preview_mono_ns = planned_start_mono_ns
    preview_publish_failed = False
    last_storage_check_ns = planned_start_mono_ns

    metadata: dict[str, Any] = {
        "camera": binding.info,
        "label": label,
        "clip_index": clip_index,
        "planned_start_utc": clip_start_utc.isoformat(),
        "planned_duration_s": (planned_stop_mono_ns - planned_start_mono_ns) / 1e9,
        "requested_settings": binding.requested,
        "actual_settings": binding.actual_settings,
        "encoding": encoding_cfg,
        "recording_preview": dataclasses.asdict(preview_settings),
        "video_path": str(final_path),
        "temporary_video_path": str(temp_path),
        "timestamps_path": str(timestamps_path),
    }

    try:
        # Grab a single frame first so output dimensions are known before FFmpeg starts.
        preview = create_preview_frame(binding)
        output_height, output_width = preview.shape[:2]
        writer = FFmpegWriter(
            ffmpeg=ffmpeg,
            temp_path=temp_path,
            final_path=final_path,
            width=output_width,
            height=output_height,
            fps=actual_fps,
            encoding_cfg=encoding_cfg,
        )
        writer.start()

        timestamp_handle = gzip.open(timestamps_path, "wt", newline="", encoding="utf-8")
        csv_writer = csv.writer(timestamp_handle)
        csv_writer.writerow(
            [
                "frame_index",
                "host_utc_ns",
                "host_utc_iso",
                "host_monotonic_ns",
                "camera_timestamp",
                "block_id",
                "skipped_images",
            ]
        )

        camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        ready_barrier.wait(timeout=30)

        # Retrieve before the planned start to clear startup frames, then only encode
        # frames whose host receipt time falls inside the common clip window.
        while camera.IsGrabbing() and not stop_event.is_set():
            now_mono_ns = time.monotonic_ns()
            if now_mono_ns >= planned_stop_mono_ns:
                break
            try:
                grab = camera.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
            except Exception as exc:
                grab_failures += 1
                LOG.warning("%s retrieve failed: %s", label, exc)
                continue

            try:
                host_mono_ns = time.monotonic_ns()
                host_utc_ns = time.time_ns()
                if not grab.GrabSucceeded():
                    grab_failures += 1
                    LOG.warning(
                        "%s grab failed: code=%s description=%s",
                        label,
                        getattr(grab, "ErrorCode", ""),
                        getattr(grab, "ErrorDescription", ""),
                    )
                    continue
                if host_mono_ns < planned_start_mono_ns:
                    continue
                if host_mono_ns >= planned_stop_mono_ns:
                    break

                image = binding.converter.Convert(grab)
                frame = apply_frame_transform(image.GetArray(), binding.requested)
                writer.write(frame)

                camera_timestamp = get_grab_value(grab, ("ChunkTimestamp", "TimeStamp", "Timestamp"))
                block_id = get_grab_value(grab, ("BlockID", "ID"))
                skipped = get_grab_value(grab, ("GetNumberOfSkippedImages", "NumberOfSkippedImages"))
                csv_writer.writerow(
                    [
                        frame_count,
                        host_utc_ns,
                        iso_utc_from_ns(host_utc_ns),
                        host_mono_ns,
                        camera_timestamp,
                        block_id,
                        skipped,
                    ]
                )

                if first_host_utc_ns is None:
                    first_host_utc_ns = host_utc_ns
                    first_host_mono_ns = host_mono_ns
                    first_camera_timestamp = camera_timestamp
                last_host_utc_ns = host_utc_ns
                last_host_mono_ns = host_mono_ns
                last_camera_timestamp = camera_timestamp
                frame_count += 1

                if (
                    preview_queue is not None
                    and preview_active_event.is_set()
                    and not preview_publish_failed
                    and host_mono_ns >= next_preview_mono_ns
                ):
                    try:
                        measured_preview_fps = None
                        if (
                            first_host_mono_ns is not None
                            and host_mono_ns > first_host_mono_ns
                            and frame_count > 1
                        ):
                            measured_preview_fps = (frame_count - 1) / (
                                (host_mono_ns - first_host_mono_ns) / 1e9
                            )
                        monitor_frame = resize_frame_to_fit(
                            frame,
                            preview_settings.max_width,
                            preview_settings.max_height,
                        )
                        put_latest_preview(
                            preview_queue,
                            PreviewPacket(
                                label=label,
                                clip_index=clip_index,
                                frame_index=frame_count - 1,
                                frame=monitor_frame,
                                host_monotonic_ns=host_mono_ns,
                                elapsed_s=max(0.0, (host_mono_ns - planned_start_mono_ns) / 1e9),
                                planned_duration_s=(planned_stop_mono_ns - planned_start_mono_ns) / 1e9,
                                measured_receive_fps=measured_preview_fps,
                            ),
                        )
                        next_preview_mono_ns = host_mono_ns + preview_interval_ns
                    except Exception as exc:
                        # Preview is monitoring-only. Never allow it to abort or
                        # delay the scientific recording.
                        preview_publish_failed = True
                        LOG.warning(
                            "%s recording preview disabled after an internal error: %s",
                            label,
                            exc,
                        )

                if clip_stop_threshold_bytes and host_mono_ns - last_storage_check_ns >= 5_000_000_000:
                    last_storage_check_ns = host_mono_ns
                    try:
                        current_size = temp_path.stat().st_size if temp_path.exists() else 0
                    except Exception:
                        current_size = 0
                    try:
                        local_free_bytes = shutil.disk_usage(storage_root).free
                    except Exception:
                        local_free_bytes = None

                    if current_size >= clip_stop_threshold_bytes:
                        error_message = (
                            f"Clip size guard triggered for {label}: "
                            f"temporary MKV reached {bytes_to_gib(current_size):.1f} GiB, "
                            f"threshold is {bytes_to_gib(clip_stop_threshold_bytes):.1f} GiB"
                        )
                        LOG.error(error_message)
                        stop_event.set()
                        break
                    if local_free_bytes is not None and local_free_bytes < 20 * 1024**3:
                        error_message = (
                            f"Local free space on {storage_root} dropped below 20 GiB "
                            f"while recording {label}"
                        )
                        LOG.error(error_message)
                        stop_event.set()
                        break
            finally:
                grab.Release()

        if camera.IsGrabbing():
            camera.StopGrabbing()
        if timestamp_handle is not None:
            timestamp_handle.flush()
            timestamp_handle.close()
            timestamp_handle = None

        ffmpeg_return_code, remuxed = writer.close_and_remux(keep_temp=True)
        writer = None
        if ffmpeg_return_code != 0:
            raise RuntimeError(f"FFmpeg exited with code {ffmpeg_return_code}")
        if not remuxed:
            raise RuntimeError(
                "H.264 capture succeeded, but MP4 remux failed. "
                f"The recoverable MKV was kept at {temp_path}."
            )
        if error_message is None and stop_event.is_set():
            error_message = "Recording stopped before the planned clip end"

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        LOG.exception("Recording failed for %s", label)
        stop_event.set()
    finally:
        try:
            if camera.IsGrabbing():
                camera.StopGrabbing()
        except Exception:
            pass
        if timestamp_handle is not None:
            try:
                timestamp_handle.close()
            except Exception:
                pass
        if writer is not None and writer.process is not None:
            try:
                if writer.process.stdin is not None:
                    writer.process.stdin.close()
                writer.process.terminate()
                writer.process.wait(timeout=5)
            except Exception:
                try:
                    writer.process.kill()
                except Exception:
                    pass
            if writer.stderr_handle is not None:
                try:
                    writer.stderr_handle.close()
                except Exception:
                    pass

        actual_elapsed_s = None
        measured_fps = None
        if first_host_mono_ns is not None and last_host_mono_ns is not None:
            actual_elapsed_s = max(0.0, (last_host_mono_ns - first_host_mono_ns) / 1e9)
            if actual_elapsed_s > 0 and frame_count > 1:
                measured_fps = (frame_count - 1) / actual_elapsed_s

        metadata.update(
            {
                "success": error_message is None,
                "error": error_message,
                "frame_count": frame_count,
                "grab_failures": grab_failures,
                "output_width": output_width,
                "output_height": output_height,
                "first_host_utc_ns": first_host_utc_ns,
                "first_host_utc": iso_utc_from_ns(first_host_utc_ns) if first_host_utc_ns else None,
                "last_host_utc_ns": last_host_utc_ns,
                "last_host_utc": iso_utc_from_ns(last_host_utc_ns) if last_host_utc_ns else None,
                "first_host_monotonic_ns": first_host_mono_ns,
                "last_host_monotonic_ns": last_host_mono_ns,
                "first_camera_timestamp": first_camera_timestamp,
                "last_camera_timestamp": last_camera_timestamp,
                "actual_elapsed_s": actual_elapsed_s,
                "measured_receive_fps": measured_fps,
                "ffmpeg_return_code": ffmpeg_return_code,
                "mp4_remux_succeeded": remuxed,
                "completed_utc": utc_now().isoformat(),
            }
        )
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, default=str)

        result_queue.put(
            ClipResult(
                label=label,
                success=error_message is None,
                metadata_path=metadata_path,
                video_path=final_path if final_path.exists() else (temp_path if temp_path.exists() else None),
                error=error_message,
            )
        )


def preview_camera(config: dict[str, Any], label: str) -> int:
    require_pypylon()
    camera_cfgs = config.get("cameras")
    if not isinstance(camera_cfgs, list):
        raise ValueError("config.cameras must be a list")
    selected = [item for item in camera_cfgs if str(item.get("label")) == label]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one camera with label={label!r}")

    devices = enumerate_devices()
    used: set[str] = set()
    device = match_device(selected[0], devices, used)
    binding = configure_camera(selected[0], device)
    window = f"Basler preview - {binding.label} (q quit, s snapshot, p print settings)"
    snapshot_count = 0
    try:
        camera = binding.camera
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        last_time = time.monotonic()
        displayed_fps = 0.0
        while camera.IsGrabbing():
            grab = camera.RetrieveResult(3000, pylon.TimeoutHandling_ThrowException)
            try:
                if not grab.GrabSucceeded():
                    continue
                frame = apply_frame_transform(binding.converter.Convert(grab).GetArray(), binding.requested)
                now = time.monotonic()
                instantaneous = 1.0 / max(now - last_time, 1e-9)
                displayed_fps = 0.9 * displayed_fps + 0.1 * instantaneous if displayed_fps else instantaneous
                last_time = now
                preview = frame
                max_width = 1400
                if preview.shape[1] > max_width:
                    scale = max_width / preview.shape[1]
                    preview = cv2.resize(
                        preview,
                        (round(preview.shape[1] * scale), round(preview.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.putText(
                    preview,
                    f"{binding.label} | receive {displayed_fps:.1f} fps | q quit | s snapshot | p settings",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    snapshot_count += 1
                    path = Path.cwd() / f"{binding.label}_snapshot_{snapshot_count:03d}.png"
                    cv2.imwrite(str(path), frame)
                    LOG.info("Saved %s", path)
                if key == ord("p"):
                    LOG.info(
                        "%s current settings: exposure_us=%r gain=%r fps=%r resulting_fps=%r",
                        binding.label,
                        read_setting(binding.camera, "ExposureTime")
                        or read_setting(binding.camera, "ExposureTimeAbs"),
                        read_setting(binding.camera, "Gain")
                        or read_setting(binding.camera, "GainRaw"),
                        read_setting(binding.camera, "AcquisitionFrameRate")
                        or read_setting(binding.camera, "AcquisitionFrameRateAbs"),
                        read_setting(binding.camera, "ResultingFrameRate")
                        or read_setting(binding.camera, "ResultingFrameRateAbs"),
                    )
            finally:
                grab.Release()
    finally:
        try:
            if binding.camera.IsGrabbing():
                binding.camera.StopGrabbing()
        except Exception:
            pass
        cv2.destroyAllWindows()
        close_bindings([binding])
    return 0


def validate_schedule(schedule: dict[str, Any]) -> tuple[float, float, Optional[float], Optional[int]]:
    clip_duration_s = float(schedule.get("clip_duration_s", 1800))
    interval_s = float(schedule.get("interval_s", clip_duration_s))
    if clip_duration_s <= 0:
        raise ValueError("schedule.clip_duration_s must be positive")
    if interval_s < clip_duration_s:
        raise ValueError(
            "schedule.interval_s is the start-to-start interval and must be >= clip_duration_s. "
            "Use equal values for continuous back-to-back clips."
        )

    total_duration_h = schedule.get("total_duration_h")
    total_duration_s = float(total_duration_h) * 3600 if total_duration_h is not None else None
    number_of_clips = schedule.get("number_of_clips")
    if number_of_clips is not None:
        number_of_clips = int(number_of_clips)
        if number_of_clips <= 0:
            raise ValueError("schedule.number_of_clips must be positive")
    if total_duration_s is None and number_of_clips is None:
        raise ValueError("Set schedule.total_duration_h or schedule.number_of_clips")
    return clip_duration_s, interval_s, total_duration_s, number_of_clips


def write_session_manifest(
    session_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    bindings: list[CameraBinding],
    ffmpeg: str,
    *,
    session_name: str,
    session_start_utc: dt.datetime,
) -> None:
    manifest = {
        "created_utc": utc_now().isoformat(),
        "session_name": session_name,
        "session_start_utc": session_start_utc.isoformat(),
        "platform": sys.platform,
        "python": sys.version,
        "config_source": str(config_path.resolve()),
        "ffmpeg": ffmpeg,
        "naming": {
            "version": 2,
            "session_directory_format": "YYYYMMDD_HHMMSS",
            "clip_directory_format": "clip_NNNN_HHMMSS",
            "camera_file_stem": "camera label",
            "path_time_precision": "seconds",
            "scientific_timestamp_precision": "nanoseconds where available",
        },
        "config": config,
        "cameras": [
            {
                "label": binding.label,
                "device": binding.info,
                "requested_settings": binding.requested,
                "actual_settings": binding.actual_settings,
            }
            for binding in bindings
        ],
    }
    with (session_dir / "session_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    shutil.copy2(config_path, session_dir / "config_used.yaml")


def write_session_summary(
    session_dir: Path,
    completed_clips: int,
    requested_clips: Optional[int],
    stopped_by_signal: bool,
    any_failure: bool,
    unexpected_exception: Optional[str],
    exit_status: str,
) -> None:
    summary = {
        "completed_clips": completed_clips,
        "requested_clips": requested_clips,
        "stopped_by_signal": stopped_by_signal,
        "any_failure": any_failure,
        "finished_utc": utc_now().isoformat(),
        "exit_status": exit_status,
        "unexpected_exception": unexpected_exception,
    }
    summary_path = session_dir / "session_summary.json"
    temp_path = summary_path.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, summary_path)


def run_recording(config_path: Path, config: dict[str, Any], verbose: bool, dry_run: bool) -> int:
    schedule = dict(config.get("schedule") or {})
    clip_duration_s, interval_s, total_duration_s, number_of_clips = validate_schedule(schedule)
    preview_settings = parse_recording_preview_settings(config)
    archive_settings = parse_archive_settings(config)
    requested_clips = number_of_clips

    project = sanitize_token(str(config.get("project", "caterpillar")))
    subject = sanitize_token(str(config.get("subject", "cohort")))
    output_root = Path(str(config.get("output_root", "./recordings"))).expanduser()
    session_start_utc = utc_now()
    session_dir = choose_unique_directory(output_root / project / subject / session_stamp_utc(session_start_utc))
    session_name = session_dir.name
    session_dir.mkdir(parents=True, exist_ok=False)
    setup_logging(session_dir, verbose)

    LOG.info("Session directory: %s", session_dir)
    LOG.info(
        "Schedule: %.1f s recording every %.1f s; total_duration_s=%s number_of_clips=%s",
        clip_duration_s,
        interval_s,
        total_duration_s,
        number_of_clips,
    )
    LOG.info(
        "Recording preview: enabled=%s fps=%.3g max_size=%dx%d",
        preview_settings.enabled,
        preview_settings.fps,
        preview_settings.max_width,
        preview_settings.max_height,
    )

    archive_preflight: Optional[ArchivePreflightResult] = None
    if archive_settings.enabled:
        archive_preflight = preflight_archive_settings(
            archive_settings,
            local_output_root=output_root,
            session_dir=session_dir,
            project=project,
            subject=subject,
            session_name=session_name,
        )
        LOG.info(
            "Archive: destination_root=%s mount_point=%s rsync=%s enabled=%s background_transfer=%s",
            archive_preflight.destination_root,
            archive_preflight.required_mount_point,
            archive_preflight.rsync_path,
            archive_settings.enabled,
            archive_settings.background_transfer,
        )
        LOG.info(
            "Archive preflight: ok=%s destination_free_gb=%s local_free_gb=%s archive_session_dir=%s",
            archive_preflight.ok,
            f"{archive_preflight.destination_free_gb:.1f}" if archive_preflight.destination_free_gb is not None else "n/a",
            f"{archive_preflight.local_free_gb:.1f}" if archive_preflight.local_free_gb is not None else "n/a",
            archive_preflight.archive_session_dir,
        )
        for error in archive_preflight.errors:
            LOG.error("Archive preflight issue: %s", error)

    if dry_run:
        shutil.copy2(config_path, session_dir / "config_used.yaml")
        with (session_dir / "dry_run.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "project": project,
                    "subject": subject,
                    "clip_duration_s": clip_duration_s,
                    "interval_s": interval_s,
                    "total_duration_s": total_duration_s,
                    "number_of_clips": number_of_clips,
                    "recording_preview": dataclasses.asdict(preview_settings),
                    "archive": dataclasses.asdict(archive_settings),
                    "archive_preflight": dataclasses.asdict(archive_preflight) if archive_preflight else None,
                },
                handle,
                indent=2,
            )
        LOG.info("Dry run completed; no cameras were opened.")
        if archive_preflight is not None and not archive_preflight.ok:
            return 1
        return 0

    if archive_settings.enabled and (archive_preflight is None or not archive_preflight.ok):
        LOG.error("Archive preflight failed; refusing to open cameras")
        return 1

    ffmpeg = find_ffmpeg(config.get("ffmpeg"))
    camera_cfgs = config.get("cameras")
    if not isinstance(camera_cfgs, list) or not camera_cfgs:
        raise ValueError("config.cameras must be a non-empty list")

    devices = enumerate_devices()
    if len(devices) < len(camera_cfgs):
        raise RuntimeError(
            f"Config requests {len(camera_cfgs)} cameras, but only {len(devices)} were detected."
        )

    used_serials: set[str] = set()
    bindings: list[CameraBinding] = []
    stop_event = threading.Event()
    preview_active_event = threading.Event()
    stopped_by_signal = False
    clip_index = 0
    any_failure = False
    unexpected_exception: Optional[str] = None
    exit_status = "running"
    archive_manager: Optional[ArchiveManager] = None
    if preview_settings.enabled:
        preview_active_event.set()

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stopped_by_signal
        stopped_by_signal = True
        LOG.warning("Received signal %s; stopping after the current frame", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    try:
        for camera_cfg_raw in camera_cfgs:
            camera_cfg = dict(camera_cfg_raw)
            device = match_device(camera_cfg, devices, used_serials)
            info = camera_info_dict(device)
            used_serials.add(info["serial"])
            bindings.append(configure_camera(camera_cfg, device))

        write_session_manifest(
            session_dir,
            config_path,
            config,
            bindings,
            ffmpeg,
            session_name=session_name,
            session_start_utc=session_start_utc,
        )
        if archive_settings.enabled:
            archive_manager = ArchiveManager(
                settings=archive_settings,
                local_session_dir=session_dir,
                project=project,
                subject=subject,
                session_name=session_name,
                stop_event=stop_event,
                preflight=archive_preflight or ArchivePreflightResult(
                    enabled=False,
                    ok=False,
                    errors=["archive preflight missing"],
                    rsync_path=None,
                    required_mount_point=str(archive_settings.required_mount_point),
                    required_mount_is_mount=False,
                    destination_root=str(archive_settings.destination_root),
                    destination_created=False,
                    destination_writable=False,
                    local_free_gb=None,
                    destination_free_gb=None,
                    path_conflict=False,
                    archive_session_dir=str(
                        archive_settings.destination_root / project / subject / session_name
                    ),
                ),
            )
            archive_manager.start()
        encoding_cfg = dict(config.get("encoding") or {})
        clip_stop_threshold_bytes = (
            int(archive_settings.max_clip_size_gb * 0.95 * 1024**3)
            if archive_settings.enabled
            else None
        )
        archive_max_clip_size_bytes = (
            int(archive_settings.max_clip_size_gb * 1024**3)
            if archive_settings.enabled
            else None
        )

        def ensure_ready_for_next_clip(next_clip_index: int) -> Optional[str]:
            try:
                local_free_gb = bytes_to_gib(shutil.disk_usage(session_dir).free)
            except Exception as exc:
                return f"could not read local disk usage: {exc}"
            if local_free_gb < archive_settings.min_local_free_gb_before_clip:
                return (
                    f"local free space {local_free_gb:.1f} GiB is below the minimum "
                    f"{archive_settings.min_local_free_gb_before_clip:.1f} GiB"
                )

            if not archive_settings.enabled or archive_manager is None:
                return None
            if archive_manager.failure_detected():
                return "archive manager already reported a failure"

            deadline = time.monotonic() + 300.0
            last_progress_log = 0.0
            while True:
                if stop_event.is_set():
                    return "recording stop requested before the next clip"
                try:
                    local_free_gb = bytes_to_gib(shutil.disk_usage(session_dir).free)
                except Exception as exc:
                    return f"could not read local disk usage: {exc}"
                if local_free_gb < archive_settings.min_local_free_gb_before_clip:
                    return (
                        f"local free space {local_free_gb:.1f} GiB is below the minimum "
                        f"{archive_settings.min_local_free_gb_before_clip:.1f} GiB"
                    )

                pending = archive_manager.unarchived_count()
                if archive_manager.failure_detected():
                    return "archive manager reported a failure"
                if pending <= archive_settings.max_unarchived_clips:
                    return None

                now = time.monotonic()
                if now - last_progress_log >= 10.0:
                    LOG.info(
                        "Waiting for archive worker to catch up before clip %d: pending=%d max=%d",
                        next_clip_index,
                        pending,
                        archive_settings.max_unarchived_clips,
                    )
                    last_progress_log = now
                if archive_manager.failure_detected():
                    return "archive manager reported a failure while waiting for backlog to clear"
                if now >= deadline:
                    return (
                        "archive backlog did not clear within 5 minutes "
                        f"(pending={pending}, max={archive_settings.max_unarchived_clips})"
                    )
                time.sleep(min(1.0, max(0.1, deadline - now)))

        session_start_mono_ns = time.monotonic_ns()

        while not stop_event.is_set():
            if number_of_clips is not None and clip_index >= number_of_clips:
                break
            scheduled_offset_s = clip_index * interval_s
            if total_duration_s is not None and scheduled_offset_s >= total_duration_s:
                break

            target_start_mono_ns = session_start_mono_ns + round(scheduled_offset_s * 1e9)
            while not stop_event.is_set():
                remaining_s = (target_start_mono_ns - time.monotonic_ns()) / 1e9
                if remaining_s <= 0:
                    break
                time.sleep(min(remaining_s, 1.0))
            if stop_event.is_set():
                break

            readiness_error = ensure_ready_for_next_clip(clip_index)
            if readiness_error is not None:
                any_failure = True
                LOG.error("Cannot start clip %d: %s", clip_index, readiness_error)
                stop_event.set()
                break

            # Give all writer threads one second to create files and start grabbing.
            planned_start_mono_ns = time.monotonic_ns() + 1_000_000_000
            planned_stop_mono_ns = planned_start_mono_ns + round(clip_duration_s * 1e9)
            delay_to_start_s = (planned_start_mono_ns - time.monotonic_ns()) / 1e9
            clip_start_utc = utc_now() + dt.timedelta(seconds=delay_to_start_s)
            clip_dir = session_dir / f"clip_{clip_index:04d}_{clip_clock_utc(clip_start_utc)}"
            clip_dir.mkdir(parents=True, exist_ok=False)
            LOG.info(
                "Starting clip %d at %s for %.1f s",
                clip_index,
                clip_start_utc.isoformat(),
                clip_duration_s,
            )

            barrier = threading.Barrier(len(bindings) + 1)
            results: queue.Queue[ClipResult] = queue.Queue()
            threads: list[threading.Thread] = []
            preview_queues: dict[str, queue.Queue[PreviewPacket]] = {}
            if preview_active_event.is_set():
                preview_queues = {binding.label: queue.Queue(maxsize=1) for binding in bindings}
            for binding in bindings:
                thread = threading.Thread(
                    target=record_one_camera,
                    name=f"camera-{binding.label}",
                    kwargs={
                        "binding": binding,
                        "clip_index": clip_index,
                        "clip_start_utc": clip_start_utc,
                        "planned_start_mono_ns": planned_start_mono_ns,
                        "planned_stop_mono_ns": planned_stop_mono_ns,
                        "clip_dir": clip_dir,
                        "ffmpeg": ffmpeg,
                        "encoding_cfg": encoding_cfg,
                        "ready_barrier": barrier,
                        "stop_event": stop_event,
                        "storage_root": session_dir,
                        "clip_stop_threshold_bytes": clip_stop_threshold_bytes,
                        "result_queue": results,
                        "preview_queue": preview_queues.get(binding.label),
                        "preview_settings": preview_settings,
                        "preview_active_event": preview_active_event,
                    },
                    daemon=False,
                )
                thread.start()
                threads.append(thread)

            try:
                barrier.wait(timeout=30)
            except threading.BrokenBarrierError:
                LOG.error("Camera workers did not become ready in time")
                stop_event.set()

            monitor_recording_threads(
                threads=threads,
                preview_queues=preview_queues,
                preview_settings=preview_settings,
                preview_active_event=preview_active_event,
                clip_dir=clip_dir,
            )

            clip_results: list[ClipResult] = []
            while not results.empty():
                clip_results.append(results.get())
            for result in clip_results:
                LOG.info(
                    "Clip %d camera=%s success=%s video=%s metadata=%s",
                    clip_index,
                    result.label,
                    result.success,
                    result.video_path,
                    result.metadata_path,
                )
                if not result.success:
                    any_failure = True
            if len(clip_results) != len(bindings):
                any_failure = True
                LOG.error(
                    "Expected %d camera results, received %d",
                    len(bindings),
                    len(clip_results),
                )

            archive_failure = archive_manager.failure_detected() if archive_manager else False
            clip_total_bytes_before_cleanup = tree_stats(clip_dir)[1]

            if (
                not any_failure
                and archive_settings.enabled
                and archive_max_clip_size_bytes is not None
                and clip_total_bytes_before_cleanup > archive_max_clip_size_bytes
            ):
                any_failure = True
                LOG.error(
                    "Clip %d exceeded the configured size cap before cleanup: %.1f GiB > %.1f GiB",
                    clip_index,
                    bytes_to_gib(clip_total_bytes_before_cleanup),
                    bytes_to_gib(archive_max_clip_size_bytes),
                )

            if not any_failure and not archive_failure:
                removed_capture_files = remove_capture_files(clip_dir)
                if removed_capture_files:
                    LOG.info(
                        "Removed %d temporary capture file(s) from %s",
                        len(removed_capture_files),
                        clip_dir,
                    )
                if archive_settings.enabled:
                    ready, archive_issues, clip_total_bytes_after_cleanup = clip_directory_ready_for_archive(
                        clip_dir,
                        len(bindings),
                        archive_max_clip_size_bytes or int(1e18),
                    )
                    if not ready:
                        any_failure = True
                        LOG.error(
                            "Clip %d is not ready for archive: %s",
                            clip_index,
                            "; ".join(archive_issues),
                        )
                    else:
                        LOG.info(
                            "Clip %d ready for archive at %.1f GiB",
                            clip_index,
                            bytes_to_gib(clip_total_bytes_after_cleanup),
                        )
                        try:
                            assert archive_manager is not None
                            archive_manager.enqueue_clip(clip_dir)
                        except Exception as exc:
                            any_failure = True
                            LOG.exception("Failed to enqueue clip %d for archive", clip_index)
                            if archive_manager is not None:
                                archive_manager.mark_failure(f"enqueue failed: {exc}")
                else:
                    # The local-only workflow still keeps the clip directory tidy.
                    pass
            elif archive_failure:
                any_failure = True
                LOG.error("Archive manager already reported a failure; not cleaning up clip %d", clip_index)

            if any_failure:
                stop_event.set()
                break

            clip_index += 1

        LOG.info("Recording finished after %d completed clip(s)", clip_index)
    except KeyboardInterrupt:
        stopped_by_signal = True
        exit_status = "interrupted"
        LOG.warning("KeyboardInterrupt received; stopping recording")
        stop_event.set()
    except Exception as exc:
        any_failure = True
        unexpected_exception = repr(exc)
        exit_status = "exception"
        LOG.exception("Unhandled exception during recording")
        stop_event.set()
    finally:
        if archive_manager is not None:
            try:
                archive_manager.wait_until_idle()
            except Exception as exc:
                any_failure = True
                LOG.warning("Could not wait for archive manager to become idle: %s", exc)
            if archive_manager.failure_detected():
                any_failure = True
        if exit_status == "running":
            if any_failure:
                exit_status = "failed"
            elif stopped_by_signal:
                exit_status = "interrupted"
            else:
                exit_status = "success"
        try:
            write_session_summary(
                session_dir=session_dir,
                completed_clips=clip_index,
                requested_clips=requested_clips,
                stopped_by_signal=stopped_by_signal,
                any_failure=any_failure,
                unexpected_exception=unexpected_exception,
                exit_status=exit_status,
            )
        except Exception as exc:
            LOG.warning("Could not write session summary: %s", exc)
        if archive_manager is not None:
            try:
                archive_manager.close()
            except Exception as exc:
                LOG.warning("Could not finalize archive manager: %s", exc)
        close_bindings(bindings)

    if exit_status == "interrupted":
        return 130
    if exit_status in {"failed", "exception"}:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    parser.add_argument("--list-cameras", action="store_true", help="List detected Basler cameras and exit")
    parser.add_argument("--preview", metavar="LABEL", help="Preview one configured camera and exit")
    parser.add_argument("--dry-run", action="store_true", help="Validate schedule and output paths without opening cameras")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_cameras:
        setup_logging(None, args.verbose)
        return list_cameras()

    if args.config is None:
        parser.error("--config is required unless --list-cameras is used")
    config_path: Path = args.config.expanduser().resolve()
    config = load_config(config_path)

    if args.preview:
        setup_logging(None, args.verbose)
        return preview_camera(config, args.preview)

    return run_recording(config_path, config, args.verbose, args.dry_run)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
