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


def compact_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


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
        self.stderr_path = temp_path.with_suffix(temp_path.suffix + ".ffmpeg.log")
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

    def close_and_remux(self) -> tuple[int, bool]:
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
        remux_log = self.final_path.with_suffix(self.final_path.suffix + ".remux.log")
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


def draw_recording_preview(packet: PreviewPacket, settings: RecordingPreviewSettings) -> np.ndarray:
    """Create a display-only copy with a compact recording status overlay."""

    display = packet.frame.copy()
    if not settings.show_status:
        return display

    font = cv2.FONT_HERSHEY_DUPLEX
    controls_font_scale = 0.42
    controls_text = "q hide | s snapshot"
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
        1,
    )

    overlay = display.copy()
    panel_height = 78
    cv2.rectangle(
        overlay,
        (0, 0),
        (display.shape[1], panel_height),
        (24, 24, 24),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

    y = 26
    for line in lines:
        cv2.putText(
            display,
            line,
            (12, y),
            font,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24

    controls_x = max(12, display.shape[1] - controls_width - 12)
    cv2.putText(
        display,
        controls_text,
        (controls_x, 24),
        font,
        controls_font_scale,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    bar_left = 12
    bar_top = 60
    bar_width = max(60, display.shape[1] - 24)
    bar_height = 8
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
                stamp = compact_utc(utc_now())
                for packet in latest.values():
                    snapshot_path = snapshot_dir / (
                        f"{packet.label}_clip{packet.clip_index:04d}_"
                        f"frame{packet.frame_index:08d}_{stamp}_{snapshot_count:03d}.png"
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
    result_queue: queue.Queue[ClipResult],
    preview_queue: Optional[queue.Queue[PreviewPacket]],
    preview_settings: RecordingPreviewSettings,
    preview_active_event: threading.Event,
) -> None:
    label = binding.label
    camera = binding.camera
    requested_fps = float(binding.requested.get("fps", binding.fps))
    actual_fps = float(binding.actual_settings.get("AcquisitionFrameRate") or requested_fps)
    prefix = "_".join(
        [
            sanitize_token(str(binding.requested.get("project_override") or "")),
            label,
            f"clip{clip_index:04d}",
            compact_utc(clip_start_utc),
        ]
    ).strip("_")
    final_path = clip_dir / f"{prefix}.mp4"
    temp_path = clip_dir / f"{prefix}.capture.mkv"
    timestamps_path = clip_dir / f"{prefix}.timestamps.csv.gz"
    metadata_path = clip_dir / f"{prefix}.json"

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
            finally:
                grab.Release()

        if camera.IsGrabbing():
            camera.StopGrabbing()
        if timestamp_handle is not None:
            timestamp_handle.flush()
            timestamp_handle.close()
            timestamp_handle = None

        ffmpeg_return_code, remuxed = writer.close_and_remux()
        writer = None
        if ffmpeg_return_code != 0:
            raise RuntimeError(f"FFmpeg exited with code {ffmpeg_return_code}")
        if not remuxed:
            raise RuntimeError(
                "H.264 capture succeeded, but MP4 remux failed. "
                f"The recoverable MKV was kept at {temp_path}."
            )

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
) -> None:
    manifest = {
        "created_utc": utc_now().isoformat(),
        "platform": sys.platform,
        "python": sys.version,
        "config_source": str(config_path.resolve()),
        "ffmpeg": ffmpeg,
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


def run_recording(config_path: Path, config: dict[str, Any], verbose: bool, dry_run: bool) -> int:
    schedule = dict(config.get("schedule") or {})
    clip_duration_s, interval_s, total_duration_s, number_of_clips = validate_schedule(schedule)
    preview_settings = parse_recording_preview_settings(config)

    project = sanitize_token(str(config.get("project", "caterpillar")))
    subject = sanitize_token(str(config.get("subject", "cohort")))
    output_root = Path(str(config.get("output_root", "./recordings"))).expanduser()
    session_start_utc = utc_now()
    session_name = f"{project}_{subject}_{compact_utc(session_start_utc)}"
    session_dir = output_root / project / subject / session_name
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
                },
                handle,
                indent=2,
            )
        LOG.info("Dry run completed; no cameras were opened.")
        return 0

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
    if preview_settings.enabled:
        preview_active_event.set()

    def request_stop(signum: int, _frame: Any) -> None:
        LOG.warning("Received signal %s; stopping after the current frame", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    try:
        for camera_cfg_raw in camera_cfgs:
            camera_cfg = dict(camera_cfg_raw)
            camera_cfg["project_override"] = f"{project}_{subject}"
            device = match_device(camera_cfg, devices, used_serials)
            info = camera_info_dict(device)
            used_serials.add(info["serial"])
            bindings.append(configure_camera(camera_cfg, device))

        write_session_manifest(session_dir, config_path, config, bindings, ffmpeg)
        encoding_cfg = dict(config.get("encoding") or {})

        session_start_mono_ns = time.monotonic_ns()
        clip_index = 0
        any_failure = False

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

            # Give all writer threads one second to create files and start grabbing.
            planned_start_mono_ns = time.monotonic_ns() + 1_000_000_000
            planned_stop_mono_ns = planned_start_mono_ns + round(clip_duration_s * 1e9)
            delay_to_start_s = (planned_start_mono_ns - time.monotonic_ns()) / 1e9
            clip_start_utc = utc_now() + dt.timedelta(seconds=delay_to_start_s)
            clip_dir = session_dir / f"clip_{clip_index:04d}_{compact_utc(clip_start_utc)}"
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
            if any_failure:
                stop_event.set()
                break

            clip_index += 1

        LOG.info("Recording finished after %d completed clip(s)", clip_index)
        return 1 if any_failure else 0
    finally:
        close_bindings(bindings)


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
