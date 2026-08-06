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
import hashlib
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
from pathlib import Path, PurePath
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


def local_now() -> dt.datetime:
    """Return the current timezone-aware local datetime."""

    return utc_now().astimezone()


def to_local(value: dt.datetime) -> dt.datetime:
    """Convert a timezone-aware datetime to the computer's local timezone."""

    if value.tzinfo is None:
        raise ValueError("Cannot convert a naive datetime to local time")
    return value.astimezone()


def isoformat_utc(
    value: dt.datetime,
    *,
    timespec: str = "milliseconds",
) -> str:
    """Return an ISO-8601 UTC timestamp using a trailing Z."""

    if value.tzinfo is None:
        raise ValueError("UTC timestamp must be timezone-aware")
    normalized = value.astimezone(dt.timezone.utc)
    return normalized.isoformat(timespec=timespec).replace("+00:00", "Z")


def isoformat_local(
    value: dt.datetime,
    *,
    timespec: str = "milliseconds",
) -> str:
    """Return an ISO-8601 local timestamp with a numeric UTC offset."""

    if value.tzinfo is None:
        raise ValueError("Local timestamp must be timezone-aware")
    return to_local(value).isoformat(timespec=timespec)


def local_timezone_metadata(
    reference_utc: Optional[dt.datetime] = None,
) -> dict[str, Optional[str]]:
    """Return best-effort local timezone metadata for the provided UTC instant."""

    reference = reference_utc or utc_now()
    if reference.tzinfo is None:
        raise ValueError("reference_utc must be timezone-aware")

    local_value = to_local(reference)
    offset = local_value.utcoffset()
    if offset is None:
        offset_text: Optional[str] = None
    else:
        total_minutes = int(offset.total_seconds() // 60)
        sign = "+" if total_minutes >= 0 else "-"
        absolute_minutes = abs(total_minutes)
        hours, minutes = divmod(absolute_minutes, 60)
        offset_text = f"{sign}{hours:02d}:{minutes:02d}"

    return {
        "local_timezone_label": local_value.tzname(),
        "local_utc_offset": offset_text,
    }


def timestamp_pair(value: dt.datetime) -> dict[str, str]:
    if value.tzinfo is None:
        raise ValueError("timestamp_pair requires a timezone-aware datetime")
    return {
        "utc": isoformat_utc(value),
        "local": isoformat_local(value),
    }


def filename_local_timestamp(value: dt.datetime) -> str:
    """Return a filename-safe local timestamp with UTC offset."""

    if value.tzinfo is None:
        raise ValueError("Filename timestamp must be timezone-aware")
    return to_local(value).strftime("%Y%m%d_%H%M%S%z")


def clip_clock_local(value: dt.datetime) -> str:
    """Return a clip-directory local timestamp with the active UTC offset."""

    if value.tzinfo is None:
        raise ValueError("Clip timestamp must be timezone-aware")
    return to_local(value).strftime("%H%M%S%z")


def snapshot_stamp_utc(value: dt.datetime) -> str:
    """Compact UTC timestamp for optional monitoring snapshots."""

    return value.astimezone(dt.timezone.utc).strftime("%H%M%S")


def iso_utc_from_ns(value_ns: int) -> str:
    return isoformat_utc(dt.datetime.fromtimestamp(value_ns / 1e9, tz=dt.timezone.utc))


def format_clock_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_preview_exposure(
    exposure_us: Optional[float],
    *,
    auto_exposure: bool,
    upper_us: Optional[float],
) -> tuple[str, bool]:
    if exposure_us is None:
        return ("AUTO EXP --" if auto_exposure else "EXP --", False)

    ms = exposure_us / 1000.0
    near_limit = (
        auto_exposure
        and upper_us is not None
        and upper_us > 0
        and exposure_us >= 0.95 * upper_us
    )
    prefix = "AUTO EXP" if auto_exposure else "EXP"
    suffix = "  MAX" if near_limit else ""
    return f"{prefix} {ms:.1f} ms{suffix}", near_limit


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


class LocalIsoFormatter(logging.Formatter):
    def formatTime(
        self,
        record: logging.LogRecord,
        datefmt: Optional[str] = None,
    ) -> str:
        del datefmt
        value_utc = dt.datetime.fromtimestamp(record.created, tz=dt.timezone.utc)
        return isoformat_local(value_utc)


def setup_logging(session_dir: Optional[Path], verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if session_dir is not None:
        handlers.append(logging.FileHandler(session_dir / "recorder.log", encoding="utf-8"))
    formatter = LocalIsoFormatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
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


def try_set_first_available(
    camera: Any,
    names: Iterable[str],
    value: Any,
    *,
    required: bool = False,
) -> tuple[Optional[str], Any]:
    for name in names:
        if get_node(camera, name) is None:
            continue
        ok, actual = try_set(camera, name, value)
        if ok:
            return name, actual

    if required:
        joined = ", ".join(names)
        raise RuntimeError(f"None of the required camera nodes could be set: {joined}")
    return None, None


def read_setting(camera: Any, name: str) -> Any:
    return node_value(get_node(camera, name))


def read_first_available_setting(camera: Any, names: Iterable[str]) -> Any:
    for name in names:
        value = read_setting(camera, name)
        if value is not None:
            return value
    return None


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
    layout: str = "card_panel"


@dataclasses.dataclass(frozen=True)
class SystemSettings:
    prevent_sleep_during_recording: bool = False


@dataclasses.dataclass(frozen=True)
class StatusSettings:
    terminal_interval_s: float = 60.0


@dataclasses.dataclass(frozen=True)
class ArchiveSettings:
    enabled: bool = False
    backend: str = "auto"
    destination_root: Path = dataclasses.field(
        default_factory=lambda: Path("/Volumes/Dr. Rose/Hung_MBL")
    )
    required_mount_point: Path = dataclasses.field(
        default_factory=lambda: Path("/Volumes/Dr. Rose")
    )
    rsync_executable: str = "/usr/bin/rsync"
    robocopy_executable: str = "robocopy"
    copy_timeout_s: float = 3600.0
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
    platform: str
    copy_backend: str
    copy_executable_path: Optional[str]
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
    total_clips: int
    frame_index: int
    frame: np.ndarray
    host_monotonic_ns: int
    elapsed_s: float
    planned_duration_s: float
    session_elapsed_s: float
    planned_session_duration_s: float
    planned_finish_utc: dt.datetime
    measured_receive_fps: Optional[float]
    exposure_us: Optional[float] = None
    auto_exposure: bool = False
    auto_exposure_upper_us: Optional[float] = None


def parse_system_settings(config: dict[str, Any]) -> SystemSettings:
    raw = config.get("system") or {}
    if not isinstance(raw, dict):
        raise ValueError("system must be a mapping/object")

    return SystemSettings(
        prevent_sleep_during_recording=bool(raw.get("prevent_sleep_during_recording", False)),
    )


def parse_status_settings(config: dict[str, Any]) -> StatusSettings:
    raw = config.get("status") or {}
    if not isinstance(raw, dict):
        raise ValueError("status must be a mapping/object")

    settings = StatusSettings(
        terminal_interval_s=float(raw.get("terminal_interval_s", 60.0)),
    )
    if settings.terminal_interval_s < 0:
        raise ValueError("status.terminal_interval_s must be >= 0")
    return settings


def parse_recording_preview_settings(config: dict[str, Any]) -> RecordingPreviewSettings:
    raw = config.get("recording_preview") or {}
    if not isinstance(raw, dict):
        raise ValueError("recording_preview must be a mapping/object")

    layout = str(raw.get("layout", "card_panel")).strip().lower() or "card_panel"
    if layout not in {"card_panel", "legacy_overlay"}:
        raise ValueError("recording_preview.layout must be one of: card_panel, legacy_overlay")

    settings = RecordingPreviewSettings(
        enabled=bool(raw.get("enabled", False)),
        fps=float(raw.get("fps", 1.0)),
        max_width=int(raw.get("max_width", 640)),
        max_height=int(raw.get("max_height", 720)),
        show_status=bool(raw.get("show_status", True)),
        layout=layout,
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

    backend = str(raw.get("backend", "auto")).strip().lower() or "auto"
    if backend not in {"auto", "rsync", "robocopy"}:
        raise ValueError("archive.backend must be one of: auto, rsync, robocopy")
    destination_root = Path(str(raw.get("destination_root", "/Volumes/Dr. Rose/Hung_MBL"))).expanduser()
    required_mount_point = Path(str(raw.get("required_mount_point", "/Volumes/Dr. Rose"))).expanduser()
    rsync_executable = str(raw.get("rsync_executable", "/usr/bin/rsync"))
    robocopy_executable = str(raw.get("robocopy_executable", "robocopy"))
    copy_timeout_s = float(raw.get("copy_timeout_s", 3600.0))
    verification = str(raw.get("verification", "checksum")).strip().lower() or "checksum"
    if verification not in {"checksum", "sha256"}:
        raise ValueError("archive.verification must be one of: checksum, sha256")
    if copy_timeout_s <= 0:
        raise ValueError("archive.copy_timeout_s must be positive")

    return ArchiveSettings(
        enabled=bool(raw.get("enabled", False)),
        backend=backend,
        destination_root=destination_root,
        required_mount_point=required_mount_point,
        rsync_executable=rsync_executable,
        robocopy_executable=robocopy_executable,
        copy_timeout_s=copy_timeout_s,
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


def resize_to_fit(
    frame: np.ndarray,
    *,
    max_width: int,
    max_height: int,
    allow_upscale: bool = False,
) -> np.ndarray:
    """Resize a frame to fit completely inside the requested bounding box."""

    if max_width <= 0:
        raise ValueError("preview max_width must be positive")
    if max_height <= 0:
        raise ValueError("preview max_height must be positive")

    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Cannot resize an empty frame with shape {frame.shape}")

    scale = min(max_width / width, max_height / height)
    if not allow_upscale:
        scale = min(scale, 1.0)

    if scale >= 1.0:
        resized = frame
    else:
        target_width = max(1, int(round(width * scale)))
        target_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            frame,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

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


@dataclasses.dataclass(frozen=True)
class AutoExposureSettings:
    mode_raw: str
    mode_value: str
    lower_us: float
    upper_us: float
    target: float
    initial_us: float
    roi: str


def parse_auto_exposure_settings(
    camera_cfg: dict[str, Any],
    *,
    label: str,
) -> AutoExposureSettings:
    mode_raw = str(camera_cfg.get("auto_exposure_mode", "continuous")).strip().lower()
    mode_map = {
        "continuous": "Continuous",
        "once": "Once",
    }
    if mode_raw not in mode_map:
        raise ValueError("auto_exposure_mode must be one of: continuous, once")

    fps = float(camera_cfg.get("fps", 5.0))
    lower_us = float(camera_cfg.get("auto_exposure_lower_us", 6000))
    upper_us = float(camera_cfg.get("auto_exposure_upper_us", 180000))
    target = float(camera_cfg.get("auto_target_brightness", 0.59))
    initial_us = float(camera_cfg.get("exposure_us", lower_us))
    roi = str(camera_cfg.get("auto_exposure_roi", "full")).strip().lower() or "full"

    if lower_us <= 0:
        raise ValueError("auto_exposure_lower_us must be positive")
    if upper_us <= lower_us:
        raise ValueError("auto_exposure_upper_us must be greater than auto_exposure_lower_us")
    if not lower_us <= initial_us <= upper_us:
        raise ValueError("exposure_us must fall within the auto-exposure lower/upper limits")
    if not 0.0 < target < 1.0:
        raise ValueError("auto_target_brightness must be between 0 and 1")
    if roi != "full":
        raise ValueError("auto_exposure_roi must currently be: full")
    if fps <= 0:
        raise ValueError("camera fps must be positive when auto exposure is enabled")

    frame_period_us = 1_000_000.0 / fps
    if upper_us >= frame_period_us:
        raise ValueError(
            "auto_exposure_upper_us must be below the nominal frame period "
            f"({frame_period_us:.0f} us at {fps:g} fps)"
        )
    if upper_us > frame_period_us * 0.95:
        LOG.warning(
            "%s auto-exposure upper limit %.0f us is very close to the %.0f us "
            "frame period; verify the measured receive FPS",
            label,
            upper_us,
            frame_period_us,
        )

    return AutoExposureSettings(
        mode_raw=mode_raw,
        mode_value=mode_map[mode_raw],
        lower_us=lower_us,
        upper_us=upper_us,
        target=target,
        initial_us=initial_us,
        roi=roi,
    )


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

    # Stable brightness is preferable for behavioral segmentation. This path
    # supports either manual exposure or bounded camera-side auto exposure.
    auto_exposure = bool(camera_cfg.get("auto_exposure", False))
    auto_gain = bool(camera_cfg.get("auto_gain", False))
    auto_exposure_settings = (
        parse_auto_exposure_settings(camera_cfg, label=label) if auto_exposure else None
    )

    try_set(camera, "ExposureAuto", "Off")
    try_set(camera, "GainAuto", "Off")

    if not auto_gain and camera_cfg.get("gain") is not None:
        gain_ok, _ = try_set(camera, "Gain", float(camera_cfg["gain"]))
        if not gain_ok:
            try_set(camera, "GainRaw", int(camera_cfg["gain"]))

    if auto_exposure_settings is not None:
        exposure_ok, _ = try_set(camera, "ExposureTime", auto_exposure_settings.initial_us)
        if not exposure_ok:
            try_set(camera, "ExposureTimeAbs", auto_exposure_settings.initial_us, required=True)

        lower_name, actual_lower = try_set_first_available(
            camera,
            ("AutoExposureTimeLowerLimit", "AutoExposureTimeLowerLimitRaw"),
            auto_exposure_settings.lower_us,
            required=True,
        )
        upper_name, actual_upper = try_set_first_available(
            camera,
            ("AutoExposureTimeUpperLimit", "AutoExposureTimeUpperLimitRaw"),
            auto_exposure_settings.upper_us,
            required=True,
        )

        if get_node(camera, "AutoTargetBrightness") is not None:
            target_name, actual_target = try_set_first_available(
                camera,
                ("AutoTargetBrightness",),
                auto_exposure_settings.target,
                required=True,
            )
        else:
            target_name, actual_target = try_set_first_available(
                camera,
                ("AutoTargetValue",),
                int(round(auto_exposure_settings.target * 255)),
                required=True,
            )

        image_offset_x = int(read_setting(camera, "OffsetX") or 0)
        image_offset_y = int(read_setting(camera, "OffsetY") or 0)
        image_width = int(read_setting(camera, "Width") or 0)
        image_height = int(read_setting(camera, "Height") or 0)
        if image_width <= 0 or image_height <= 0:
            raise RuntimeError("Could not determine the configured camera ROI for auto exposure")

        roi_family: str
        if get_node(camera, "AutoFunctionROISelector") is not None:
            roi_family = "ROI"
            try_set(camera, "AutoFunctionROISelector", "ROI1", required=True)
            try_set(camera, "AutoFunctionROIOffsetX", image_offset_x, required=True)
            try_set(camera, "AutoFunctionROIOffsetY", image_offset_y, required=True)
            try_set(camera, "AutoFunctionROIWidth", image_width, required=True)
            try_set(camera, "AutoFunctionROIHeight", image_height, required=True)
            try_set(camera, "AutoFunctionROIUseBrightness", True, required=True)
        elif get_node(camera, "AutoFunctionAOISelector") is not None:
            roi_family = "AOI"
            try_set(camera, "AutoFunctionAOISelector", "AOI1", required=True)
            try_set(camera, "AutoFunctionAOIOffsetX", image_offset_x, required=True)
            try_set(camera, "AutoFunctionAOIOffsetY", image_offset_y, required=True)
            try_set(camera, "AutoFunctionAOIWidth", image_width, required=True)
            try_set(camera, "AutoFunctionAOIHeight", image_height, required=True)
            try_set(camera, "AutoFunctionAOIUsageIntensity", True, required=True)
        else:
            raise RuntimeError("Auto exposure requires a complete AutoFunction ROI/AOI node family")
    else:
        if camera_cfg.get("exposure_us") is not None:
            exposure_ok, _ = try_set(camera, "ExposureTime", float(camera_cfg["exposure_us"]))
            if not exposure_ok:
                try_set(camera, "ExposureTimeAbs", float(camera_cfg["exposure_us"]))

    if auto_gain:
        first_settable_enum(camera, "GainAuto", ("Continuous", "Once"))

    if auto_exposure_settings is not None:
        try_set(camera, "ExposureAuto", auto_exposure_settings.mode_value, required=True)
        LOG.info(
            "%s auto exposure: mode=%s target=%0.3f limits=%s..%s us seed=%.0f us "
            "ROI=%d,%d %dx%d GainAuto=%s gain=%r",
            label,
            auto_exposure_settings.mode_value,
            float(actual_target if actual_target is not None else auto_exposure_settings.target),
            actual_lower if actual_lower is not None else auto_exposure_settings.lower_us,
            actual_upper if actual_upper is not None else auto_exposure_settings.upper_us,
            auto_exposure_settings.initial_us,
            image_offset_x,
            image_offset_y,
            image_width,
            image_height,
            read_setting(camera, "GainAuto"),
            read_setting(camera, "Gain") if read_setting(camera, "Gain") is not None else read_setting(camera, "GainRaw"),
        )

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
        "AutoExposureTimeLowerLimit",
        "AutoExposureTimeUpperLimit",
        "AutoExposureTimeLowerLimitRaw",
        "AutoExposureTimeUpperLimitRaw",
        "AutoTargetBrightness",
        "AutoTargetValue",
        "AutoFunctionROISelector",
        "AutoFunctionROIOffsetX",
        "AutoFunctionROIOffsetY",
        "AutoFunctionROIWidth",
        "AutoFunctionROIHeight",
        "AutoFunctionROIUseBrightness",
        "AutoFunctionAOISelector",
        "AutoFunctionAOIOffsetX",
        "AutoFunctionAOIOffsetY",
        "AutoFunctionAOIWidth",
        "AutoFunctionAOIHeight",
        "AutoFunctionAOIUsageIntensity",
        "BslEffectiveExposureTime",
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


def regular_file_manifest(root: Path) -> dict[str, tuple[int, str]]:
    """Return relative path -> (size_bytes, sha256_hex) for all regular files."""

    manifest: dict[str, tuple[int, str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Archive verification does not permit symlinks: {path}")
        if not path.is_file():
            continue

        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        manifest[relative] = (path.stat().st_size, digest.hexdigest())
    return manifest


@dataclasses.dataclass(frozen=True)
class ArchiveVerificationSummary:
    success: bool
    error: str
    source_file_count: int
    destination_file_count: int
    source_bytes: int
    destination_bytes: int


def verify_file_trees(source: Path, destination: Path) -> ArchiveVerificationSummary:
    if not source.exists():
        return ArchiveVerificationSummary(
            success=False,
            error=f"source directory is missing: {source}",
            source_file_count=0,
            destination_file_count=0,
            source_bytes=0,
            destination_bytes=0,
        )
    if not source.is_dir():
        return ArchiveVerificationSummary(
            success=False,
            error=f"source path is not a directory: {source}",
            source_file_count=0,
            destination_file_count=0,
            source_bytes=0,
            destination_bytes=0,
        )
    if not destination.exists():
        return ArchiveVerificationSummary(
            success=False,
            error=f"destination directory is missing: {destination}",
            source_file_count=0,
            destination_file_count=0,
            source_bytes=0,
            destination_bytes=0,
        )
    if not destination.is_dir():
        return ArchiveVerificationSummary(
            success=False,
            error=f"destination path is not a directory: {destination}",
            source_file_count=0,
            destination_file_count=0,
            source_bytes=0,
            destination_bytes=0,
        )

    source_manifest = regular_file_manifest(source)
    destination_manifest = regular_file_manifest(destination)

    source_names = set(source_manifest)
    destination_names = set(destination_manifest)
    source_bytes = sum(size for size, _hash in source_manifest.values())
    destination_bytes = sum(size for size, _hash in destination_manifest.values())

    missing = sorted(source_names - destination_names)
    extra = sorted(destination_names - source_names)
    if missing:
        return ArchiveVerificationSummary(
            success=False,
            error=f"missing destination files: {missing}",
            source_file_count=len(source_manifest),
            destination_file_count=len(destination_manifest),
            source_bytes=source_bytes,
            destination_bytes=destination_bytes,
        )
    if extra:
        return ArchiveVerificationSummary(
            success=False,
            error=f"unexpected destination files: {extra}",
            source_file_count=len(source_manifest),
            destination_file_count=len(destination_manifest),
            source_bytes=source_bytes,
            destination_bytes=destination_bytes,
        )

    for relative in sorted(source_names):
        source_size, source_hash = source_manifest[relative]
        destination_size, destination_hash = destination_manifest[relative]
        if source_size != destination_size:
            return ArchiveVerificationSummary(
                success=False,
                error=(
                    f"size mismatch for {relative}: "
                    f"source={source_size} destination={destination_size}"
                ),
                source_file_count=len(source_manifest),
                destination_file_count=len(destination_manifest),
                source_bytes=source_bytes,
                destination_bytes=destination_bytes,
            )
        if source_hash != destination_hash:
            return ArchiveVerificationSummary(
                success=False,
                error=f"SHA-256 mismatch for {relative}",
                source_file_count=len(source_manifest),
                destination_file_count=len(destination_manifest),
                source_bytes=source_bytes,
                destination_bytes=destination_bytes,
            )

    return ArchiveVerificationSummary(
        success=True,
        error="",
        source_file_count=len(source_manifest),
        destination_file_count=len(destination_manifest),
        source_bytes=source_bytes,
        destination_bytes=destination_bytes,
    )


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


def path_is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def read_json_mapping(path: Path) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        return None, f"could not read JSON: {exc}"
    if not isinstance(data, dict):
        return None, "JSON root is not an object"
    return data, None


def json_default(value: Any) -> Any:
    """Convert explicitly supported objects at JSON boundaries."""

    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    """Atomically write a complete JSON metadata file."""

    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                default=json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


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


def resolve_archive_backend(settings: ArchiveSettings) -> tuple[str, Optional[str]]:
    requested = settings.backend
    if requested == "auto":
        backend = "robocopy" if os.name == "nt" else "rsync"
    else:
        backend = requested

    if backend == "robocopy":
        executable = resolve_executable(settings.robocopy_executable)
    elif backend == "rsync":
        executable = resolve_executable(settings.rsync_executable)
    else:
        raise ValueError(f"Unsupported archive backend: {backend}")

    return backend, executable


class SleepInhibitor:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._caffeinate_process: Optional[subprocess.Popen[Any]] = None
        self._windows_active = False

    def __enter__(self) -> "SleepInhibitor":
        if not self.enabled:
            return self

        if os.name == "nt":
            import ctypes

            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001

            result = ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            )
            if result == 0:
                raise RuntimeError("Windows SetThreadExecutionState failed")

            self._windows_active = True
            LOG.info("Windows system sleep prevention enabled")
        elif sys.platform == "darwin":
            caffeinate = shutil.which("caffeinate")
            if caffeinate is None:
                raise RuntimeError("caffeinate was not found on macOS")

            self._caffeinate_process = subprocess.Popen(
                [caffeinate, "-i", "-w", str(os.getpid())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            LOG.info("macOS system sleep prevention enabled")
        else:
            LOG.warning("Sleep prevention is not implemented for platform %s", sys.platform)

        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._windows_active:
            import ctypes

            ES_CONTINUOUS = 0x80000000
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)  # type: ignore[attr-defined]
            self._windows_active = False
            LOG.info("Windows system sleep prevention released")

        if self._caffeinate_process is not None:
            self._caffeinate_process.terminate()
            try:
                self._caffeinate_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._caffeinate_process.kill()
            self._caffeinate_process = None
            LOG.info("macOS system sleep prevention released")


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

    copy_backend, copy_executable_path = resolve_archive_backend(settings)
    if copy_executable_path is None:
        configured = (
            settings.robocopy_executable
            if copy_backend == "robocopy"
            else settings.rsync_executable
        )
        errors.append(f"{copy_backend} executable was not found: {configured}")

    required_mount_exists = required_mount_point.exists()
    required_mount_is_dir = required_mount_exists and required_mount_point.is_dir()
    required_mount_is_mount = required_mount_is_dir and os.path.ismount(required_mount_point)
    if not required_mount_exists:
        errors.append(f"required mount point does not exist: {required_mount_point}")
    elif not required_mount_is_dir:
        errors.append(f"required mount point is not a directory: {required_mount_point}")
    elif not required_mount_is_mount:
        errors.append(f"required mount point is not a mounted filesystem: {required_mount_point}")

    destination_created = False
    destination_writable = False
    destination_free_gb: Optional[float] = None
    local_free_gb: Optional[float] = None
    path_conflict = paths_overlap(destination_root, local_output_root)
    destination_within_mount = path_is_within(destination_root, required_mount_point)

    if not destination_within_mount:
        errors.append(
            f"archive destination {destination_root} is not inside required mount point {required_mount_point}"
        )

    if path_conflict:
        errors.append(
            f"archive destination {destination_root} conflicts with local output root {local_output_root}"
        )

    if required_mount_is_mount and destination_within_mount and not path_conflict:
        try:
            destination_root.mkdir(parents=True, exist_ok=True)
            destination_created = True
        except Exception as exc:
            errors.append(f"could not create archive destination root {destination_root}: {exc}")

    if required_mount_is_mount and destination_within_mount and destination_root.exists():
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
        platform=os.name,
        copy_backend=copy_backend,
        copy_executable_path=copy_executable_path,
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
    started_local: str
    completed_utc: Optional[str]
    completed_local: Optional[str]
    copy_backend: str
    copy_return_code: Optional[int]
    copy_output_tail: Optional[str]
    verification_method: str
    source_file_count: Optional[int]
    destination_file_count: Optional[int]
    source_bytes: Optional[int]
    destination_bytes: Optional[int]
    verification_succeeded: bool
    promoted_from_partial: bool
    local_deleted: bool
    error: Optional[str]


@dataclasses.dataclass(frozen=True)
class ArchiveCopyRun:
    backend: str
    return_code: int
    output_tail: str


def archive_copy_succeeded(run: ArchiveCopyRun) -> bool:
    if run.backend == "robocopy":
        return run.return_code < 8
    if run.backend == "rsync":
        return run.return_code == 0
    raise ValueError(f"Unsupported archive backend: {run.backend}")


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
        archive_failure_event: threading.Event,
        preflight: ArchivePreflightResult,
    ) -> None:
        self.settings = settings
        self.local_session_dir = local_session_dir
        self.project = project
        self.subject = subject
        self.session_name = session_name
        self.archive_failure_event = archive_failure_event
        self.preflight = preflight
        self.copy_backend = preflight.copy_backend
        self.copy_executable_path = preflight.copy_executable_path
        self.archive_session_dir = Path(preflight.archive_session_dir)
        self.incoming_dir = self.archive_session_dir / ".incoming"
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
        self._failure_message: Optional[str] = None
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

    def close_transfers(self) -> None:
        if not self.settings.enabled or self._closed:
            return
        self._closed = True
        if self._started:
            self._queue.put(None)
            self._queue.join()
            if self._worker.is_alive():
                self._worker.join(timeout=30)
            if self._worker.is_alive():
                raise RuntimeError("Archive worker did not stop within 30 seconds")

        summary = self._build_summary()
        write_json(self.local_summary_path, summary)

    def copy_final_metadata(self) -> None:
        if not self.settings.enabled:
            return
        flush_logging_handlers()
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
        completed_at = utc_now()
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
            "failure_message": self._failure_message,
            "results": [dataclasses.asdict(result) for result in self._results],
            "completed_utc": isoformat_utc(completed_at),
            "completed_local": isoformat_local(completed_at),
        }

    def _ensure_robocopy_partial_destination(self, destination: Path) -> None:
        expected_parent = self.incoming_dir.resolve(strict=False)
        actual_parent = destination.parent.resolve(strict=False)
        if actual_parent != expected_parent:
            raise RuntimeError(
                f"Refusing Robocopy /MIR outside archive incoming directory: {destination}"
            )
        if not destination.name.endswith(".partial"):
            raise RuntimeError(
                f"Refusing Robocopy /MIR to a non-partial directory: {destination}"
            )

    def _copy_into_partial(self, source: Path, destination: Path) -> ArchiveCopyRun:
        backend = self.copy_backend
        executable = self.copy_executable_path
        if executable is None:
            raise RuntimeError(f"{backend} executable path is unavailable")

        if backend == "rsync":
            command = [
                executable,
                "-a",
                "--partial",
                f"{source}/",
                f"{destination}/",
            ]
        elif backend == "robocopy":
            self._ensure_robocopy_partial_destination(destination)
            command = [
                executable,
                str(source),
                str(destination),
                "/MIR",
                "/Z",
                "/R:3",
                "/W:5",
                "/COPY:DAT",
                "/DCOPY:DAT",
                "/XJ",
                "/NP",
                "/NFL",
                "/NDL",
                "/NJH",
                "/NJS",
            ]
        else:
            raise RuntimeError(f"Unsupported archive backend: {backend}")

        try:
            copy = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                check=False,
                timeout=self.settings.copy_timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{backend} timed out after {self.settings.copy_timeout_s:.0f} seconds"
            ) from exc

        return ArchiveCopyRun(
            backend=backend,
            return_code=copy.returncode,
            output_tail=(copy.stdout or "")[-8000:],
        )

    def _append_transfer_record(self, result: ArchiveResult) -> None:
        self.local_transfers_path.parent.mkdir(parents=True, exist_ok=True)
        with self.local_transfers_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    dataclasses.asdict(result),
                    separators=(",", ":"),
                    default=json_default,
                )
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _mark_failure(self, error: str) -> None:
        if not self._failure:
            LOG.error("Archive failure detected: %s", error)
        self._failure = True
        self._failure_message = error
        self.archive_failure_event.set()

    def _verify_tree(self, source: Path, destination: Path) -> ArchiveVerificationSummary:
        try:
            summary = verify_file_trees(source, destination)
        except Exception as exc:
            return ArchiveVerificationSummary(
                success=False,
                error=f"verification raised {type(exc).__name__}: {exc}",
                source_file_count=0,
                destination_file_count=0,
                source_bytes=0,
                destination_bytes=0,
            )

        if summary.success:
            LOG.info(
                "Verified archive %s: %d files, %d bytes, SHA-256 content match",
                source.name,
                summary.source_file_count,
                summary.source_bytes,
            )
        else:
            LOG.error(
                "Archive verification failed for %s -> %s: %s",
                source,
                destination,
                summary.error,
            )
        return summary

    def _transfer_clip(self, clip_dir: Path) -> ArchiveResult:
        started_at = utc_now()
        started_utc = isoformat_utc(started_at)
        started_local = isoformat_local(started_at)
        clip_name = clip_dir.name
        final_destination = self.archive_session_dir / clip_name
        partial_destination = self.incoming_dir / f"{clip_name}.partial"
        bytes_transferred: Optional[int] = None
        completed_utc: Optional[str] = None
        completed_local: Optional[str] = None
        copy_return_code: Optional[int] = None
        copy_output_tail: Optional[str] = None
        verification_method = "sha256_manifest"
        source_file_count: Optional[int] = None
        destination_file_count: Optional[int] = None
        source_bytes: Optional[int] = None
        destination_bytes: Optional[int] = None
        verification_succeeded = False
        promoted_from_partial = False
        local_deleted = False
        error: Optional[str] = None
        success = False

        try:
            if not clip_dir.exists():
                raise FileNotFoundError(f"Source clip directory does not exist: {clip_dir}")

            source_file_count, source_bytes = tree_stats(clip_dir)
            bytes_transferred = source_bytes

            if final_destination.exists():
                verification = self._verify_tree(clip_dir, final_destination)
                source_file_count = verification.source_file_count
                destination_file_count = verification.destination_file_count
                source_bytes = verification.source_bytes
                destination_bytes = verification.destination_bytes
                if not verification.success:
                    raise RuntimeError(
                        f"destination already exists but does not match source: {verification.error}"
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
                    LOG.warning("Checking existing partial archive directory %s", partial_destination)
                    verification = self._verify_tree(clip_dir, partial_destination)
                    source_file_count = verification.source_file_count
                    destination_file_count = verification.destination_file_count
                    source_bytes = verification.source_bytes
                    destination_bytes = verification.destination_bytes
                    if verification.success:
                        partial_destination.replace(final_destination)
                        if not final_destination.is_dir():
                            raise RuntimeError(
                                f"Archive promotion failed; destination missing: {final_destination}"
                            )
                        verification_succeeded = True
                        promoted_from_partial = True
                        success = True
                        self._successful += 1
                        self._bytes_transferred += bytes_transferred or 0
                    else:
                        LOG.warning(
                            "Existing partial archive does not yet match source; retrying %s into %s",
                            self.copy_backend,
                            partial_destination,
                        )

                if not success:
                    reserve_bytes = int(
                        self.settings.min_external_free_gb_before_transfer * 1024**3
                    )
                    destination_free_bytes = shutil.disk_usage(self.archive_session_dir).free
                    required_free_bytes = (source_bytes or 0) + reserve_bytes
                    if destination_free_bytes < required_free_bytes:
                        raise RuntimeError(
                            "archive destination does not have enough free space: "
                            f"free={bytes_to_gib(destination_free_bytes):.1f} GiB, "
                            f"clip={bytes_to_gib(source_bytes or 0):.1f} GiB, "
                            f"reserve={self.settings.min_external_free_gb_before_transfer:.1f} GiB"
                        )

                    copy_run = self._copy_into_partial(clip_dir, partial_destination)
                    copy_return_code = copy_run.return_code
                    copy_output_tail = copy_run.output_tail or None
                    if not archive_copy_succeeded(copy_run):
                        raise RuntimeError(
                            f"{copy_run.backend} returned {copy_run.return_code}: "
                            f"{(copy_output_tail or 'no output').strip()}"
                        )

                    verification = self._verify_tree(clip_dir, partial_destination)
                    source_file_count = verification.source_file_count
                    destination_file_count = verification.destination_file_count
                    source_bytes = verification.source_bytes
                    destination_bytes = verification.destination_bytes
                    if not verification.success:
                        raise RuntimeError(f"verification failed: {verification.error}")

                    if final_destination.exists():
                        raise RuntimeError(
                            f"Final archive destination unexpectedly exists: {final_destination}"
                        )
                    partial_destination.replace(final_destination)
                    if not final_destination.is_dir():
                        raise RuntimeError(
                            f"Archive promotion failed; destination missing: {final_destination}"
                        )
                    promoted_from_partial = True
                    success = True
                    self._successful += 1
                    self._bytes_transferred += bytes_transferred or 0

                verification_succeeded = True

            if success and self.settings.delete_local_clip_after_verified_transfer:
                shutil.rmtree(clip_dir)
                local_deleted = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._failed += 1
            self._mark_failure(error)
            LOG.exception("Archive failed for %s", clip_dir)
        else:
            completed_at = utc_now()
            completed_utc = isoformat_utc(completed_at)
            completed_local = isoformat_local(completed_at)
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
                with self._lock:
                    self._pending_local_clips = max(0, self._pending_local_clips - 1)
            result = ArchiveResult(
                clip_name=clip_name,
                source_path=str(clip_dir),
                destination_path=str(final_destination),
                success=success,
                bytes_transferred=bytes_transferred,
                started_utc=started_utc,
                started_local=started_local,
                completed_utc=completed_utc,
                completed_local=completed_local,
                copy_backend=self.copy_backend,
                copy_return_code=copy_return_code,
                copy_output_tail=copy_output_tail,
                verification_method=verification_method,
                source_file_count=source_file_count,
                destination_file_count=destination_file_count,
                source_bytes=source_bytes,
                destination_bytes=destination_bytes,
                verification_succeeded=verification_succeeded,
                promoted_from_partial=promoted_from_partial,
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
                    started_utc=isoformat_utc(utc_now()),
                    started_local=isoformat_local(utc_now()),
                    completed_utc=None,
                    completed_local=None,
                    copy_backend=self.copy_backend,
                    copy_return_code=None,
                    copy_output_tail=None,
                    verification_method="sha256_manifest",
                    source_file_count=None,
                    destination_file_count=None,
                    source_bytes=None,
                    destination_bytes=None,
                    verification_succeeded=False,
                    promoted_from_partial=False,
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


def draw_preview_status(
    preview: np.ndarray,
    *,
    label: str,
    receive_fps_text: str,
) -> None:
    _height, width = preview.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45 if width < 600 else 0.55
    thickness = 1

    cv2.putText(
        preview,
        f"{label} | {receive_fps_text}",
        (10, 20),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "q quit | s snapshot | p settings",
        (10, 40),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


def update_setup_preview_fps(
    *,
    now: float,
    fps_window_start: Optional[float],
    fps_window_frames: int,
    displayed_fps: Optional[float],
) -> tuple[Optional[float], Optional[float], int]:
    if fps_window_start is None or now < fps_window_start or now - fps_window_start > 5.0:
        return None, now, 1

    fps_window_frames += 1
    elapsed = now - fps_window_start
    if elapsed >= 1.0:
        return (fps_window_frames - 1) / elapsed, now, 1
    return displayed_fps, fps_window_start, fps_window_frames


def log_recording_heartbeat(
    *,
    clip_index: int,
    total_clips: int,
    planned_start_mono_ns: int,
    planned_stop_mono_ns: int,
    session_start_mono_ns: int,
    planned_session_duration_s: float,
    planned_finish_utc: dt.datetime,
) -> None:
    now_mono_ns = time.monotonic_ns()
    clip_total_s = (planned_stop_mono_ns - planned_start_mono_ns) / 1e9
    clip_elapsed_s = max(
        0.0,
        min(
            clip_total_s,
            (now_mono_ns - planned_start_mono_ns) / 1e9,
        ),
    )
    session_elapsed_s = max(0.0, (now_mono_ns - session_start_mono_ns) / 1e9)
    remaining_s = max(0.0, planned_session_duration_s - session_elapsed_s)

    LOG.info(
        "STATUS | recording active | clip %d/%d | clip %s/%s | session %s/%s | remaining ~%s | finish ~%s local",
        clip_index + 1,
        total_clips,
        format_clock_duration(clip_elapsed_s),
        format_clock_duration(clip_total_s),
        format_clock_duration(session_elapsed_s),
        format_clock_duration(planned_session_duration_s),
        format_clock_duration(remaining_s),
        format_local_finish_time(planned_finish_utc),
    )


def monitor_recording_threads(
    threads: list[threading.Thread],
    preview_queues: dict[str, queue.Queue[PreviewPacket]],
    preview_settings: RecordingPreviewSettings,
    preview_active_event: threading.Event,
    clip_dir: Path,
    *,
    clip_index: int,
    total_clips: int,
    planned_start_mono_ns: int,
    planned_stop_mono_ns: int,
    session_start_mono_ns: int,
    planned_session_duration_s: float,
    planned_finish_utc: dt.datetime,
    terminal_interval_s: float,
) -> None:
    """Keep recording windows responsive while camera workers acquire frames."""

    latest: dict[str, PreviewPacket] = {}
    window_names: set[str] = set()
    snapshot_count = 0
    next_terminal_status_mono = time.monotonic()

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

            now_mono = time.monotonic()
            if terminal_interval_s > 0 and now_mono >= next_terminal_status_mono:
                log_recording_heartbeat(
                    clip_index=clip_index,
                    total_clips=total_clips,
                    planned_start_mono_ns=planned_start_mono_ns,
                    planned_stop_mono_ns=planned_stop_mono_ns,
                    session_start_mono_ns=session_start_mono_ns,
                    planned_session_duration_s=planned_session_duration_s,
                    planned_finish_utc=planned_finish_utc,
                )
                next_terminal_status_mono = now_mono + terminal_interval_s

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
    total_clips: int,
    clip_start_utc: dt.datetime,
    planned_start_mono_ns: int,
    planned_stop_mono_ns: int,
    session_start_mono_ns: int,
    planned_session_duration_s: float,
    planned_finish_utc: dt.datetime,
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
    actual_fps_value = binding.actual_settings.get("AcquisitionFrameRate")
    if actual_fps_value is None:
        actual_fps_value = requested_fps
    actual_fps = float(actual_fps_value)
    requested_auto_exposure = bool(binding.requested.get("auto_exposure", False))
    requested_auto_exposure_upper_us = (
        float(binding.requested["auto_exposure_upper_us"])
        if binding.requested.get("auto_exposure_upper_us") is not None
        else None
    )
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
    clip_timezone = local_timezone_metadata(clip_start_utc)

    metadata: dict[str, Any] = {
        "camera": binding.info,
        "label": label,
        "clip_index": clip_index,
        "planned_start_utc": isoformat_utc(clip_start_utc),
        "planned_start_local": isoformat_local(clip_start_utc),
        "planned_finish_utc": isoformat_utc(planned_finish_utc),
        "planned_finish_local": isoformat_local(planned_finish_utc),
        "planned_duration_s": (planned_stop_mono_ns - planned_start_mono_ns) / 1e9,
        "requested_settings": binding.requested,
        "actual_settings": binding.actual_settings,
        "encoding": encoding_cfg,
        "recording_preview": dataclasses.asdict(preview_settings),
        "timestamp_policy": {
            "canonical_wall_clock": "UTC",
            "human_display": "local_with_numeric_offset",
            "duration_clock": "monotonic",
            "local_timezone_label": clip_timezone["local_timezone_label"],
            "local_utc_offset_at_start": clip_timezone["local_utc_offset"],
        },
        "local_timezone_label": clip_timezone["local_timezone_label"],
        "local_utc_offset_at_clip_start": clip_timezone["local_utc_offset"],
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
                        current_exposure_us: Optional[float] = None
                        if (
                            first_host_mono_ns is not None
                            and host_mono_ns > first_host_mono_ns
                            and frame_count > 1
                        ):
                            measured_preview_fps = (frame_count - 1) / (
                                (host_mono_ns - first_host_mono_ns) / 1e9
                            )
                        try:
                            current_exposure_us = read_first_available_setting(
                                camera,
                                ("ExposureTime", "ExposureTimeAbs"),
                            )
                            if current_exposure_us is not None:
                                current_exposure_us = float(current_exposure_us)
                        except Exception:
                            current_exposure_us = None
                        monitor_frame = resize_to_fit(
                            frame,
                            max_width=preview_settings.max_width,
                            max_height=preview_settings.max_height,
                        )
                        put_latest_preview(
                            preview_queue,
                            PreviewPacket(
                                label=label,
                                clip_index=clip_index,
                                total_clips=total_clips,
                                frame_index=frame_count - 1,
                                frame=monitor_frame,
                                host_monotonic_ns=host_mono_ns,
                                elapsed_s=max(0.0, (host_mono_ns - planned_start_mono_ns) / 1e9),
                                planned_duration_s=(planned_stop_mono_ns - planned_start_mono_ns) / 1e9,
                                session_elapsed_s=max(0.0, (host_mono_ns - session_start_mono_ns) / 1e9),
                                planned_session_duration_s=planned_session_duration_s,
                                planned_finish_utc=planned_finish_utc,
                                measured_receive_fps=measured_preview_fps,
                                exposure_us=current_exposure_us,
                                auto_exposure=requested_auto_exposure,
                                auto_exposure_upper_us=requested_auto_exposure_upper_us,
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
        completed_at = utc_now()

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
                "first_host_local": (
                    isoformat_local(dt.datetime.fromtimestamp(first_host_utc_ns / 1e9, tz=dt.timezone.utc))
                    if first_host_utc_ns
                    else None
                ),
                "last_host_utc_ns": last_host_utc_ns,
                "last_host_utc": iso_utc_from_ns(last_host_utc_ns) if last_host_utc_ns else None,
                "last_host_local": (
                    isoformat_local(dt.datetime.fromtimestamp(last_host_utc_ns / 1e9, tz=dt.timezone.utc))
                    if last_host_utc_ns
                    else None
                ),
                "first_host_monotonic_ns": first_host_mono_ns,
                "last_host_monotonic_ns": last_host_mono_ns,
                "actual_start_utc": iso_utc_from_ns(first_host_utc_ns) if first_host_utc_ns else None,
                "actual_start_local": (
                    isoformat_local(dt.datetime.fromtimestamp(first_host_utc_ns / 1e9, tz=dt.timezone.utc))
                    if first_host_utc_ns
                    else None
                ),
                "actual_stop_utc": iso_utc_from_ns(last_host_utc_ns) if last_host_utc_ns else None,
                "actual_stop_local": (
                    isoformat_local(dt.datetime.fromtimestamp(last_host_utc_ns / 1e9, tz=dt.timezone.utc))
                    if last_host_utc_ns
                    else None
                ),
                "first_camera_timestamp": first_camera_timestamp,
                "last_camera_timestamp": last_camera_timestamp,
                "actual_elapsed_s": actual_elapsed_s,
                "measured_receive_fps": measured_fps,
                "ffmpeg_return_code": ffmpeg_return_code,
                "mp4_remux_succeeded": remuxed,
                "completed_utc": isoformat_utc(completed_at),
                "completed_local": isoformat_local(completed_at),
            }
        )
        write_json(metadata_path, metadata)

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
    preview_raw = config.get("preview")
    if preview_raw is None:
        preview_raw = config.get("recording_preview", {})
    if preview_raw is None:
        preview_raw = {}
    if not isinstance(preview_raw, dict):
        raise ValueError("config.preview must be a mapping/object")
    preview_max_width = int(preview_raw.get("max_width", 600))
    preview_max_height = int(preview_raw.get("max_height", 640))
    preview_show_status = bool(preview_raw.get("show_status", True))
    if preview_max_width <= 0:
        raise ValueError("preview.max_width must be positive")
    if preview_max_height <= 0:
        raise ValueError("preview.max_height must be positive")

    window = f"Basler preview - {binding.label} (q quit, s snapshot, p print settings)"
    snapshot_count = 0
    logged_preview_size = False
    resized_window = False
    try:
        camera = binding.camera
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        fps_window_start: Optional[float] = None
        fps_window_frames = 0
        displayed_fps: Optional[float] = None
        while camera.IsGrabbing():
            grab = camera.RetrieveResult(3000, pylon.TimeoutHandling_ThrowException)
            try:
                if not grab.GrabSucceeded():
                    continue
                frame = apply_frame_transform(binding.converter.Convert(grab).GetArray(), binding.requested)
                now = time.monotonic()
                displayed_fps, fps_window_start, fps_window_frames = update_setup_preview_fps(
                    now=now,
                    fps_window_start=fps_window_start,
                    fps_window_frames=fps_window_frames,
                    displayed_fps=displayed_fps,
                )
                fps_text = "measuring..." if displayed_fps is None else f"{displayed_fps:.1f} fps"
                preview = resize_to_fit(
                    frame,
                    max_width=preview_max_width,
                    max_height=preview_max_height,
                )
                if preview_show_status:
                    draw_preview_status(
                        preview,
                        label=binding.label,
                        receive_fps_text=fps_text,
                    )
                if not logged_preview_size:
                    LOG.info(
                        "Preview display size for %s: source=%dx%d display=%dx%d limits=%dx%d",
                        binding.label,
                        frame.shape[1],
                        frame.shape[0],
                        preview.shape[1],
                        preview.shape[0],
                        preview_max_width,
                        preview_max_height,
                    )
                    logged_preview_size = True
                if not resized_window:
                    cv2.resizeWindow(window, preview.shape[1], preview.shape[0])
                    resized_window = True
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
                    exposure_value = read_first_available_setting(
                        binding.camera,
                        ("ExposureTime", "ExposureTimeAbs"),
                    )
                    gain_value = read_first_available_setting(binding.camera, ("Gain", "GainRaw"))
                    fps_value = read_first_available_setting(
                        binding.camera,
                        ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"),
                    )
                    resulting_fps_value = read_first_available_setting(
                        binding.camera,
                        ("ResultingFrameRate", "ResultingFrameRateAbs"),
                    )
                    LOG.info(
                        "%s current settings: exposure_us=%r gain=%r fps=%r resulting_fps=%r",
                        binding.label,
                        exposure_value,
                        gain_value,
                        fps_value,
                        resulting_fps_value,
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


def expected_clip_count(
    *,
    interval_s: float,
    total_duration_s: Optional[float],
    number_of_clips: Optional[int],
) -> int:
    """Return the effective number of clips allowed by the configured schedule."""

    limits: list[int] = []
    if number_of_clips is not None:
        limits.append(number_of_clips)
    if total_duration_s is not None:
        limits.append(max(1, math.ceil(total_duration_s / interval_s)))
    if not limits:
        raise ValueError("At least one schedule limit is required")
    return min(limits)


def planned_session_span_s(
    *,
    clip_count: int,
    clip_duration_s: float,
    interval_s: float,
) -> float:
    """Return the planned session wall-clock span from first start to final planned end."""

    if clip_count <= 0:
        return 0.0
    return ((clip_count - 1) * interval_s) + clip_duration_s


def format_local_finish_time(
    finish_utc: dt.datetime,
    *,
    now_utc: Optional[dt.datetime] = None,
) -> str:
    """Format an approximate finish time in the computer's local timezone."""

    if finish_utc.tzinfo is None:
        raise ValueError("finish_utc must be timezone-aware")

    reference_utc = now_utc or utc_now()
    local_finish = finish_utc.astimezone()
    local_now = reference_utc.astimezone()
    if local_finish.date() == local_now.date():
        return local_finish.strftime("%H:%M")
    return local_finish.strftime("%Y-%m-%d %H:%M")


PREVIEW_PANEL_BACKGROUND = (42, 27, 24)
PREVIEW_FOOTER_BACKGROUND = (56, 36, 32)
PREVIEW_CARD_FILL = (71, 48, 43)
PREVIEW_CARD_BORDER = (106, 79, 72)
PREVIEW_PRIMARY_TEXT = (250, 246, 244)
PREVIEW_SECONDARY_TEXT = (198, 180, 173)
PREVIEW_PURPLE = (255, 108, 155)
PREVIEW_TRACK = (97, 74, 68)
PREVIEW_RED = (94, 77, 255)
PREVIEW_GREEN = (115, 210, 82)
PREVIEW_AMBER = (77, 180, 240)


@dataclasses.dataclass(frozen=True)
class PreviewCardPanelLayout:
    panel_x: int
    panel_width: int
    footer_y: int
    footer_height: int
    recording_card: tuple[int, int, int, int]
    session_card: tuple[int, int, int, int]
    mode: str
    recording_mode: str

    @property
    def compact_mode(self) -> bool:
        return self.mode != "full"


def clamp_progress(value: float) -> float:
    return max(0.0, min(1.0, value))


def _text_size(
    text: str,
    *,
    scale: float,
    thickness: int,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> tuple[int, int, int]:
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    return width, height, baseline


def _put_preview_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> None:
    cv2.putText(image, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def _draw_solid_rounded_rect(
    image: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    if width <= 0 or height <= 0:
        return

    image_h, image_w = image.shape[:2]
    if image_w <= 0 or image_h <= 0:
        return

    x = max(0, min(x, image_w - 1))
    y = max(0, min(y, image_h - 1))
    width = min(width, image_w - x)
    height = min(height, image_h - y)
    if width <= 0 or height <= 0:
        return

    radius = max(0, min(radius, width // 2, height // 2))
    x2 = x + width - 1
    y2 = y + height - 1

    if radius == 0:
        cv2.rectangle(image, (x, y), (x2, y2), color, -1)
        return

    cv2.rectangle(image, (x + radius, y), (x2 - radius, y2), color, -1)
    cv2.rectangle(image, (x, y + radius), (x2, y2 - radius), color, -1)
    cv2.circle(image, (x + radius, y + radius), radius, color, -1)
    cv2.circle(image, (x2 - radius, y + radius), radius, color, -1)
    cv2.circle(image, (x + radius, y2 - radius), radius, color, -1)
    cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)


def _draw_filled_rounded_rect(
    image: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    radius: int,
    fill_color: tuple[int, int, int],
    border_color: Optional[tuple[int, int, int]] = None,
    border_width: int = 1,
) -> None:
    if border_color is not None and border_width > 0:
        _draw_solid_rounded_rect(
            image,
            x=x,
            y=y,
            width=width,
            height=height,
            radius=radius,
            color=border_color,
        )
        inset_x = x + border_width
        inset_y = y + border_width
        inset_width = width - (2 * border_width)
        inset_height = height - (2 * border_width)
        if inset_width <= 0 or inset_height <= 0:
            return
        _draw_solid_rounded_rect(
            image,
            x=inset_x,
            y=inset_y,
            width=inset_width,
            height=inset_height,
            radius=max(0, radius - border_width),
            color=fill_color,
        )
        return

    _draw_solid_rounded_rect(
        image,
        x=x,
        y=y,
        width=width,
        height=height,
        radius=radius,
        color=fill_color,
    )


def _draw_preview_progress_bar(
    image: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    progress: float,
    track_color: tuple[int, int, int],
    fill_color: tuple[int, int, int],
    border_color: Optional[tuple[int, int, int]] = None,
) -> None:
    if width <= 0 or height <= 0:
        return

    progress = clamp_progress(progress)
    radius = max(0, min(height // 2, width // 2))
    _draw_filled_rounded_rect(
        image,
        x=x,
        y=y,
        width=width,
        height=height,
        radius=radius,
        fill_color=track_color,
        border_color=border_color,
        border_width=1 if border_color is not None else 0,
    )

    inner_x = x + (1 if border_color is not None else 0)
    inner_y = y + (1 if border_color is not None else 0)
    inner_width = width - (2 if border_color is not None else 0)
    inner_height = height - (2 if border_color is not None else 0)
    if inner_width <= 0 or inner_height <= 0:
        return

    fill_width = int(round(inner_width * progress))
    if fill_width <= 0:
        return
    _draw_solid_rounded_rect(
        image,
        x=inner_x,
        y=inner_y,
        width=fill_width,
        height=inner_height,
        radius=max(0, min(inner_height // 2, fill_width // 2)),
        color=fill_color,
    )


def _ellipsize_preview_text(
    text: str,
    *,
    max_width: int,
    font: int,
    font_scale: float,
    thickness: int,
) -> str:
    if max_width <= 0:
        return ""

    text_width, _text_height, _baseline = _text_size(
        text,
        scale=font_scale,
        thickness=thickness,
        font=font,
    )
    if text_width <= max_width:
        return text

    ellipsis = "..."
    ellipsis_width, _ellipsis_height, _ellipsis_baseline = _text_size(
        ellipsis,
        scale=font_scale,
        thickness=thickness,
        font=font,
    )
    if ellipsis_width > max_width:
        return ""

    low = 0
    high = len(text)
    best = ellipsis
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid].rstrip() + ellipsis
        candidate_width, _candidate_height, _candidate_baseline = _text_size(
            candidate,
            scale=font_scale,
            thickness=thickness,
            font=font,
        )
        if candidate_width <= max_width:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def _preview_card_scales(card_width: int, *, mode: str) -> tuple[float, float, float]:
    if card_width < 180:
        if mode == "full":
            return 0.34, 0.36, 0.42
        if mode == "compact":
            return 0.28, 0.30, 0.36
        return 0.24, 0.26, 0.32
    if mode == "full":
        return 0.38, 0.40, 0.46
    if mode == "compact":
        return 0.30, 0.32, 0.38
    return 0.26, 0.28, 0.34


def _preview_text_line_height(
    text: str,
    *,
    scale: float,
    thickness: int = 1,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> int:
    return _text_size(text, scale=scale, thickness=thickness, font=font)[1]


def _preview_block_height(
    lines: list[str],
    *,
    scale: float,
    thickness: int = 1,
    line_gap: int = 2,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> int:
    if not lines:
        return 0
    height = 0
    for index, line in enumerate(lines):
        if index:
            height += line_gap
        height += _preview_text_line_height(line, scale=scale, thickness=thickness, font=font)
    return height


def _measure_preview_group_height(
    title: str,
    value_lines: list[str],
    *,
    title_scale: float,
    value_scale: float,
    mode: str,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> int:
    title_height = _preview_text_line_height(title, scale=title_scale, font=font)
    value_height = _preview_block_height(value_lines, scale=value_scale, font=font)
    if mode == "minimal":
        return title_height + 2 + value_height + 4
    if mode == "compact":
        return title_height + 3 + value_height + 5
    return title_height + 4 + value_height + 6


def _measure_recording_card_height(card_width: int, *, mode: str) -> int:
    _, small_scale, value_scale = _preview_card_scales(card_width, mode=mode)
    font = cv2.FONT_HERSHEY_SIMPLEX
    status_height = _preview_text_line_height("RECORDING" if mode == "full" else "REC", scale=small_scale, font=font)
    clip_height = _preview_text_line_height("00:00 / 00:00", scale=value_scale, font=font)
    fps_height = _preview_text_line_height("0.00 fps", scale=small_scale, font=font)
    exposure_height = _preview_text_line_height("AUTO EXP 180.0 ms  MAX", scale=small_scale, font=font)
    bar_height = 7
    top_pad = 16
    between_status_and_clip = 10
    between_clip_and_fps = 10
    bottom_pad = 13
    if mode == "full":
        return (
            top_pad
            + status_height
            + between_status_and_clip
            + clip_height
            + between_clip_and_fps
            + fps_height
            + 4
            + exposure_height
            + 10
            + bar_height
            + bottom_pad
        )
    return top_pad + status_height + between_status_and_clip + clip_height + 10 + bar_height + bottom_pad


def _measure_session_card_height(card_width: int, *, mode: str) -> int:
    label_scale, _small_scale, value_scale = _preview_card_scales(card_width, mode=mode)
    font = cv2.FONT_HERSHEY_SIMPLEX
    separator_gap = 12
    block_gap = 8
    current_height = 13 if mode == "minimal" else 14

    current_height += _measure_preview_group_height(
        "SESSION",
        ["00:00 / 00:00"],
        title_scale=label_scale,
        value_scale=value_scale,
        mode=mode,
        font=font,
    )
    current_height += 7 + (8 if mode == "minimal" else 10)
    if mode == "minimal":
        current_height += _measure_preview_group_height(
            "REMAINING",
            ["00:00:00"],
            title_scale=label_scale,
            value_scale=value_scale,
            mode=mode,
            font=font,
        )
        return current_height + 4

    current_height += separator_gap
    current_height += _measure_preview_group_height(
        "REMAINING",
        ["00:00:00"],
        title_scale=label_scale,
        value_scale=value_scale,
        mode=mode,
        font=font,
    )
    current_height += separator_gap

    finish_lines = ["2026-08-06", "06:59"] if mode == "full" else ["2026-08-06 06:59"]
    current_height += _measure_preview_group_height(
        "EST. FINISH",
        finish_lines,
        title_scale=label_scale,
        value_scale=value_scale,
        mode=mode,
        font=font,
    )
    current_height += separator_gap

    current_height += _measure_preview_group_height(
        "CLIP",
        ["1 / 14"],
        title_scale=label_scale,
        value_scale=value_scale,
        mode=mode,
        font=font,
    )
    if mode == "full":
        current_height += separator_gap
        current_height += _measure_preview_group_height(
            "CAMERA",
            ["camera1"],
            title_scale=label_scale,
            value_scale=value_scale,
            mode=mode,
            font=font,
        )

    return current_height + block_gap


def _calculate_card_panel_layout(
    image_width: int,
    image_height: int,
    footer_height: int,
) -> PreviewCardPanelLayout:
    panel_width = int(round(image_width * 0.42))
    panel_width = max(176, min(panel_width, 216))
    panel_x = image_width
    outer_padding = 12
    gap = 10
    panel_top = outer_padding
    panel_bottom = image_height - outer_padding
    available_panel_height = max(0, panel_bottom - panel_top)
    card_x = panel_x + outer_padding
    card_width = max(0, panel_width - (2 * outer_padding))

    recording_full_required = _measure_recording_card_height(card_width, mode="full")
    recording_min_required = _measure_recording_card_height(card_width, mode="minimal")
    session_full_required = _measure_session_card_height(card_width, mode="full")
    session_compact_required = _measure_session_card_height(card_width, mode="compact")
    session_min_required = _measure_session_card_height(card_width, mode="minimal")

    recording_card_height = min(132, max(120, int(round(image_height * 0.19))))
    recording_card_height = min(recording_card_height, max(0, available_panel_height))
    recording_card_height = max(recording_min_required, recording_card_height)
    recording_card_height = min(recording_card_height, max(0, available_panel_height))
    session_card_height = max(0, available_panel_height - gap - recording_card_height)

    mode = "minimal"
    if session_card_height >= session_full_required:
        mode = "full"
    elif session_card_height >= session_compact_required:
        mode = "compact"
    elif session_card_height >= session_min_required:
        mode = "minimal"
    else:
        recording_card_height = max(
            recording_min_required,
            min(recording_card_height, max(0, available_panel_height - gap - session_min_required)),
        )
        recording_card_height = min(recording_card_height, max(0, available_panel_height))
        session_card_height = max(0, available_panel_height - gap - recording_card_height)
        if session_card_height >= session_full_required:
            mode = "full"
        elif session_card_height >= session_compact_required:
            mode = "compact"
        else:
            mode = "minimal"

    recording_card = (
        card_x,
        panel_top,
        card_width,
        recording_card_height,
    )
    session_card = (
        card_x,
        panel_top + recording_card_height + gap,
        card_width,
        session_card_height,
    )
    return PreviewCardPanelLayout(
        panel_x=panel_x,
        panel_width=panel_width,
        footer_y=image_height,
        footer_height=footer_height,
        recording_card=recording_card,
        session_card=session_card,
        mode=mode,
        recording_mode="full" if recording_card_height >= recording_full_required else "minimal",
    )


def draw_recording_preview(packet: PreviewPacket, settings: RecordingPreviewSettings) -> np.ndarray:
    """Create a display-only copy with the configured recording status layout."""

    if not settings.show_status:
        return packet.frame.copy()

    if settings.layout == "legacy_overlay":
        return _draw_recording_preview_legacy(packet)

    return _draw_recording_preview_card_panel(packet)


def _draw_recording_preview_legacy(packet: PreviewPacket) -> np.ndarray:
    display = packet.frame.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    status_font_scale = 0.40 if display.shape[1] < 500 else 0.48
    controls_font_scale = 0.34
    text_thickness = 1
    controls_text = "q hide | s snapshot"
    fps_text = (
        f"{packet.measured_receive_fps:.2f} fps"
        if packet.measured_receive_fps is not None
        else "starting"
    )
    clip_number = packet.clip_index + 1
    clip_elapsed_text = format_clock_duration(packet.elapsed_s)
    clip_total_text = format_clock_duration(packet.planned_duration_s)
    session_elapsed_text = format_clock_duration(packet.session_elapsed_s)
    session_total_text = format_clock_duration(packet.planned_session_duration_s)
    remaining_s = max(0.0, packet.planned_session_duration_s - packet.session_elapsed_s)
    remaining_text = format_clock_duration(remaining_s)
    finish_text = format_local_finish_time(packet.planned_finish_utc)
    clip_progress = 0.0
    if packet.planned_duration_s > 0:
        clip_progress = clamp_progress(packet.elapsed_s / packet.planned_duration_s)
    session_progress = 0.0
    if packet.planned_session_duration_s > 0:
        session_progress = clamp_progress(
            packet.session_elapsed_s / packet.planned_session_duration_s
        )
    lines = [
        f"REC {packet.label} | clip {clip_number}/{packet.total_clips}",
        f"clip {clip_elapsed_text}/{clip_total_text} | {fps_text}",
        f"session {session_elapsed_text}/{session_total_text}",
        f"remaining ~{remaining_text} | finish ~{finish_text}",
    ]
    (controls_width, _), _ = cv2.getTextSize(
        controls_text,
        font,
        controls_font_scale,
        text_thickness,
    )

    overlay = display.copy()
    panel_height = 116
    cv2.rectangle(
        overlay,
        (0, 0),
        (display.shape[1], panel_height),
        (24, 24, 24),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

    first_line_y = 20
    line_spacing = 21
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
        (controls_x, panel_height - 14),
        font,
        controls_font_scale,
        (235, 235, 235),
        text_thickness,
        cv2.LINE_AA,
    )

    bar_left = 12
    bar_top = 92
    bar_width = max(60, display.shape[1] - 24)
    bar_height = 6
    gap = 10

    cv2.putText(
        display,
        "clip",
        (bar_left, bar_top - 4),
        font,
        0.28,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        display,
        (bar_left, bar_top),
        (bar_left + bar_width, bar_top + bar_height),
        (82, 82, 82),
        -1,
    )
    filled_width = int(round(bar_width * clip_progress))
    if filled_width > 0:
        cv2.rectangle(
            display,
            (bar_left, bar_top),
            (bar_left + filled_width, bar_top + bar_height),
            (90, 190, 255),
            -1,
        )

    session_bar_top = bar_top + bar_height + gap
    cv2.putText(
        display,
        "all",
        (bar_left, session_bar_top - 4),
        font,
        0.28,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        display,
        (bar_left, session_bar_top),
        (bar_left + bar_width, session_bar_top + bar_height),
        (82, 82, 82),
        -1,
    )
    session_filled_width = int(round(bar_width * session_progress))
    if session_filled_width > 0:
        cv2.rectangle(
            display,
            (bar_left, session_bar_top),
            (bar_left + session_filled_width, session_bar_top + bar_height),
            (100, 220, 140),
            -1,
        )
    return display


def _draw_recording_preview_card_panel(packet: PreviewPacket) -> np.ndarray:
    frame = packet.frame
    image_h, image_w = frame.shape[:2]
    footer_height = 42 if image_w >= 360 else 38
    layout = _calculate_card_panel_layout(image_w, image_h, footer_height)

    canvas_height = image_h + footer_height
    canvas_width = image_w + layout.panel_width
    canvas = np.full((canvas_height, canvas_width, 3), PREVIEW_PANEL_BACKGROUND, dtype=np.uint8)
    canvas[:image_h, :image_w] = frame
    canvas[layout.footer_y:, :] = PREVIEW_FOOTER_BACKGROUND
    cv2.line(canvas, (0, layout.footer_y), (canvas_width - 1, layout.footer_y), PREVIEW_CARD_BORDER, 1)

    recording_x, recording_y, recording_w, recording_h = layout.recording_card
    session_x, session_y, session_w, session_h = layout.session_card
    outer_radius = 9
    if recording_w > 0 and recording_h > 0:
        _draw_filled_rounded_rect(
            canvas,
            x=recording_x,
            y=recording_y,
            width=recording_w,
            height=recording_h,
            radius=outer_radius,
            fill_color=PREVIEW_CARD_FILL,
            border_color=PREVIEW_CARD_BORDER,
            border_width=1,
        )
    if session_w > 0 and session_h > 0:
        _draw_filled_rounded_rect(
            canvas,
            x=session_x,
            y=session_y,
            width=session_w,
            height=session_h,
            radius=outer_radius,
            fill_color=PREVIEW_CARD_FILL,
            border_color=PREVIEW_CARD_BORDER,
            border_width=1,
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    recording_mode = layout.recording_mode
    session_mode = layout.mode
    label_scale, small_scale, value_scale = _preview_card_scales(recording_w, mode=recording_mode)
    session_label_scale, _session_small_scale, session_value_scale = _preview_card_scales(
        session_w,
        mode=session_mode,
    )
    large_scale = 0.52 if recording_mode == "minimal" else 0.58
    inner_pad = 13

    def put_text(
        text: str,
        x: int,
        y: int,
        *,
        scale: float,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> tuple[int, int, int]:
        text_width, text_height, baseline = _text_size(
            text,
            scale=scale,
            thickness=thickness,
            font=font,
        )
        _put_preview_text(
            canvas,
            text,
            (x, y + text_height),
            scale=scale,
            color=color,
            thickness=thickness,
            font=font,
        )
        return text_width, text_height, baseline

    # Recording card.
    status_label = "REC" if recording_mode == "minimal" else (
        "RECORDING" if packet.measured_receive_fps is not None else "STARTING"
    )
    status_color = PREVIEW_RED if packet.measured_receive_fps is not None else PREVIEW_AMBER
    status_row_y = recording_y + 16
    cv2.circle(canvas, (recording_x + inner_pad + 5, status_row_y + 6), 5, status_color, -1)
    put_text(
        status_label,
        recording_x + inner_pad + 16,
        status_row_y,
        scale=small_scale,
        color=PREVIEW_PRIMARY_TEXT,
    )

    clip_elapsed_text = format_clock_duration(packet.elapsed_s)
    clip_total_text = format_clock_duration(packet.planned_duration_s)
    clip_value_text = f"{clip_elapsed_text} / {clip_total_text}"
    clip_value_y = recording_y + 42
    _clip_value_width, clip_value_height, _ = put_text(
        clip_value_text,
        recording_x + inner_pad,
        clip_value_y,
        scale=large_scale,
        color=PREVIEW_PRIMARY_TEXT,
    )
    fps_text = (
        f"{packet.measured_receive_fps:.2f} fps"
        if packet.measured_receive_fps is not None
        else "Starting"
    )
    exposure_text, exposure_near_limit = format_preview_exposure(
        packet.exposure_us,
        auto_exposure=packet.auto_exposure,
        upper_us=packet.auto_exposure_upper_us,
    )
    if recording_mode != "minimal":
        fps_y = clip_value_y + clip_value_height + 10
        put_text(
            fps_text,
            recording_x + inner_pad,
            fps_y,
            scale=small_scale,
            color=PREVIEW_SECONDARY_TEXT,
        )
        exposure_y = fps_y + _preview_text_line_height(fps_text, scale=small_scale, font=font) + 4
        put_text(
            exposure_text,
            recording_x + inner_pad,
            exposure_y,
            scale=small_scale,
            color=PREVIEW_AMBER if exposure_near_limit else PREVIEW_SECONDARY_TEXT,
        )
    clip_progress = 0.0
    if packet.planned_duration_s > 0:
        clip_progress = clamp_progress(packet.elapsed_s / packet.planned_duration_s)
    bar_height = 7
    bar_width = max(0, recording_w - (2 * inner_pad))
    bar_y = recording_y + recording_h - inner_pad - bar_height
    if bar_width > 0:
        _draw_preview_progress_bar(
            canvas,
            x=recording_x + inner_pad,
            y=bar_y,
            width=bar_width,
            height=bar_height,
            progress=clip_progress,
            track_color=PREVIEW_TRACK,
            fill_color=PREVIEW_PURPLE,
        )

    # Session card.
    if session_w > 0 and session_h > 0:
        group_x = session_x + inner_pad
        group_w = max(0, session_w - (2 * inner_pad))
        current_y = session_y + (13 if session_mode == "minimal" else 14)

        def draw_separator(y: int) -> None:
            cv2.line(canvas, (group_x, y), (group_x + group_w, y), PREVIEW_CARD_BORDER, 1)

        def draw_section(
            title: str,
            lines: list[str],
            *,
            title_scale: float,
            value_scale_local: float,
            after_gap: int,
        ) -> None:
            nonlocal current_y
            title_h = _preview_text_line_height(title, scale=title_scale, font=font)
            put_text(
                title,
                group_x,
                current_y,
                scale=title_scale,
                color=PREVIEW_SECONDARY_TEXT,
            )
            current_y += title_h + 4
            for index, line in enumerate(lines):
                line_h = _preview_text_line_height(line, scale=value_scale_local, font=font)
                put_text(
                    line,
                    group_x,
                    current_y,
                    scale=value_scale_local,
                    color=PREVIEW_PRIMARY_TEXT,
                )
                current_y += line_h
                if index < len(lines) - 1:
                    current_y += 2
            current_y += after_gap

        session_value_text = (
            f"{format_clock_duration(packet.session_elapsed_s)} / "
            f"{format_clock_duration(packet.planned_session_duration_s)}"
        )
        draw_section(
            "SESSION",
            [session_value_text],
            title_scale=session_label_scale,
            value_scale_local=session_value_scale,
            after_gap=4,
        )
        session_progress = 0.0
        if packet.planned_session_duration_s > 0:
            session_progress = clamp_progress(
                packet.session_elapsed_s / packet.planned_session_duration_s
            )
        _draw_preview_progress_bar(
            canvas,
            x=group_x,
            y=current_y,
            width=group_w,
            height=7,
            progress=session_progress,
            track_color=PREVIEW_TRACK,
            fill_color=PREVIEW_PURPLE,
        )
        current_y += 7 + (8 if session_mode == "minimal" else 10)
        remaining_text = format_clock_duration(
            max(0.0, packet.planned_session_duration_s - packet.session_elapsed_s)
        )
        if session_mode == "minimal":
            draw_section(
                "REMAINING",
                [remaining_text],
                title_scale=session_label_scale,
                value_scale_local=session_value_scale,
                after_gap=4,
            )
        else:
            draw_separator(current_y)
            current_y += 12
            draw_section(
                "REMAINING",
                [remaining_text],
                title_scale=session_label_scale,
                value_scale_local=session_value_scale,
                after_gap=4,
            )
            draw_separator(current_y)
            current_y += 12

            finish_text = format_local_finish_time(packet.planned_finish_utc)
            if session_mode == "compact":
                finish_lines = [finish_text]
            elif " " in finish_text:
                finish_lines = finish_text.rsplit(" ", 1)
            else:
                finish_lines = [finish_text]
            draw_section(
                "EST. FINISH",
                finish_lines,
                title_scale=session_label_scale,
                value_scale_local=session_value_scale,
                after_gap=4,
            )
            draw_separator(current_y)
            current_y += 12

            clip_display = f"{packet.clip_index + 1} / {packet.total_clips}"
            draw_section(
                "CLIP",
                [clip_display],
                title_scale=session_label_scale,
                value_scale_local=session_value_scale,
                after_gap=4,
            )
            if session_mode == "full":
                draw_separator(current_y)
                current_y += 12
                camera_label = _ellipsize_preview_text(
                    packet.label,
                    max_width=group_w,
                    font=font,
                    font_scale=session_value_scale,
                    thickness=1,
                )
                draw_section(
                    "CAMERA",
                    [camera_label],
                    title_scale=session_label_scale,
                    value_scale_local=session_value_scale,
                    after_gap=4,
                )

    footer_status_text = "Recording active" if packet.measured_receive_fps is not None else "Starting"
    footer_status_color = PREVIEW_GREEN if packet.measured_receive_fps is not None else PREVIEW_AMBER
    footer_text_scale = 0.34 if image_w < 480 else 0.38
    footer_secondary_scale = 0.30 if image_w < 480 else 0.34
    cv2.circle(canvas, (16, layout.footer_y + 14), 5, footer_status_color, -1)
    _put_preview_text(
        canvas,
        footer_status_text,
        (28, layout.footer_y + 20),
        scale=footer_text_scale,
        color=PREVIEW_PRIMARY_TEXT,
    )

    snapshot_text = "S snapshot"
    hide_text = "Q hide"
    snapshot_width, _snapshot_height, _ = _text_size(
        snapshot_text,
        scale=footer_secondary_scale,
        thickness=1,
    )
    hide_width, _hide_height, _ = _text_size(
        hide_text,
        scale=footer_secondary_scale,
        thickness=1,
    )
    snapshot_x = canvas_width - 14 - snapshot_width
    hide_x = snapshot_x - 18 - hide_width
    _put_preview_text(
        canvas,
        snapshot_text,
        (snapshot_x, layout.footer_y + 20),
        scale=footer_secondary_scale,
        color=PREVIEW_SECONDARY_TEXT,
    )
    _put_preview_text(
        canvas,
        hide_text,
        (hide_x, layout.footer_y + 20),
        scale=footer_secondary_scale,
        color=PREVIEW_SECONDARY_TEXT,
    )

    return canvas


def write_session_manifest(
    session_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    bindings: list[CameraBinding],
    ffmpeg: str,
    *,
    recording_plan: Optional[dict[str, Any]] = None,
    session_name: str,
    session_start_utc: dt.datetime,
) -> None:
    created_at = utc_now()
    session_timezone = local_timezone_metadata(session_start_utc)
    manifest = {
        "created_utc": isoformat_utc(created_at),
        "created_local": isoformat_local(created_at),
        "session_name": session_name,
        "session_start_utc": isoformat_utc(session_start_utc),
        "session_start_local": isoformat_local(session_start_utc),
        "local_timezone_label": session_timezone["local_timezone_label"],
        "local_utc_offset_at_session_start": session_timezone["local_utc_offset"],
        "platform": sys.platform,
        "python": sys.version,
        "config_source": str(config_path.resolve()),
        "ffmpeg": ffmpeg,
        "naming": {
            "version": 3,
            "session_directory_format": "YYYYMMDD_HHMMSS+ZZZZ (local time with numeric UTC offset)",
            "clip_directory_format": "clip_NNNN_HHMMSS+ZZZZ (local time with numeric UTC offset)",
            "camera_file_stem": "camera label",
            "path_time_precision": "seconds",
            "scientific_timestamp_precision": "nanoseconds where available",
        },
        "timestamp_policy": {
            "directory_naming": "local_time_with_numeric_utc_offset",
            "log_display": "local_time_with_numeric_utc_offset",
            "canonical_metadata_time": "UTC",
            "human_display": "local_with_numeric_offset",
            "duration_clock": "monotonic",
            "local_timezone_label": session_timezone["local_timezone_label"],
            "local_utc_offset_at_start": session_timezone["local_utc_offset"],
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
    if recording_plan is not None:
        manifest["recording_plan"] = recording_plan
    write_json(session_dir / "session_manifest.json", manifest)
    shutil.copy2(config_path, session_dir / "config_used.yaml")


def write_session_summary(
    session_dir: Path,
    completed_clips: int,
    requested_clips: Optional[int],
    planned_session_span_s: Optional[float],
    stopped_by_signal: bool,
    any_failure: bool,
    unexpected_exception: Optional[str],
    exit_status: str,
) -> None:
    finished_at = utc_now()
    summary = {
        "completed_clips": completed_clips,
        "requested_clips": requested_clips,
        "planned_session_span_s": planned_session_span_s,
        "stopped_by_signal": stopped_by_signal,
        "any_failure": any_failure,
        "finished_utc": isoformat_utc(finished_at),
        "finished_local": isoformat_local(finished_at),
        "exit_status": exit_status,
        "unexpected_exception": unexpected_exception,
    }
    write_json(session_dir / "session_summary.json", summary)


def run_recording(config_path: Path, config: dict[str, Any], verbose: bool, dry_run: bool) -> int:
    schedule = dict(config.get("schedule") or {})
    clip_duration_s, interval_s, total_duration_s, number_of_clips = validate_schedule(schedule)
    system_settings = parse_system_settings(config)
    status_settings = parse_status_settings(config)
    preview_settings = parse_recording_preview_settings(config)
    archive_settings = parse_archive_settings(config)
    total_clips = expected_clip_count(
        interval_s=interval_s,
        total_duration_s=total_duration_s,
        number_of_clips=number_of_clips,
    )
    planned_session_duration_s = planned_session_span_s(
        clip_count=total_clips,
        clip_duration_s=clip_duration_s,
        interval_s=interval_s,
    )
    schedule_start_utc = utc_now()
    planned_finish_utc = schedule_start_utc + dt.timedelta(seconds=planned_session_duration_s)
    schedule_timezone = local_timezone_metadata(schedule_start_utc)
    recording_plan = {
        "expected_clips": total_clips,
        "clip_duration_s": clip_duration_s,
        "interval_s": interval_s,
        "planned_session_span_s": planned_session_duration_s,
        "schedule_start_utc": isoformat_utc(schedule_start_utc),
        "schedule_start_local": isoformat_local(schedule_start_utc),
        "planned_finish_utc": isoformat_utc(planned_finish_utc),
        "planned_finish_local": isoformat_local(planned_finish_utc),
        "timestamp_policy": {
            "canonical_wall_clock": "UTC",
            "human_display": "local_with_numeric_offset",
            "duration_clock": "monotonic",
            "local_timezone_label": schedule_timezone["local_timezone_label"],
            "local_utc_offset_at_start": schedule_timezone["local_utc_offset"],
        },
    }
    requested_clips = total_clips

    project = sanitize_token(str(config.get("project", "caterpillar")))
    subject = sanitize_token(str(config.get("subject", "cohort")))
    output_root = Path(str(config.get("output_root", "./recordings"))).expanduser()
    session_start_utc = utc_now()
    session_dir = choose_unique_directory(
        output_root / project / subject / filename_local_timestamp(session_start_utc)
    )
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
    LOG.info(
        "System: prevent_sleep_during_recording=%s",
        system_settings.prevent_sleep_during_recording,
    )
    LOG.info(
        "Recording plan: clips=%d planned_span=%s planned_finish_local=%s",
        total_clips,
        format_clock_duration(planned_session_duration_s),
        format_local_finish_time(planned_finish_utc, now_utc=schedule_start_utc),
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
            "Archive: platform=%s backend=%s executable=%s destination_root=%s mount_point=%s enabled=%s background_transfer=%s",
            archive_preflight.platform,
            archive_preflight.copy_backend,
            archive_preflight.copy_executable_path,
            archive_preflight.destination_root,
            archive_preflight.required_mount_point,
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
        write_json(
            session_dir / "dry_run.json",
            {
                "project": project,
                "subject": subject,
                "clip_duration_s": clip_duration_s,
                "interval_s": interval_s,
                "total_duration_s": total_duration_s,
                "number_of_clips": number_of_clips,
                "status": dataclasses.asdict(status_settings),
                "system": dataclasses.asdict(system_settings),
                "recording_preview": dataclasses.asdict(preview_settings),
                "archive": dataclasses.asdict(archive_settings),
                "archive_preflight": dataclasses.asdict(archive_preflight) if archive_preflight else None,
                "recording_plan": recording_plan,
            },
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
    recording_stop_event = threading.Event()
    archive_failure_event = threading.Event()
    preview_active_event = threading.Event()
    stopped_by_signal = False
    clip_index = 0
    completed_clips = 0
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
        recording_stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    try:
        with SleepInhibitor(system_settings.prevent_sleep_during_recording):
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
                recording_plan=recording_plan,
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
                    archive_failure_event=archive_failure_event,
                    preflight=archive_preflight or ArchivePreflightResult(
                        enabled=False,
                        ok=False,
                        errors=["archive preflight missing"],
                        platform=os.name,
                        copy_backend=resolve_archive_backend(archive_settings)[0],
                        copy_executable_path=None,
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
                    if recording_stop_event.is_set():
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
                    if pending < archive_settings.max_unarchived_clips:
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

            while not recording_stop_event.is_set():
                if number_of_clips is not None and clip_index >= number_of_clips:
                    break
                scheduled_offset_s = clip_index * interval_s
                if total_duration_s is not None and scheduled_offset_s >= total_duration_s:
                    break

                target_start_mono_ns = session_start_mono_ns + round(scheduled_offset_s * 1e9)
                next_wait_status_mono = time.monotonic()
                while not recording_stop_event.is_set():
                    remaining_s = (target_start_mono_ns - time.monotonic_ns()) / 1e9
                    if remaining_s <= 0:
                        break
                    now_mono = time.monotonic()
                    if (
                        status_settings.terminal_interval_s > 0
                        and now_mono >= next_wait_status_mono
                        and remaining_s > 2.0
                    ):
                        session_elapsed_s = max(
                            0.0,
                            (time.monotonic_ns() - session_start_mono_ns) / 1e9,
                        )
                        LOG.info(
                            "STATUS | waiting for clip %d/%d | starts in %s | session %s/%s | finish ~%s local",
                            clip_index + 1,
                            total_clips,
                            format_clock_duration(remaining_s),
                            format_clock_duration(session_elapsed_s),
                            format_clock_duration(planned_session_duration_s),
                            format_local_finish_time(planned_finish_utc),
                        )
                        next_wait_status_mono = now_mono + status_settings.terminal_interval_s
                    time.sleep(min(remaining_s, 1.0))
                if recording_stop_event.is_set():
                    break

                readiness_error = ensure_ready_for_next_clip(clip_index)
                if readiness_error is not None:
                    any_failure = True
                    LOG.error("Cannot start clip %d: %s", clip_index, readiness_error)
                    break

                # Give all writer threads one second to create files and start grabbing.
                planned_start_mono_ns = time.monotonic_ns() + 1_000_000_000
                planned_stop_mono_ns = planned_start_mono_ns + round(clip_duration_s * 1e9)
                delay_to_start_s = (planned_start_mono_ns - time.monotonic_ns()) / 1e9
                clip_start_utc = utc_now() + dt.timedelta(seconds=delay_to_start_s)
                clip_dir = session_dir / f"clip_{clip_index:04d}_{clip_clock_local(clip_start_utc)}"
                clip_dir.mkdir(parents=True, exist_ok=False)
                LOG.info(
                    "Starting clip %d at %s for %.1f s",
                    clip_index,
                    isoformat_local(clip_start_utc),
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
                            "total_clips": total_clips,
                            "clip_start_utc": clip_start_utc,
                            "planned_start_mono_ns": planned_start_mono_ns,
                            "planned_stop_mono_ns": planned_stop_mono_ns,
                            "session_start_mono_ns": session_start_mono_ns,
                            "planned_session_duration_s": planned_session_duration_s,
                            "planned_finish_utc": planned_finish_utc,
                            "clip_dir": clip_dir,
                            "ffmpeg": ffmpeg,
                            "encoding_cfg": encoding_cfg,
                            "ready_barrier": barrier,
                            "stop_event": recording_stop_event,
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
                    recording_stop_event.set()

                monitor_recording_threads(
                    threads=threads,
                    preview_queues=preview_queues,
                    preview_settings=preview_settings,
                    preview_active_event=preview_active_event,
                    clip_dir=clip_dir,
                    clip_index=clip_index,
                    total_clips=total_clips,
                    planned_start_mono_ns=planned_start_mono_ns,
                    planned_stop_mono_ns=planned_stop_mono_ns,
                    session_start_mono_ns=session_start_mono_ns,
                    planned_session_duration_s=planned_session_duration_s,
                    planned_finish_utc=planned_finish_utc,
                    terminal_interval_s=status_settings.terminal_interval_s,
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
                clip_finalized_successfully = len(clip_results) == len(bindings) and all(
                    result.success for result in clip_results
                )
                if clip_finalized_successfully:
                    completed_clips += 1

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

                if not any_failure:
                    removed_capture_files = remove_capture_files(clip_dir)
                    if removed_capture_files:
                        LOG.info(
                            "Removed %d temporary capture file(s) from %s",
                            len(removed_capture_files),
                            clip_dir,
                        )

                    if archive_settings.enabled and not archive_failure:
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
                    elif archive_failure:
                        any_failure = True
                        LOG.error(
                            "Archive failure detected. Clip %d finished locally and will not be archived in this run.",
                            clip_index,
                        )

                if (
                    archive_failure_event.is_set()
                    and archive_settings.stop_before_next_clip_on_transfer_failure
                ):
                    any_failure = True
                    LOG.error(
                        "Archive failure detected. The active clip has finished; no additional clip will be started. Local data is preserved."
                    )
                    break

                if any_failure:
                    break

                clip_index += 1

            LOG.info("Recording finished after %d completed clip(s)", completed_clips)
    except KeyboardInterrupt:
        stopped_by_signal = True
        exit_status = "interrupted"
        LOG.warning("KeyboardInterrupt received; stopping recording")
        recording_stop_event.set()
    except Exception as exc:
        any_failure = True
        unexpected_exception = repr(exc)
        exit_status = "exception"
        LOG.exception("Unhandled exception during recording")
        recording_stop_event.set()
    finally:
        if archive_manager is not None:
            try:
                archive_manager.wait_until_idle()
                archive_manager.close_transfers()
            except Exception as exc:
                any_failure = True
                if unexpected_exception is None:
                    unexpected_exception = f"archive finalization failed: {exc!r}"
                LOG.warning("Could not finalize archive manager transfers: %s", exc)
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
                completed_clips=completed_clips,
                requested_clips=requested_clips,
                planned_session_span_s=planned_session_duration_s,
                stopped_by_signal=stopped_by_signal,
                any_failure=any_failure,
                unexpected_exception=unexpected_exception,
                exit_status=exit_status,
            )
        except Exception as exc:
            LOG.warning("Could not write session summary: %s", exc)
        if archive_manager is not None:
            try:
                archive_manager.copy_final_metadata()
            except Exception as exc:
                LOG.warning("Could not copy final archive metadata: %s", exc)
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
