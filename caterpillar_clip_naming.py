#!/usr/bin/env python3
"""Shared timestamp parsing for caterpillar crop and rename tools."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

SESSION_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})(?P<rest>.*)$")
CLIP_RE = re.compile(
    r"^clip_(?P<index>\d+)_(?P<time>\d{6})(?P<rest>.*)$",
    re.IGNORECASE,
)
OLDCOMMIT_RE = re.compile(r"^[_-]?old(?:[_-]?commit)$", re.IGNORECASE)
OFFSET_RE = re.compile(r"^[+-]\d{4}$")

WOODS_HOLE_OLDCOMMIT_TZ = dt.timezone(dt.timedelta(hours=-4))


class ClipParseError(ValueError):
    """Raised when a nested clip path cannot be interpreted safely."""


@dataclass(frozen=True)
class ClipMetadata:
    local_datetime: dt.datetime
    clip_index: str
    source_format: str

    @property
    def stem(self) -> str:
        return f"{self.local_datetime.strftime('%Y%m%d_%H%M%S%z')}_clip_{self.clip_index}"


@dataclass(frozen=True)
class SourceNaming:
    stem: str | None
    error: str | None = None
    metadata: ClipMetadata | None = None


def _parse_offset(text: str) -> dt.tzinfo | None:
    if not text or not OFFSET_RE.fullmatch(text):
        return None

    sign = 1 if text[0] == "+" else -1
    hours = int(text[1:3])
    minutes = int(text[3:5])
    return dt.timezone(sign * dt.timedelta(hours=hours, minutes=minutes))


def _make_aware(date_text: str, time_text: str, tzinfo: dt.tzinfo) -> dt.datetime:
    return dt.datetime.strptime(
        f"{date_text}{time_text}", "%Y%m%d%H%M%S"
    ).replace(tzinfo=tzinfo)


def _is_oldcommit_suffix(text: str) -> bool:
    return bool(text and OLDCOMMIT_RE.fullmatch(text))


def parse_clip_metadata(source: Path) -> ClipMetadata:
    """Parse a nested recorder clip source into canonical local metadata.

    The function only accepts nested ``camera1.mp4`` sources under a ``clip_*``
    directory. Flat legacy clips are intentionally handled by the caller.
    """

    if source.name.lower() != "camera1.mp4":
        raise ClipParseError(f"not a nested camera1.mp4 source: {source}")

    clip_dir = source.parent.name
    session_dir = source.parent.parent.name

    clip_match = CLIP_RE.fullmatch(clip_dir)
    if not clip_match:
        raise ClipParseError(f"unrecognized clip folder name: {clip_dir}")

    session_match = SESSION_RE.fullmatch(session_dir)
    if not session_match:
        raise ClipParseError(f"unrecognized session folder name: {session_dir}")

    clip_index = clip_match.group("index")
    clip_time = clip_match.group("time")
    clip_rest = clip_match.group("rest")
    session_date = session_match.group("date")
    session_time = session_match.group("time")
    session_rest = session_match.group("rest")

    if _is_oldcommit_suffix(session_rest):
        session_utc = _make_aware(session_date, session_time, dt.timezone.utc)
        clip_utc = _make_aware(session_date, clip_time, dt.timezone.utc)
        if clip_utc < session_utc:
            clip_utc += dt.timedelta(days=1)
        local_dt = clip_utc.astimezone(WOODS_HOLE_OLDCOMMIT_TZ)
        return ClipMetadata(
            local_datetime=local_dt,
            clip_index=clip_index,
            source_format="oldcommit_utc",
        )

    session_tz = _parse_offset(session_rest)
    clip_tz = _parse_offset(clip_rest)
    tzinfo = clip_tz or session_tz

    if tzinfo is None:
        raise ClipParseError(
            "missing timezone offset in session and clip folder names"
        )

    session_dt = _make_aware(session_date, session_time, tzinfo)
    clip_dt = _make_aware(session_date, clip_time, tzinfo)
    if clip_dt < session_dt:
        clip_dt += dt.timedelta(days=1)

    return ClipMetadata(
        local_datetime=clip_dt,
        clip_index=clip_index,
        source_format="current_local",
    )


def resolve_source_naming(source: Path) -> SourceNaming:
    """Return a canonical stem or a controlled parse failure reason."""

    if source.name.lower() != "camera1.mp4":
        return SourceNaming(stem=source.stem)

    if not source.parent.name.lower().startswith("clip_"):
        return SourceNaming(
            stem=None,
            error=f"nested camera1.mp4 is not inside a clip_ folder: {source}",
        )

    try:
        metadata = parse_clip_metadata(source)
    except ClipParseError as exc:
        return SourceNaming(stem=None, error=str(exc))

    return SourceNaming(stem=metadata.stem, metadata=metadata)

