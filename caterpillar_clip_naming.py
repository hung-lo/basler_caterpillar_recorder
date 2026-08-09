#!/usr/bin/env python3
"""Compatibility shim for timestamp helpers.

The canonical implementation lives in `crop_caterpillar_videos.py`.
"""

from __future__ import annotations

from crop_caterpillar_videos import (  # noqa: F401
    CLIP_RE,
    ClipMetadata,
    ClipParseError,
    SESSION_CURRENT_RE,
    SESSION_LEGACY_UTC_RE,
    SESSION_OLDCOMMIT_RE,
    SourceNaming,
    WOODS_HOLE_LEGACY_TZ,
    canonical_nested_recording_stem,
    parse_clip_metadata,
    resolve_source_naming,
    source_metadata,
)

