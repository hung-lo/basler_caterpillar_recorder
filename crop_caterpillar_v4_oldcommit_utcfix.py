#!/usr/bin/env python3
"""Compatibility wrapper for the unified cropper.

Use `crop_caterpillar_videos.py` as the canonical entry point.
"""

from __future__ import annotations

from crop_caterpillar_videos import main


if __name__ == "__main__":
    raise SystemExit(main())
