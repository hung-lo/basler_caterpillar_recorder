from __future__ import annotations

import csv
import datetime as dt
import gzip
import logging
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

LOG = logging.getLogger("analysis_timing")

UTC = dt.timezone.utc
DEFAULT_TIMEZONE = "America/New_York"
TIMESTAMP_SUFFIXES = (".timestamps.csv.gz", ".timestamps.csv")


def utc_from_ns(value_ns: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(value_ns / 1e9, tz=UTC)


def coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_iso_datetime(value: str) -> dt.datetime:
    text = value.strip()
    if not text:
        raise ValueError("timestamp is empty")
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    parsed = dt.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("timestamp is missing a timezone offset")
    return parsed


def parse_utc_value(value: Any) -> dt.datetime:
    if value is None:
        raise ValueError("missing UTC timestamp")
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = utc_from_ns(int(value))
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("missing UTC timestamp")
        try:
            parsed = utc_from_ns(int(text))
        except ValueError:
            parsed = parse_iso_datetime(text)
    return parsed.astimezone(UTC)


def parse_timestamp_row(row: dict[str, Any]) -> dt.datetime:
    for field in ("host_utc_ns", "utc_ns", "timestamp_ns"):
        value = row.get(field)
        if value not in (None, ""):
            parsed_ns = coerce_int(value)
            if parsed_ns is None:
                raise ValueError(f"{field} is not an integer")
            return utc_from_ns(parsed_ns)
    for field in ("host_utc_iso", "utc_iso", "timestamp_iso"):
        value = row.get(field)
        if value not in (None, ""):
            return parse_utc_value(value)
    raise ValueError("missing UTC timestamp columns")


def open_text_file(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def timezone_label(tz: dt.tzinfo) -> str:
    label = getattr(tz, "key", None)
    if label:
        return str(label)
    sample = dt.datetime.now(UTC).astimezone(tz)
    name = sample.tzname()
    return name or str(tz)


def load_timezone(name: str) -> dt.tzinfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        local_tz = dt.datetime.now().astimezone().tzinfo
        if local_tz is None:
            LOG.warning(
                "Could not load timezone %s and the local timezone is unavailable; falling back to UTC",
                name,
            )
            return UTC
        LOG.warning(
            "Could not load timezone %s; falling back to the local system timezone (%s)",
            name,
            timezone_label(local_tz),
        )
        return local_tz


def to_plot_local(value_utc: dt.datetime, tz: dt.tzinfo) -> dt.datetime:
    return value_utc.astimezone(tz).replace(tzinfo=None)


def format_utc(value_utc: dt.datetime) -> str:
    return value_utc.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def format_local(value_utc: dt.datetime, tz: dt.tzinfo) -> str:
    return value_utc.astimezone(tz).isoformat(sep=" ", timespec="microseconds")


def load_timestamp_series(path: Path) -> list[dt.datetime]:
    timestamps: list[dt.datetime] = []
    with open_text_file(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header in {path}")
        for row_index, row in enumerate(reader, start=2):
            try:
                timestamps.append(parse_timestamp_row(row))
            except Exception as exc:
                raise ValueError(f"row {row_index}: {exc}") from exc
    if not timestamps:
        raise ValueError(f"no timestamp rows in {path}")
    return timestamps
