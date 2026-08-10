#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError:  # pragma: no cover - optional frame-count helper
    cv2 = None

from crop_caterpillar_videos import CROPS, discover_sources, output_path, resolve_source_naming

LOG = logging.getLogger("prepare_cropped_timestamps")
TIMESTAMP_SUFFIXES = (".timestamps.csv.gz", ".timestamps.csv")
MANIFEST_FIELDS = [
    "clip_key",
    "animal_id",
    "cropped_video",
    "source_video",
    "source_timestamp_file",
    "copied_timestamp_file",
    "source_layout",
    "timestamp_rows",
    "video_frames_reported",
    "frame_count_status",
]


@dataclass(frozen=True)
class ManifestRow:
    clip_key: str
    animal_id: str
    cropped_video: str
    source_video: str
    source_timestamp_file: str
    copied_timestamp_file: str
    source_layout: str
    timestamp_rows: str
    video_frames_reported: str
    frame_count_status: str

    def as_dict(self) -> dict[str, str]:
        return {
            "clip_key": self.clip_key,
            "animal_id": self.animal_id,
            "cropped_video": self.cropped_video,
            "source_video": self.source_video,
            "source_timestamp_file": self.source_timestamp_file,
            "copied_timestamp_file": self.copied_timestamp_file,
            "source_layout": self.source_layout,
            "timestamp_rows": self.timestamp_rows,
            "video_frames_reported": self.video_frames_reported,
            "frame_count_status": self.frame_count_status,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy one authoritative timestamp sidecar per raw source clip into "
            "cropped_by_caterpillar/timestamps and write crop_manifest.csv."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Dataset root containing raw recordings and cropped_by_caterpillar/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite conflicting copied timestamp files when their contents differ.",
    )
    return parser


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def adjacent_timestamp_file(video_path: Path) -> Optional[Path]:
    for suffix in TIMESTAMP_SUFFIXES:
        candidate = video_path.with_name(f"{video_path.stem}{suffix}")
        if candidate.exists():
            return candidate
    return None


def copied_timestamp_name(clip_key: str, source_timestamp: Path) -> str:
    if source_timestamp.name.endswith(".timestamps.csv.gz"):
        return f"{clip_key}.timestamps.csv.gz"
    if source_timestamp.name.endswith(".timestamps.csv"):
        return f"{clip_key}.timestamps.csv"
    raise ValueError(f"unsupported timestamp filename: {source_timestamp.name}")


def count_timestamp_rows(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def sha256_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def copy_timestamp_file(source: Path, destination: Path, *, force: bool) -> tuple[str, bool]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_digest(source) == sha256_digest(destination):
            return "unchanged", False
        if not force:
            return "conflict", False
    temp_path = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temp_path)
    temp_path.replace(destination)
    return ("overwritten" if destination.exists() else "copied"), True


def cheap_video_frame_count(path: Path) -> Optional[int]:
    if cv2 is None:
        return None
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        value = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        capture.release()
    if value is None or value <= 0:
        return None
    return int(round(float(value)))


def manifest_row_for_crop(
    *,
    clip_key: str,
    animal_id: str,
    cropped_video: Path,
    root: Path,
    source_video: Path,
    source_layout: str,
    source_timestamp_file: Optional[Path],
    copied_timestamp_file: Optional[Path],
    timestamp_rows: Optional[int],
    status: str,
) -> ManifestRow:
    video_frames_reported = cheap_video_frame_count(cropped_video)
    frame_count_status = status
    if status == "ready" and timestamp_rows is not None:
        if video_frames_reported is None:
            frame_count_status = "unknown"
        elif video_frames_reported == timestamp_rows:
            frame_count_status = "match"
        else:
            frame_count_status = "mismatch"

    return ManifestRow(
        clip_key=clip_key,
        animal_id=animal_id,
        cropped_video=relative_to_root(cropped_video, root),
        source_video=relative_to_root(source_video, root),
        source_timestamp_file=(
            relative_to_root(source_timestamp_file, root) if source_timestamp_file else ""
        ),
        copied_timestamp_file=(
            relative_to_root(copied_timestamp_file, root) if copied_timestamp_file else ""
        ),
        source_layout=source_layout,
        timestamp_rows="" if timestamp_rows is None else str(timestamp_rows),
        video_frames_reported="" if video_frames_reported is None else str(video_frames_reported),
        frame_count_status=frame_count_status,
    )


def prepare_manifest(root: Path, *, force: bool) -> tuple[list[ManifestRow], list[str], list[str]]:
    cropped_dir = root / "cropped_by_caterpillar"
    timestamp_dir = cropped_dir / "timestamps"
    rows: list[ManifestRow] = []
    warnings: list[str] = []
    errors: list[str] = []

    if not cropped_dir.exists():
        raise FileNotFoundError(f"missing cropped output directory: {cropped_dir}")

    for source in discover_sources(root):
        naming = resolve_source_naming(source)
        if naming.stem is None:
            warning = f"Skipping source with unresolved crop naming: {source} ({naming.error})"
            LOG.warning(warning)
            warnings.append(warning)
            continue

        crop_paths = []
        for animal_id in sorted(CROPS):
            candidate = output_path(cropped_dir, animal_id, naming)
            if candidate.exists():
                crop_paths.append((animal_id, candidate))
        if not crop_paths:
            continue

        clip_key = naming.stem
        source_layout = naming.layout or ""
        source_timestamp = adjacent_timestamp_file(source)
        copied_timestamp: Optional[Path] = None
        timestamp_rows: Optional[int] = None
        row_status = "ready"

        if source_timestamp is None:
            row_status = "missing_timestamp"
            warning = f"Missing adjacent timestamp sidecar for source video: {source}"
            LOG.warning(warning)
            warnings.append(warning)
        else:
            timestamp_rows = count_timestamp_rows(source_timestamp)
            copied_timestamp = timestamp_dir / copied_timestamp_name(clip_key, source_timestamp)
            copy_status, _copied = copy_timestamp_file(
                source_timestamp,
                copied_timestamp,
                force=force,
            )
            if copy_status == "conflict":
                row_status = "timestamp_conflict"
                copied_timestamp = None
                error = (
                    "Conflicting copied timestamp file already exists for "
                    f"{clip_key}; rerun with --force to overwrite."
                )
                LOG.error(error)
                errors.append(error)

        for animal_id, cropped_video in crop_paths:
            rows.append(
                manifest_row_for_crop(
                    clip_key=clip_key,
                    animal_id=animal_id,
                    cropped_video=cropped_video,
                    root=root,
                    source_video=source,
                    source_layout=source_layout,
                    source_timestamp_file=source_timestamp,
                    copied_timestamp_file=copied_timestamp,
                    timestamp_rows=timestamp_rows,
                    status=row_status,
                )
            )

    rows.sort(key=lambda row: (row.clip_key, row.animal_id))
    return rows, warnings, errors


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    temp_path.replace(path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        parser.error(f"root must be an existing directory: {root}")

    try:
        rows, warnings, errors = prepare_manifest(root, force=args.force)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    manifest_path = root / "cropped_by_caterpillar" / "crop_manifest.csv"
    write_manifest(rows, manifest_path)
    LOG.info("Wrote crop manifest: %s", manifest_path)
    if warnings:
        LOG.info("Finished with %d warning(s)", len(warnings))
    if errors:
        LOG.info("Finished with %d error(s)", len(errors))
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
