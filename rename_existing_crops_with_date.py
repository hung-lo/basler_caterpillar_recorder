#!/usr/bin/env python3
r"""
Safely rename existing legacy caterpillar crops to timestamped names.

Example legacy crop:
    C01_clip_0000_134218-0400.mp4

Corresponding raw source:
    ROOT\20260809_134217-0400\clip_0000_134218-0400\camera1.mp4

New name:
    C01_20260809_134218-0400_clip_0000.mp4

Safety:
- Does not touch raw camera1.mp4 files.
- Does not overwrite an existing new-name crop.
- If the same legacy clip name occurs in more than one recording session,
  it is considered ambiguous and is NOT renamed automatically.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

from crop_caterpillar_videos import resolve_source_naming

DEFAULT_ROOT = Path(
    r"D:\Hung_MBL\monarch_behavior_windows\new_cohort_C01-08"
)

CATERPILLAR_IDS = [f"C{i:02d}" for i in range(1, 9)]

def discover_sources(root: Path) -> list[Path]:
    found = []
    for p in root.rglob("camera1.mp4"):
        if (
            p.is_file()
            and p.parent.name.lower().startswith("clip_")
            and "cropped_by_caterpillar"
            not in {part.lower() for part in p.parts}
        ):
            found.append(p)
    return sorted(found, key=lambda p: str(p).lower())


def legacy_stem(source: Path) -> str:
    return source.parent.name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rename existing legacy crops without re-encoding."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    output_dir = root / "cropped_by_caterpillar"

    print("=" * 68)
    print("MBL existing-crop timestamp renamer")
    print("=" * 68)
    print(f"Root: {root}")
    print(f"Crop folder: {output_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'RENAME'}")
    print("Naming: CXX_YYYYMMDD_HHMMSS-TZ_clip_NNNN.mp4")

    if not root.exists():
        print(f"ERROR root does not exist: {root}")
        return 2

    if not output_dir.exists():
        print(f"ERROR crop folder does not exist: {output_dir}")
        return 2

    sources = discover_sources(root)
    print(f"Raw source videos discovered: {len(sources)}")

    # A legacy filename lacks the date. Group raw sources by the exact old
    # clip stem so we can detect cases where the old name is ambiguous.
    by_legacy_stem: dict[str, list[Path]] = defaultdict(list)
    for source in sources:
        by_legacy_stem[legacy_stem(source)].append(source)

    renamed = 0
    ambiguous = 0
    target_exists = 0
    unmatched = 0

    for old_stem, matching_sources in sorted(by_legacy_stem.items()):
        legacy_files = [
            output_dir / f"{cid}_{old_stem}.mp4"
            for cid in CATERPILLAR_IDS
        ]
        existing_legacy = [p for p in legacy_files if p.exists()]

        if not existing_legacy:
            continue

        if len(matching_sources) != 1:
            ambiguous += len(existing_legacy)
            print(
                f"AMBIGUOUS: {old_stem} matches "
                f"{len(matching_sources)} raw sources; leaving "
                f"{len(existing_legacy)} crop(s) untouched."
            )
            for source in matching_sources:
                print(f"  raw: {source}")
            continue

        source = matching_sources[0]
        naming = resolve_source_naming(source)

        if naming.error or naming.stem is None:
            unmatched += len(existing_legacy)
            print(
                f"UNPARSED raw directory names; leaving old crop(s): "
                f"{source}"
            )
            print(f"  reason: {naming.error or 'missing canonical stem'}")
            continue

        stem = naming.stem

        for old_path in existing_legacy:
            cid = old_path.name.split("_", 1)[0]
            new_path = output_dir / f"{cid}_{stem}.mp4"

            if new_path.exists():
                target_exists += 1
                print(
                    f"SKIP target already exists: {new_path.name}"
                )
                continue

            if args.dry_run:
                print(
                    f"WOULD RENAME: {old_path.name} -> {new_path.name}"
                )
            else:
                os.replace(old_path, new_path)
                print(
                    f"RENAMED: {old_path.name} -> {new_path.name}"
                )

            renamed += 1

    print("-" * 68)
    print(
        f"{'Would rename' if args.dry_run else 'Renamed'}: {renamed}"
    )
    print(f"Ambiguous, left untouched: {ambiguous}")
    print(f"Target already existed: {target_exists}")
    print(f"Unparsed, left untouched: {unmatched}")

    if ambiguous:
        print(
            "Ambiguous legacy names are intentionally not guessed. "
            "The canonical cropper can regenerate those raw clips with unique "
            "timestamped names if needed."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
