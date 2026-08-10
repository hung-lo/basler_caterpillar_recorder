#!/usr/bin/env python3

from __future__ import annotations

import csv
import datetime as dt
import gzip
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - dependency availability varies by interpreter
    cv2 = None

try:
    import extract_motion_energy as motion
except ImportError as exc:  # pragma: no cover - dependency availability varies by interpreter
    if cv2 is None and "cv2" in str(exc):
        motion = None
    else:
        raise


@unittest.skipIf(cv2 is None or motion is None, "opencv-python is not available")
class ExtractMotionEnergyTests(unittest.TestCase):
    def write_video(self, path: Path, frames: list[np.ndarray], *, fps: float = 1.0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            self.skipTest("OpenCV MP4 writer is unavailable in this environment")
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()

    def write_timestamp_sidecar(self, path: Path, timestamps: list[dt.datetime]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_index", "host_utc_ns", "host_utc_iso"])
            for index, timestamp in enumerate(timestamps):
                ns = int(timestamp.timestamp() * 1e9)
                writer.writerow([index, ns, timestamp.isoformat().replace("+00:00", "Z")])

    def write_manifest(
        self,
        root: Path,
        rows: list[dict[str, str]],
    ) -> None:
        manifest_path = root / "cropped_by_caterpillar" / "crop_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=motion.prep.MANIFEST_FIELDS if hasattr(motion, "prep") else [
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
            ])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def make_manifest_row(
        self,
        *,
        clip_key: str,
        animal_id: str,
        cropped_video: str,
        copied_timestamp_file: str,
        timestamp_rows: int = 0,
    ) -> dict[str, str]:
        return {
            "clip_key": clip_key,
            "animal_id": animal_id,
            "cropped_video": cropped_video,
            "source_video": "",
            "source_timestamp_file": copied_timestamp_file,
            "copied_timestamp_file": copied_timestamp_file,
            "source_layout": "current",
            "timestamp_rows": str(timestamp_rows),
            "video_frames_reported": "",
            "frame_count_status": "match",
        }

    def make_trace_file(
        self,
        path: Path,
        rows: list[motion.MotionTraceRow],
        timezone: dt.tzinfo,
    ) -> None:
        motion.write_gzip_csv(
            path,
            motion.TRACE_FIELDS,
            motion.trace_rows_to_dicts(rows, timezone=timezone),
        )

    def read_summary_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def build_video_dataset(
        self,
        root: Path,
        *,
        clip_name: str,
        animal_id: str,
        frames: list[np.ndarray],
        start_utc: dt.datetime,
        timestamp_count: int | None = None,
    ) -> tuple[Path, Path]:
        cropped_rel = f"cropped_by_caterpillar/{animal_id}_{clip_name}.mp4"
        timestamp_rel = f"cropped_by_caterpillar/timestamps/{clip_name}.timestamps.csv.gz"
        video_path = root / cropped_rel
        timestamp_path = root / timestamp_rel
        self.write_video(video_path, frames)
        count = len(frames) if timestamp_count is None else timestamp_count
        timestamps = [start_utc + dt.timedelta(seconds=index) for index in range(count)]
        self.write_timestamp_sidecar(timestamp_path, timestamps)
        self.write_manifest(
            root,
            [
                self.make_manifest_row(
                    clip_key=clip_name,
                    animal_id=animal_id,
                    cropped_video=cropped_rel,
                    copied_timestamp_file=timestamp_rel,
                    timestamp_rows=count,
                )
            ],
        )
        return video_path, timestamp_path

    def test_local_motion_scores_higher_than_static_and_global_brightness_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = np.zeros((64, 64, 3), dtype=np.uint8)
            static_frames = [base.copy() for _ in range(20)]
            brightness_frames = [np.full((64, 64, 3), fill_value=i * 4, dtype=np.uint8) for i in range(20)]
            moving_frames = []
            for index in range(20):
                frame = base.copy()
                x0 = 4 + index
                frame[20:32, x0 : x0 + 8] = 255
                moving_frames.append(frame)

            for animal_id, clip_name, frames in [
                ("C01", "static_clip", static_frames),
                ("C02", "brightness_clip", brightness_frames),
                ("C03", "moving_clip", moving_frames),
            ]:
                cropped_rel = f"cropped_by_caterpillar/{animal_id}_{clip_name}.mp4"
                timestamp_rel = f"cropped_by_caterpillar/timestamps/{clip_name}.timestamps.csv.gz"
                self.write_video(root / cropped_rel, frames)
                timestamps = [
                    dt.datetime(2026, 8, 9, 17, 0, 0, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=index)
                    for index in range(len(frames))
                ]
                self.write_timestamp_sidecar(root / timestamp_rel, timestamps)

            self.write_manifest(
                root,
                [
                    self.make_manifest_row(
                        clip_key="static_clip",
                        animal_id="C01",
                        cropped_video="cropped_by_caterpillar/C01_static_clip.mp4",
                        copied_timestamp_file="cropped_by_caterpillar/timestamps/static_clip.timestamps.csv.gz",
                        timestamp_rows=20,
                    ),
                    self.make_manifest_row(
                        clip_key="brightness_clip",
                        animal_id="C02",
                        cropped_video="cropped_by_caterpillar/C02_brightness_clip.mp4",
                        copied_timestamp_file="cropped_by_caterpillar/timestamps/brightness_clip.timestamps.csv.gz",
                        timestamp_rows=20,
                    ),
                    self.make_manifest_row(
                        clip_key="moving_clip",
                        animal_id="C03",
                        cropped_video="cropped_by_caterpillar/C03_moving_clip.mp4",
                        copied_timestamp_file="cropped_by_caterpillar/timestamps/moving_clip.timestamps.csv.gz",
                        timestamp_rows=20,
                    ),
                ],
            )

            rc = motion.main([str(root)])

            self.assertEqual(rc, 0)
            static_trace = motion.load_motion_trace_rows(
                root / "cropped_by_caterpillar" / "motion_energy" / "traces" / "C01_static_clip.motion.csv.gz"
            )
            brightness_trace = motion.load_motion_trace_rows(
                root / "cropped_by_caterpillar" / "motion_energy" / "traces" / "C02_brightness_clip.motion.csv.gz"
            )
            moving_trace = motion.load_motion_trace_rows(
                root / "cropped_by_caterpillar" / "motion_energy" / "traces" / "C03_moving_clip.motion.csv.gz"
            )

            static_mean = float(np.mean([row.motion_energy for row in static_trace]))
            brightness_mean = float(np.mean([row.motion_energy for row in brightness_trace]))
            moving_mean = float(np.mean([row.motion_energy for row in moving_trace]))

            self.assertLess(static_mean, moving_mean)
            self.assertLess(brightness_mean, moving_mean)
            self.assertGreater(moving_mean, max(static_mean, brightness_mean) * 3.0)

    def test_exact_timestamp_match_succeeds_and_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(10)]

            self.build_video_dataset(
                root,
                clip_name="matched_clip",
                animal_id="C01",
                frames=frames,
                start_utc=dt.datetime(2026, 8, 9, 18, 0, 0, tzinfo=dt.timezone.utc),
            )
            rc = motion.main([str(root)])
            self.assertEqual(rc, 0)
            summary_rows = self.read_summary_rows(
                root / "cropped_by_caterpillar" / "motion_energy" / "motion_summary.csv"
            )
            self.assertEqual(summary_rows[0]["status"], "computed")
            self.assertEqual(summary_rows[0]["decoded_frames"], "10")
            self.assertTrue(
                (
                    root
                    / "cropped_by_caterpillar"
                    / "motion_energy"
                    / "traces"
                    / "C01_matched_clip.motion.csv.gz"
                ).exists()
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(10)]
            self.build_video_dataset(
                root,
                clip_name="mismatched_clip",
                animal_id="C01",
                frames=frames,
                start_utc=dt.datetime(2026, 8, 9, 18, 0, 0, tzinfo=dt.timezone.utc),
                timestamp_count=9,
            )

            rc = motion.main([str(root)])

            self.assertEqual(rc, 0)
            summary_rows = self.read_summary_rows(
                root / "cropped_by_caterpillar" / "motion_energy" / "motion_summary.csv"
            )
            self.assertEqual(summary_rows[0]["status"], "frame_mismatch")
            self.assertFalse(
                (
                    root
                    / "cropped_by_caterpillar"
                    / "motion_energy"
                    / "traces"
                    / "C01_mismatched_clip.motion.csv.gz"
                ).exists()
            )

    def test_trace_caching_skips_redecode_and_classify_only_avoids_video_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(6)]
            self.build_video_dataset(
                root,
                clip_name="cached_clip",
                animal_id="C01",
                frames=frames,
                start_utc=dt.datetime(2026, 8, 9, 19, 0, 0, tzinfo=dt.timezone.utc),
            )

            self.assertEqual(motion.main([str(root)]), 0)

            with mock.patch("extract_motion_energy.cv2.VideoCapture", side_effect=AssertionError("should not decode")):
                self.assertEqual(motion.main([str(root)]), 0)
                self.assertEqual(motion.main([str(root), "--classify-only"]), 0)

    def test_trace_files_do_not_compare_across_clip_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_time = dt.datetime(2026, 8, 9, 20, 0, 0, tzinfo=dt.timezone.utc)
            frames_a = [np.zeros((48, 48, 3), dtype=np.uint8) for _ in range(3)]
            frames_b = [np.zeros((48, 48, 3), dtype=np.uint8) for _ in range(3)]

            for clip_name, frames, start in [
                ("clip_a", frames_a, base_time),
                ("clip_b", frames_b, base_time + dt.timedelta(minutes=10)),
            ]:
                cropped_rel = f"cropped_by_caterpillar/C01_{clip_name}.mp4"
                timestamp_rel = f"cropped_by_caterpillar/timestamps/{clip_name}.timestamps.csv.gz"
                self.write_video(root / cropped_rel, frames)
                self.write_timestamp_sidecar(
                    root / timestamp_rel,
                    [start + dt.timedelta(seconds=index) for index in range(len(frames))],
                )

            self.write_manifest(
                root,
                [
                    self.make_manifest_row(
                        clip_key="clip_a",
                        animal_id="C01",
                        cropped_video="cropped_by_caterpillar/C01_clip_a.mp4",
                        copied_timestamp_file="cropped_by_caterpillar/timestamps/clip_a.timestamps.csv.gz",
                        timestamp_rows=3,
                    ),
                    self.make_manifest_row(
                        clip_key="clip_b",
                        animal_id="C01",
                        cropped_video="cropped_by_caterpillar/C01_clip_b.mp4",
                        copied_timestamp_file="cropped_by_caterpillar/timestamps/clip_b.timestamps.csv.gz",
                        timestamp_rows=3,
                    ),
                ],
            )

            self.assertEqual(motion.main([str(root)]), 0)
            trace_a = motion.load_motion_trace_rows(
                root / "cropped_by_caterpillar" / "motion_energy" / "traces" / "C01_clip_a.motion.csv.gz"
            )
            trace_b = motion.load_motion_trace_rows(
                root / "cropped_by_caterpillar" / "motion_energy" / "traces" / "C01_clip_b.motion.csv.gz"
            )

            self.assertEqual(trace_a[0].frame_index_start, 0)
            self.assertEqual(trace_b[0].frame_index_start, 0)

    def test_classification_uses_per_animal_thresholds_and_merges_short_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            timezone = dt.timezone(dt.timedelta(hours=-4))
            manifest_rows = [
                self.make_manifest_row(
                    clip_key="clip_gap",
                    animal_id="C01",
                    cropped_video="cropped_by_caterpillar/C01_clip_gap.mp4",
                    copied_timestamp_file="cropped_by_caterpillar/timestamps/clip_gap.timestamps.csv.gz",
                ),
                self.make_manifest_row(
                    clip_key="clip_threshold",
                    animal_id="C02",
                    cropped_video="cropped_by_caterpillar/C02_clip_threshold.mp4",
                    copied_timestamp_file="cropped_by_caterpillar/timestamps/clip_threshold.timestamps.csv.gz",
                ),
            ]
            self.write_manifest(root, manifest_rows)

            traces_dir = root / "cropped_by_caterpillar" / "motion_energy" / "traces"
            base = dt.datetime(2026, 8, 9, 21, 0, 0, tzinfo=dt.timezone.utc)
            self.make_trace_file(
                traces_dir / "C01_clip_gap.motion.csv.gz",
                [
                    motion.MotionTraceRow("C01", "clip_gap", 0, 1, base, base + dt.timedelta(seconds=1), 9.0, 1.0, 0.0),
                    motion.MotionTraceRow(
                        "C01",
                        "clip_gap",
                        1,
                        2,
                        base + dt.timedelta(seconds=1),
                        base + dt.timedelta(seconds=2),
                        1.0,
                        1.0,
                        0.0,
                    ),
                    motion.MotionTraceRow(
                        "C01",
                        "clip_gap",
                        2,
                        3,
                        base + dt.timedelta(seconds=2),
                        base + dt.timedelta(seconds=3),
                        8.0,
                        1.0,
                        0.0,
                    ),
                ],
                timezone,
            )
            self.make_trace_file(
                traces_dir / "C02_clip_threshold.motion.csv.gz",
                [
                    motion.MotionTraceRow("C02", "clip_threshold", 0, 1, base, base + dt.timedelta(seconds=1), 4.0, 1.0, 0.0),
                ],
                timezone,
            )

            thresholds = {
                "C01": motion.MotionThreshold("C01", 5.0, "manual", None, None, None, None, None, 3),
                "C02": motion.MotionThreshold("C02", 5.0, "manual", None, None, None, None, None, 1),
            }
            for animal_id in motion.ANIMAL_ORDER:
                thresholds.setdefault(
                    animal_id,
                    motion.MotionThreshold(animal_id, None, "manual", None, None, None, None, None, 0),
                )
            motion.write_thresholds(
                root / "cropped_by_caterpillar" / "motion_energy" / "motion_thresholds.csv",
                thresholds,
            )

            self.assertEqual(motion.main([str(root), "--classify-only"]), 0)
            states = motion.load_motion_states(
                root / "cropped_by_caterpillar" / "motion_energy" / "motion_states.csv"
            )

            c01_states = [state for state in states if state.animal_id == "C01"]
            c02_states = [state for state in states if state.animal_id == "C02"]
            self.assertEqual(len(c01_states), 1)
            self.assertEqual(c01_states[0].state, "mobile")
            self.assertEqual((c01_states[0].end_utc - c01_states[0].start_utc).total_seconds(), 3.0)
            self.assertEqual(len(c02_states), 1)
            self.assertEqual(c02_states[0].state, "immobile")

    def test_states_do_not_merge_across_clip_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            timezone = dt.timezone(dt.timedelta(hours=-4))
            self.write_manifest(
                root,
                [
                    self.make_manifest_row(
                        clip_key="clip_a",
                        animal_id="C01",
                        cropped_video="cropped_by_caterpillar/C01_clip_a.mp4",
                        copied_timestamp_file="cropped_by_caterpillar/timestamps/clip_a.timestamps.csv.gz",
                    ),
                    self.make_manifest_row(
                        clip_key="clip_b",
                        animal_id="C01",
                        cropped_video="cropped_by_caterpillar/C01_clip_b.mp4",
                        copied_timestamp_file="cropped_by_caterpillar/timestamps/clip_b.timestamps.csv.gz",
                    ),
                ],
            )
            traces_dir = root / "cropped_by_caterpillar" / "motion_energy" / "traces"
            base = dt.datetime(2026, 8, 9, 22, 0, 0, tzinfo=dt.timezone.utc)
            self.make_trace_file(
                traces_dir / "C01_clip_a.motion.csv.gz",
                [motion.MotionTraceRow("C01", "clip_a", 0, 1, base, base + dt.timedelta(seconds=1), 9.0, 1.0, 0.0)],
                timezone,
            )
            self.make_trace_file(
                traces_dir / "C01_clip_b.motion.csv.gz",
                [
                    motion.MotionTraceRow(
                        "C01",
                        "clip_b",
                        0,
                        1,
                        base + dt.timedelta(minutes=30),
                        base + dt.timedelta(minutes=30, seconds=1),
                        9.0,
                        1.0,
                        0.0,
                    )
                ],
                timezone,
            )

            thresholds = {
                animal_id: motion.MotionThreshold(
                    animal_id,
                    5.0 if animal_id == "C01" else None,
                    "manual",
                    None,
                    None,
                    None,
                    None,
                    None,
                    1,
                )
                for animal_id in motion.ANIMAL_ORDER
            }
            motion.write_thresholds(
                root / "cropped_by_caterpillar" / "motion_energy" / "motion_thresholds.csv",
                thresholds,
            )

            self.assertEqual(motion.main([str(root), "--classify-only"]), 0)
            states = motion.load_motion_states(
                root / "cropped_by_caterpillar" / "motion_energy" / "motion_states.csv"
            )

            self.assertEqual(len([state for state in states if state.animal_id == "C01"]), 2)
            self.assertEqual({state.clip_key for state in states if state.animal_id == "C01"}, {"clip_a", "clip_b"})


if __name__ == "__main__":
    unittest.main()
