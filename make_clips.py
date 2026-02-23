"""
Convert per-second labels to clip-level binary labels.

Each clip: 16 frames at 2fps = 8 seconds of video.
Label: 1 (golden) if >50% of the clip's seconds are Golden Standard, else 0.

Usage:
  python make_clips.py

Reads: egocentric/{train,val,test}/ and labels/
Writes: clips/{train,val,test}_manifest.json
"""

import glob
import json
import os

import numpy as np

from config import LABELS_DIR, TASKS, CLIP_DURATION_S

CLIPS_DIR = "clips"

# Choice: non-overlapping clips (stride = clip duration = 8s).
# Alternative: 50% overlap (stride=4s) doubles training data but
# introduces correlation between adjacent clips. Start simple,
# can add overlap later if training data is insufficient.
STRIDE_S = int(CLIP_DURATION_S)  # 8 seconds


def parse_time(t):
    p = t.split(":")
    return int(p[0]) * 60 + int(p[1])


def load_labels(label_path):
    with open(label_path) as f:
        segments = json.load(f)

    max_t = 0
    for seg in segments:
        max_t = max(max_t, parse_time(seg["end_time"]))
    duration_s = max_t + 1

    # Per-second array: 1 = golden, 0 = not good
    labels = np.zeros(duration_s, dtype=np.int32)
    for seg in segments:
        start = parse_time(seg["start_time"])
        end = parse_time(seg["end_time"])
        if seg.get("label") == "Golden Standard":
            labels[start:end + 1] = 1

    return labels, duration_s


def find_label_file(factory_key, video_id):
    # Subfolder (API-labeled)
    p = os.path.join(LABELS_DIR, factory_key, f"{video_id}.json")
    if os.path.exists(p):
        return p
    # Flat (calibration labels)
    p = os.path.join(LABELS_DIR, f"{video_id}.json")
    if os.path.exists(p):
        return p
    return None


def make_clips_for_video(video_path, split):
    basename = os.path.splitext(os.path.basename(video_path))[0]
    parts = basename.split("_")
    factory_key = f"factory{parts[1]}"

    if factory_key not in TASKS:
        return []

    label_path = find_label_file(factory_key, basename)
    if label_path is None:
        return []

    per_second, duration_s = load_labels(label_path)

    clips = []
    clip_start = 0
    while clip_start + CLIP_DURATION_S <= duration_s:
        clip_end = int(clip_start + CLIP_DURATION_S)

        segment = per_second[clip_start:clip_end]
        golden_ratio = segment.mean()

        # Choice: threshold at 0.5 (majority golden = golden clip).
        # Alternative: stricter threshold (0.75) for higher precision.
        label = 1 if golden_ratio > 0.5 else 0

        clips.append({
            "video_id": basename,
            "factory": factory_key,
            "split": split,
            "clip_start_s": clip_start,
            "clip_end_s": clip_end,
            "label": label,
            "golden_ratio": float(golden_ratio),
        })

        clip_start += STRIDE_S

    return clips


def process_split(split):
    pattern = os.path.join("egocentric", split, "*", "*.mp4")
    videos = sorted(glob.glob(pattern))

    all_clips = []
    for video_path in videos:
        clips = make_clips_for_video(video_path, split)
        all_clips.extend(clips)

    manifest_path = os.path.join(CLIPS_DIR, f"{split}_manifest.json")
    os.makedirs(CLIPS_DIR, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(all_clips, f, indent=2)

    n_golden = sum(1 for c in all_clips if c["label"] == 1)
    n_total = len(all_clips)
    print(f"{split}: {n_total} clips ({n_golden} golden, {n_total - n_golden} not-good)")

    return all_clips


for split in ["train", "val", "test"]:
    process_split(split)