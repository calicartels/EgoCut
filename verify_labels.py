"""
Overlay label timestamps on a video for visual verification.

Draws a colored bar + text at the top of the video:
  Green  = "Golden Standard"
  Red    = "not good for collecting"

Usage:
  python verify_labels.py labels/factory_001_worker_001_0076.json \
    gemini_review/factory_001_worker_001_0076.mp4

Outputs: verify_output/factory_001_worker_001_0076_overlay.mp4
"""

import json
import os
import subprocess
import sys


def parse_time(t):
    """Convert "MM:SS" to seconds."""
    parts = t.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def build_drawtext_filters(segments):
    """Build ffmpeg drawtext filter chain from label segments."""
    filters = []
    for seg in segments:
        start = parse_time(seg["start_time"])
        end = parse_time(seg["end_time"])
        is_golden = seg.get("label") == "Golden Standard"

        color = "0x00AA00" if is_golden else "0xAA0000"
        text = "GOLDEN" if is_golden else "NOT GOOD"
        seg_id = seg.get("segment_id", "")
        if seg_id:
            text = f"GOLDEN #{seg_id}"

        # Background bar
        filters.append(
            f"drawbox=x=0:y=0:w=iw:h=32:color={color}@0.7:t=fill"
            f":enable='between(t,{start},{end})'"
        )
        # Text label
        filters.append(
            f"drawtext=text='{text}':x=10:y=6:fontsize=18"
            f":fontcolor=white:enable='between(t,{start},{end})'"
        )
        # Timestamp display
        filters.append(
            f"drawtext=text='{seg['start_time']}-{seg['end_time']}'"
            f":x=w-120:y=6:fontsize=18"
            f":fontcolor=white:enable='between(t,{start},{end})'"
        )

    return ",".join(filters)


def verify(label_path, video_path):
    with open(label_path) as f:
        segments = json.load(f)

    basename = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = "verify_output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{basename}_overlay.mp4")

    vf = build_drawtext_filters(segments)

    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vf", vf,
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-c:a", "copy",
         out_path],
        check=True,
    )

    print(f"Output: {out_path}")


label_path = sys.argv[1]
video_path = sys.argv[2]
verify(label_path, video_path)