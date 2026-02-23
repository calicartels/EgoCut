import json
import os
import subprocess
import sys


def parse_time(t):
    parts = t.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def build_filter(segments):
    parts = []
    for seg in segments:
        start = parse_time(seg["start_time"])
        end = parse_time(seg["end_time"])
        is_golden = seg.get("label") == "Golden Standard"

        color = "0x00AA00" if is_golden else "0xAA0000"
        seg_id = seg.get("segment_id")

        if is_golden and seg_id:
            text = f"GOLDEN {seg_id}"
        elif is_golden:
            text = "GOLDEN"
        else:
            text = "NOT GOOD"

        ts = f"{seg['start_time']}-{seg['end_time']}".replace(":", r"\:")
        enable = f"between(t\\,{start}\\,{end})"

        parts.append(
            f"drawbox=x=0:y=0:w=iw:h=32:color={color}@0.7:t=fill:enable='{enable}'"
        )
        parts.append(
            f"drawtext=text='{text}':x=10:y=6:fontsize=18:fontcolor=white:enable='{enable}'"
        )
        parts.append(
            f"drawtext=text='{ts}':x=w-120:y=6:fontsize=18:fontcolor=white:enable='{enable}'"
        )

    return ",".join(parts)


def verify(label_path, video_path):
    with open(label_path) as f:
        segments = json.load(f)

    basename = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = "verify_output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{basename}_overlay.mp4")

    vf = build_filter(segments)

    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vf", vf,
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-an", out_path],
        check=True,
    )

    print(f"Output: {out_path}")


label_path = sys.argv[1]
video_path = sys.argv[2]
verify(label_path, video_path)