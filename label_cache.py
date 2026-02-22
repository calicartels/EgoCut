"""
Label one video using the Gemini context cache.

Crops the video (256x256 with 20px top trim), uploads it, and sends
to Gemini with the cached calibration context.

Usage:
  python label_cached.py egocentric/val/factory001/factory_001_worker_001_0076.mp4

Requires: cache_name.txt (created by create_cache.py)
"""

import json
import os
import re
import subprocess
import sys
import time

from google import genai
from google.genai import types

from config import CROPPED_DIR, FFMPEG_VF, GEMINI_MODEL, LABELS_DIR, TASKS


def parse_video_path(path):
    basename = os.path.splitext(os.path.basename(path))[0]
    parts = basename.split("_")
    factory_key = f"factory{parts[1]}"
    return factory_key, basename


def get_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def crop_video(input_path, factory_key, video_id):
    out_dir = os.path.join(CROPPED_DIR, factory_key)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{video_id}.mp4")

    if os.path.exists(out_path):
        return out_path

    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-vf", FFMPEG_VF,
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-an", out_path],
        capture_output=True, check=True,
    )
    return out_path


def upload_and_wait(client, path):
    f = client.files.upload(file=path)
    while f.state.name == "PROCESSING":
        time.sleep(2)
        f = client.files.get(name=f.name)
    return f


def build_prompt(factory_key, video_id, duration_s):
    task = TASKS[factory_key]
    dur_int = int(duration_s)

    return (
        f"Video: {video_id}.mp4\n\n"
        f"Task: {task['description']}\n"
        f"Average cycle time: {task['avg_cycle_s']}s\n\n"
        f"Apply the Golden Standard criteria to this video. "
        f"Cover every second from 00:00 to {dur_int // 60:02d}:{dur_int % 60:02d}. "
        f"Return ONLY the JSON list."
    )


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def validate(label, duration_s):
    def parse_t(t):
        p = t.split(":")
        return int(p[0]) * 60 + int(p[1])

    golden_s = sum(parse_t(s["end_time"]) - parse_t(s["start_time"]) + 1
                   for s in label if s.get("label") == "Golden Standard")
    total_s = sum(parse_t(s["end_time"]) - parse_t(s["start_time"]) + 1
                  for s in label)
    coverage = total_s / duration_s
    n_golden = sum(1 for s in label if s.get("label") == "Golden Standard")

    print(f"Golden: {golden_s}s ({n_golden} segments), Coverage: {coverage:.0%}")

    if coverage < 0.9:
        print(f"WARNING: {coverage:.0%} coverage - gaps in labeling")
    return label


def label_video(path):
    factory_key, video_id = parse_video_path(path)
    duration_s = get_duration(path)

    # Read cache name
    with open("cache_name.txt") as f:
        cache_name = f.read().strip()

    # Crop
    cropped = crop_video(path, factory_key, video_id)

    # Upload
    client = genai.Client()
    print(f"Uploading {cropped}...")
    video_file = upload_and_wait(client, cropped)

    # Label
    prompt = build_prompt(factory_key, video_id, duration_s)
    print(f"Labeling with cache {cache_name}...")

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[video_file, prompt],
        config=types.GenerateContentConfig(
            cached_content=cache_name,
        ),
    )

    # Clean up remote file
    client.files.delete(name=video_file.name)

    # Parse and save
    label = extract_json(response.text)
    label = validate(label, duration_s)

    out_dir = os.path.join(LABELS_DIR, factory_key)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{video_id}.json")
    with open(out_path, "w") as f:
        json.dump(label, f, indent=2)

    print(f"Saved: {out_path}")
    return label


label = label_video(sys.argv[1])
print(json.dumps(label, indent=2))