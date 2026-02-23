import glob
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
        time.sleep(2.5)
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

    return golden_s, n_golden, coverage


def find_all_videos():
    videos = []
    for split in ["train", "val", "test"]:
        pattern = os.path.join("egocentric", split, "*", "*.mp4")
        videos.extend(sorted(glob.glob(pattern)))
    return videos


def label_path_for(factory_key, video_id):
    return os.path.join(LABELS_DIR, factory_key, f"{video_id}.json")


def is_labeled(factory_key, video_id):
    if os.path.exists(label_path_for(factory_key, video_id)):
        return True
    if os.path.exists(os.path.join(LABELS_DIR, f"{video_id}.json")):
        return True
    return False


def label_one(client, cache_name, video_path):
    factory_key, video_id = parse_video_path(video_path)
    duration_s = get_duration(video_path)

    cropped = crop_video(video_path, factory_key, video_id)
    video_file = upload_and_wait(client, cropped)

    prompt = build_prompt(factory_key, video_id, duration_s)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[video_file, prompt],
        config=types.GenerateContentConfig(
            cached_content=cache_name,
        ),
    )

    client.files.delete(name=video_file.name)

    label = extract_json(response.text)
    golden_s, n_golden, coverage = validate(label, duration_s)

    out_dir = os.path.join(LABELS_DIR, factory_key)
    os.makedirs(out_dir, exist_ok=True)
    out_path = label_path_for(factory_key, video_id)
    with open(out_path, "w") as f:
        json.dump(label, f, indent=2)

    return golden_s, n_golden, coverage


dry_run = "--dry-run" in sys.argv
factory_filter = None
for i, arg in enumerate(sys.argv):
    if arg == "--factory" and i + 1 < len(sys.argv):
        factory_filter = sys.argv[i + 1]

all_videos = find_all_videos()
to_label = []
for v in all_videos:
    fk, vid = parse_video_path(v)
    if fk not in TASKS:
        continue
    if factory_filter and fk != factory_filter:
        continue
    if is_labeled(fk, vid):
        continue
    to_label.append(v)

print(f"Found {len(all_videos)} total videos, {len(to_label)} need labeling")

if dry_run:
    for v in to_label:
        fk, vid = parse_video_path(v)
        print(f"  {fk}: {vid}")
    sys.exit(0)

if not to_label:
    print("Nothing to label")
    sys.exit(0)

with open("cache_name.txt") as f:
    cache_name = f.read().strip()

client = genai.Client()

cache = client.caches.get(name=cache_name)
print(f"Cache: {cache.name} (expires {cache.expire_time})")

DELAY_S = 5
failed = []

for i, video_path in enumerate(to_label):
    fk, vid = parse_video_path(video_path)
    print(f"[{i+1}/{len(to_label)}] {fk}/{vid}...", end=" ", flush=True)

    try:
        golden_s, n_golden, coverage = label_one(client, cache_name, video_path)
        print(f"golden={golden_s}s ({n_golden} segs), coverage={coverage:.0%}")
    except Exception as e:
        print(f"FAILED: {e}")
        failed.append((video_path, str(e)))
        time.sleep(15)
        continue

    if i < len(to_label) - 1:
        time.sleep(DELAY_S)

labeled = len(to_label) - len(failed)
print(f"\nDone: {labeled}/{len(to_label)} labeled")
if failed:
    print(f"Failed ({len(failed)}):")
    for path, err in failed:
        print(f"  {path}: {err}")