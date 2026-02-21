"""
Crop and collect the 11 already-labeled videos (one per factory) into
a single folder for manual Gemini review.

Applies: center crop 456→256 wide, 20px top trim, resize to 256x256.
"""

import os
import subprocess

DATA_DIR = "egocentric"
OUT_DIR = "gemini_review"
os.makedirs(OUT_DIR, exist_ok=True)

# The last val video per factory — these have existing annotations
VIDEOS = {
    "factory001": "factory_001_worker_001_0076.mp4",
    "factory002": "factory_002_worker_001_0075.mp4",
    "factory003": "factory_003_worker_001_0053.mp4",
    "factory004": "factory_004_worker_001_0090.mp4",
    "factory005": "factory_005_worker_001_0086.mp4",
    "factory006": "factory_006_worker_001_0078.mp4",
    "factory007": "factory_007_worker_001_0085.mp4",
    "factory008": "factory_008_worker_001_0068.mp4",
    "factory009": "factory_009_worker_001_0093.mp4",
    "factory010": "factory_010_worker_001_0077.mp4",
    "factory011": "factory_011_worker_001_0079.mp4",
}

# crop=256:236:100:20 → center 256 wide, skip top 20px, take 236 tall
# scale=256:256 → resize back to square for VJEPA2
VF = "crop=256:236:100:20,scale=256:256"

for factory, filename in sorted(VIDEOS.items()):
    src = os.path.join(DATA_DIR, "val", factory, filename)
    dst = os.path.join(OUT_DIR, filename)

    subprocess.run(
        ["ffmpeg", "-y", "-i", src,
         "-vf", VF,
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-c:a", "copy",
         dst],
        capture_output=True, check=True,
    )

    size_mb = os.path.getsize(dst) / 1e6
    print(f"{factory}: {filename} ({size_mb:.1f} MB)")

print(f"\n{len(VIDEOS)} videos in {OUT_DIR}/")