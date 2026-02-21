"""
Test center crop with top trim. Outputs 3 full cropped videos.

Usage:
  python crop_test.py egocentric/val/factory001/factory_001_worker_001_0076.mp4
"""

import os
import subprocess
import sys

path = sys.argv[1]
out_dir = "crop_test"
os.makedirs(out_dir, exist_ok=True)
basename = os.path.splitext(os.path.basename(path))[0]

# Original: 456x256. Center crop removes 100px each side → 256 wide.
# Top trim options: how many pixels to cut from top before resizing to 256x256.
trims = [0, 20, 36]

for trim in trims:
    h = 256 - trim
    vf = f"crop=256:{h}:100:{trim},scale=256:256"
    out_path = os.path.join(out_dir, f"{basename}_trim{trim}.mp4")

    subprocess.run(
        ["ffmpeg", "-y", "-i", path,
         "-vf", vf,
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-c:a", "copy",
         out_path],
        capture_output=True, check=True,
    )

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"trim={trim}px → {out_path} ({size_mb:.1f} MB)")