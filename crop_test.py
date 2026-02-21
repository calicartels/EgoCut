"""
Test center crop with top trim. Produces side-by-side comparisons.

Tries 3 top-trim values so you can pick the best one:
  - 0px  (current: 256x256, no top trim)
  - 20px (256x236 crop, then resize to 256x256)
  - 36px (256x220 crop, then resize to 256x256)

Usage:
  python crop_test.py egocentric/val/factory001/factory_001_worker_001_0076.mp4
"""

import os
import subprocess
import sys

path = sys.argv[1]
out_dir = "crop_test"
os.makedirs(out_dir, exist_ok=True)

# Original is 456x256. Center crop width: (456-256)/2 = 100px each side.
trims = [0, 20, 36]
timestamp = "90"

# Extract original frame
before = os.path.join(out_dir, "before.png")
subprocess.run(
    ["ffmpeg", "-y", "-ss", timestamp, "-i", path,
     "-frames:v", "1", before],
    capture_output=True, check=True,
)

for trim in trims:
    tag = f"trim{trim}"
    frame_path = os.path.join(out_dir, f"after_{tag}.png")

    # Choice: crop then scale back to 256x256.
    # crop=256:(256-trim):100:trim takes 256 wide centered, (256-trim) tall
    # starting trim pixels from top. Then scale back to 256x256.
    # The slight vertical stretch from scaling is negligible (<15%).
    h = 256 - trim
    vf = f"crop=256:{h}:100:{trim},scale=256:256"

    subprocess.run(
        ["ffmpeg", "-y", "-ss", timestamp, "-i", path,
         "-vf", vf, "-frames:v", "1", frame_path],
        capture_output=True, check=True,
    )

    # Side by side: before (456x256) | after (256x256)
    # Resize before to same height for clean comparison
    sidebyside = os.path.join(out_dir, f"compare_{tag}.png")
    subprocess.run(
        ["ffmpeg", "-y",
         "-i", before, "-i", frame_path,
         "-filter_complex",
         "[0]scale=-1:256[a];[1]scale=-1:256[b];[a][b]hstack[out]",
         "-map", "[out]", sidebyside],
        capture_output=True, check=True,
    )
    print(f"trim={trim}px: {sidebyside}")

print(f"\nOriginal frame: {before}")
print("Compare the 3 side-by-side images and pick the best trim value.")