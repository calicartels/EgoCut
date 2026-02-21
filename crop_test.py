"""
Test center crop on one video. Produces:
  - cropped video (256x256)
  - before.png (original 456x256 frame)
  - after.png  (cropped 256x256 frame)

Usage:
  python crop_test.py egocentric/val/factory001/factory_001_worker_001_0076.mp4
"""

import os
import subprocess
import sys

path = sys.argv[1]
basename = os.path.splitext(os.path.basename(path))[0]
out_dir = "crop_test"
os.makedirs(out_dir, exist_ok=True)

cropped_path = os.path.join(out_dir, f"{basename}_cropped.mp4")
before_path = os.path.join(out_dir, "before.png")
after_path = os.path.join(out_dir, "after.png")

# Extract one frame from middle of original (at 90s)
subprocess.run(
    ["ffmpeg", "-y", "-ss", "90", "-i", path,
     "-frames:v", "1", before_path],
    capture_output=True, check=True,
)
print(f"Before: {before_path}")

# Center crop 456x256 → 256x256
# crop=out_w:out_h:x:y — omitting x,y centers automatically
subprocess.run(
    ["ffmpeg", "-y", "-i", path,
     "-vf", "crop=256:256",
     "-c:v", "libx264", "-preset", "fast", "-crf", "23",
     "-c:a", "copy",
     cropped_path],
    capture_output=True, check=True,
)
print(f"Cropped video: {cropped_path}")

# Extract same frame from cropped video
subprocess.run(
    ["ffmpeg", "-y", "-ss", "90", "-i", cropped_path,
     "-frames:v", "1", after_path],
    capture_output=True, check=True,
)
print(f"After: {after_path}")

# Print dimensions for verification
for label, p in [("Original", before_path), ("Cropped", after_path)]:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", p],
        capture_output=True, text=True,
    )
    print(f"  {label}: {result.stdout.strip()}")