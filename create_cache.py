"""
Create explicit Gemini context cache with calibration conversation.

Uploads the full cropped factory001 and factory002 videos (same ones
used in the original Gemini chat). The conversation references golden
timestamps within those videos, matching the original calibration flow.

Factory003 deliberately excluded — it's our validation target.

Usage:
  export GEMINI_API_KEY=your_key
  python create_cache.py
"""

import json
import os
import time

from google import genai
from google.genai import types

from config import GEMINI_MODEL

# Choice: use the already-cropped (256x256) videos from gemini_review/.
# These are the same videos Gemini saw during calibration.
FACTORY001_VIDEO = "gemini_review/factory_001_worker_001_0076.mp4"
FACTORY002_VIDEO = "gemini_review/factory_002_worker_001_0075.mp4"

# Choice: 2h TTL — enough to label a full batch in one sitting.
# Alternative: 48h for multi-day, but storage costs scale with TTL.
CACHE_TTL = "7200s"

SYSTEM_INSTRUCTION = (
    "You are an expert Video Data Curator for Egocentric AI training.\n\n"
    "Your job is to identify 'Golden Standard' action segments within factory "
    "worker egocentric (first-person headcam) videos. A Golden Standard segment "
    "is a high-clarity, uninterrupted execution of the worker's primary "
    "repetitive task.\n\n"
    "Selection Criteria (Strict):\n\n"
    "1. Full Temporal Cycle: The segment must contain a complete cycle of the "
    "task. It starts when hands begin the action and ends when the object is "
    "released and hands return to neutral position.\n\n"
    "2. Gaze Stability: The camera (worker's head) must remain steady and locked "
    "on the manipulation zone. Reject segments where the worker looks away, "
    "turns their head, or the camera blurs significantly (>0.5 seconds).\n\n"
    "3. Zero Occlusion: The primary action (point of contact between hands and "
    "objects) must be fully visible. Reject segments where tools, equipment, "
    "support beams, bins, or other structures block the view.\n\n"
    "4. Optimal Framing: The hands must remain within the central 60% of the "
    "frame throughout. If the action happens at the extreme edges or goes "
    "off-screen, discard the segment.\n\n"
    "Output Format: Return ONLY a JSON list covering every second of the video. "
    "Golden Standard segments get a segment_id and description. All other "
    "segments are labeled 'not good for collecting'. No gaps, no overlaps."
)

# Factory001 corrected labels (ground truth from manual review)
FACTORY001_LABELS = [
    {"start_time": "00:00", "end_time": "00:04", "label": "not good for collecting"},
    {"segment_id": 1, "start_time": "00:05", "end_time": "00:08", "label": "Golden Standard",
     "description": "Single perfect repetition; centered framing and clear thumb-press."},
    {"start_time": "00:09", "end_time": "00:15", "label": "not good for collecting"},
    {"segment_id": 2, "start_time": "00:16", "end_time": "00:19", "label": "Golden Standard",
     "description": "Unobstructed view of assembly; zero occlusion from assembly line structure."},
    {"start_time": "00:20", "end_time": "00:25", "label": "not good for collecting"},
    {"segment_id": 3, "start_time": "00:26", "end_time": "00:28", "label": "Golden Standard",
     "description": "High-speed repetition with focused gaze and stable camera."},
    {"start_time": "00:29", "end_time": "00:30", "label": "not good for collecting"},
    {"segment_id": 4, "start_time": "00:31", "end_time": "00:34", "label": "Golden Standard",
     "description": "Full acquisition-to-release cycle within the central work zone."},
    {"start_time": "00:35", "end_time": "00:49", "label": "not good for collecting"},
    {"segment_id": 5, "start_time": "00:50", "end_time": "00:53", "label": "Golden Standard",
     "description": "Clear component contrast; steady egocentric perspective."},
    {"start_time": "00:54", "end_time": "01:41", "label": "not good for collecting"},
    {"segment_id": 6, "start_time": "01:42", "end_time": "01:45", "label": "Golden Standard",
     "description": "Optimal lighting on the insertion slot; distinct tactile press visible."},
    {"segment_id": 7, "start_time": "01:46", "end_time": "01:49", "label": "Golden Standard",
     "description": "Consistent repetition; no head movement or peripheral distractions."},
    {"start_time": "01:50", "end_time": "02:03", "label": "not good for collecting"},
    {"segment_id": 8, "start_time": "02:04", "end_time": "02:07", "label": "Golden Standard",
     "description": "Symmetrical hand positioning; perfectly centered manipulation."},
    {"start_time": "02:08", "end_time": "02:21", "label": "not good for collecting"},
    {"segment_id": 9, "start_time": "02:22", "end_time": "02:29", "label": "Golden Standard",
     "description": "Master Golden Segment; two full, uninterrupted textbook cycles."},
    {"start_time": "02:30", "end_time": "02:59", "label": "not good for collecting"},
]

# Factory002 labels (Gemini got these right without correction)
FACTORY002_LABELS = [
    {"start_time": "00:00", "end_time": "00:02", "label": "not good for collecting",
     "reason": "Segment starts mid-action; camera moves away from primary workspace."},
    {"segment_id": 1, "start_time": "00:03", "end_time": "00:24", "label": "Golden Standard",
     "description": "Primary Golden Template: two full lamination cycles with centered framing, steady rolling motion, and clear bimanual coordination."},
    {"start_time": "00:25", "end_time": "00:26", "label": "not good for collecting",
     "reason": "Transition period; camera gaze shifts briefly."},
    {"segment_id": 2, "start_time": "00:27", "end_time": "00:46", "label": "Golden Standard",
     "description": "Two high-speed repetitions; consistent lighting and zero occlusion."},
    {"start_time": "00:47", "end_time": "00:48", "label": "not good for collecting",
     "reason": "Gaze shift away from task."},
    {"segment_id": 3, "start_time": "00:49", "end_time": "01:06", "label": "Golden Standard",
     "description": "Single long cycle with focus on panel alignment and roller pressure."},
    {"start_time": "01:07", "end_time": "01:10", "label": "not good for collecting",
     "reason": "Camera blur and significant head movement."},
    {"segment_id": 4, "start_time": "01:11", "end_time": "01:29", "label": "Golden Standard",
     "description": "Final clean repetition; matches the Golden Template trajectory."},
    {"start_time": "01:30", "end_time": "02:09", "label": "not good for collecting",
     "reason": "Out-of-distribution: worker interacts with bottle, looks at instructions."},
]


def upload_and_wait(client, path):
    f = client.files.upload(file=path)
    while f.state.name == "PROCESSING":
        time.sleep(2.5)
        f = client.files.get(name=f.name)
    print(f"Uploaded: {os.path.basename(path)} -> {f.uri}")
    return f


def build_conversation(f001_file, f002_file):
    """Build multi-turn calibration conversation with full videos.

    Matches the original Gemini chat flow:
    1. Upload full factory001 video, ask for labels
    2. Gemini gives labels (simulated as too-loose first attempt)
    3. User corrects with right labels, points to 02:22-02:29
    4. Gemini acknowledges and articulates hard-fail criteria
    5. Upload full factory002 video, provide task description + golden template
    6. Gemini produces correct labels without correction
    """

    f001_part = types.Part(
        file_data=types.FileData(file_uri=f001_file.uri, mime_type=f001_file.mime_type),
    )
    f002_part = types.Part(
        file_data=types.FileData(file_uri=f002_file.uri, mime_type=f002_file.mime_type),
    )

    contents = [
        # Turn 1: User uploads full factory001 video and asks for labels
        types.Content(
            role="user",
            parts=[
                f001_part,
                types.Part(text=
                    "This is factory_001_worker_001_0076.mp4 (full 3-minute video). "
                    "The worker is performing circular ring installation into electronic "
                    "casings. They pick up a rectangular casing, retrieve a circular ring "
                    "from a pile, press the ring into a slot using their thumbs, and release "
                    "the completed unit onto a moving green conveyor belt.\n\n"
                    "Apply the Golden Standard protocol to identify all high-quality segments."
                ),
            ],
        ),

        # Turn 2: Model gives a too-loose first attempt
        types.Content(
            role="model",
            parts=[types.Part(text=
                "I have analyzed the video. Here are the Golden Standard segments I identified:\n\n"
                + json.dumps([
                    {"start": "00:00", "end": "00:02", "label": "Golden Standard",
                     "confidence_score": 0.95, "reason": "Complete cycle with stable gaze."},
                    {"start": "00:05", "end": "00:08", "label": "Golden Standard",
                     "confidence_score": 0.92, "reason": "High-clarity bimanual coordination."},
                    {"start": "00:13", "end": "00:15", "label": "Golden Standard",
                     "confidence_score": 0.88, "reason": "Clean execution with hands in central ROI."},
                    {"start": "00:21", "end": "00:23", "label": "Golden Standard",
                     "confidence_score": 0.90, "reason": "Steady camera; clear ring alignment."},
                    {"start": "00:24", "end": "02:59", "label": "Discard/Negative",
                     "reason": "Remaining duration contains frequent occlusions."},
                ], indent=2)
            )],
        ),

        # Turn 3: User corrects — points to 02:22-02:29 as master golden segment
        types.Content(
            role="user",
            parts=[types.Part(text=
                "That's wrong. You included 00:00-00:02 which has camera instability, "
                "and 00:13-00:15 which has the white support beam occluding the view. "
                "You also missed the best segments entirely.\n\n"
                "The PERFECT golden segment in this video is 02:22 to 02:29 — watch it again. "
                "Two full, uninterrupted textbook cycles with perfect centering, zero occlusion, "
                "and steady gaze. That is what Golden Standard looks like.\n\n"
                "Here is the CORRECT labeling I verified by hand:\n\n"
                + json.dumps(FACTORY001_LABELS, indent=2)
            )],
        ),

        # Turn 4: Model acknowledges correction and articulates hard-fail criteria
        types.Content(
            role="model",
            parts=[types.Part(text=
                "Understood. I see where my previous attempt failed. I was too loose with the "
                "criteria. Looking at your corrections and re-watching the video, I now understand "
                "three hard-fail rules:\n\n"
                "1. Gaze Shift: If the worker looks away from the task zone for even 0.5 seconds, "
                "the segment is immediately cut. I incorrectly included 00:00-00:02 where there is "
                "a brief camera settle.\n"
                "2. Structural Occlusion: The white support beam at center-left frequently blocks "
                "the Pick phase. Any frame where hands disappear behind this beam means discard.\n"
                "3. Boundary Truncation: If the hand starts a Pick or finishes a Place action "
                "off-camera or at the frame edge, the segment is discarded. I missed that 00:13-00:15 "
                "has hands at the extreme edge.\n\n"
                "The Master Golden Segment at 02:22-02:29 is the benchmark — it shows what perfect "
                "looks like: centered thumb-press, both hands visible, zero occlusion, steady gaze "
                "locked on the work zone, and two complete cycles.\n\n"
                "I will apply these strict criteria to all future videos."
            )],
        ),

        # Turn 5: User uploads full factory002 video with task description
        types.Content(
            role="user",
            parts=[
                f002_part,
                types.Part(text=
                    "Next video: factory_002_worker_001_0075.mp4\n\n"
                    "0:03 to 0:24 is the golden template for this task.\n\n"
                    "The worker performs a lamination/adhesion task for electronic panel "
                    "components. They position a base component, align a thin dark rectangular "
                    "panel over it, use a blue hand roller to press for bubble-free adhesion, "
                    "and move the finished piece to a completed stack. The process emphasizes "
                    "uniform pressure and surface sealing.\n\n"
                    "Apply the same Golden Standard criteria."
                ),
            ],
        ),

        # Turn 6: Model produces correct factory002 labels (no correction needed)
        types.Content(
            role="model",
            parts=[types.Part(text=json.dumps(FACTORY002_LABELS, indent=2))],
        ),
    ]

    return contents


def create_cache():
    client = genai.Client()

    # Upload both full calibration videos
    print("Uploading factory001 video...")
    f001_file = upload_and_wait(client, FACTORY001_VIDEO)

    print("Uploading factory002 video...")
    f002_file = upload_and_wait(client, FACTORY002_VIDEO)

    # Build conversation referencing both videos
    contents = build_conversation(f001_file, f002_file)

    # Create the explicit cache
    print("Creating cache...")
    cache = client.caches.create(
        model=GEMINI_MODEL,
        config=types.CreateCachedContentConfig(
            display_name="egocut-golden-standard-v1",
            system_instruction=SYSTEM_INSTRUCTION,
            contents=contents,
            ttl=CACHE_TTL,
        ),
    )

    print(f"Cache: {cache.name}")
    print(f"Model: {cache.model}")
    print(f"Usage: {cache.usage_metadata}")
    print(f"Expires: {cache.expire_time}")

    # Save cache name
    with open("cache_name.txt", "w") as f:
        f.write(cache.name)

    # Clean up uploaded files — cache retains its own copies
    client.files.delete(name=f001_file.name)
    client.files.delete(name=f002_file.name)

    return cache.name


name = create_cache()