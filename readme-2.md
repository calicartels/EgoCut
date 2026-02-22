# EgoCut — Factory Cycle Detection Dataset

Curated dataset of factory worker egocentric (first-person headcam) video
for detecting clean, complete work cycles. Built from the Egocentric-100K
dataset using VJEPA2 as the target vision backbone.

## Problem

Factory workers perform short repeating tasks (2-22 second cycles) at
assembly stations. A headcam captures their point-of-view. We need to
classify each moment as either a clean, complete action cycle or not.

Previous attempts using single-frame classification failed because static
hand poses are ambiguous — "hand reaching for headgear adjustment" looks
identical to "hand reaching for device casing" in a single frame. VJEPA2
solves this by encoding 16-frame temporal clips, capturing the full
motion trajectory.

A prior attempt using VJEPA2 with cosine similarity sliding windows also
failed: it detected anomalies initially but then normalized to them. After
2 seconds of "not working," the sliding window treated "not working" as the
new baseline. The fix is discriminative classification (frozen VJEPA2
backbone + linear probe), not similarity-based detection.


## Source Dataset

**[builddotai/Egocentric-100K](https://huggingface.co/datasets/builddotai/Egocentric-100K)**

- 100K+ hours of factory egocentric video
- 11 factories, 7-57 workers each
- Videos: ~180 seconds, 30fps native, 456×256 pixels, fisheye lens, H.265 codec
- Stored as WebDataset tar archives on HuggingFace, one tar per worker

We initially surveyed all 11 factories. Factory 010 (cable packaging) was
dropped due to poor video quality — the task happens mostly outside the
camera's field of view. 10 factories remain active.


## Dataset Extraction

`extract_dataset.py` downloads tars from HuggingFace and extracts specific
videos into a train/val/test split.

### Split Design

| Split | Worker | Videos per Factory | Total Videos | Purpose |
|-------|--------|--------------------|-------------|---------|
| Train | worker_001 | 3 (sequential: 0000-0002) | 33 | Learn cycle classification |
| Val   | worker_001 | 2 (0003 + last video) | 22 | Tune threshold, early stopping |
| Test  | worker_002 | 2 (first 2 available) | 22 | Cross-worker generalization |

Total: 77 videos, ~2.2 GB.

### Why This Split

**Sequential training videos** (0000, 0001, 0002) from the same shift
ensure consistent lighting, camera angle, and worker behavior. Random
sampling would risk crossing shift boundaries where conditions change.

**worker_001 for train+val, worker_002 for test** is the minimum viable
generalization test. If the model can't transfer from one person to another
doing the same task at the same station, the approach is broken. This is a
harder test than same-worker held-out videos but easier than cross-factory
transfer.

**The last val video per factory** is the one with existing Gemini
calibration labels (see Labeling section below), enabling direct comparison
between human-verified labels and model predictions.

### Directory Structure

```
egocentric/
├── train/
│   ├── factory001/   (3 videos)
│   ├── factory002/   (3 videos)
│   └── ...
├── val/
│   ├── factory001/   (2 videos)
│   └── ...
└── test/
    ├── factory001/   (2 videos)
    └── ...
```


## Video Preprocessing

### Center Crop: 456×256 → 256×256

The raw videos are 456 pixels wide with a fisheye lens. Two problems:

1. The edges capture neighboring workers at adjacent stations doing
   different tasks, which confuses both Gemini (during labeling) and
   VJEPA2 (during training)
2. Fisheye distortion is worst at the edges

We center-crop to remove 100 pixels from each side of the width and 20
pixels from the top (where the adjacent station's worker torso bleeds in).

ffmpeg filter: `crop=256:236:100:20,scale=256:256`

The crop was validated by generating full videos at three trim levels
(0px, 20px, 36px top trim) using `crop_test.py` and manually reviewing.
20px was selected as the best balance — removes the neighbor while keeping
the full workspace visible. Factory 007 has slight neighbor bleed-through
even at 20px, but this was deemed acceptable.

The 236→256 vertical rescale introduces ~8% stretch, which is imperceptible
and within VJEPA2's augmentation tolerance.

This crop also aligns Gemini labeling with VJEPA2 training — both see the
exact same 256×256 frame, so there's no mismatch between what was labeled
and what the model learns from.

### Fixed 2fps Sampling

All 10 tasks are sampled at a fixed 2 frames per second, regardless of
cycle time. VJEPA2 uses 16-frame clips, so each clip spans 8 seconds of
real time.

| Task Speed | Cycle Time | What 8s Window Shows |
|-----------|-----------|---------------------|
| Fast | 2.2-4.4s | 2-3 full cycles |
| Medium | 6.1-13.4s | ~1 full cycle |
| Slow | 17.2-22.0s | ~half a cycle |

All sufficient for binary golden/not-golden classification.

**Why fixed, not per-task:** At deployment, the system won't know which
task is being performed (that's what it's trying to learn). A fixed FPS
means the model sees a consistent temporal resolution regardless of task.
FPS becomes a deployment config, not a model hyperparameter.

**Why 2fps specifically:** 4fps gives a 4-second window (misses context on
slow tasks). 1fps gives a 16-second window (too coarse for fast 2.2s
cycles). 2fps is the sweet spot.

Each ~180s video yields ~22 non-overlapping clips at 2fps. 77 videos
produce ~1,700 clips total.


## Labeling

### Label Categories

**"Golden Standard"** — A segment where:
1. A complete temporal cycle is visible (pick-process-place or equivalent)
2. The camera (worker's gaze) is steady and locked on the manipulation zone
3. Zero occlusion — hands and the contact point are fully visible
4. Hands remain within the central 60% of the frame

**"not good for collecting"** — Everything else:
- Headgear adjustments, looking at neighbors, restocking materials
- Water breaks, chatting, idle hands, walking
- Incomplete cycles, fumbled attempts
- Camera blur from rapid head movement
- Structural occlusion (support beams, equipment blocking view)
- Hands leaving the frame, off-screen action
- Sorting bulk materials, organizing workspace

This is stricter than simple "working vs not-working." A worker can be
actively working but if the camera is turned away, the action is occluded,
or the cycle is incomplete, it's labeled "not good for collecting." The
model learns what a clean, countable, fully-visible cycle looks like.

### Calibration Labels (10 videos, hand-verified)

One video per factory was reviewed manually using Gemini chat. The process:

1. Upload the cropped (256×256) video to Gemini
2. Describe the task and what constitutes a full cycle
3. Ask Gemini for Golden Standard segments
4. Gemini's first attempt was too loose — it included partial cycles and
   segments with gaze shifts
5. Correct Gemini by identifying the true golden segment (e.g., "02:22 to
   02:29 is actually the perfect segment")
6. Gemini articulates the hard-fail criteria it learned from the correction
7. On subsequent factories, Gemini applies the learned criteria without
   further correction

Key finding: Gemini learned the meta-criteria (gaze stability, zero
occlusion, full cycle, optimal framing) after one correction on factory001.
By factory003, it was producing accurate labels from just a golden clip
reference and task description, with no corrections needed.

### Label Files

Located in `labels/`, one JSON per calibration video:

```
labels/
├── factory_001_worker_001_0076.json
├── factory_002_worker_001_0075.json
├── factory_003_worker_001_0053.json
├── factory_004_worker_001_0090.json
├── factory_005_worker_001_0086.json
├── factory_006_worker_001_0078.json
├── factory_007_worker_001_0085.json
├── factory_008_worker_001_0068.json
├── factory_009_worker_001_0093.json
└── factory_011_worker_001_0079.json
```

### Label Format

```json
[
  {
    "start_time": "00:00",
    "end_time": "00:04",
    "label": "not good for collecting"
  },
  {
    "segment_id": 1,
    "start_time": "00:05",
    "end_time": "00:08",
    "label": "Golden Standard",
    "description": "Single perfect repetition; centered framing."
  }
]
```

- Timestamps are `MM:SS` strings, inclusive
- Every second of the video is covered — no gaps, no overlaps
- Golden segments have a `segment_id` (integer) and `description`
- Non-golden segments have no `segment_id`
- Some segments include a `confidence_score` (0-1) and `reason` for discard

### Golden Standard Coverage per Factory

| Factory | Task | Golden Seconds | Total Video | % Golden |
|---------|------|---------------|-------------|----------|
| 001 | Electronics assembly | ~30s | 179s | 17% |
| 002 | Panel lamination | ~88s | 129s | 68% |
| 003 | PCB frame assembly | ~85s | 179s | 47% |
| 004 | Industrial sewing | ~100s | 179s | 56% |
| 005 | Box packaging | ~48s | 179s | 27% |
| 006 | Lever press | ~20s | 179s | 11% |
| 007 | Motor housing | ~45s | 179s | 25% |
| 008 | Battery insertion | ~82s | 179s | 46% |
| 009 | Glue sealing | ~143s | 179s | 80% |
| 011 | Cylinder scraping | ~139s | 179s | 78% |

Factories 001 and 006 have thin golden coverage from the calibration
video. This is expected to improve when additional videos are labeled.

### Label Verification

`verify_labels.py` overlays the label JSON on the corresponding video,
drawing a green bar during Golden Standard segments and a red bar during
"not good for collecting" segments. This allows scrubbing through the
full video to check if timestamp boundaries are accurate.

### Gemini Conversation for Calibration

The full Gemini chat transcript where labels were calibrated is preserved.
The conversation shows:
- The initial prompt (Golden Standard protocol with four criteria)
- Gemini's first (inaccurate) attempt on factory001
- The user's correction identifying the true golden segment
- Gemini's articulated hard-fail rules after correction
- Subsequent factories labeled accurately without correction

This conversation is the basis for the API scaling pipeline — it will be
used as explicit cached context so the API reproduces the same labeling
quality across all remaining videos.

### Scaling Plan (Context Cache Architecture)

The remaining ~67 unlabeled videos will be labeled via Gemini API using
explicit context caching. The cache contains the calibration conversation
(factories 001-003, including the correction loop) as multi-turn message
history. Each new video is sent as a request referencing the cache.

Staged rollout:
- Gate 1: Label 2-3 new videos from one factory, manually verify
- Gate 2: Label all remaining train/val videos, spot-check
- Gate 3: Label test split (worker_002, hardest — different person)


## Task Reference

| Factory | Task | Description | Avg Cycle |
|---------|------|-------------|-----------|
| 001 | Electronics assembly | Ring installation into casings on conveyor | 3.1s |
| 002 | Panel lamination | Dark panel adhesion with blue hand roller | 22.0s |
| 003 | PCB frame assembly | Green PCB insertion into black plastic frames | 12.0s |
| 004 | Industrial sewing | Edge stitching blue striped shirts | 13.4s |
| 005 | Box packaging | Manual folding, cable insertion, box sealing | 11.5s |
| 006 | Lever press | Stamping pink cylindrical components | 4.4s |
| 007 | Motor housing | Wired motor routing and seating into housings | 11.3s |
| 008 | Battery insertion | Blue cell + wire tucking into white casings | 3.0s |
| 009 | Glue sealing | Adhesive application and sleeve pinch-sealing | 6.1s |
| 011 | Cylinder scraping | Batch deburring with green tool (5 per cycle) | 17.2s |


## Scripts

| File | What It Does |
|------|-------------|
| `config.py` | All paths, crop settings, task descriptions, FPS, model config |
| `extract_dataset.py` | Download 77 videos from HuggingFace into train/val/test splits |
| `prep_review.py` | Crop and collect 10 calibration videos into `gemini_review/` for manual labeling |
| `crop_test.py` | Generate cropped videos at different trim levels (0/20/36px) for comparison |
| `verify_labels.py` | Overlay label JSON on video with colored bar for timestamp verification |
| `label_one.py` | Label one video via Gemini API (center-crop + upload + parse response) |
| `inspect_tars.py` | List tar archive contents without extracting |
| `discover_workers.py` | Map all workers and tar files via HuggingFace API |


## Dependencies

```
google-genai
huggingface_hub
```

ffmpeg and ffprobe required for all video processing (cropping, overlay,
duration detection). Python 3.10+.