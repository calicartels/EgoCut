# EgoCut — Factory Cycle Detection

Binary classification of factory worker activity from egocentric headcam
video: is the worker performing their repeating task, or not?

## What This Detects

Workers wear headcams while doing short, repeating tasks (2-22 second cycles).
The classifier watches the video stream and outputs a binary label per clip:
**working** (hands actively doing the repetitive task) or **not working**
(everything else — headgear adjustment, restocking, breaks, chatting, idle).

This gives utilization rate, break frequency, and time-on-task per shift.

## Source Dataset

[builddotai/Egocentric-100K](https://huggingface.co/datasets/builddotai/Egocentric-100K)
on HuggingFace. 100K+ hours of factory egocentric video across 11 factories.
Videos are ~180s, 30fps native, 456x256, fisheye lens, H.265 codec.
Stored as WebDataset tar archives, one per worker.

## Extracted Subset

77 videos, 2.2 GB. Extracted by `extract_dataset.py`.

| Split | Worker | Videos/Factory | Total | Purpose |
|-------|--------|---------------|-------|---------|
| Train | worker_001 | 3 | 33 | Learn binary classification |
| Val   | worker_001 | 2 | 22 | Tune threshold, early stopping |
| Test  | worker_002 | 2 | 22 | Cross-worker generalization |

Train on one worker, test on a different worker doing the same task.
If it can't generalize across two people, the approach is broken.

```
egocentric/
├── train/{factory001..011}/   3 sequential videos per factory
├── val/{factory001..011}/     4th video + last video (has existing labels)
└── test/{factory001..011}/    worker_002, first 2 videos per factory
```

### Selection Rationale

**Sequential videos** (0000-0003) from the same shift ensure consistent
lighting, camera angle, and worker behavior. Random sampling would risk
crossing shift boundaries.

**worker_002 for test** is the easiest generalization test — same task,
same station, different person. If it fails here, harder tests are pointless.

## Labeling

Gemini 2.5 Pro watches each ~180s video and outputs time ranges for
working vs not-working segments.

### Running the Labeler

```bash
pip install google-genai
export GEMINI_API_KEY=your_key_here
python label_one.py egocentric/val/factory001/factory_001_worker_001_0076.mp4
```

### Label Format

```json
{
  "video_id": "factory_001_worker_001_0076",
  "factory": "factory001",
  "task_id": "electronics_assembly_01",
  "duration_s": 179.7,
  "segments": {
    "working": [[35, 67], [72, 140]],
    "not_working": [[0, 34], [68, 71], [141, 179]]
  }
}
```

Segments are `[start_second, end_second]` inclusive. Every second of the
video must be in exactly one list — no gaps, no overlaps. Saved to
`labels/{factoryXXX}/{video_id}.json`.

### Labeling Rules

- "Not working" = any break from the core repetitive cycle
- If unsure about a 2-3s segment, call it not_working
- Sanity check: sum of all segment durations ≈ video duration

### Cost

66 new Gemini calls needed. Each video ≈ 47K input tokens (180s × 258
tokens/s) + ~3K output tokens. At $1.25/M input + $10/M output ≈ $6 total.

## Video Processing

Fixed **2fps** sampling for all tasks. VJEPA2 uses 16-frame clips, so each
clip spans **8 seconds** of real time.

At 8s window: fast tasks (2.2-4.4s) show 2-3 full cycles, medium tasks
(6-13s) show ~1 cycle, slow tasks (17-22s) show half a cycle. All sufficient
for binary working/not-working.

Each ~180s video produces ~22 non-overlapping clips. 77 videos → ~1,700
clips total.

## Task Reference

| Factory | Task | Avg Cycle | Steps |
|---------|------|-----------|-------|
| 001 | electronics_assembly_01 | 3.1s | 4 |
| 002 | panel_roller_assembly_01 | 22.0s | 4 |
| 003 | pcb_frame_assembly_01 | 12.0s | 4 |
| 004 | industrial_sewing_01 | 13.4s | 4 |
| 005 | box_packaging_01 | 11.5s | 4 |
| 006 | manual_lever_press_01 | 4.4s | 4 |
| 007 | motor_housing_assembly_01 | 11.3s | 5 |
| 008 | battery_assembly_01 | 3.0s | 4 |
| 009 | glue_sealing_assembly_01 | 6.1s | 4 |
| 010 | cable_packaging_01 | 2.2s | 3 |
| 011 | cylinder_scraping_01 | 17.2s | 3 |

## Files

| File | Purpose |
|------|---------|
| `config.py` | Paths, task descriptions, model settings, FPS |
| `extract_dataset.py` | Download and organize 77 videos from HuggingFace |
| `label_one.py` | Label one video with Gemini, save JSON |
| `inspect_tars.py` | List tar contents without extracting |
| `discover_workers.py` | Map all workers/tars via HuggingFace API |

## Dependencies

```
google-genai
huggingface_hub
```

ffprobe required for video duration detection (comes with ffmpeg).