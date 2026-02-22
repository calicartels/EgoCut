import os

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR = "egocentric"
LABELS_DIR = "labels"
CROPPED_DIR = "cropped"

# ── Gemini ───────────────────────────────────────────────────────────
# Choice: gemini-2.5-pro for best video understanding accuracy.
# Alternative: gemini-2.5-flash is 10x cheaper but less reliable on
# fine-grained temporal boundary detection in 180s videos.
GEMINI_MODEL = "gemini-2.5-pro"

# ── Video preprocessing ──────────────────────────────────────────────
# Choice: center-crop 456x256 → 256x256 with 20px top trim.
# Removes 100px from each side (neighboring workers, fisheye distortion)
# and 20px from top (adjacent station bleed-through).
# Final: crop=256:236:100:20 then scale back to 256x256.
# Alternative: no top trim (keeps neighboring worker's torso in frame).
# Tested: trim=0, trim=20, trim=36. User verified trim=20 is optimal.
CROP_W = 256
CROP_H = 236  # 256 - 20px top trim
CROP_X = 100  # (456 - 256) / 2
CROP_Y = 20   # top trim
FFMPEG_VF = f"crop={CROP_W}:{CROP_H}:{CROP_X}:{CROP_Y},scale=256:256"

# ── Sampling ─────────────────────────────────────────────────────────
# Choice: fixed 2fps for all tasks.
# At 2fps, 16-frame VJEPA2 clip = 8 seconds.
# Alternative: per-task FPS (1-4fps) but requires knowing task at deploy time.
SAMPLE_FPS = 2
FRAMES_PER_CLIP = 16
CLIP_DURATION_S = FRAMES_PER_CLIP / SAMPLE_FPS  # 8.0 seconds

# ── Factories (10 active, factory010 dropped due to poor video quality) ──
TASKS = {
    "factory001": {
        "task_id": "electronics_assembly_01",
        "description": (
            "Circular ring installation into electronic casings. Worker picks up "
            "a rectangular casing, retrieves a circular ring from a pile, presses "
            "the ring into a slot using thumbs, releases onto conveyor belt."
        ),
        "avg_cycle_s": 3.1,
        "calibration_video": "factory_001_worker_001_0076",
    },
    "factory002": {
        "task_id": "panel_roller_assembly_01",
        "description": (
            "Panel lamination with hand roller. Worker positions a base component, "
            "aligns a dark panel over it, uses a blue roller to press for bubble-free "
            "adhesion, moves finished piece to completed stack."
        ),
        "avg_cycle_s": 22.0,
        "calibration_video": "factory_002_worker_001_0075",
    },
    "factory003": {
        "task_id": "pcb_frame_assembly_01",
        "description": (
            "PCB to plastic frame assembly. Worker retrieves a black frame, exposes "
            "adhesive strips, aligns a green circuit board into the cavity, presses "
            "to seat it, places completed assembly on conveyor."
        ),
        "avg_cycle_s": 12.0,
        "calibration_video": "factory_003_worker_001_0053",
    },
    "factory004": {
        "task_id": "industrial_sewing_01",
        "description": (
            "Industrial sewing of shirt edges. Worker picks up fabric, aligns folded "
            "edge under presser foot, guides fabric through machine for straight seam, "
            "cuts thread, transfers to finished pile."
        ),
        "avg_cycle_s": 13.4,
        "calibration_video": "factory_004_worker_001_0090",
    },
    "factory005": {
        "task_id": "box_packaging_01",
        "description": (
            "Manual insertion and box sealing. Worker folds instruction manual, inserts "
            "into product box with cable, folds and tucks top flaps to seal, moves "
            "completed package to output stack."
        ),
        "avg_cycle_s": 11.5,
        "calibration_video": "factory_005_worker_001_0086",
    },
    "factory006": {
        "task_id": "manual_lever_press_01",
        "description": (
            "Manual lever press stamping. Left hand feeds pink cylindrical components, "
            "right hand actuates heavy lever. Place component in die, pull lever to "
            "stamp, remove processed item, slot into output tray."
        ),
        "avg_cycle_s": 4.4,
        "calibration_video": "factory_006_worker_001_0078",
    },
    "factory007": {
        "task_id": "motor_housing_assembly_01",
        "description": (
            "Motor and housing assembly. Worker picks up light blue housing, retrieves "
            "wired motor, routes red/black wires through internal slot, presses motor "
            "into cavity until seated, places in output area."
        ),
        "avg_cycle_s": 11.3,
        "calibration_video": "factory_007_worker_001_0085",
    },
    "factory008": {
        "task_id": "battery_assembly_01",
        "description": (
            "Battery insertion into casings. Worker picks up white casing from conveyor, "
            "retrieves blue battery cell with wires, presses battery in while tucking "
            "wires, places completed unit back on belt. Extremely fast when uninterrupted."
        ),
        "avg_cycle_s": 3.0,
        "calibration_video": "factory_008_worker_001_0068",
    },
    "factory009": {
        "task_id": "glue_sealing_assembly_01",
        "description": (
            "Adhesive application and sleeve sealing. Worker retrieves blue component "
            "from pink-illuminated chamber, dips wooden stick into adhesive, applies "
            "glue to tip, pinches seam to seal, drops into bin."
        ),
        "avg_cycle_s": 6.1,
        "calibration_video": "factory_009_worker_001_0093",
    },
    "factory011": {
        "task_id": "cylinder_scraping_01",
        "description": (
            "Batch deburring of metal cylinders. Worker grabs handful from white bin, "
            "aligns side-by-side in left hand, uses green tool to scrape circular base "
            "of each cylinder sequentially, releases batch into clear container."
        ),
        "avg_cycle_s": 17.2,
        "calibration_video": "factory_011_worker_001_0079",
    },
}