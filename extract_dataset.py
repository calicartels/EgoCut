"""
extract_dataset.py — Egocentric-100K subset for Tier 1 cycle detection.

Downloads tars from HuggingFace, extracts only selected videos into
train/val/test folders, deletes tar cache when done.

Split:
  Train: worker_001, first 3 sequential videos per factory  (33 videos)
  Val:   worker_001, 4th + last video per factory            (22 videos)
  Test:  worker_002, first 2 videos per factory              (22 videos)
                                                        Total: 77 videos

All downstream processing uses fixed 2fps sampling (16 frames = 8s window).
This script only extracts raw mp4s — sampling happens in the clip extraction
step.

Choice: worker_002 for test instead of a random high-numbered worker.
  - Easiest generalization test. Same task, same station, different person.
  - If it fails on worker_002, no point testing worker_015.
  - If it passes, next step is testing on worker_005+ for harder generalization.

Choice: sequential videos (0,1,2,3) rather than random sampling.
  - Same shift segment = consistent lighting, camera angle, behavior.
  - Random risks crossing shift boundaries with condition changes.

Choice: 3 train + 2 val + 2 test = 7 per factory.
  - 3 train videos × ~22 clips/video = ~66 training clips per factory.
  - Linear probe on frozen VJEPA2 needs 50-200 examples per class. 66 clips
    with ~60/40 working/not-working split gives ~40 working and ~26
    not-working per factory. Tight but workable for a linear probe.
  - Pooled across 11 factories: ~1,700 clips total. Comfortable.
"""

import os
import tarfile
from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir

REPO = "builddotai/Egocentric-100K"
OUT_DIR = "egocentric"

# ── Worker 001 tars (known from inspection) ──────────────────────────
WORKER1_TARS = {
    "factory001": "factory001/worker001/part000.tar",
    "factory002": "factory002/worker001/part152.tar",
    "factory003": "factory003/worker001/part187.tar",
    "factory004": "factory004/worker001/part200.tar",
    "factory005": "factory005/worker001/part278.tar",
    "factory006": "factory006/worker001/part345.tar",
    "factory007": "factory007/worker001/part362.tar",
    "factory008": "factory008/worker001/part449.tar",
    "factory009": "factory009/worker001/part566.tar",
    "factory010": "factory010/worker001/part682.tar",
    "factory011": "factory011/worker001/part728.tar",
}

# ── Worker 002 tars ──────────────────────────────────────────────────
# factory001/worker002 tar path unknown (inspection output was truncated).
# Discovered at runtime via HuggingFace API. All others from inspection.
WORKER2_TARS = {
    "factory001": None,  # discovered at runtime
    "factory002": "factory002/worker002/part153.tar",
    "factory003": "factory003/worker002/part188.tar",
    "factory004": "factory004/worker002/part203.tar",
    "factory005": "factory005/worker002/part281.tar",
    "factory006": "factory006/worker002/part348.tar",
    "factory007": "factory007/worker002/part365.tar",
    "factory008": "factory008/worker002/part452.tar",
    "factory009": "factory009/worker002/part568.tar",
    "factory010": "factory010/worker002/part685.tar",
    "factory011": "factory011/worker002/part731.tar",
}

# ── Worker 001 video index ranges (first, last) from inspection ─────
WORKER1_RANGE = {
    "factory001": (0, 76),
    "factory002": (1, 75),
    "factory003": (1, 53),
    "factory004": (0, 90),
    "factory005": (0, 86),
    "factory006": (0, 78),
    "factory007": (0, 85),
    "factory008": (0, 68),
    "factory009": (0, 93),
    "factory010": (0, 77),
    "factory011": (0, 79),
}


def discover_factory001_worker002():
    """Find factory001/worker002 tar path via HF API."""
    print("Discovering factory001/worker002 tar path...")
    api = HfApi()
    files = list(api.list_repo_tree(
        repo_id=REPO,
        repo_type="dataset",
        path_in_repo="factory001/worker002",
    ))
    tars = sorted([
        f.path for f in files
        if hasattr(f, "path") and f.path.endswith(".tar")
    ])
    if not tars:
        raise RuntimeError("No tars found for factory001/worker002!")
    print(f"  Found: {tars[0]}")
    return tars[0]


def vid_filename(factory_key, worker_num, vid_num):
    """e.g. factory_001_worker_002_0005.mp4"""
    fnum = factory_key.replace("factory", "")
    return f"factory_{fnum}_worker_{worker_num:03d}_{vid_num:04d}.mp4"


def extract_named_videos(tar_path, out_dir, wanted_basenames):
    """Extract specific mp4s by name from a tar. Returns count extracted."""
    os.makedirs(out_dir, exist_ok=True)
    wanted = set(wanted_basenames)
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            basename = os.path.basename(m.name)
            if basename in wanted:
                m.name = basename
                tf.extract(m, path=out_dir)
                wanted.discard(basename)
    if wanted:
        print(f"  WARNING missing: {wanted}")
    return len(wanted_basenames) - len(wanted)


def extract_first_n_videos(tar_path, out_dir, n):
    """Extract first N mp4s (alphabetically) from a tar. Returns filenames."""
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        mp4s = sorted(
            [m for m in tf.getmembers()
             if os.path.basename(m.name).endswith(".mp4")],
            key=lambda m: os.path.basename(m.name),
        )
        extracted = []
        for m in mp4s[:n]:
            m.name = os.path.basename(m.name)
            tf.extract(m, path=out_dir)
            extracted.append(m.name)
    return extracted


def download_tar(tar_repo_path):
    """Download a tar from HF hub, return local path."""
    return hf_hub_download(
        repo_id=REPO,
        filename=tar_repo_path,
        repo_type="dataset",
    )


def clear_hf_cache():
    """Delete all cached files for this repo."""
    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_id == REPO:
            for rev in repo.revisions:
                strategy = cache.delete_revisions(rev.commit_hash)
                strategy.execute()
                print(f"Deleted {REPO} from cache ({strategy.freed_size_str})")
                return
    print("No cache found for", REPO)


# ── Main ─────────────────────────────────────────────────────────────

if WORKER2_TARS["factory001"] is None:
    WORKER2_TARS["factory001"] = discover_factory001_worker002()

total = {"train": 0, "val": 0, "test": 0}

for fk in sorted(WORKER1_TARS):
    print(f"\n{'=' * 60}")
    print(fk)
    print("=" * 60)

    first, last = WORKER1_RANGE[fk]

    # ── Train: worker_001, videos [first, first+1, first+2] ─────
    train_indices = [first, first + 1, first + 2]
    train_names = [vid_filename(fk, 1, i) for i in train_indices]
    train_dir = os.path.join(OUT_DIR, "train", fk)

    print(f"  TRAIN: indices {train_indices}")
    tar_path = download_tar(WORKER1_TARS[fk])
    total["train"] += extract_named_videos(tar_path, train_dir, train_names)

    # ── Val: worker_001, videos [first+3, last] ─────────────────
    val_indices = [first + 3, last]
    val_names = [vid_filename(fk, 1, i) for i in val_indices]
    val_dir = os.path.join(OUT_DIR, "val", fk)

    print(f"  VAL:   indices {val_indices}")
    total["val"] += extract_named_videos(tar_path, val_dir, val_names)

    # ── Test: worker_002, first 2 mp4s from their tar ───────────
    print(f"  TEST:  worker_002")
    tar2_path = download_tar(WORKER2_TARS[fk])
    test_dir = os.path.join(OUT_DIR, "test", fk)
    extracted = extract_first_n_videos(tar2_path, test_dir, n=2)
    for name in extracted:
        print(f"    {name}")
    total["test"] += len(extracted)

# ── Summary ──────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("FINAL DATASET")
print("=" * 60)

for split in ["train", "val", "test"]:
    split_dir = os.path.join(OUT_DIR, split)
    split_total = 0
    print(f"\n{split.upper()}/")
    for fk in sorted(os.listdir(split_dir)):
        fd = os.path.join(split_dir, fk)
        if not os.path.isdir(fd):
            continue
        mp4s = sorted(f for f in os.listdir(fd) if f.endswith(".mp4"))
        split_total += len(mp4s)
        mb = sum(os.path.getsize(os.path.join(fd, f)) / 1e6 for f in mp4s)
        print(f"  {fk}/  ({len(mp4s)} videos, {mb:.0f} MB)")
        for f in mp4s:
            sz = os.path.getsize(os.path.join(fd, f)) / 1e6
            print(f"    {f}  ({sz:.1f} MB)")
    print(f"  → {split_total} videos")

grand = sum(total.values())
print(f"\nGrand total: {grand} videos")
print(f"  Train: {total['train']}, Val: {total['val']}, Test: {total['test']}")

# ── Cleanup ──────────────────────────────────────────────────────────

print("\nCleaning up HF cache...")
clear_hf_cache()
print("Done.")