"""
Extract frozen VJEPA2 (ViT-g 384px) features for all clips.

Loads 16 frames per clip from cropped videos, resizes to 384x384,
runs through frozen VJEPA2 encoder, spatially averages patch features
to get temporal token sequence, saves per-split.

Usage:
  python extract.py

Reads: clips/{train,val,test}_manifest.json, cropped/ or egocentric/
Writes: embeddings/{train,val,test}.pt
  Each .pt contains:
    'features': (N, T_tokens, embed_dim) float32  — temporal features
    'labels':   (N,) long                          — binary labels

Requires: GPU with ~12GB VRAM for ViT-g.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

from config import (
    SAMPLE_FPS, FRAMES_PER_CLIP, TASKS,
    VJEPA2_MODEL, VJEPA2_IMG_SIZE, VJEPA2_EMBED_DIM,
    VJEPA2_T_TOKENS, VJEPA2_H_TOKENS, VJEPA2_W_TOKENS,
)

CLIPS_DIR = "clips"
EMB_DIR = "embeddings"


def load_model():
    """Load VJEPA2 encoder via torch.hub.

    Choice: torch.hub over HuggingFace AutoModel.
    Reason: torch.hub gives the raw encoder with no constraints on
    frame count (HF models have fpc=64 baked in, we use 16 frames).
    Alternative: HuggingFace AutoModel with facebook/vjepa2-vitg-fpc64-384,
    but would need to handle frame count mismatch.
    """
    print(f"Loading {VJEPA2_MODEL} via torch.hub...")
    model = torch.hub.load(
        "facebookresearch/vjepa2", VJEPA2_MODEL,
        trust_repo=True,
    )
    model = model.cuda().eval()
    print(f"  embed_dim={model.embed_dim}, params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    return model


def load_preprocessor():
    """Load VJEPA2 video preprocessor via torch.hub.

    Returns a callable transform that normalizes and resizes frames.
    """
    processor = torch.hub.load(
        "facebookresearch/vjepa2", "vjepa2_preprocessor",
        trust_repo=True,
    )
    return processor


def read_clip_frames(video_path, start_s, n_frames, target_fps):
    """Read n_frames from video starting at start_s, sampled at target_fps."""
    vr = VideoReader(video_path, ctx=cpu(0))
    native_fps = vr.get_avg_fps()

    start_frame = int(start_s * native_fps)
    step = max(1, int(native_fps / target_fps))
    indices = [start_frame + i * step for i in range(n_frames)]

    max_idx = len(vr) - 1
    indices = [min(i, max_idx) for i in indices]

    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C) uint8
    return frames


def preprocess_frames(frames, processor):
    """Preprocess frames: resize to 384x384, normalize.

    The VJEPA2 preprocessor expects (T, H, W, C) uint8 and returns
    a tensor ready for the encoder.
    """
    # The torch.hub preprocessor is a torchvision transform pipeline.
    # It expects a tensor of shape (C, T, H, W) or similar.
    # We manually handle: uint8 (T,H,W,C) → float (C,T,H,W) → resize → normalize

    # (T, H, W, C) → (T, C, H, W)
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

    # Resize each frame to 384x384
    # Choice: bilinear interpolation for upscaling 256→384.
    # Alternative: bicubic (slightly sharper but slower).
    T = t.shape[0]
    t = t.view(T, 3, frames.shape[1], frames.shape[2])
    t = F.interpolate(t, size=(VJEPA2_IMG_SIZE, VJEPA2_IMG_SIZE),
                      mode="bilinear", align_corners=False)

    # Normalize with ImageNet stats (standard for VJEPA2)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    t = (t - mean) / std

    # (T, C, H, W) → (1, C, T, H, W) for the encoder
    t = t.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, T, 384, 384)
    return t


def get_cropped_video_path(video_id, factory_key):
    """Find the cropped video."""
    cropped = os.path.join("cropped", factory_key, f"{video_id}.mp4")
    if os.path.exists(cropped):
        return cropped
    for split in ["train", "val", "test"]:
        p = os.path.join("egocentric", split, factory_key, f"{video_id}.mp4")
        if os.path.exists(p):
            return p
    return None


@torch.no_grad()
def extract_features(model, video_tensor):
    """Run frozen encoder and return spatially-averaged temporal features.

    Input: (1, 3, T, 384, 384)
    Encoder output: (1, T_tok * H_tok * W_tok, embed_dim)
    After spatial avg: (T_tok, embed_dim)

    Choice: spatial average pooling to reduce 4608 tokens → 8 temporal tokens.
    Reason: storing full 4608 × 1408 per clip = ~26MB each, ~13GB for 500 clips.
    Spatial avg retains temporal patterns (which matter for golden vs not-good)
    while being practical to store (~45KB per clip).
    Alternative: store full patch features for maximum information retention,
    but requires ~13GB disk and much more probe training memory.
    """
    video_tensor = video_tensor.cuda().half()

    # Forward through frozen encoder
    features = model(video_tensor)  # (1, n_patches, embed_dim)

    # Reshape: (1, T*H*W, D) → (1, T, H*W, D)
    T = VJEPA2_T_TOKENS
    HW = VJEPA2_H_TOKENS * VJEPA2_W_TOKENS
    D = VJEPA2_EMBED_DIM

    features = features.float()
    features = features.view(1, T, HW, D)

    # Spatial average pool: (1, T, H*W, D) → (1, T, D)
    features = features.mean(dim=2)

    return features.squeeze(0).cpu()  # (T, D)


def extract_split(split, model):
    manifest_path = os.path.join(CLIPS_DIR, f"{split}_manifest.json")
    with open(manifest_path) as f:
        clips = json.load(f)

    if not clips:
        print(f"{split}: no clips")
        return

    all_features = []
    all_labels = []
    skipped = 0

    for i, clip in enumerate(clips):
        video_path = get_cropped_video_path(clip["video_id"], clip["factory"])
        if video_path is None:
            skipped += 1
            continue

        # Read and preprocess frames
        frames = read_clip_frames(
            video_path, clip["clip_start_s"],
            FRAMES_PER_CLIP, SAMPLE_FPS,
        )
        video_tensor = preprocess_frames(frames, None)

        # Extract features
        feat = extract_features(model, video_tensor)  # (T, D)
        all_features.append(feat)
        all_labels.append(clip["label"])

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  {split}: {i+1}/{len(clips)}")

    features = torch.stack(all_features)  # (N, T, D)
    labels = torch.tensor(all_labels, dtype=torch.long)

    os.makedirs(EMB_DIR, exist_ok=True)
    out_path = os.path.join(EMB_DIR, f"{split}.pt")
    torch.save({"features": features, "labels": labels}, out_path)

    n_golden = labels.sum().item()
    size_mb = features.element_size() * features.nelement() / 1e6
    print(f"{split}: {len(labels)} clips, {n_golden} golden, "
          f"shape={tuple(features.shape)}, size={size_mb:.1f}MB"
          + (f", skipped={skipped}" if skipped else ""))


model = load_model()
for split in ["train", "val", "test"]:
    extract_split(split, model)