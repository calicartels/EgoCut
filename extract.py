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
    print(f"Loading {VJEPA2_MODEL} via torch.hub...")
    result = torch.hub.load(
        "facebookresearch/vjepa2", VJEPA2_MODEL,
        trust_repo=True,
    )
    model = result[0] if isinstance(result, tuple) else result
    model = model.cuda().eval()
    print(f"  embed_dim={model.embed_dim}, params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    return model


def read_clip_frames(video_path, start_s, n_frames, target_fps):
    vr = VideoReader(video_path, ctx=cpu(0))
    native_fps = vr.get_avg_fps()

    start_frame = int(start_s * native_fps)
    step = max(1, int(native_fps / target_fps))
    indices = [start_frame + i * step for i in range(n_frames)]

    max_idx = len(vr) - 1
    indices = [min(i, max_idx) for i in indices]

    frames = vr.get_batch(indices).asnumpy()
    return frames


def preprocess_frames(frames):
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

    T = t.shape[0]
    t = t.view(T, 3, frames.shape[1], frames.shape[2])
    t = F.interpolate(t, size=(VJEPA2_IMG_SIZE, VJEPA2_IMG_SIZE),
                      mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    t = (t - mean) / std

    t = t.permute(1, 0, 2, 3).unsqueeze(0)
    return t


def get_cropped_video_path(video_id, factory_key):
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
    video_tensor = video_tensor.cuda()

    with torch.amp.autocast("cuda"):
        features = model(video_tensor)

    T = VJEPA2_T_TOKENS
    HW = VJEPA2_H_TOKENS * VJEPA2_W_TOKENS
    D = VJEPA2_EMBED_DIM

    features = features.float()
    features = features.view(1, T, HW, D)
    features = features.mean(dim=2)

    return features.squeeze(0).cpu()


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

        frames = read_clip_frames(
            video_path, clip["clip_start_s"],
            FRAMES_PER_CLIP, SAMPLE_FPS,
        )
        video_tensor = preprocess_frames(frames)

        feat = extract_features(model, video_tensor)
        all_features.append(feat)
        all_labels.append(clip["label"])

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  {split}: {i+1}/{len(clips)}")

    features = torch.stack(all_features)
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