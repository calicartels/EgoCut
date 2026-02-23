"""
Train an attentive probe on frozen VJEPA2 features for golden/not-good classification.

Architecture (from the V-JEPA 2 paper):
  - 4 transformer blocks on temporal features
  - Last block uses cross-attention with a learnable query token
  - Query output → LayerNorm → Linear → 2-class prediction

Usage:
  python train.py

Reads: embeddings/{train,val,test}.pt, clips/{train,val,test}_manifest.json
Writes:
  - model/probe.pt         (trained probe weights)
  - predictions/val/       (JSON per video, same format as Gemini labels)
  - predictions/test/      (JSON per video, same format as Gemini labels)
  - predictions/metrics.json (val + test metrics + per-factory breakdown)
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

from config import VJEPA2_EMBED_DIM, VJEPA2_T_TOKENS

EMB_DIR = "embeddings"
PRED_DIR = "predictions"
MODEL_DIR = "model"

# ── Attentive Probe Architecture ─────────────────────────────────────
# Matches the V-JEPA 2 paper: "4-layer attentive probe composed of four
# transformer blocks, the last of which replaces standard self-attention
# with a cross-attention layer using a learnable query token."
#
# Choice: project 1408→384 before the transformer blocks.
# Reason: we have ~500 training clips; a full 1408-dim, 4-layer probe
# has ~80M params and will overfit badly. Projecting to 384 gives ~5M
# params, better suited to our data size.
# Alternative: keep full 1408-dim (as in the paper's SSv2 eval with 170K
# training videos). Switch to this if we get more data.

PROBE_DIM = 384
PROBE_HEADS = 6
PROBE_DEPTH = 4
PROBE_DROPOUT = 0.2
NUM_CLASSES = 2

# ── Training hyperparameters ─────────────────────────────────────────
# Choice: AdamW with cosine schedule, 100 epochs, early stopping on val F1.
# Alternative: SGD with momentum (more stable but slower convergence).
LR = 1e-3
WEIGHT_DECAY = 0.05
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 15  # early stopping patience


class CrossAttentionBlock(nn.Module):
    """Final probe block: cross-attention from query token to features."""

    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, kv):
        # Cross-attention: query attends to kv
        q = self.norm_q(query)
        kv_normed = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q, kv_normed, kv_normed)
        query = query + attn_out
        query = query + self.ffn(self.norm_ff(query))
        return query


class AttentiveProbe(nn.Module):
    """4-layer attentive probe following the V-JEPA 2 paper.

    Blocks 1-3: standard self-attention transformer encoder layers.
    Block 4: cross-attention from a learnable query token to features.
    Query output → LayerNorm → Linear → class logits.
    """

    def __init__(
        self,
        input_dim=VJEPA2_EMBED_DIM,
        hidden_dim=PROBE_DIM,
        n_heads=PROBE_HEADS,
        depth=PROBE_DEPTH,
        n_classes=NUM_CLASSES,
        dropout=PROBE_DROPOUT,
    ):
        super().__init__()

        # Project from encoder dim to probe dim
        self.proj = nn.Linear(input_dim, hidden_dim)

        # First (depth-1) blocks: self-attention
        self_attn_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.self_attn_blocks = nn.TransformerEncoder(
            self_attn_layer, num_layers=depth - 1,
        )

        # Last block: cross-attention with learnable query
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cross_attn_block = CrossAttentionBlock(hidden_dim, n_heads, dropout)

        # Classification head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, T, input_dim) — temporal features from VJEPA2 encoder
        returns: (B, n_classes) — logits
        """
        x = self.proj(x)  # (B, T, hidden_dim)

        # Self-attention blocks
        x = self.self_attn_blocks(x)  # (B, T, hidden_dim)

        # Cross-attention: query token attends to temporal features
        B = x.shape[0]
        query = self.query_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        query = self.cross_attn_block(query, x)      # (B, 1, hidden_dim)

        # Classify from query output
        cls_out = query.squeeze(1)        # (B, hidden_dim)
        cls_out = self.norm(cls_out)
        return self.head(cls_out)         # (B, n_classes)


# ── Data loading ─────────────────────────────────────────────────────

def load_split(split):
    path = os.path.join(EMB_DIR, f"{split}.pt")
    data = torch.load(path, weights_only=True)
    return data["features"], data["labels"]


def load_manifest(split):
    with open(os.path.join("clips", f"{split}_manifest.json")) as f:
        return json.load(f)


# ── Prediction → JSON conversion ─────────────────────────────────────

def clips_to_label_json(clips, predictions):
    """Convert clip-level predictions back to per-second Gemini-format JSON."""
    by_video = defaultdict(list)
    for clip, pred in zip(clips, predictions):
        by_video[clip["video_id"]].append({**clip, "pred": int(pred)})

    results = {}
    for video_id, video_clips in by_video.items():
        video_clips.sort(key=lambda c: c["clip_start_s"])

        max_t = max(c["clip_end_s"] for c in video_clips)
        per_second = np.zeros(max_t, dtype=np.int32)
        for c in video_clips:
            if c["pred"] == 1:
                per_second[c["clip_start_s"]:c["clip_end_s"]] = 1

        segments = []
        segment_id = 0
        i = 0
        while i < len(per_second):
            label = per_second[i]
            start = i
            while i < len(per_second) and per_second[i] == label:
                i += 1
            end = i - 1

            start_str = f"{start // 60:02d}:{start % 60:02d}"
            end_str = f"{end // 60:02d}:{end % 60:02d}"

            if label == 1:
                segment_id += 1
                segments.append({
                    "segment_id": segment_id,
                    "start_time": start_str,
                    "end_time": end_str,
                    "label": "Golden Standard",
                    "description": "Predicted by VJEPA2 attentive probe.",
                })
            else:
                segments.append({
                    "start_time": start_str,
                    "end_time": end_str,
                    "label": "not good for collecting",
                })

        results[video_id] = {
            "factory": video_clips[0]["factory"],
            "segments": segments,
        }

    return results


def compute_metrics(y_true, y_pred, name):
    return {
        "split": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_clips": int(len(y_true)),
        "n_golden_true": int(y_true.sum()),
        "n_golden_pred": int(y_pred.sum()),
    }


# ── Training loop ────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    train_clips = load_manifest("train")
    val_clips = load_manifest("val")
    test_clips = load_manifest("test")

    print(f"Train: {len(y_train)} clips ({y_train.sum()} golden, {len(y_train) - y_train.sum()} not-good)")
    print(f"Val:   {len(y_val)} clips ({y_val.sum()} golden, {len(y_val) - y_val.sum()} not-good)")
    print(f"Test:  {len(y_test)} clips ({y_test.sum()} golden, {len(y_test) - y_test.sum()} not-good)")

    # DataLoaders
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # Model
    probe = AttentiveProbe().to(device)
    n_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"\nAttentive probe: {n_params / 1e6:.2f}M trainable params")
    print(f"  dim={PROBE_DIM}, heads={PROBE_HEADS}, depth={PROBE_DEPTH}, dropout={PROBE_DROPOUT}")

    # Choice: class-weighted loss to handle imbalanced golden/not-good ratio.
    # Alternative: no weighting (biases toward majority class).
    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    print(f"  class weights: not-good={weight[0]:.2f}, golden={weight[1]:.2f}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop with early stopping
    best_val_f1 = 0
    best_epoch = 0
    best_state = None

    X_val_d = X_val.to(device)
    X_test_d = X_test.to(device)

    for epoch in range(EPOCHS):
        # Train
        probe.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = probe(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()

            # Choice: gradient clipping at 1.0 for training stability.
            # Alternative: no clipping (risk of exploding gradients with small batches).
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_correct += (logits.argmax(1) == batch_y).sum().item()
            train_total += batch_x.size(0)

        scheduler.step()

        # Validate
        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val_d)
            val_preds = val_logits.argmax(1).cpu().numpy()
            val_f1 = f1_score(y_val.numpy(), val_preds, zero_division=0)
            val_acc = accuracy_score(y_val.numpy(), val_preds)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  epoch {epoch+1:3d}: loss={train_loss/train_total:.4f}, "
                  f"train_acc={train_correct/train_total:.3f}, "
                  f"val_acc={val_acc:.3f}, val_f1={val_f1:.3f}, lr={lr:.2e}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

        if epoch + 1 - best_epoch >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (best was {best_epoch})")
            break

    print(f"\nBest val F1={best_val_f1:.3f} at epoch {best_epoch}")

    # Load best model
    probe.load_state_dict(best_state)
    probe = probe.to(device).eval()

    # ── Save model ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "config": {
            "input_dim": VJEPA2_EMBED_DIM,
            "hidden_dim": PROBE_DIM,
            "n_heads": PROBE_HEADS,
            "depth": PROBE_DEPTH,
            "n_classes": NUM_CLASSES,
            "dropout": PROBE_DROPOUT,
        },
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
    }, os.path.join(MODEL_DIR, "probe.pt"))
    print(f"Model saved to {MODEL_DIR}/probe.pt")

    # ── Evaluate ──
    with torch.no_grad():
        val_logits = probe(X_val_d)
        test_logits = probe(X_test_d)

    y_val_pred = val_logits.argmax(1).cpu().numpy()
    y_test_pred = test_logits.argmax(1).cpu().numpy()

    y_val_np = y_val.numpy()
    y_test_np = y_test.numpy()

    print(f"\n{'='*50}")
    print(f"=== Val ===")
    print(classification_report(y_val_np, y_val_pred, target_names=["not-good", "golden"]))

    print(f"=== Test (worker_002 — cross-worker generalization) ===")
    print(classification_report(y_test_np, y_test_pred, target_names=["not-good", "golden"]))

    # ── Per-factory test breakdown ──
    print("Per-factory test breakdown:")
    per_factory_metrics = []
    factories = sorted(set(c["factory"] for c in test_clips))
    for factory in factories:
        mask = [i for i, c in enumerate(test_clips) if c["factory"] == factory]
        if not mask:
            continue
        y_true_f = y_test_np[mask]
        y_pred_f = y_test_pred[mask]
        acc = accuracy_score(y_true_f, y_pred_f)
        f1 = f1_score(y_true_f, y_pred_f, zero_division=0)
        print(f"  {factory}: acc={acc:.3f}, f1={f1:.3f} (n={len(mask)}, golden={y_true_f.sum()})")
        per_factory_metrics.append({
            "factory": factory, "accuracy": float(acc), "f1": float(f1),
            "n_clips": len(mask), "n_golden": int(y_true_f.sum()),
        })

    # ── Save predictions as JSON (Gemini label format) ──
    for split, clips, preds in [("val", val_clips, y_val_pred), ("test", test_clips, y_test_pred)]:
        label_jsons = clips_to_label_json(clips, preds)
        for video_id, data in label_jsons.items():
            out_dir = os.path.join(PRED_DIR, split, data["factory"])
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{video_id}.json")
            with open(out_path, "w") as f:
                json.dump(data["segments"], f, indent=2)

    print(f"\nPredictions saved to {PRED_DIR}/val/ and {PRED_DIR}/test/")

    # ── Save metrics ──
    metrics = {
        "model": VJEPA2_MODEL,
        "probe": {
            "dim": PROBE_DIM, "heads": PROBE_HEADS,
            "depth": PROBE_DEPTH, "dropout": PROBE_DROPOUT,
            "params": n_params,
        },
        "training": {
            "best_epoch": best_epoch, "best_val_f1": best_val_f1,
            "lr": LR, "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
        },
        "val": compute_metrics(y_val_np, y_val_pred, "val"),
        "test": compute_metrics(y_test_np, y_test_pred, "test"),
        "per_factory_test": per_factory_metrics,
    }
    os.makedirs(PRED_DIR, exist_ok=True)
    with open(os.path.join(PRED_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {PRED_DIR}/metrics.json")
    print(f"\nTo overlay predictions on video:")
    print(f"  python verify_labels.py predictions/test/factory001/factory_001_worker_002_0000.json \\")
    print(f"    egocentric/test/factory001/factory_001_worker_002_0000.mp4")


train()