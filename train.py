"""
Train classifiers on frozen VJEPA2 features for golden/not-good classification.

Two stages:
  Stage 1: Logistic regression on mean-pooled features (baseline sanity check).
  Stage 2: Attentive probe (only if stage 1 shows the features separate).

Usage:
  python train.py

Reads: embeddings/{train,val,test}.pt, clips/{train,val,test}_manifest.json
Writes:
  - model/probe.pt              (trained probe weights)
  - predictions/val/            (JSON per video, Gemini label format)
  - predictions/test/           (JSON per video, Gemini label format)
  - predictions/metrics.json    (val + test metrics + per-factory breakdown)
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

from config import VJEPA2_EMBED_DIM, VJEPA2_T_TOKENS, VJEPA2_MODEL

EMB_DIR = "embeddings"
PRED_DIR = "predictions"
MODEL_DIR = "model"

# ── Attentive Probe Architecture ─────────────────────────────────────
# Choice: project 1408→128, depth=2 (1 self-attn + 1 cross-attn).
# Reason: ~591 training clips (or ~1100 with overlap). A 4-layer 384-dim
# probe had 7.64M params and collapsed. This has ~0.5M params.
# Alternative: depth=4, dim=384 as in the paper (designed for 170K SSv2
# training videos). We can scale up if we get more data.
PROBE_DIM = 128
PROBE_HEADS = 4
PROBE_DEPTH = 2  # 1 self-attn block + 1 cross-attn block
PROBE_DROPOUT = 0.4
NUM_CLASSES = 2

# ── Training hyperparameters ─────────────────────────────────────────
LR = 3e-4
WEIGHT_DECAY = 0.1
EPOCHS = 200
BATCH_SIZE = 64
PATIENCE = 30


# ── Model definition ─────────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
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
        q = self.norm_q(query)
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q, kv_n, kv_n)
        query = query + attn_out
        query = query + self.ffn(self.norm_ff(query))
        return query


class AttentiveProbe(nn.Module):
    """Lightweight attentive probe for small datasets.

    Architecture: projection → (depth-1) self-attn blocks → 1 cross-attn
    block with learnable query → LayerNorm → Linear → logits.
    """

    def __init__(
        self, input_dim=VJEPA2_EMBED_DIM, hidden_dim=PROBE_DIM,
        n_heads=PROBE_HEADS, depth=PROBE_DEPTH,
        n_classes=NUM_CLASSES, dropout=PROBE_DROPOUT,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        if depth > 1:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.self_attn = nn.TransformerEncoder(layer, num_layers=depth - 1)
        else:
            self.self_attn = nn.Identity()

        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cross_attn_block = CrossAttentionBlock(hidden_dim, n_heads, dropout)

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.proj(x)
        x = self.self_attn(x)
        B = x.shape[0]
        query = self.query_token.expand(B, -1, -1)
        query = self.cross_attn_block(query, x)
        return self.head(self.norm(query.squeeze(1)))


# ── Utilities ────────────────────────────────────────────────────────

def load_split(split):
    data = torch.load(os.path.join(EMB_DIR, f"{split}.pt"), weights_only=True)
    return data["features"], data["labels"]


def load_manifest(split):
    with open(os.path.join("clips", f"{split}_manifest.json")) as f:
        return json.load(f)


def clips_to_label_json(clips, predictions):
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
                    "start_time": start_str, "end_time": end_str,
                    "label": "Golden Standard",
                    "description": "Predicted by VJEPA2 attentive probe.",
                })
            else:
                segments.append({
                    "start_time": start_str, "end_time": end_str,
                    "label": "not good for collecting",
                })
        results[video_id] = {"factory": video_clips[0]["factory"], "segments": segments}
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


def per_factory_eval(clips, y_true, y_pred):
    results = []
    factories = sorted(set(c["factory"] for c in clips))
    for factory in factories:
        mask = [i for i, c in enumerate(clips) if c["factory"] == factory]
        if not mask:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, zero_division=0)
        print(f"  {factory}: acc={acc:.3f}, f1={f1:.3f} (n={len(mask)}, golden={yt.sum()})")
        results.append({"factory": factory, "accuracy": float(acc), "f1": float(f1),
                        "n_clips": len(mask), "n_golden": int(yt.sum())})
    return results


# ── Stage 1: Logistic Regression ─────────────────────────────────────

def stage1_logreg():
    """Baseline: mean-pool temporal features → logistic regression.

    This checks whether the VJEPA2 features actually separate golden from not-good.
    If this fails, the features don't encode the distinction and we need a different approach.
    """
    print("=" * 60)
    print("STAGE 1: Logistic Regression Baseline")
    print("=" * 60)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    # Mean pool temporal dim: (N, 8, 1408) → (N, 1408)
    X_train_flat = X_train.mean(dim=1).numpy()
    X_val_flat = X_val.mean(dim=1).numpy()
    X_test_flat = X_test.mean(dim=1).numpy()
    y_train_np = y_train.numpy()
    y_val_np = y_val.numpy()
    y_test_np = y_test.numpy()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_flat)
    X_val_s = scaler.transform(X_val_flat)
    X_test_s = scaler.transform(X_test_flat)

    best_f1 = 0
    best_C = 1.0
    print("\nTuning C on val:")
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(X_train_s, y_train_np)
        yp = clf.predict(X_val_s)
        f1 = f1_score(y_val_np, yp, zero_division=0)
        acc = accuracy_score(y_val_np, yp)
        prec = precision_score(y_val_np, yp, zero_division=0)
        rec = recall_score(y_val_np, yp, zero_division=0)
        print(f"  C={C:>6}: acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_C = C

    clf = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs")
    clf.fit(X_train_s, y_train_np)

    y_val_pred = clf.predict(X_val_s)
    y_test_pred = clf.predict(X_test_s)

    print(f"\nBest C={best_C}")
    print(f"\nVal:")
    print(classification_report(y_val_np, y_val_pred, target_names=["not-good", "golden"]))
    print(f"Test:")
    print(classification_report(y_test_np, y_test_pred, target_names=["not-good", "golden"]))

    val_f1 = f1_score(y_val_np, y_val_pred)
    test_f1 = f1_score(y_test_np, y_test_pred)

    print(f"Stage 1 summary: val_f1={val_f1:.3f}, test_f1={test_f1:.3f}")
    features_separate = val_f1 > 0.3
    print(f"Features separate classes: {'YES' if features_separate else 'NO'}")

    return features_separate


# ── Stage 2: Attentive Probe ─────────────────────────────────────────

def stage2_probe():
    print("\n" + "=" * 60)
    print("STAGE 2: Attentive Probe")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    train_clips = load_manifest("train")
    val_clips = load_manifest("val")
    test_clips = load_manifest("test")

    print(f"Train: {len(y_train)} clips ({y_train.sum()} golden, {len(y_train) - y_train.sum()} not-good)")
    print(f"Val:   {len(y_val)} clips ({y_val.sum()} golden, {len(y_val) - y_val.sum()} not-good)")
    print(f"Test:  {len(y_test)} clips ({y_test.sum()} golden, {len(y_test) - y_test.sum()} not-good)")

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    probe = AttentiveProbe().to(device)
    n_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"\nProbe: {n_params / 1e3:.1f}K params (dim={PROBE_DIM}, heads={PROBE_HEADS}, "
          f"depth={PROBE_DEPTH}, dropout={PROBE_DROPOUT})")

    # Class-weighted loss
    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    print(f"  class weights: not-good={weight[0]:.2f}, golden={weight[1]:.2f}")

    optimizer = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1 = 0
    best_epoch = 0
    best_state = None

    X_val_d = X_val.to(device)
    X_test_d = X_test.to(device)

    for epoch in range(EPOCHS):
        probe.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits = probe(bx)
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * bx.size(0)
            train_correct += (logits.argmax(1) == by).sum().item()
            train_total += bx.size(0)

        scheduler.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val_d)
            val_preds = val_logits.argmax(1).cpu().numpy()
            val_f1 = f1_score(y_val.numpy(), val_preds, zero_division=0)
            val_acc = accuracy_score(y_val.numpy(), val_preds)
            # Also check it's not degenerate (predicting all one class)
            val_pred_pos = val_preds.sum()
            val_pred_neg = len(val_preds) - val_pred_pos

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}: loss={train_loss/train_total:.4f}, "
                  f"train_acc={train_correct/train_total:.3f}, "
                  f"val_acc={val_acc:.3f}, val_f1={val_f1:.3f}, "
                  f"val_pred=[{val_pred_neg}neg/{val_pred_pos}pos]")

        # Only count as best if predicting both classes
        if val_f1 > best_val_f1 and val_pred_pos > 0 and val_pred_neg > 0:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

        if epoch + 1 - best_epoch >= PATIENCE and best_state is not None:
            print(f"  Early stopping at epoch {epoch+1} (best was {best_epoch})")
            break

    if best_state is None:
        print("WARNING: Probe never predicted both classes. Using final state.")
        best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
        best_val_f1 = val_f1
        best_epoch = epoch + 1

    print(f"\nBest val F1={best_val_f1:.3f} at epoch {best_epoch}")

    probe.load_state_dict(best_state)
    probe = probe.to(device).eval()

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "config": {
            "input_dim": VJEPA2_EMBED_DIM, "hidden_dim": PROBE_DIM,
            "n_heads": PROBE_HEADS, "depth": PROBE_DEPTH,
            "n_classes": NUM_CLASSES, "dropout": PROBE_DROPOUT,
        },
        "best_epoch": best_epoch, "best_val_f1": best_val_f1,
    }, os.path.join(MODEL_DIR, "probe.pt"))
    print(f"Model saved to {MODEL_DIR}/probe.pt")

    # Evaluate
    with torch.no_grad():
        y_val_pred = probe(X_val_d).argmax(1).cpu().numpy()
        y_test_pred = probe(X_test_d).argmax(1).cpu().numpy()

    y_val_np = y_val.numpy()
    y_test_np = y_test.numpy()

    print(f"\n=== Val ===")
    print(classification_report(y_val_np, y_val_pred, target_names=["not-good", "golden"]))

    print(f"=== Test (worker_002 — cross-worker generalization) ===")
    print(classification_report(y_test_np, y_test_pred, target_names=["not-good", "golden"]))

    print("Per-factory test breakdown:")
    pf_metrics = per_factory_eval(test_clips, y_test_np, y_test_pred)

    # Save predictions as JSON
    for split, clips, preds in [("val", val_clips, y_val_pred), ("test", test_clips, y_test_pred)]:
        label_jsons = clips_to_label_json(clips, preds)
        for video_id, data in label_jsons.items():
            out_dir = os.path.join(PRED_DIR, split, data["factory"])
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"{video_id}.json"), "w") as f:
                json.dump(data["segments"], f, indent=2)

    print(f"\nPredictions saved to {PRED_DIR}/val/ and {PRED_DIR}/test/")

    # Save metrics
    metrics = {
        "model": VJEPA2_MODEL,
        "probe": {"dim": PROBE_DIM, "heads": PROBE_HEADS,
                  "depth": PROBE_DEPTH, "dropout": PROBE_DROPOUT, "params": n_params},
        "training": {"best_epoch": best_epoch, "best_val_f1": best_val_f1,
                     "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE},
        "val": compute_metrics(y_val_np, y_val_pred, "val"),
        "test": compute_metrics(y_test_np, y_test_pred, "test"),
        "per_factory_test": pf_metrics,
    }
    os.makedirs(PRED_DIR, exist_ok=True)
    with open(os.path.join(PRED_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {PRED_DIR}/metrics.json")
    print(f"\nTo overlay predictions on video:")
    print(f"  python verify_labels.py predictions/test/factory001/factory_001_worker_002_0000.json \\")
    print(f"    egocentric/test/factory001/factory_001_worker_002_0000.mp4")


# ── Run ──
features_ok = stage1_logreg()
if features_ok:
    stage2_probe()
else:
    print("\nStage 1 failed — features don't separate classes.")
    print("Consider: different pooling, unfreezing encoder layers, or more training data.")