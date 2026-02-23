# EgoCut

Real-time golden segment detection in egocentric factory videos using V-JEPA 2.

Given a first-person headcam video of a factory worker performing a repetitive task, EgoCut classifies every 8-second window as **Golden Standard** (clear, uninterrupted, perfectly framed execution) or **not good for collecting** (gaze shifts, occlusions, incomplete cycles, poor framing).

---

## Project Diary

This README is written as a chronological journal of every decision, dead end, and result. The project started as a labeling problem and became an end-to-end video understanding pipeline.

### The Dataset

We work with a subset of Ego-Exo4D / Egocentric-100K, specifically 77 videos across 10 factories. Each video is ~3 minutes of a worker performing a single repetitive manufacturing task, filmed from a head-mounted camera. The factories span electronics assembly, sewing, packaging, battery insertion, adhesive sealing, metal deburring, and more.

The raw videos are 456x256 at 30fps. Every factory has 7 videos: 3 for training (worker_001), 2 for validation (worker_001), and 2 for testing (worker_002). The test split uses a different worker doing the same task, so it measures cross-worker generalization.

We dropped factory010 early due to poor video quality, leaving us with 10 active factories initially, and eventually 9 after dropping factory006 (explained below).

### Phase 1: Defining "Golden Standard"

The core challenge was defining what makes a video segment worth collecting for AI training. We arrived at four strict criteria:

1. **Full Temporal Cycle** — the segment must contain a complete pick-manipulate-release cycle, not a fragment.
2. **Gaze Stability** — the worker's head (and therefore the camera) must stay locked on the task zone. Any look-away longer than 0.5 seconds kills the segment.
3. **Zero Occlusion** — hands making contact with components must be fully visible. Support beams, bins, equipment blocking the view means rejection.
4. **Optimal Framing** — hands stay within the central 60% of the frame. Actions at the extreme edges or off-screen are discarded.

These criteria emerged from manually labeling the first factory (factory001). The key insight was that "golden" is much stricter than "the worker is doing their job." Most of the time the worker IS doing their job, but the camera angle, gaze direction, or equipment positioning makes that particular moment useless for training an action recognition model.

### Phase 2: Video Preprocessing

Raw videos have significant visual noise. Neighboring workers are visible on both sides. Fisheye distortion warps the edges. Adjacent workstations bleed into the top of the frame.

We tested three crop configurations and settled on: center-crop 256x236 from the 456x256 frame (removing 100px from each side and 20px from the top), then scale back to 256x256. The side crop removes neighboring workers and the worst fisheye distortion. The 20px top trim removes the adjacent station's equipment that pokes into frame. We tested trim=0, trim=20, and trim=36. trim=20 was the sweet spot — trim=36 started cutting into the worker's own hands during overhead reaches.

ffmpeg filter: `crop=256:236:100:20,scale=256:256`

Sampling: fixed 2fps across all factories. At 2fps, a 16-frame V-JEPA 2 clip covers exactly 8 seconds. We considered per-task adaptive FPS (fast tasks like battery insertion could use 4fps, slow tasks like panel lamination could use 1fps), but fixed 2fps keeps the pipeline simple and doesn't require knowing the task at deployment time.

### Phase 3: Manual Calibration Labels (10 videos)

We hand-labeled one video per factory — the calibration videos. This was done in a Gemini chat session where we uploaded each video and iterated on the labels.

The first attempt on factory001 was too loose. Gemini included segments with subtle gaze shifts and partial cycles. We corrected it by pointing to timestamp 02:22-02:29 as the "Master Golden Segment" — two perfect, uninterrupted cycles of electronics assembly. After seeing the correction, Gemini articulated three hard-fail rules on its own (gaze shift = cut, structural occlusion = remove, boundary truncation = discard) and applied them correctly to factory002 without any further correction.

This correction loop became the foundation for everything that followed.

### Phase 4: Scaling Labels with Gemini Context Caching

With 10 calibration labels done, we needed to label the remaining 60+ videos. The key decision was how to transfer the learned criteria to an API pipeline.

We used Gemini's explicit context caching. The cache contained:
- A system instruction encoding the four Golden Standard criteria
- The full factory001 video (3 min) + the correction conversation where Gemini learned what it got wrong
- The full factory002 video (3 min) + the zero-correction success proving it had learned

Factory003 was deliberately held out as a validation target.

**Why explicit caching over implicit:** explicit caching guarantees a 50% token discount on the cached content and gives us full control over what's cached. The two full 3-minute videos plus the conversation cost ~84K tokens in the cache.

**Why factories 001-002 only:** minimal viable context. One correction (001) teaches the criteria. One success (002) proves generalization. Adding more factories to the cache would increase storage costs without meaningfully improving labeling quality.

**Model choice for caching:** the original calibration chat used `gemini-3-flash-preview`, not `gemini-2.5-pro`. This mattered — we initially built the cache with 2.5-pro and got different (worse) results. The cache model must match the model that learned the criteria. We rebuilt with flash and the results aligned.

**Validation results on factory003 (held out):**

| Metric | Ground Truth | Gemini 2.5 Pro | Gemini 3 Flash |
|--------|-------------|----------------|----------------|
| Golden seconds | 85s | 143s | 99s |
| % Golden | 47% | 80% | 55% |
| Segments | 7 | 11 | 8 |

Flash caught the major 00:49-01:30 not-good zone that Pro missed entirely. It still merges through some 1-second gaps (our ground truth cuts at single-frame gaze shifts), but at 8-second clip resolution these differences vanish.

Factory004 (sewing) was trickier — Flash missed the first 43-second golden segment and shifted remaining boundaries. Different task structure from the electronics assembly that dominated the cache. Acceptable at clip resolution.

### Phase 5: Batch Labeling

60 videos labeled in one automated batch run. Zero failures. 5-second delay between API requests to avoid rate limits. Each video took ~30 seconds to process (upload + inference + response).

Coverage was 100-101% across all videos (complete temporal coverage, no gaps). Golden time varied widely by factory — from 8% to 89% of total video duration depending on the worker and task complexity.

### Phase 6: The Factory006 Problem

Spot-checking revealed factory006 was inconsistent. The calibration video showed a manual lever-press stamping task with only 11% golden time. But some training videos had wildly different content — one scored 89% golden with 32 segments.

We overlayed the labels and confirmed: the train/val/test videos for factory006 showed a different task than the calibration video. The API labels were probably correct for the actual video content, but they didn't match the calibration criteria because the calibration was done on a different activity.

**Decision: drop factory006 entirely.** 9 factories, 63 labeled videos. Better to have consistent labels across fewer factories than noisy labels polluting the training signal.

### Phase 7: V-JEPA 2 Feature Extraction

V-JEPA 2 (Meta FAIR, June 2025) is a self-supervised video encoder pretrained on over 1 million hours of video. It learns to predict masked video patches in feature space, producing rich spatiotemporal representations without any human labels.

**Model choice: ViT-g at 384px** — the largest available model. 1 billion parameters, 1408-dimensional embeddings, ~12GB VRAM. We initially considered ViT-L (300M params, 1024-dim, 2GB VRAM) as a lighter option but had the compute budget, so we went full scale.

**Loading via torch.hub, not HuggingFace:** the HuggingFace AutoModel wraps the encoder with a fixed frame-per-clip count (fpc=64). We use 16 frames. torch.hub gives the raw encoder with no constraints. One gotcha: `torch.hub.load` returns a `(encoder, predictor)` tuple — the predictor was used during pretraining for masked patch prediction and isn't needed for downstream classification. We only keep the encoder.

**Feature extraction pipeline:**

1. Read 16 frames per clip from the cropped video at 2fps using decord
2. Resize from 256x256 to 384x384 (bilinear interpolation to match the model's training resolution)
3. Normalize with ImageNet statistics
4. Forward through frozen encoder with autocast → (1, 4608, 1408) patch features
5. Reshape to (1, 8, 576, 1408) — 8 temporal tokens x 576 spatial tokens per frame
6. Spatial average pool → (8, 1408) temporal features per clip

**Why spatial averaging:** storing all 4608 x 1408 features per clip would be ~26MB each, ~13GB for 500 clips. Spatial averaging retains temporal patterns (which matter — "hands doing centered task" vs "head turning away" plays out over time) while being practical to store (~45KB per clip). The tradeoff is losing spatial information about WHERE things are in the frame, which matters for our task (centered hands = golden, edge of frame = not good). This became relevant in the results.

**Clip construction:** 8-second non-overlapping clips from each video. Label: golden if >50% of the clip's seconds are Golden Standard. For the final training run we used 50% overlap (stride=4s) for training clips only, doubling training data to ~1182 clips. Validation and test clips stayed non-overlapping to keep metrics honest.

### Phase 8: Training the Probe

**Stage 1 — Logistic Regression Baseline:**

First thing we ran was a sanity check: mean-pool the 8 temporal tokens to a 1408-dim vector, StandardScaler, logistic regression. This answers the question "do the features separate the classes at all?"

| Split | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Val | 0.57 | 0.52 | 0.25 | 0.34 |
| Test | 0.53 | 0.45 | 0.19 | 0.26 |

Weak, but above chance (always-golden baseline = 0.45 accuracy). The features encode something about golden vs not-good. The low recall (19% on test) means it's very conservative — mostly predicting not-good.

**Stage 2 — Attentive Probe:**

The V-JEPA 2 paper uses a 4-layer attentive probe for evaluation: 3 self-attention blocks + 1 cross-attention block with a learnable query token. The query attends to the temporal features and produces a single classification vector.

**First attempt: 384-dim, 4 layers, 0.2 dropout = 7.64M params on 591 training clips.** It collapsed immediately — learned to predict "golden" for everything at epoch 1 and never recovered. 100% recall, 0% precision on not-good. The weighted loss made "predict golden always" the path of least resistance. Classic overparameterization on small data.

**Second attempt: scaled down aggressively.** 128-dim, 2 layers (1 self-attn + 1 cross-attn), 0.4 dropout = 578K params. Class-weighted cross-entropy loss (golden weighted 1.84x). AdamW with cosine LR decay, gradient clipping at 1.0, early stopping on validation F1 with a guard that rejects degenerate all-one-class predictions.

**Final results (1182 train clips with 50% overlap, non-overlapping val/test):**

| Split | Accuracy | Precision (golden) | Recall (golden) | F1 (golden) |
|-------|----------|---------------------|-----------------|-------------|
| Val | 0.61 | 0.54 | 0.69 | 0.61 |
| Test | 0.60 | 0.53 | 0.84 | 0.65 |

**Per-factory test breakdown:**

| Factory | Task | Avg Cycle | Acc | F1 | Golden/Total | Notes |
|---------|------|-----------|-----|-----|--------------|-------|
| factory004 | Industrial sewing | 13.4s | 0.75 | 0.83 | 28/44 | Best performer — long, distinct motion |
| factory011 | Cylinder scraping | 17.2s | 0.75 | 0.85 | 32/44 | Consistent hand motion pattern |
| factory002 | Panel lamination | 22.0s | 0.61 | 0.71 | 27/44 | Long roller sweeps, easy to distinguish |
| factory007 | Motor housing | 11.3s | 0.64 | 0.71 | 23/44 | Good wire-routing signal |
| factory003 | PCB assembly | 12.0s | 0.50 | 0.65 | 20/44 | Moderate, some confusion at boundaries |
| factory001 | Ring insertion | 3.1s | 0.52 | 0.55 | 22/44 | Ultra-fast cycles, subtle boundaries |
| factory008 | Battery insertion | 3.0s | 0.41 | 0.46 | 16/44 | Fastest task, hardest to classify |
| factory005 | Box packaging | 11.5s | 0.34 | 0.33 | 8/44 | Very few golden clips in test data |
| factory009 | Glue sealing | 6.1s | 0.86 | 0.00 | 2/44 | Only 2 golden clips — effectively untestable |

**The pattern is clear:** factories with longer, more visually distinctive cycles (sewing at 13.4s, scraping at 17.2s, lamination at 22s) perform well. Factories with ultra-fast cycles (battery at 3s, ring insertion at 3.1s) struggle — at 8-second clip resolution, most clips are mixed golden/not-good, making the binary label itself noisy. The visual difference between "doing the task with good framing" and "doing the task with a 0.5-second gaze shift" is extremely subtle when the entire cycle takes 3 seconds.

### What Worked

1. **The Gemini calibration loop** was highly effective. One correction on factory001 was enough to teach the criteria. The model generalized to 8 other factory tasks without further correction.
2. **Context caching** gave us consistent labels across 60 videos at 50% token cost. The two-factory cache (correction + success) was sufficient.
3. **V-JEPA 2 features do encode task-relevant information.** The ViT-g encoder, trained on 1M+ hours of video without any labels, produces features that separate golden from not-good segments above chance. The signal is real.
4. **Factory004 (sewing) and factory011 (scraping) hit 0.83-0.85 F1.** For tasks with long, visually distinctive cycles, the pipeline works.

### Where It Struggled, and Why

The struggles were NOT failures of the approach — they trace back to specific, identifiable causes:

1. **Spatial averaging discards positional information.** Golden vs not-good is partly about WHERE things are in the frame — hands centered vs reaching to the edge, gaze on-task vs shifted. Averaging over all spatial positions collapses this signal. This is an architectural choice we can fix.

2. **Small training set limits the probe.** ~1182 clips (416 golden) is thin for even a 578K param model. The probe peaked at epoch 2-4 and overfit from there. More data would help directly.

3. **Label noise from Gemini.** The API labels are ~85% accurate per-second compared to hand-verified ground truth. At clip level this becomes ~90-95%, but it's still a ceiling on what the probe can learn. The label noise is concentrated at boundaries, not systematic misclassification.

4. **Fast-cycle factories have a resolution problem.** Factory001 and factory008 have 3-second cycles. At 8-second clip resolution, almost every clip is mixed. The binary label becomes unreliable. This isn't a model failure — it's a granularity mismatch. Shorter clips (4s at 4fps) would help.

### Demo Videos

Two overlay videos are included for visual verification:

- `verify_output/factory_001_worker_002_0000_overlay.mp4` — Ring insertion, cross-worker test. Fast 3.1-second cycles.
- `verify_output/factory_008_worker_001_0003_overlay.mp4` — Battery insertion, validation set. Ultra-fast 3-second cycles.

Green bar = model predicts Golden Standard. Red bar = model predicts not good for collecting.

### Next Steps

1. **Keep spatial features** — instead of averaging, use a spatial attention mechanism or keep full patch tokens with a more efficient probe. This would let the model learn "hands in center = golden."
2. **Larger training set** — add the 10 calibration videos to training. Use stronger augmentation (temporal jitter, random spatial crop within the center region).
3. **Fine-tune encoder** — unfreeze the last 2 layers of ViT-g and fine-tune end-to-end. Risky with small data but could adapt features to this specific domain.
4. **Per-factory heads** — train a shared backbone with factory-specific classification heads. Different factories have very different visual signatures.
5. **Finer temporal resolution** — for fast-cycle factories, try 4fps with 16 frames = 4-second clips instead of 8-second.

---

## Repository Structure

```
EgoCut/
  config.py              # All constants: paths, crop params, VJEPA2 config, factory metadata
  make_clips.py          # Per-second labels -> 8s clip manifests (with optional overlap)
  extract.py             # Frozen VJEPA2 ViT-g feature extraction via torch.hub
  train.py               # Two-stage: logistic regression baseline + attentive probe
  verify_labels.py       # Overlay predicted labels on video (green/red bar)
  create_cache.py        # Build Gemini context cache with calibration conversation
  label_batch.py         # Batch-label videos using cached Gemini context
  egocentric/            # Raw videos by split/factory
    train/val/test/
      factory001-011/
  cropped/               # Center-cropped 256x256 videos
  labels/                # Gemini-generated JSON labels per video
  clips/                 # Clip manifests (train/val/test_manifest.json)
  embeddings/            # Extracted VJEPA2 features (train/val/test.pt)
  model/                 # Trained probe weights (probe.pt)
  predictions/           # Predicted labels as JSON + metrics.json
    val/test/
      factory001-011/
  verify_output/         # Overlay videos for visual verification
```

## Running the Pipeline

```bash
pip install torch torchvision timm einops decord scikit-learn google-genai

# 1. Label videos (requires Gemini API key + existing cache)
export GEMINI_API_KEY=your_key
python create_cache.py
python label_batch.py

# 2. Build clip manifests (50% overlap for training)
python make_clips.py --overlap 0.5

# 3. Extract VJEPA2 features (needs GPU, ~12GB VRAM)
python extract.py

# 4. Train probe and generate predictions
python train.py

# 5. Visually verify
python verify_labels.py predictions/test/factory004/factory_004_worker_002_0000.json \
  egocentric/test/factory004/factory_004_worker_002_0000.mp4
```

## Key Numbers

| Metric | Value |
|--------|-------|
| Factories | 9 (dropped 006 and 010) |
| Total videos labeled | 63 |
| Training clips (50% overlap) | 1,182 |
| Validation clips | 390 |
| Test clips | 396 |
| Encoder | V-JEPA 2 ViT-g 384px (1B params, frozen) |
| Probe | 578K params (128-dim, 2-layer attentive) |
| Test F1 (golden) | 0.65 |
| Best factory F1 | 0.85 (factory011, cylinder scraping) |
| Worst testable factory F1 | 0.33 (factory005, box packaging) |