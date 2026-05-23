# Stage 2 — Breakline Model (Pilot)

**Type:** ML
**Status:** Not started
**Depends on:** Stage 1
**Maps to:** roadmap Phase A.0 Weeks 3-4

## Objective

Train the first breakline-detection model end to end on the pilot projects, prove the learning machinery works, and build the post-processing that turns model output into breakline polylines. The goal is a **working pipeline**, not a useful model — the model will overfit, and that is intended.

## Scope

### In scope
- A PyTorch Lightning `DataModule` wrapping the Stage 1 TorchGeo dataset.
- A `LightningModule` with a small `segmentation_models_pytorch` U-Net (ResNet-18/34 encoder, ImageNet initialization).
- Loss: BCE + Dice (handles the severe class imbalance — most pixels are not breaklines).
- Weights & Biases logging wired up from the first run.
- A deliberate overfit sanity check on a handful of tiles.
- Post-processing: probability threshold → skeletonize → vectorize → simplify (Douglas-Peucker) → smooth → polylines.
- Evaluation: buffer-based IoU and F1 against expert breaklines.

### Out of scope (deferred)
- Bulk-corpus training and generalization (Stage 3).
- Resolution prediction (Stage 5).
- Any HEC-RAS run (Phase A evaluates against expert breaklines, not runs).

## Tasks

1. Build the Lightning `DataModule` and `LightningModule`.
2. Implement the BCE + Dice loss.
3. Wire up W&B logging (metrics, sample prediction images).
4. Overfit run: train on a few tiles, confirm loss drives to near-zero — this proves the machinery learns.
5. Full pilot training run on Muncie + Bald Eagle.
6. Implement the post-processing chain to polylines.
7. Implement buffer-based IoU / F1 metrics.
8. Notebook: predicted polylines overlaid on expert breaklines and terrain.

## Checkpoint — exit criteria

- [ ] The overfit sanity check succeeds — loss collapses on a few tiles, confirming the gradient path works.
- [ ] A full pilot training run completes without error and logs to W&B.
- [ ] The end-to-end path runs: terrain → feature stack → model → probability map → polylines.
- [ ] IoU and F1 are computed and recorded against expert breaklines.
- [ ] Predicted polylines are visualized against ground truth.
- [ ] `pytest` green; pre-commit clean.

## Notes & risks

- Overfitting on the pilot is expected and is the point — do not chase generalization here (that is Stage 3).
- The post-processing chain (raster probability → clean polylines) is fiddly; budget real time for the skeletonize/vectorize/simplify steps.
- Record a short retrospective at the end: what surprised you, what to change before scaling.
