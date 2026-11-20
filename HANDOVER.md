# Project Handover Plan

## Current next-plan (next implementation sprint)

This is the plan we should execute next to turn the project from a working prototype into a reliable demo and evaluation pipeline.

1. Finalize model validation before claiming quality.
   - Check the available EC-YOLO checkpoint and training logs in `artifacts/ec_yolo_training/`.
   - Run a proper evaluation on the validation/test split using precision, recall, mAP50, and mAP50-95.
   - Record a single clean benchmark summary so the current detector quality is measurable and comparable.

2. Validate the active ROI HARP-UNet checkpoint against detector-generated crops.
   - The active model is trained on `prepared_roi_dataset` at 128x128; use the matching ROI inference mode and checkpoint.
   - Evaluate with predicted detector boxes as well as ground-truth ROI crops to measure detector/segmenter interaction.
   - Use GPU training when available; the current workspace has a CPU-only PyTorch build.

3. Harden the inference pipeline and checkpoint compatibility.
   - Validate the detector runs successfully on the active checkpoint without silent loading failures.
   - Confirm the pipeline can accept both a custom checkpoint and the fallback YOLO model path without breaking the UI.
   - Fix any checkpoint-state mismatch, incompatible key warnings, or partial-load issues discovered during testing.

4. Validate ROI mask restoration quality.
   - Keep ROI inference preprocessing aligned with the 128x128 square-resize training transform.
   - Ensure the final pothole mask aligns with the original image geometry before area and severity calculations.
   - Keep full-image inference explicitly selectable for the legacy full-image checkpoint.

5. Lock down the physical-area math and calibration assumptions.
   - Treat `meters_per_pixel` as the calibration input we already support, not as a guaranteed default.
   - Keep calibration guidance explicit in the UI and docs: without a known scale, we can report pixel area but not true physical area.
   - Do not overstate measurement accuracy beyond the calibration data currently available.

6. Reduce UI scope to the features we actually use.
   - Keep the Streamlit interface focused on image/video processing, model selection, confidence threshold, BEV mode, and output inspection.
   - Remove or defer experimental controls that are not part of the current demo workflow.
   - Keep metadata fields minimal and avoid adding unused geospatial inputs back in until there is a real requirement.

7. Measure performance and stability on video.
   - Benchmark FPS, latency, GPU memory, and output quality on a representative video.
   - Confirm the app remains responsive during multi-frame processing and that downloaded reports are consistent.
   - Document the pass/fail operating envelope of the current pipeline on available hardware.

8. Prepare the demo handoff package.
   - Save a final curated checkpoint set for the active detector and segmenter.
   - Keep one reproducible run command for each major workflow: dataset prep, training, evaluation, and inference.
   - Package a concise final report with the current model status, known limitations, and the next actions for the next contributor.

## Short priority order

1. Detector + segmentation validation
2. Safe checkpoint loading and runtime stability
3. Detector-to-ROI segmentation validation
4. Calibration and severity honesty
5. Video benchmarking and final demo polish
6. Documentation and handoff cleanup

## Project direction

The project should remain focused on a credible, deployable pothole-monitoring demo rather than broad experimental features. The key near-term objective is to prove that the trained detector and segmenter work together reliably on real images and video, with honest calibration and reproducible evaluation metrics.
