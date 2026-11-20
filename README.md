# Pothole Detection, Segmentation, and Road Health Assessment

Pipeline:

`UAV image/video -> EC-YOLO detection -> ROI -> HARP-UNet segmentation -> aggregate road damage area -> severity and road-health score`

## Current architecture

See the [project architecture diagram](docs/architecture_diagram.md) for the implemented runtime and offline training flows. The more detailed [integrated system architecture](docs/integrated_system_architecture.md) also documents report-level extension points.

- **EC-YOLO:** EfficientNet-B0-inspired MBC backbone, SPPF, C2PSA, PAN/FPN neck, CoordAtt detection heads.
- **HARP-UNet:** ROI-crop segmentation with 128x128 input, MSCAG encoder blocks, DSConv decoder blocks, and area estimation. Full-image segmentation remains available for the legacy checkpoint.
- **Area:** pixel area is always available. Physical area uses a calibrated ground-surface scale, such as an ArUco marker or known camera geometry.
- **Road damage severity:** assigned once per analyzed road from the union of all segmented pothole masks, so overlapping detections are not double-counted. With metres-per-pixel calibration, aggregate damage is Low `<0.05 m²`, Medium `0.05–<0.20 m²`, High `0.20–<1.00 m²`, or Critical `≥1.00 m²`. Video summaries report the highest rated severity observed in any analyzed frame, and frame exports include each frame's severity. Without calibration, aggregate pixel area remains available and severity is `Unrated`. These are initial project thresholds, not a universal road-maintenance standard; validate them against local guidance before operational use.
- **Road Damage Percentage (RDP) / RHI:** the IDD-fine-tuned SegFormer-B0 is preferred when available, followed by the earlier local fine-tune and then the Cityscapes baseline. RDP uses the union of segmented pothole pixels as damage and the union of the predicted road region with those pothole pixels as the analyzed area. This prevents a road-segmentation mask from silently removing a pothole when it fails to classify damaged pavement as road. If road segmentation errors, returns a shape-mismatched mask, or returns an empty mask, the system falls back to the full image area and labels the RDP as an estimate. RHI is `clamp(100 - RDP, 0, 100)`. The UI reports road-mask overlap so low overlap can be reviewed.
- **Repair priority:** follows the RHI rule (RHI `< 40` = high, `40–70` inclusive = medium, and `> 70` = low). Pothole count does not affect the priority. Queue ranking uses priority band, then lower RHI.
- **Road-segmentation data:** the Kaggle Road Segmentation Dataset is stored under `dataset/road_segmentation_kaggle/` (31 image/annotation pairs) and has a local fine-tuning script. See its `DATASET_LICENSE.md`; the dataset is licensed CC BY-NC-ND 4.0.
- **Larger road-segmentation data:** the public IDD20KII dataset is available under `dataset/idd20kii/` and supplies polygon masks for road, separately from fallback drivable areas. Its Kaggle listing states CC BY 4.0; see [`DATASET_LICENSE.md`](dataset/idd20kii/DATASET_LICENSE.md) and keep attribution with any trained artifact.

## Dataset policy

Only matching files in `dataset/JPEGImages/` and `dataset/SegmentationClass/` are segmentation samples. Images without masks are not assumed to be negative examples until manually verified.

For a stronger training set, combine multiple pothole datasets before training. The dataset preparation script now accepts repeated `--source` entries, so this can be done in one pass:

```powershell
python run_prepare_dataset.py --source D:/datasets/pothole_a --source D:/datasets/pothole_b
```

This is the correct workflow when a mentor says the current set is too small: gather another public pothole dataset that uses the same segmentation style, merge it into the same `JPEGImages/SegmentationClass` or `images/masks` layout, then retrain with the combined prepared dataset. A paper-level benchmark usually uses external pothole datasets plus a local set, not just one small collection.

Run the audit after installing dependencies:

```powershell
python tools/audit_dataset.py
```

The audit creates a report and a reproducible manifest under `artifacts/dataset_audit/`.

## GPU workflow

```powershell
pip install -r requirements.txt
python run_prepare_dataset.py
python tools/prepare_roi_dataset.py --source prepared_dataset --output prepared_roi_dataset
python run_train_ec_yolo.py --epochs 50 --batch-size 8 --image-size 640
python run_train_harp.py --epochs 50 --batch-size 8 --image-size 128
streamlit run app.py
```

For recorded video:

```powershell
python run_video.py input.mp4 output.mp4 --detection-checkpoint artifacts/ec_yolo_training/last.pt --segmentation-checkpoint artifacts/harp_roi_training/best.pt --segmentation-mode roi
```

Video summaries count unique potholes using persistent Ultralytics ByteTrack
IDs when the EC-YOLO detector is selected; the per-frame report still includes
raw detections and cumulative unique counts. Other detector backends use
box-overlap/proximity matching, which is less reliable with camera motion. The
dashboard analyzes every frame by default to give tracking the best chance to
maintain identities. YOLO box suppression is set to an IoU threshold of 0.45
to reduce overlapping boxes for the same pothole. Tracking can still split or
merge identities when detections are missed or potholes overlap heavily.

Without a local fine-tuned checkpoint, the default road-surface model is
`nvidia/segformer-b0-finetuned-cityscapes-1024-1024`; pass
`--road-segmentation-model` to use another compatible semantic-segmentation
checkpoint.

To fine-tune SegFormer-B0 on the included CVAT polygon labels (private,
non-commercial use only), run:

```powershell
python tools/train_road_segformer.py --license-acknowledged
```

The script holds out a validation split, reports its road-class IoU against the
untuned checkpoint, and saves the best local model under
`artifacts/road_segformer_finetuned/`. The app and video runner automatically
prefer this local checkpoint when present. The training set is small and
frames may be correlated; inspect its validation metrics and road-mask overlays
before relying on results. Do not distribute the fine-tuned checkpoint or use
it commercially without permission from the dataset provider.

For the larger IDD road dataset, extract it so `dataset/idd20kii/` contains
`gtFine/` and `leftImg8bit/`, then run a 5,000-image CPU pilot:

```powershell
python tools/train_road_segformer.py --dataset-format idd --dataset-dir dataset/idd20kii --output-dir artifacts/road_segformer_idd20k --epochs 1 --max-train-samples 5000 --max-validation-samples 500 --license-acknowledged
```

IDD uses its provided train/validation directories; only polygons labeled
`road` are included in the road mask. Drivable fallback polygons are ignored in
the loss and IoU, while other classes are background. In the saved
5,000-train/500-validation, one-epoch pilot, road IoU increased from 0.8109
(Cityscapes baseline) to 0.9508. These results apply to this held-out subset.
The checkpoint is saved under `artifacts/road_segformer_idd20k/` and is
preferred by the app and video runner when present. Retain the IDD attribution.

To continue the saved IDD fine-tune to 50 total epochs, resume from its latest
weights and run this from the project root:

```powershell
& ".\.venv\Scripts\python.exe" tools\train_road_segformer.py --dataset-format idd --dataset-dir dataset\idd20kii --output-dir artifacts\road_segformer_idd20k --epochs 50 --max-train-samples 5000 --max-validation-samples 500 --license-acknowledged --resume
```

`--epochs 50` is the total target, so the existing six epochs are continued.
Metrics are updated after every epoch in
`artifacts/road_segformer_idd20k/training_metrics.json`; the best model is kept
in the output directory and the latest weights are kept in its `last/` folder.

For the legacy full-image checkpoint, use `--segmentation-checkpoint artifacts/harp_training/best.pt --segmentation-mode full_image`.

Inverse perspective mapping is built into the pipeline and runs before detection
by default. For calibrated geometry, provide four source-image points in
clockwise order, starting at the top-left road-plane corner:

```powershell
python run_video.py input.mp4 output.mp4 --bev-source-points '[[120,80],[520,80],[620,460],[20,460]]' --bev-width 640 --bev-height 640
```

The Streamlit sidebar accepts the same JSON point format and defaults to
`auto`, which applies a visual road-plane trapezoid. These points define a
road-plane trapezoid that is rectified into a top-down bird's-eye image before
detection. Use manual points for a calibrated IPM; automatic points are not
suitable for metric area measurement. Detector boxes are projected back to the
original frame before HARP-UNet segmentation and visualization.

The custom detector trainer is an initial research implementation. Validate its metrics against the paper's official loss and target-assignment details when those are available.


python -m streamlit run app.py