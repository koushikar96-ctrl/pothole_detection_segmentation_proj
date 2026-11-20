"""Pothole Detection and Road Health — Streamlit Dashboard
============================================================
Tabs:
  📷 Image  — upload a single road image and inspect detections, masks, areas,
              and a colour-coded health card.
  🎥 Video  — upload a recorded road video, process it frame-by-frame with a
              progress bar, and download the annotated output.
"""
from __future__ import annotations

import io
import csv
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from cache.detector_profiles import DETECTOR_OPTIONS, resolve_detector_config, resolve_project_model_path
from src.area_health import highest_road_damage_severity
from src.road_priority import (
    inspection_priority,
    priority_from_rhi,
    priority_rank,
    select_worst_record,
)
from src.road_segmentation import DEFAULT_ROAD_SEGMENTATION_MODEL
from src.video_tracking import PotholeTracker

# ---------------------------------------------------------------------------
# Page config (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Road Pothole Inspection Dashboard",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

road_queue_path = Path("artifacts") / "video_reports" / "road_priority_queue.json"
if "road_inspections" not in st.session_state:
    if road_queue_path.exists():
        try:
            st.session_state.road_inspections = json.loads(road_queue_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            st.session_state.road_inspections = []
    else:
        st.session_state.road_inspections = []

priorities_updated = False
for inspection in st.session_state.road_inspections:
    current_priority = inspection_priority(inspection)
    if inspection.get("repair_priority") != current_priority:
        inspection["repair_priority"] = current_priority
        priorities_updated = True
if priorities_updated:
    try:
        road_queue_path.write_text(
            json.dumps(st.session_state.road_inspections, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        st.warning(f"Could not update saved inspection priorities: {exc}")

if st.session_state.road_inspections:
    if st.sidebar.button("Clear saved road queue"):
        st.session_state.road_inspections = []
        if road_queue_path.exists():
            road_queue_path.unlink()
        st.rerun()

# ---------------------------------------------------------------------------
# Sidebar — shared configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    detector_source = DETECTOR_OPTIONS[0]
    detector_config = resolve_detector_config(detector_source)
    detector_kind = detector_config["kind"]
    detector_backend = str(detector_config["backend"])
    is_custom_detector = bool(detector_config["is_custom_detector"])
    custom_runtime_model_path = str(detector_config.get("runtime_checkpoint", detector_config["checkpoint"]))
    custom_reuntime_model_id = str(detector_config.get("model_id", ""))
    st.header("Inspection settings")
    detection_checkpoint = st.text_input(
        "Detector Checkpoint",
        value=str(detector_config["checkpoint"]),
        help="Path to the active detector checkpoint used for inference.",
    )
    segmentation_checkpoint = st.text_input(
        "Segmentation checkpoint",
        value=resolve_project_model_path("artifacts/harp_roi_training/best.pt"),
        help="Path to HARP-UNet *.pt file",
    )
    segmentation_mode = "roi"
    threshold = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Minimum detection confidence",
    )
    meters_per_pixel_raw = st.number_input(
        "Metres per pixel (0 = pixel units only)",
        min_value=0.0,
        value=0.0,
        format="%.6f",
        help="Set from an ArUco marker or ruler calibration",
    )
    meters_per_pixel: float | None = meters_per_pixel_raw if meters_per_pixel_raw > 0 else None
    segment_name = st.text_input("Road segment", value="", placeholder="Optional segment name")
    latitude = None
    longitude = None
    video_frame_step = st.number_input(
        "Video frame step",
        min_value=1,
        value=1,
        step=1,
        help="Analyze every frame by default for more reliable pothole tracking. Larger steps run faster but can count repeat sightings separately.",
    )

    st.markdown("---")
    st.caption("EC-YOLO + HARP-UNet pothole pipeline")


# ---------------------------------------------------------------------------
# Pipeline loader (cached so it isn't reloaded on every widget interaction)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models …")
def load_pipeline(
    det_ckpt: str,
    seg_ckpt: str,
    conf: float,
    detector_backend: str = "ecyolo",
    segmentation_mode: str = "roi",
    runtime_model_path: str | None = None,
    runtime_model_id: str | None = None,
):
    # Import here so Streamlit's module scanner doesn't fail before src is on path
    import sys
    src = str((Path(__file__).resolve().parent / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    from src.pipeline import PotholePipeline  # noqa: PLC0415

    detection_path = None
    detection_model_id = None
    if detector_backend == "ecyolo":
        runtime_source = runtime_model_path or runtime_model_id or det_ckpt
        detection_model_id = str(runtime_source).strip() if runtime_source and str(runtime_source).strip() else "yolov8s.pt"
    else:
        detection_path = Path(det_ckpt) if det_ckpt and str(det_ckpt).strip() else None

    segmentation_path = Path(seg_ckpt) if seg_ckpt and str(seg_ckpt).strip() else None

    return PotholePipeline(
        detection_checkpoint=detection_path,
        segmentation_checkpoint=segmentation_path,
        threshold=conf,
        detection_backend=detector_backend,
        detection_model_id=detection_model_id,
        segmentation_mode=segmentation_mode,
        road_segmentation_model_id=DEFAULT_ROAD_SEGMENTATION_MODEL,
    )


def run_analysis(image_bgr: np.ndarray, *, detector_kind: str, detection_checkpoint: str,
                 segmentation_checkpoint: str,
                 confidence_threshold: float, meters_per_pixel: float | None,
                 segmentation_mode: str,
                 runtime_model_path: str | None = None, runtime_model_id: str | None = None,
                 video_tracking: bool = False, start_video: bool = False):
    """Run detection and segmentation on the original image."""
    pipeline = load_pipeline(
        resolve_project_model_path(detection_checkpoint) if detector_backend != "ecyolo" else detection_checkpoint,
        resolve_project_model_path(segmentation_checkpoint),
        confidence_threshold,
        detector_backend,
        segmentation_mode,
        runtime_model_path=runtime_model_path,
        runtime_model_id=runtime_model_id,
    )
    track_ids = None
    if video_tracking and detector_backend == "ecyolo":
        if start_video:
            pipeline.reset_video_tracker()
        detections, track_ids = pipeline.track_video(image_bgr)
    else:
        _, _, detections = pipeline.detect_stages(image_bgr)
    road_mask = pipeline.segment_road(image_bgr)
    mask, results, health = pipeline.process_detections(
        image_bgr,
        detections,
        meters_per_pixel,
        road_mask=road_mask,
    )
    stages = {"source_detections": detections, "road_mask": road_mask}
    if track_ids is not None:
        stages["track_ids"] = track_ids
    return mask, results, health, stages


# ---------------------------------------------------------------------------
# Shared drawing utilities
# ---------------------------------------------------------------------------

_CONDITION_COLORS = {
    "good": ("🟢", "#2ecc71"),
    "fair": ("🟡", "#f1c40f"),
    "poor": ("🟠", "#e67e22"),
    "critical": ("🔴", "#e74c3c"),
}

_PRIORITY_LABELS = {
    "low": "🟢 Low",
    "medium": "🟡 Medium",
    "high": "🔴 High",
    "urgent": "🔴 High",
}


def priority_label(priority: object) -> str:
    """Return a consistent, human-readable priority label."""
    value = str(priority or "unknown").lower()
    return _PRIORITY_LABELS.get(value, value.capitalize())


def draw_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    detections: list[dict],
    track_ids: list[int | str] | None = None,
) -> np.ndarray:
    """Return an annotated BGR image with mask overlay and bounding boxes."""
    overlay = image_bgr.copy()
    # Red mask overlay
    overlay[mask > 0] = (0, 0, 220)
    output = cv2.addWeighted(image_bgr, 0.65, overlay, 0.35, 0)
    # Bounding boxes + confidence labels
    for index, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        conf = det["confidence"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)
        track_label = f" #{track_ids[index]}" if track_ids is not None else ""
        label = f"pothole{track_label} {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(output, (x1, y1 - lh - 6), (x1 + lw + 4, y1), (0, 220, 0), -1)
        cv2.putText(output, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return output


def draw_detection_boxes(
    image_bgr: np.ndarray,
    detections: list[tuple[float, float, float, float, float]],
) -> np.ndarray:
    """Return a frame with detector boxes before ROI segmentation."""
    output = image_bgr.copy()
    for x1, y1, x2, y2, confidence in detections:
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
        cv2.putText(
            output,
            f"pothole {confidence:.2f}",
            (int(x1), max(18, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
        )
    return output


def show_extracted_rois(image_bgr: np.ndarray, detections: list[dict]) -> None:
    """Show margin-expanded detector crops used for inspection and measurement."""
    if not detections:
        return
    from src.roi import extract_roi  # noqa: PLC0415

    with st.expander("Detected pothole regions", expanded=True):
        st.caption("These are the detector crops segmented by HARP-UNet.")
        columns = st.columns(min(len(detections), 4))
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["box"]
            roi = extract_roi(image_bgr, (x1, y1, x2, y2), detection["confidence"])
            with columns[index % len(columns)]:
                st.image(
                    cv2.cvtColor(roi.image, cv2.COLOR_BGR2RGB),
                    caption=f"ROI {index + 1}: {roi.image.shape[1]} x {roi.image.shape[0]} px",
                    width="stretch",
                )


def health_card(health: dict) -> None:
    """Render inspection KPIs and a combined RHI/RDP donut visualization."""
    condition = health.get("condition", "unknown")
    priority = health.get("repair_priority", "unknown")
    score = health.get("health_score", 0.0)
    coverage = health.get("damage_coverage", 0.0)
    count = health.get("pothole_count", 0)
    rdp_percent = float(health.get("rdp_percent", coverage * 100))
    damaged_area_pixels = int(health.get("damaged_area_pixels", 0))
    total_pothole_area_m2 = health.get("total_pothole_area_m2")
    road_damage_severity = health.get("road_damage_severity", "Unrated")

    icon, _ = _CONDITION_COLORS.get(condition, ("⚪", "#95a5a6"))

    with st.container(border=True):
        st.markdown(
            f"### {icon} Road condition: **{condition.capitalize()}**"
            f" · Repair priority: **{_PRIORITY_LABELS.get(priority, priority)}**"
        )
        metric_column, chart_column = st.columns([1.25, 1])
        with metric_column:
            with st.container(horizontal=True):
                st.metric("Road Health Index (RHI)", f"{score:.1f}", border=True)
                st.metric("Road Damage Percentage (RDP)", f"{rdp_percent:.2f}%", border=True)
                st.metric("Detected potholes", count, border=True)
            st.caption(
                "RDP compares segmented pothole area with the estimated road area. "
                "Physical area estimates require image calibration."
            )
            total_area_label = (
                f"{float(total_pothole_area_m2):.4f} m²"
                if total_pothole_area_m2 is not None
                else f"{damaged_area_pixels:,} px (calibration required for severity)"
            )
            st.caption(
                f"Total pothole area: **{total_area_label}** · "
                f"Road damage severity: **{road_damage_severity}**"
            )
            if total_pothole_area_m2 is not None:
                st.caption(
                    "Severity uses aggregate segmented area: Low <0.05 m²; "
                    "Medium 0.05–<0.20 m²; High 0.20–<1.00 m²; Critical ≥1.00 m²."
                )
            if health.get("road_area_fallback_used"):
                st.warning(
                    "Road segmentation failed; RDP is estimated against the full image instead. "
                    f"Details: {health.get('road_area_fallback_reason', 'unknown error')}"
                )
        with chart_column:
            st.altair_chart(
                rhi_rdp_donut(float(health.get("rhi", score)), rdp_percent),
                width="stretch",
                theme=None,
            )
            st.markdown(":blue[●] **RHI** · Road Health Index")
            st.markdown(":orange[●] **RDP** · Road Damage Percentage")


def rhi_rdp_donut(rhi: float, rdp: float) -> alt.LayerChart:
    """Build one donut ring split into complementary RHI and RDP portions."""
    chart_data = pd.DataFrame(
        [
            {"Metric": "Road Health Index (RHI)", "Value": max(0.0, min(100.0, rhi))},
            {"Metric": "Road Damage Percentage (RDP)", "Value": max(0.0, min(100.0, rdp))},
        ]
    )
    ring = (
        alt.Chart(chart_data)
        .mark_arc(innerRadius=66, outerRadius=94, cornerRadius=5)
        .encode(
            theta=alt.Theta("Value:Q", stack=True),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(
                    domain=[
                        "Road Health Index (RHI)",
                        "Road Damage Percentage (RDP)",
                    ],
                    range=["#2563EB", "#F97316"],
                ),
                legend=None,
            ),
            order=alt.Order("Metric:N", sort="ascending"),
            tooltip=[
                alt.Tooltip("Metric:N", title="Measure"),
                alt.Tooltip("Value:Q", title="Percentage", format=".2f"),
            ],
        )
    )
    center_label = (
        alt.Chart(pd.DataFrame({"label": [f"{rhi:.1f}"], "subtitle": ["RHI"]}))
        .mark_text(fontSize=23, fontWeight="bold", color="#1E3A8A")
        .encode(text="label:N")
    )
    return alt.layer(ring, center_label).properties(
        width=200,
        height=190,
    )


def detections_table(detections: list[dict], mpp: float | None) -> None:
    """Render a per-pothole details table."""
    if not detections:
        st.info("No potholes detected.")
        return
    rows = []
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["box"]
        row = {
            "#": i + 1,
            "Box (x1,y1,x2,y2)": f"({x1},{y1},{x2},{y2})",
            "Confidence": f"{d['confidence']:.3f}",
            "Pothole area (px)": d["pixel_area"],
            "Equiv. diameter (px)": f"{d['equivalent_diameter_pixels']:.1f}",
            "Damage ratio": f"{float(d.get('damage_ratio_percent', 0.0)):.4f}%",
        }
        if mpp and d.get("area_m2") is not None:
            row["Area (m²)"] = f"{d['area_m2']:.4f}"
        rows.append(row)
    st.dataframe(rows, width="stretch")


# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_dashboard, tab_image, tab_video = st.tabs(["Overview", "📷 Image", "🎥 Video"])


# ===== IMAGE TAB ============================================================
with tab_image:
    st.subheader("Upload a road image")
    uploaded_image = st.file_uploader(
        "Drag & drop or browse",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )

    if uploaded_image is not None:
        file_bytes = np.frombuffer(uploaded_image.read(), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("Could not decode the uploaded image.")
        else:
            with st.spinner("Running pipeline …"):
                try:
                    mask, detections, health, stages = run_analysis(
                        image_bgr,
                        detector_kind=detector_kind,
                        detection_checkpoint=detection_checkpoint,
                        segmentation_checkpoint=segmentation_checkpoint,
                        confidence_threshold=threshold,
                        meters_per_pixel=meters_per_pixel,
                        segmentation_mode=segmentation_mode,
                        runtime_model_path=custom_runtime_model_path,
                        runtime_model_id=custom_reuntime_model_id,
                    )
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")
                    st.stop()

            annotated = draw_overlay(image_bgr, mask, detections)

            st.markdown("### Processing stages")
            image_area = image_bgr.shape[0] * image_bgr.shape[1]
            oversized_detections = [
                (x2 - x1) * (y2 - y1) / image_area
                for x1, y1, x2, y2, _ in stages["source_detections"]
                if image_area > 0
            ]
            if any(box_fraction >= 0.65 for box_fraction in oversized_detections):
                st.warning(
                    "At least one detector box covers 65% or more of the image. "
                    "The box and its mask may be too broad; verify the overlay before relying on its measurements."
                )
            st.image(
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                caption="1. Original image",
                width="stretch",
            )
            show_extracted_rois(image_bgr, detections)
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption="2. HARP-UNet segmentation and mask overlay",
                width="stretch",
            )

            st.markdown("### 🩺 Road Health")
            health_card(health)

            st.markdown("### 📋 Detected Potholes")
            detections_table(detections, meters_per_pixel)

            with st.expander("Raw JSON output"):
                st.json({
                    "metadata": {
                        "road_segment": segment_name.strip() or None,
                        "latitude": latitude,
                        "longitude": longitude,
                    },
                    "health": health,
                    "potholes": [
                        {k: v for k, v in d.items() if k != "mask"} for d in detections
                    ],
                })


# ===== VIDEO TAB ============================================================
with tab_video:
    st.subheader("Upload a recorded road video")
    uploaded_video = st.file_uploader(
        "Drag & drop or browse (MP4 / AVI)",
        type=["mp4", "avi", "mov"],
        key="video_upload",
    )

    if uploaded_video is not None:
        # Write the upload to a temp file so OpenCV can read it
        suffix = Path(uploaded_video.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            tmp_in.write(uploaded_video.read())
            tmp_in_path = tmp_in.name

        capture = cv2.VideoCapture(tmp_in_path)
        if not capture.isOpened():
            st.error("Could not open the uploaded video.")
        else:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            st.info(f"Video: {width}×{height} @ {fps:.1f} FPS — {total_frames} frames")
            process_btn = st.button("▶ Process video", type="primary")

            if process_btn:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                    tmp_out_path = tmp_out.name

                writer = cv2.VideoWriter(
                    tmp_out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

                try:
                    pipeline = None
                except Exception as exc:
                    st.error(f"Could not load pipeline: {exc}")
                    st.stop()

                progress_bar = st.progress(0, text="Processing frames …")
                status_text = st.empty()
                frame_idx = 0
                source_frame_idx = 0
                health_scores: list[float] = []
                tracker = PotholeTracker()
                seen_video_track_ids: set[int] = set()
                total_detections = 0
                video_records: list[dict[str, object]] = []
                t0 = time.time()

                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    source_frame_idx += 1
                    if source_frame_idx % int(video_frame_step) != 0:
                        writer.write(frame)
                        continue

                    mask, detections, health, stages = run_analysis(
                        frame,
                        detector_kind=detector_kind,
                        detection_checkpoint=detection_checkpoint,
                        segmentation_checkpoint=segmentation_checkpoint,
                        confidence_threshold=threshold,
                        meters_per_pixel=meters_per_pixel,
                        segmentation_mode=segmentation_mode,
                        runtime_model_path=custom_runtime_model_path,
                        runtime_model_id=custom_reuntime_model_id,
                        video_tracking=True,
                        start_video=frame_idx == 0,
                    )
                    native_track_ids = stages.get("track_ids")
                    if native_track_ids is None:
                        track_ids, new_potholes = tracker.update(stages["source_detections"])
                        unique_potholes = tracker.unique_count
                    else:
                        valid_ids = [int(track_id) for track_id in native_track_ids if track_id is not None]
                        new_potholes = len(set(valid_ids) - seen_video_track_ids)
                        seen_video_track_ids.update(valid_ids)
                        track_ids = [
                            track_id if track_id is not None else "?"
                            for track_id in native_track_ids
                        ]
                        unique_potholes = len(seen_video_track_ids)
                    annotated = draw_overlay(frame, mask, detections, track_ids)

                    # HUD overlay
                    condition = health.get("condition", "")
                    priority = health.get("repair_priority", "")
                    cv2.putText(
                        annotated,
                        f"Health: {condition.capitalize()}  |  Priority: {priority.capitalize()}  |  Unique potholes: {unique_potholes}",
                        (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )
                    writer.write(annotated)

                    health_scores.append(float(health.get("health_score", 0.0)))
                    total_detections += len(detections)
                    frame_idx += 1
                    video_records.append({
                        "frame": frame_idx,
                        "source_frame": source_frame_idx,
                        "timestamp_seconds": round(source_frame_idx / max(fps, 1e-3), 2),
                        "potholes": len(detections),
                        "new_potholes": new_potholes,
                        "unique_potholes": unique_potholes,
                        "health_score": round(float(health.get("health_score", 0.0)), 2),
                        "condition": condition,
                        "repair_priority": priority,
                        "rhi": round(float(health.get("rhi", health.get("health_score", 0.0))), 2),
                        "damaged_area_pixels": int(health.get("damaged_area_pixels", 0)),
                        "damage_coverage": round(float(health.get("damage_coverage", 0.0)), 4),
                        "rdp_percent": round(float(health.get("rdp_percent", 0.0)), 2),
                        "total_pothole_area_m2": health.get("total_pothole_area_m2"),
                        "road_damage_severity": health.get("road_damage_severity", "Unrated"),
                        "road_area_pixels": int(health.get("road_area_pixels", 0)),
                        "road_area_fallback_used": bool(health.get("road_area_fallback_used", False)),
                    })

                    if total_frames > 0:
                        pct = min(source_frame_idx / total_frames, 1.0)
                        elapsed = time.time() - t0
                        fps_proc = frame_idx / max(elapsed, 1e-3)
                        progress_bar.progress(pct, text=f"Frame {frame_idx}/{total_frames} — {fps_proc:.1f} FPS")
                    else:
                        progress_bar.progress(0, text=f"Frame {frame_idx} processed")

                capture.release()
                writer.release()
                progress_bar.progress(1.0, text="Done!")

                elapsed_total = time.time() - t0
                proc_fps = frame_idx / max(elapsed_total, 1e-3)

                st.success(f"Processed {frame_idx} frames in {elapsed_total:.1f}s ({proc_fps:.1f} FPS throughput)")
                fallback_frames = sum(
                    bool(record["road_area_fallback_used"]) for record in video_records
                )
                if fallback_frames:
                    st.warning(
                        f"Road segmentation failed on {fallback_frames} analyzed frame(s). "
                        "Their RDP values use the full image area as a fallback."
                    )

                # Summary metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Frames processed", frame_idx)
                col2.metric("Avg RHI", f"{np.mean(health_scores):.1f}" if health_scores else "N/A")
                unique_potholes = (
                    len(seen_video_track_ids)
                    if detector_backend == "ecyolo"
                    else tracker.unique_count
                )
                col3.metric("Unique potholes", unique_potholes)
                highest_severity = highest_road_damage_severity(
                    str(record.get("road_damage_severity", "Unrated"))
                    for record in video_records
                )
                col4.metric("Highest damage severity", highest_severity)
                col5.metric("Throughput", f"{proc_fps:.1f} FPS")
                st.caption(
                    f"{total_detections} frame detections in total; repeated sightings of a tracked pothole "
                    "are counted once."
                )
                if video_records:
                    trend_data = pd.DataFrame(video_records).melt(
                        id_vars=["timestamp_seconds"],
                        value_vars=["rhi", "rdp_percent"],
                        var_name="Measure",
                        value_name="Percentage",
                    )
                    trend_data["Measure"] = trend_data["Measure"].map(
                        {
                            "rhi": "Road Health Index (RHI)",
                            "rdp_percent": "Road Damage Percentage (RDP)",
                        }
                    )
                    with st.container(border=True):
                        st.subheader("Road health and damage comparison")
                        trend_chart = (
                            alt.Chart(trend_data)
                            .mark_line(point=True, strokeWidth=2.5)
                            .encode(
                                x=alt.X("timestamp_seconds:Q", title="Video time (seconds)"),
                                y=alt.Y(
                                    "Percentage:Q",
                                    title="Percentage",
                                    scale=alt.Scale(domain=[0, 100]),
                                ),
                                color=alt.Color(
                                    "Measure:N",
                                    scale=alt.Scale(
                                        domain=[
                                            "Road Health Index (RHI)",
                                            "Road Damage Percentage (RDP)",
                                        ],
                                        range=["#2563EB", "#F97316"],
                                    ),
                                    title="Measure",
                                ),
                                tooltip=[
                                    alt.Tooltip("timestamp_seconds:Q", title="Time (s)", format=".2f"),
                                    alt.Tooltip("Measure:N", title="Measure"),
                                    alt.Tooltip("Percentage:Q", title="Value", format=".2f"),
                                ],
                            )
                            .properties(height=280)
                        )
                        st.altair_chart(trend_chart, width="stretch")
                # Store one summary per uploaded road video, not one repair item per frame.
                priority_order = ("high", "medium", "low")
                worst_record = select_worst_record(video_records)
                highest_severity = highest_road_damage_severity(
                    str(record.get("road_damage_severity", "Unrated"))
                    for record in video_records
                )
                road_record = {
                    "road": segment_name.strip() or Path(uploaded_video.name).stem,
                    "video": uploaded_video.name,
                    "latitude": latitude,
                    "longitude": longitude,
                    "frames_analyzed": frame_idx,
                    "unique_potholes": unique_potholes,
                    "total_potholes": unique_potholes,
                    "total_detections": total_detections,
                    "average_rhi": round(float(np.mean([record["rhi"] for record in video_records])), 2) if video_records else 0.0,
                    "worst_rhi": round(float(min(record["rhi"] for record in video_records)), 2) if video_records else 0.0,
                    "damage_coverage": round(float(max((record["damage_coverage"] for record in video_records), default=0.0)), 4),
                    "rdp_percent": round(float(max((record["rdp_percent"] for record in video_records), default=0.0)), 2),
                    "condition": worst_record.get("condition", "unknown"),
                    "repair_priority": (
                        priority_from_rhi(float(worst_record["rhi"]))
                        if worst_record
                        else "low"
                    ),
                    "road_damage_severity": highest_severity,
                    "worst_timestamp_seconds": worst_record.get("timestamp_seconds", 0.0),
                }
                st.session_state.road_inspections.append(road_record)
                road_queue_path.parent.mkdir(parents=True, exist_ok=True)
                road_queue_path.write_text(
                    json.dumps(st.session_state.road_inspections, indent=2),
                    encoding="utf-8",
                )

                st.markdown("### Road repair priority queue")
                ranked_roads = sorted(
                    st.session_state.road_inspections,
                    key=lambda road: (
                        priority_rank(inspection_priority(road)),
                        float(road["worst_rhi"]),
                    ),
                )
                priority_counts = {
                    priority: sum(
                        inspection_priority(road) == priority
                        for road in st.session_state.road_inspections
                    )
                    for priority in priority_order
                }
                pcols = st.columns(3)
                for column, priority in zip(pcols, priority_order):
                    column.metric(priority.capitalize(), priority_counts[priority])
                st.caption("Each row is one uploaded road/video. Priority follows the RHI bands; lower RHI ranks first within each band.")
                if ranked_roads:
                    queue_display = [
                        {
                            "Rank": index,
                            "Road": road["road"],
                            "Priority": priority_label(inspection_priority(road)),
                            "Road Damage Percentage (%)": road.get("rdp_percent", float(road.get("damage_coverage", 0.0)) * 100),
                            "Damage Severity": road.get("road_damage_severity", "Unrated"),
                            "RHI": road["worst_rhi"],
                        }
                        for index, road in enumerate(ranked_roads, start=1)
                    ]
                    st.dataframe(queue_display, width="stretch", hide_index=True)
                    first_road = ranked_roads[0]
                    st.warning(
                        f"Queue position 1: {first_road['road']} ({priority_label(inspection_priority(first_road))} priority, "
                        f"RDP {float(first_road.get('rdp_percent', float(first_road.get('damage_coverage', 0.0)) * 100)):.2f}%, "
                        f"RHI {first_road['worst_rhi']:.1f})."
                    )
                    st.caption(
                        "Queue position is the order for follow-up, not a separate severity rating. "
                        "Priority is assigned from RHI; lower RHI is ranked first within each priority band."
                    )
                    st.download_button(
                        "Download road priority queue",
                        data=json.dumps(ranked_roads, indent=2),
                        file_name="road_priority_queue.json",
                        mime="application/json",
                        on_click="ignore",
                    )

                csv_buffer = io.StringIO()
                if video_records:
                    writer_csv = csv.DictWriter(csv_buffer, fieldnames=list(video_records[0]))
                    writer_csv.writeheader()
                    writer_csv.writerows(video_records)
                download_col1, download_col2 = st.columns(2)
                download_col1.download_button(
                    "Download frame report (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="video_frame_report.csv",
                    mime="text/csv",
                    on_click="ignore",
                )
                download_col2.download_button(
                    "Download frame report (JSON)",
                    data=json.dumps(video_records, indent=2),
                    file_name="video_frame_report.json",
                    mime="application/json",
                    on_click="ignore",
                )

                # Download button
                with open(tmp_out_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download annotated video",
                        data=f.read(),
                        file_name="annotated_" + uploaded_video.name,
                        mime="video/mp4",
                        on_click="ignore",
                    )

                # Keep a local, reviewable inspection record beside the output video.
                report_dir = Path("artifacts") / "video_reports"
                report_dir.mkdir(parents=True, exist_ok=True)
                report_path = report_dir / f"{Path(uploaded_video.name).stem}_report.json"
                rhi_values = [float(record["rhi"]) for record in video_records]
                trend_per_minute = 0.0
                if len(rhi_values) >= 2:
                    timestamps = np.array([float(record["timestamp_seconds"]) for record in video_records])
                    trend_per_second = float(np.polyfit(timestamps, np.array(rhi_values), 1)[0])
                    trend_per_minute = round(trend_per_second * 60.0, 3)
                report_payload = {
                    "road_segment": segment_name,
                    "latitude": latitude,
                    "longitude": longitude,
                    "video_name": uploaded_video.name,
                    "video_fps": fps,
                    "frame_step": int(video_frame_step),
                    "frames_in_video": total_frames,
                    "frames_analyzed": frame_idx,
                    "rhi_trend_per_minute": trend_per_minute,
                    "trend_interpretation": "worsening" if trend_per_minute < -1 else "improving" if trend_per_minute > 1 else "stable",
                    "road_summary": road_record,
                    "records": video_records,
                }
                report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
                st.caption(f"Inspection report saved to `{report_path}`")


# ===== DASHBOARD TAB ========================================================
# Render after the video workflow so a newly saved inspection is reflected here.
with tab_dashboard:
    inspections = st.session_state.road_inspections
    ranked_roads = sorted(
        inspections,
        key=lambda road: (
            priority_rank(inspection_priority(road)),
            float(road.get("worst_rhi", 0.0)),
        ),
    )
    priority_counts = {
        "High": sum(inspection_priority(road) == "high" for road in inspections),
        "Medium": sum(inspection_priority(road) == "medium" for road in inspections),
        "Low": sum(inspection_priority(road) == "low" for road in inspections),
    }
    average_rhi = (
        float(np.mean([float(road.get("average_rhi", road.get("worst_rhi", 0.0))) for road in inspections]))
        if inspections
        else None
    )
    tracked_inspections = [
        road for road in inspections if road.get("unique_potholes") is not None
    ]
    total_potholes = sum(int(road["unique_potholes"]) for road in tracked_inspections)

    st.title("Road Pothole Inspection Dashboard")
    st.caption("Operational overview of saved road inspections and repair follow-up.")
    if len(tracked_inspections) != len(inspections):
        untracked_count = len(inspections) - len(tracked_inspections)
        st.caption(
            f"{untracked_count} earlier inspection(s) predate tracking; their saved pothole counts may include repeat sightings."
        )

    with st.container(horizontal=True):
        st.metric("Saved inspections", len(inspections), border=True)
        st.metric("Unique potholes", total_potholes, border=True)
        st.metric("Average inspection RHI", f"{average_rhi:.1f}" if average_rhi is not None else "N/A", border=True)
        st.metric("High-priority inspections", priority_counts["High"], border=True)

    if not inspections:
        with st.container(border=True):
            st.subheader("No inspection records yet")
            st.write("Process a road video to create the first inspection record and populate the repair queue.")
            st.info("Use the Video tab to upload a recording and start an inspection.")
    else:
        first_road = ranked_roads[0]
        with st.container(border=True):
            st.subheader("Next inspection for follow-up")
            followup_column, donut_column = st.columns([1.5, 1])
            with followup_column:
                st.markdown(
                    f"**Queue position 1 — {first_road.get('road', 'Unknown')}**  \n"
                    f"{priority_label(inspection_priority(first_road))} priority · "
                    f"Lowest observed RHI: **{float(first_road.get('worst_rhi', 0.0)):.1f}** · "
                    f"RDP: **{float(first_road.get('rdp_percent', float(first_road.get('damage_coverage', 0.0)) * 100)):.2f}%**"
                )
                st.caption(
                    "Queue position indicates follow-up order, not severity by itself. "
                    "The queue sorts by priority band, then by lowest observed RHI."
                )
            with donut_column:
                st.altair_chart(
                    rhi_rdp_donut(
                        float(first_road.get("worst_rhi", 0.0)),
                        float(first_road.get("rdp_percent", float(first_road.get("damage_coverage", 0.0)) * 100)),
                    ),
                    width="stretch",
                    theme=None,
                )
                st.markdown(":blue[●] **RHI** · Road Health Index")
                st.markdown(":orange[●] **RDP** · Road Damage Percentage")

        st.subheader("Priority bands")
        with st.container(horizontal=True):
            st.metric("High · RHI <40", priority_counts["High"], border=True)
            st.metric("Medium · RHI 40–70", priority_counts["Medium"], border=True)
            st.metric("Low · RHI >70", priority_counts["Low"], border=True)
        st.caption(
            "Priority depends only on RHI: High <40; Medium 40–70 inclusive; Low >70. "
            "Higher RHI indicates better road condition."
        )

        with st.container(border=True):
            st.subheader("Inspection comparison")
            comparison_data = pd.DataFrame(
                [
                    {
                        "Inspection": str(road.get("road", "Unknown")),
                        "Measure": measure,
                        "Percentage": value,
                    }
                    for road in ranked_roads[:10]
                    for measure, value in (
                        ("Road Health Index (RHI)", float(road.get("worst_rhi", 0.0))),
                        (
                            "Road Damage Percentage (RDP)",
                            float(road.get("rdp_percent", float(road.get("damage_coverage", 0.0)) * 100)),
                        ),
                    )
                ]
            )
            comparison_chart = (
                alt.Chart(comparison_data)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    x=alt.X(
                        "Percentage:Q",
                        title="Percentage",
                        scale=alt.Scale(domain=[0, 100]),
                    ),
                    y=alt.Y("Inspection:N", title=None, sort="-x"),
                    yOffset=alt.YOffset("Measure:N"),
                    color=alt.Color(
                        "Measure:N",
                        scale=alt.Scale(
                            domain=[
                                "Road Health Index (RHI)",
                                "Road Damage Percentage (RDP)",
                            ],
                            range=["#2563EB", "#F97316"],
                        ),
                        title="Measure",
                    ),
                    tooltip=[
                        alt.Tooltip("Inspection:N", title="Road / segment"),
                        alt.Tooltip("Measure:N", title="Measure"),
                        alt.Tooltip("Percentage:Q", title="Percentage", format=".2f"),
                    ],
                )
                .properties(height=max(180, 44 * min(len(ranked_roads), 10)))
            )
            st.altair_chart(comparison_chart, width="stretch")
            if len(ranked_roads) > 10:
                st.caption("Showing the 10 highest-priority inspections. Filter or search the queue below for other records.")

        st.subheader("Repair follow-up queue")
        filter_column, search_column = st.columns([1, 2])
        with filter_column:
            selected_priority = st.selectbox(
                "Filter by priority",
                ["All", "High", "Medium", "Low"],
            )
        with search_column:
            road_search = st.text_input(
                "Find a road or segment",
                placeholder="Search inspection names",
            )

        filtered_roads = list(enumerate(ranked_roads, start=1))
        if selected_priority != "All":
            filtered_roads = [
                (position, road) for position, road in filtered_roads
                if priority_label(inspection_priority(road)).split(" ", maxsplit=1)[-1]
                == selected_priority
            ]
        if road_search.strip():
            query = road_search.strip().casefold()
            filtered_roads = [
                (position, road) for position, road in filtered_roads
                if query in str(road.get("road", "")).casefold()
                or query in str(road.get("video", "")).casefold()
            ]

        queue_display = [
            {
                "Queue position": index,
                "Road / segment": road.get("road", "Unknown"),
                "Priority": priority_label(inspection_priority(road)),
                "Worst RHI": round(float(road.get("worst_rhi", 0.0)), 1),
                "Road Damage Percentage (%)": round(float(road.get("rdp_percent", float(road.get("damage_coverage", 0.0)) * 100)), 2),
                "Unique potholes": (
                    int(road["unique_potholes"])
                    if road.get("unique_potholes") is not None
                    else "Not tracked"
                ),
                "Frames analyzed": int(road.get("frames_analyzed", 0)),
            }
            for index, road in filtered_roads
        ]
        if queue_display:
            st.dataframe(queue_display, width="stretch", hide_index=True)
        else:
            st.info("No inspections match the selected filters.")

        st.subheader("Recent inspections")
        latest_display = [
            {
                "Road / segment": road.get("road", "Unknown"),
                "Video": road.get("video", ""),
                "Priority": priority_label(inspection_priority(road)),
                "Average RHI": round(float(road.get("average_rhi", road.get("worst_rhi", 0.0))), 1),
                "Worst RHI": round(float(road.get("worst_rhi", 0.0)), 1),
                "Unique potholes": (
                    int(road["unique_potholes"])
                    if road.get("unique_potholes") is not None
                    else "Not tracked"
                ),
                "Frames analyzed": int(road.get("frames_analyzed", 0)),
            }
            for road in reversed(inspections[-10:])
        ]
        st.dataframe(latest_display, width="stretch", hide_index=True)
