# Project Architecture Diagram

This diagram shows the implemented runtime and offline training paths for the pothole road-health project. Solid arrows are data flow; dashed arrows are model-loading or configuration relationships.

```mermaid
flowchart TB
    subgraph INPUTS[Inputs]
        IMAGE[Road image]
        VIDEO[Recorded video]
        CONFIG[Thresholds, BEV points, calibration]
    end

    subgraph INTERFACES[User Interfaces]
        STREAMLIT[Streamlit dashboard\napp.py]
        VIDEO_CLI[Video CLI\nrun_video.py]
        IMAGE_CLI[Image CLI\nrun_pipeline.py]
    end

    subgraph RUNTIME[Runtime Analysis Pipeline]
        DECODE[Decode BGR frame]
        BEV[Optional BEV / IPM warp\nsrc/bev.py]
        DETECT[Detector selection]
        EC[Custom EC-YOLO\nsrc/ec_yolo.py]
        NMS[Decode, confidence filter\nand box mapping]
        ROI[Margin-expanded ROI\nsrc/roi.py]
        LETTERBOX[Resize ROI to 128 x 128]
        HARP[HARP-UNet\nsrc/harp_unet.py]
        RESTORE[Remove padding and paste\nROI masks into full frame]
        AREA[Pixel area / optional m2\nsrc/area_health.py]
        HEALTH[Health score, condition\nand repair priority]
        OVERLAY[Annotated image or video\nplus structured metrics]
    end

    subgraph OUTPUTS[Outputs]
        IMAGE_RESULT[Annotated image]
        VIDEO_RESULT[Annotated MP4]
        METRICS[Detection, mask, area\nand health metrics]
    end

    subgraph TRAINING[Offline Dataset and Training]
        SOURCE[dataset/\nimages + segmentation masks]
        AUDIT[Dataset audit\ntools/audit_dataset.py]
        PREP[Prepared paired dataset\nand ROI crops]
        DET_TRAIN[EC-YOLO training\nrun_train_ec_yolo.py]
        SEG_TRAIN[HARP-UNet training\nrun_train_harp.py]
        DET_MODEL[(EC-YOLO checkpoint)]
        SEG_MODEL[(HARP-UNet checkpoint)]
    end

    IMAGE --> STREAMLIT
    VIDEO --> STREAMLIT
    VIDEO --> VIDEO_CLI
    IMAGE --> IMAGE_CLI
    CONFIG -.-> STREAMLIT
    CONFIG -.-> VIDEO_CLI
    STREAMLIT --> DECODE
    VIDEO_CLI --> DECODE
    IMAGE_CLI --> DECODE
    DECODE --> BEV --> DETECT
    DETECT --> EC
    DETECT --> YOLO
    EC --> NMS
    YOLO --> NMS
    NMS --> ROI --> LETTERBOX --> HARP --> RESTORE
    NMS --> RESTORE
    RESTORE --> AREA --> HEALTH --> OVERLAY
    RESTORE --> OVERLAY
    OVERLAY --> IMAGE_RESULT
    OVERLAY --> VIDEO_RESULT
    OVERLAY --> METRICS
    IMAGE_RESULT --> STREAMLIT
    VIDEO_RESULT --> STREAMLIT
    METRICS --> STREAMLIT

    SOURCE --> AUDIT --> PREP
    PREP --> DET_TRAIN --> DET_MODEL
    PREP --> SEG_TRAIN --> SEG_MODEL
    DET_MODEL -.-> EC
    SEG_MODEL -.-> HARP

    classDef input fill:#e8f1f3,stroke:#32606b,color:#14272c
    classDef interface fill:#f2eee5,stroke:#8a6a3d,color:#2e2417
    classDef runtime fill:#edf4ed,stroke:#47704d,color:#1d2d1e
    classDef model fill:#e7eef8,stroke:#41658f,color:#18263a
    classDef output fill:#fff3d8,stroke:#a87920,color:#352706
    classDef training fill:#f0ebf5,stroke:#725783,color:#2a1d31
    classDef store fill:#f4f0f8,stroke:#725783,color:#2a1d31

    class IMAGE,VIDEO,CONFIG input
    class STREAMLIT,VIDEO_CLI,IMAGE_CLI interface
    class DECODE,BEV,DETECT,NMS,ROI,LETTERBOX,RESTORE,AREA,HEALTH,OVERLAY runtime
    class EC,YOLO,HARP model
    class IMAGE_RESULT,VIDEO_RESULT,METRICS output
    class SOURCE,AUDIT,PREP,DET_TRAIN,SEG_TRAIN training
    class DET_MODEL,SEG_MODEL store
```

## Runtime sequence

1. A dashboard upload or CLI input supplies an image or video frame.
2. The pipeline optionally rectifies the frame with BEV/IPM, then runs either custom EC-YOLO or the optional Ultralytics detector.
3. Detector boxes are mapped back to source-image coordinates and expanded into ROIs.
4. Each ROI is resized to 128 x 128 for HARP-UNet, matching the ROI-training transform. The predicted binary masks are resized to the crop and restored to full-frame coordinates. Full-image inference remains available for its matching checkpoint.
5. Pixel area is always calculated. Physical area is calculated only when metres-per-pixel calibration is provided.
6. The health module produces the score, condition, and repair priority used by the dashboard or annotated output.

The offline path audits paired image/mask files, prepares YOLO and segmentation data, trains both models, and supplies their checkpoints to the runtime pipeline.