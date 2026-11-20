# GeoVision Road Surface Intelligence
## IEEE-Style Integrated System Architecture

The diagram below separates the implemented prototype path from report-level extension points. Solid arrows are runtime data flow; dashed arrows are optional configuration or planned interfaces.

```mermaid
flowchart TB
    %% ================================================================
    %% External environment and acquisition
    %% ================================================================
    subgraph EXT[External Environment]
        CAM[Dashboard camera / UAV camera]
        IMG[Road image]
        VID[Recorded road video]
        META[Optional metadata\nsegment name, GPS, timestamp]
        CAL[Optional calibration\nmetres per pixel]
        AUTH[Road-maintenance user]
    end

    %% ================================================================
    %% Offline dataset and model lifecycle
    %% ================================================================
    subgraph TRAIN[Offline Dataset and Model Lifecycle]
        DATASET[Paired road images and masks]
        AUDIT[Dataset audit]
        PREP[Dataset preparation\nYOLO boxes + train/val/test split]
        DETTRAIN[EC-YOLO training]
        SEGTRAIN[HARP-UNet training]
        DETCKPT[(EC-YOLO checkpoint)]
        SEGCKPT[(HARP-UNet checkpoint)]
    end

    %% ================================================================
    %% Application boundary
    %% ================================================================
    subgraph APP[GeoVision Road Surface Intelligence - Streamlit Application]
        UI[Input and visualization interface\nImage tab | Video tab | configuration]
        VIDEOCLI[Recorded-video CLI\nrun_video.py]
        ROUTER[Analysis controller\nrun_analysis]
        CACHE[Model resource cache\ncheckpoint loading and device selection]

        subgraph PRE[Preprocessing and Data Preparation]
            DECODE[Decode uploaded media\nBGR image / sampled video frame]
            BEV[Optional calibrated IPM / BEV warp\nclockwise source points -> bird's-eye frame]
            NORMALIZE[Resize and normalize\nRGB, [0,1], 640 x 640 detector input]
        end

        subgraph DET[Detection Subsystem]
            EC[EC-YOLO detector\nMBC backbone | SPPF | C2PSA | PAN/FPN | CoordAtt]
            DEC[Prediction decoder and NMS\nconfidence filtering]
        end

        subgraph ROI[ROI and Segmentation Subsystem]
            BOX[Bounding boxes + confidence\nabsolute image coordinates]
            EXTRACT[Margin-expanded ROI extraction\n5 percent context padding]
            LB[Resize ROI to 128 x 128\nmatching ROI training]
            HARP[HARP-UNet segmenter\nresidual encoder | pyramid context | attention | decoder]
            MASK[Sigmoid + threshold\nbinary pothole mask]
            RESTORE[Resize mask back to ROI\nand paste into full frame]
        end

        subgraph SCORE[Damage Quantification and Decision Support]
            AREA[Area estimation\npixel area | calibrated physical area]
            FEATURES[Damage features\ncoverage | equivalent diameter | count | confidence]
            HEALTH[Road health assessment\nhealth score / RHI-like score]
            PRIORITY[Repair priority\ncondition and priority class]
            OUTPUT[Structured inspection result\nvisual overlay + metrics]
        end

        subgraph VIDEO[Video Processing Path]
            SAMPLE[Frame sampling]
            AGG[Frame-wise result aggregation\nannotated output video]
        end
    end

    %% ================================================================
    %% Persistence and future deployment boundary
    %% ================================================================
    subgraph EXTENSION[Report-Level Extension Interfaces]
        HIST[(Inspection history store\nSQLite or service database)]
        PRED[Temporal deterioration model\nrequires repeated segment inspections]
        RANK[Geo-spatial road ranking\nRHI / RPS queue]
        DASH[Government maintenance dashboard\nFlask or production web frontend]
        REVIEW[Human verification and engineering review]
    end

    %% Offline model lifecycle
    DATASET --> AUDIT --> PREP
    PREP --> DETTRAIN --> DETCKPT
    PREP --> SEGTRAIN --> SEGCKPT
    DETCKPT -. loaded at startup .-> CACHE
    SEGCKPT -. loaded at startup .-> CACHE

    %% Acquisition and UI
    CAM --> IMG
    CAM --> VID
    IMG --> UI
    VID --> UI
    META --> UI
    CAL --> UI
    AUTH --> UI
    UI --> ROUTER
    VID --> VIDEOCLI
    VIDEOCLI --> ROUTER
    ROUTER --> CACHE
    ROUTER --> DECODE
    VID --> SAMPLE --> DECODE
    IMG --> DECODE
    DECODE --> BEV --> NORMALIZE

    %% Detector selection
    NORMALIZE --> EC
    NORMALIZE --> HF
    EC --> DEC
    HF --> DEC
    DEC --> BOX
    BOX --> EXTRACT

    %% Segmentation and restoration
    EXTRACT --> LB --> HARP --> MASK --> RESTORE
    BOX --> RESTORE
    RESTORE --> AREA
    RESTORE --> FEATURES
    BOX --> FEATURES
    AREA --> FEATURES
    FEATURES --> HEALTH
    FEATURES --> PRIORITY
    HEALTH --> OUTPUT
    PRIORITY --> OUTPUT
    OUTPUT --> UI

    %% Video path
    OUTPUT --> AGG
    AGG --> UI

    %% Optional production extensions
    OUTPUT -. persist inspection .-> HIST
    HIST -. historical features .-> PRED
    HEALTH -. current RHI .-> PRED
    PRED -. predicted RHI / risk .-> RANK
    OUTPUT -. current metrics .-> RANK
    RANK -. ranked maintenance queue .-> DASH
    HIST -. inspection history .-> DASH
    DASH -. authorized review .-> AUTH
    OUTPUT -. high-cost decisions .-> REVIEW
    REVIEW -. confirmation / correction .-> HIST

    %% Styling: restrained publication palette
    classDef external fill:#eef4f7,stroke:#405a64,color:#17252b,stroke-width:1px
    classDef app fill:#f7f9fa,stroke:#1f4e5f,color:#17252b,stroke-width:1px
    classDef model fill:#e7f1f4,stroke:#1f7185,color:#10252c,stroke-width:1px
    classDef decision fill:#fff4dc,stroke:#9a6a00,color:#302300,stroke-width:1px
    classDef extension fill:#f4f0f8,stroke:#6a4c83,color:#251a30,stroke-width:1px,stroke-dasharray: 5 5
    classDef datastore fill:#f4f0f8,stroke:#6a4c83,color:#251a30,stroke-width:1px

    class CAM,IMG,VID,META,CAL,AUTH external
    class DATASET,AUDIT,PREP,DETTRAIN,SEGTRAIN app
    class DETCKPT,SEGCKPT datastore
    class UI,VIDEOCLI,ROUTER,CACHE,DECODE,BEV,NORMALIZE,SAMPLE,AGG,BOX,EXTRACT,LB,MASK,RESTORE,OUTPUT app
    class EC,HF,HARP,DEC model
    class AREA,FEATURES,HEALTH,PRIORITY decision
    class PRED,RANK,DASH,REVIEW extension
    class HIST datastore
```

## Interface and Data Contract Summary

| Boundary | Input | Output | Status |
|---|---|---|---|
| Application input | Image/video, optional calibration | Decoded BGR frame(s) | Implemented |
| Detector adapter | Normalized `3 x 640 x 640` tensor | `(x1, y1, x2, y2, confidence)` | Implemented |
| ROI adapter | Full-frame image and detection box | Margin-expanded crop and box metadata | Implemented |
| HARP-UNet | Resized `3 x 128 x 128` ROI | Binary ROI mask | Implemented |
| Mask restoration | ROI mask and source coordinates | Full-frame binary union mask | Implemented |
| Quantification | Mask, optional metres-per-pixel scale | Pixel area, optional `m^2`, equivalent diameter | Implemented |
| Health decision | Pothole count, mask area, image area | Health score, condition, repair priority | Implemented |
| IPM / BEV | Four ordered source-image points and output size | Perspective-normalized frame | Implemented when configured |
| History / prediction | Repeated segment inspections | Future RHI and deterioration risk | Report-level extension |
| Geo-spatial dashboard | Stored inspections and metadata | Ranked maintenance queue | Report-level extension |

## IEEE Presentation Notes

- Use the diagram as the system-level figure and keep the interface table as its companion caption material.
- Rename the figure in the report as **Fig. X. Integrated architecture of the GeoVision road-surface intelligence system**.
- The current health implementation is a prototype decision score. It should not be labelled as an official Pavement Condition Index or as a measured engineering standard.
- Metric area is emitted only when calibration is supplied; otherwise the output remains in pixel units.
- BEV is implemented as an optional calibrated preprocessing step. With no four-point calibration, the input frame passes unchanged to the detector.
- Dashed extension blocks preserve the report vision without claiming that Flask, SQLite, temporal prediction, or government ranking are already present in this repository.
