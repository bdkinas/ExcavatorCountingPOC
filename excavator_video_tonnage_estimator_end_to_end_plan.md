# Excavator Video Tonnage Estimator – End‑to‑End Plan

---
## 1. Objective
Design and deploy an application that ingests MP4 video of an excavator in operation and outputs an accurate estimate of total tonnage excavated over time.

---
## 2. High‑Level Architecture
1. **Ingestion Service** – Accepts MP4 video uploads (web UI, REST endpoint, or S3 drop).
2. **Pre‑Processing Pipeline** – Uses FFmpeg/OpenCV to extract frames and metadata.
3. **Detection & Tracking Module** – Deep‑learning model (YOLOv8 + ByteTrack) detects and tracks:
   - Excavator bucket
   - Bucket state (empty / filled / dumping)
4. **Cycle & State Analysis** – Identifies dig→lift→dump cycles; determines fill ratio.
5. **Volume Estimator** – Estimates bucket volume per cycle:
   - Option A: Fixed bucket geometry × fill ratio
   - Option B: Instance segmentation (Mask R‑CNN) + monocular depth estimation for partial fills
6. **Tonnage Calculator** – Volume × material density (user‑supplied or lookup table) = mass per cycle; aggregate totals.
7. **Result Store & API** – Persist results (PostgreSQL); expose via REST/GraphQL.
8. **Dashboard / Reporting UI** – React‑based frontend with live metrics and historical reports.
9. **Orchestration & Monitoring** – Dockerised micro‑services on AWS ECS Fargate (or Kubernetes), with CloudWatch & Prometheus.

---
## 3. Component Details
### 3.1 Ingestion Service
- Handles authentication, file validation, virus scan
- Pushes job metadata to message queue (AWS SQS / Kafka)

### 3.2 Pre‑Processing Pipeline
- Extract ~5 FPS keyframes (configurable)
- Store frames in /tmp or S3 for stateless processing

### 3.3 Detection & Tracking
- Model: YOLOv8‑x pre‑trained on COCO, fine‑tuned with 3k labelled frames of excavators
- Tracker: ByteTrack for ID consistency across frames
- Outputs: bounding box, object ID, confidence, frame ID

### 3.4 Bucket State Classifier
- Light CNN head attached to YOLO to classify each detection into {empty, loaded, dumping}
- Training data: 1k images per class; use focal loss to balance

### 3.5 Cycle Detection Logic
- Define state machine per bucket ID:
  1. **Digging** (start): loaded state enters scene below ground plane
  2. **Loaded Transport**: loaded state above ground plane and moving
  3. **Dump**: dumping state detected; end of cycle
- Record `t_start`, `t_end`, fill_ratio

### 3.6 Volume & Density Estimation
- **Calibration**: one‑time extrinsic calibration using Aruco board placed near excavation area → pixel‑to‑meter scale
- **Fill Ratio**: from classifier probability or segmentation mask area ÷ maximum bucket area
- **Volume**: `bucket_capacity [m³] × fill_ratio`
- **Density**: choose by material type (sand, clay, gravel) entered by user
- **Mass**: `volume × density` → convert to tonnes (÷ 1000)

### 3.7 Data Store & API
- PostgreSQL (TimescaleDB extension) for time‑series tonnage/cycle data
- FastAPI backend exposes `/jobs`, `/metrics`, `/cycles` endpoints

### 3.8 Dashboard UI
- React + Recharts for plots (cycles per hour, cumulative tonnage)
- WebSocket for live feed; export CSV/PDF reports

### 3.9 Infrastructure
- **Compute**: GPU T4 spot instances for model inference
- **Storage**: S3 for raw video & frames; RDS/Postgres for structured data
- **CI/CD**: GitHub Actions → ECR → ECS blue‑green deployments

---
## 4. Technology Stack
| Layer | Choice | Notes |
|-------|--------|-------|
| CV / ML | PyTorch 2 + YOLOv8, TorchVision, Detectron2 | CUDA 12; export to ONNX/TensorRT for prod |
| Orchestration | Docker, AWS ECS Fargate (alt: K8s) | One service per module |
| Messaging | AWS SQS | Decoupled, durable |
| Backend | FastAPI | Async I/O, Pydantic models |
| Frontend | React + Vite | Tailwind CSS, Recharts |
| Monitoring | CloudWatch, Prometheus, Grafana | Latency & GPU utilisation |

---
## 5. Data & Labeling
1. Collect ≥ 10 hrs of varied excavator footage (angles, lighting, materials)
2. Label frames in CVAT:
   - Bucket bounding box & segmentation mask
   - State tags (empty/full/dumping)
3. Split 70 / 15 / 15 (train/val/test)
4. Augment: random brightness, motion blur, dust overlay

---
## 6. Implementation Phases & Timeline (≈ 16 weeks)
| Phase | Duration | Key Deliverables |
|-------|----------|-----------------|
| **P0 – Setup** | 1 wk | Repo, CI/CD, cloud infra scaffold |
| **P1 – Data & Models** | 4 wks | Labeled dataset, trained detector & classifier, evaluation report (> 90 mAP, 95 cycle detect F1) |
| **P2 – Core Pipeline** | 4 wks | Frame extractor, inference service, cycle logic, volume computation |
| **P3 – API & Dashboard MVP** | 3 wks | REST endpoints, React dashboard, live charts |
| **P4 – Calibration & Accuracy** | 2 wks | Aruco‑based scaling, density UI input, < ±7 % tonnage error |
| **P5 – Hardening & Deploy** | 2 wks | Load tests, auto‑scaling, alerting |

---
## 7. Validation & Metrics
- **Detection**: mAP50 ≥ 0.90
- **Cycle Detection**: F1 ≥ 0.95 vs manually counted cycles
- **Tonnage**: ≤ ±7 % vs weighbridge reference over ≥ 50 cycles

---
## 8. Risks & Mitigations
| Risk | Mitigation |
|------|-----------|
| Camera angle obstructed | Enforce mounting guidelines; multi‑camera fusion |
| Variable bucket sizes | Parameterise bucket capacity per machine ID |
| Material density variance | Allow real‑time density input or integrate moisture sensor |
| Low‑light/night | Train on IR imagery or add floodlights |

---
## 9. Future Enhancements
- Stereo or LiDAR integration for direct volume reconstruction
- On‑edge inference with NVIDIA Jetson for offline sites
- Predictive analytics: cycle time optimisation, operator benchmarking
- Integration with fleet management (Cat VisionLink, Komatsu Smart Construction)

---
## 10. Next Steps
1. Confirm hardware & cloud budget
2. Secure sample videos & bucket specs
3. Kick‑off data collection & labeling sprint
4. Revisit plan after P1 demo for adjustments

