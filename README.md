# Equipment Utilization & Activity Classification System

Real-time, microservices-based pipeline for analysing construction equipment in video footage.

---

That's it. Docker handles Kafka, TimescaleDB, the CV service, and the Streamlit UI.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         Video Source                         │
│               (file on disk  or  RTSP stream)                │
└────────────────────────────┬─────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   CV Service    │
                    │  (Python/OpenCV)│
                    │                 │
                    │  YOLOv8 detect  │
                    │  ByteTrack      │
                    │  Optical Flow   │
                    │  Activity Class │
                    └────────┬────────┘
                             │ JSON events
                    ┌────────▼────────┐
                    │   Apache Kafka  │
                    │ topic: equip-   │
                    │        events   │
                    └────────┬────────┘
                             │
               ┌─────────────┴──────────────┐
               │                            │
      ┌────────▼────────┐        ┌──────────▼────────┐
      │ Consumer Service│        │   (future)         │
      │  Kafka → DB     │        │  Alert Service     │
      └────────┬────────┘        └────────────────────┘
               │
      ┌────────▼────────┐
      │  TimescaleDB    │
      │  (PostgreSQL)   │
      └────────┬────────┘
               │
      ┌────────▼────────┐
      │ Streamlit UI    │
      │  localhost:8501 │
      └─────────────────┘
```

### Services

| Service | Technology | Role |
|---|---|---|
| `cv_service` | Python, OpenCV, YOLOv8, confluent-kafka | Video analysis + Kafka producer |
| `consumer_service` | Python, psycopg2, confluent-kafka | Kafka consumer + DB writer |
| `ui_service` | Streamlit, pandas | Live dashboard |
| `kafka` | Confluent Platform (KRaft) | Message broker |
| `timescaledb` | TimescaleDB (PostgreSQL 16) | Time-series storage |

---

## Technical Write-up: Design Decisions

### Problem 1 — Articulated Motion (The Core Challenge)

**The problem:** An excavator's arm can be actively digging while its tracks are
completely stationary.  A naive approach — "did the bounding-box region change
between frames?" — would wrongly label this machine as INACTIVE.

**Our solution: Region-Based Optical Flow**

We split each bounding box into two horizontal halves:

```
 ┌─────────────────────┐
 │   UPPER  (0-50%)    │  ← cab, boom, arm, bucket
 ├─────────────────────┤
 │   LOWER  (50-100%)  │  ← undercarriage, tracks, wheels
 └─────────────────────┘
```

Dense optical flow (Farneback algorithm) is computed independently on each region.
The `motion_source` field in the Kafka payload reflects which region(s) are active:

| `motion_source` | Meaning |
|---|---|
| `full_body` | Both regions moving — machine is driving + working |
| `arm_only` | Only upper region — arm digging, tracks still (typical for stationary excavating) |
| `tracks_only` | Only lower region — machine repositioning but arm idle |
| `none` | Neither region — machine is genuinely idle |

A machine is considered **ACTIVE** if *any* region exceeds the motion threshold.
This correctly handles the arm-only scenario.

**Why Farneback over frame-differencing?**

Frame-differencing is fast but produces binary masks that merge foreground/background
noise and lose directional information (dx, dy) that we need for activity classification.
Farneback gives us per-pixel velocity vectors, which are the input to the classifier.

**Why not keypoint tracking or instance segmentation?**

Both are more accurate but have trade-offs at prototype scale:
- Keypoint tracking (e.g., ViTPose) requires a fine-tuned model for construction machinery — there are no public pretrained weights for excavator keypoints.
- Instance segmentation (Mask R-CNN / YOLOv8-seg) is ~3× slower per frame with limited benefit for the utilisation-state binary decision.

Region-based flow is a good first-principles approach that works on CPU in real-time
and degrades gracefully on low-quality video.

---

### Problem 2 — Activity Classification

**Without a labelled training set** for this specific domain, we use geometric
heuristics derived from the optical flow direction in the upper (arm) region:

```
 Upper region mean flow vector (dx, dy):

   dy < 0  (upward)   + dx != 0  →  DUMPING   (arm lifting and rotating)
   dy > 0  (downward)             →  DIGGING   (arm pressing into ground)
   |dx| >> |dy|                   →  SWINGING  (lateral swing to drop zone)
   magnitude < threshold          →  WAITING
```

For dump trucks, the vocabulary simplifies to DUMPING (bed raising = arm_only motion) or SWINGING_LOADING (driving to drop zone).

**In production** this should be replaced by a sequence classifier:
- Collect 30-frame clips with activity labels
- Extract per-frame motion vectors (upper dx, upper dy, upper mag, lower mag)
- Train a small GRU (≈50k parameters) — generalises well with limited data
- Expected accuracy: 85-92% on held-out footage from the same site

---

### Kafka Payload Format

```json
{
  "frame_id": 450,
  "equipment_id": "EX-001",
  "equipment_class": "excavator",
  "timestamp": "00:00:15.000",
  "utilization": {
    "current_state":    "ACTIVE",
    "current_activity": "DIGGING",
    "motion_source":    "arm_only"
  },
  "time_analytics": {
    "total_tracked_seconds": 15.0,
    "total_active_seconds":  12.5,
    "total_idle_seconds":    2.5,
    "utilization_percent":   83.3
  },
  "bbox": [120, 80, 540, 460]
}
```

The `equipment_id` key is used as the Kafka partition key, so all events for
the same machine land on the same partition in arrival order.

---

### TimescaleDB Schema

```sql
equipment_events  (hypertable, partitioned by recorded_at)
  ├── recorded_at              TIMESTAMPTZ   -- wall clock time of DB insert
  ├── equipment_id             TEXT
  ├── current_state            TEXT          -- ACTIVE | INACTIVE
  ├── current_activity         TEXT          -- DIGGING | SWINGING_LOADING | DUMPING | WAITING
  ├── motion_source            TEXT          -- full_body | arm_only | tracks_only | none
  ├── utilization_percent      DOUBLE
  └── ...time analytics fields...

equipment_utilization_1min  (continuous aggregate)
  └── 1-minute rollups for dashboard charts
```

Raw events are retained for 7 days; the continuous aggregate is kept indefinitely.

---

### Trade-offs & Limitations

| Decision | Trade-off |
|---|---|
| YOLOv8n (nano) | Fast on CPU but lower accuracy than larger models. Swap to yolov8m.pt for better detection at the cost of ~3× more compute. |
| Farneback optical flow | Robust and no training needed, but slower than sparse Lucas-Kanade. For >25fps real-time on CPU, reduce `winsize` or increase `--skip`. |
| Heuristic classifier | Zero training data required but brittle to unusual postures. Replace with a GRU sequence model when labelled data is available. |
| Kafka KRaft (no Zookeeper) | Simpler deployment (one less service) but requires Kafka 3.3+. Works with confluentinc/cp-kafka:7.6+. |
| TimescaleDB continuous aggregate | Near-zero query cost for dashboard charts, but ~30s lag before fresh data appears in the aggregate. |

---

## Extending the System

**Plug in a fine-tuned YOLO model:**
```bash
# Download your custom weights and mount them
docker compose run cv_service python main.py --model /data/excavator_yolov8.pt
```

**Switch to a live RTSP camera:**
```yaml
# In docker-compose.yml
environment:
  VIDEO_SOURCE: "rtsp://admin:password@192.168.1.100:554/stream1"
```

**Add a new equipment class:**
1. Add class IDs to `EQUIPMENT_CLASS_MAP` in `detector.py`
2. Add prefix to `PREFIX_MAP`
3. Add activity heuristics to `activity_classifier.py`

---

## Requirements

- Docker + Docker Compose v2
- 8 GB RAM recommended (Kafka + TimescaleDB + CV are each memory-hungry)
- A GPU is optional; everything runs on CPU with frame skipping enabled

For GPU support, change the base image in `cv_service/Dockerfile` to:
```dockerfile
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
```
and set `device=0` in the YOLO model call inside `detector.py`.
