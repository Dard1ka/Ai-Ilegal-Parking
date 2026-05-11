# 🅿️ Illegal Parking Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Version-1.0_(Prototype)-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Active_Development-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFAA?style=flat-square"/>
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/React-18.3-61DAFB?style=flat-square&logo=react&logoColor=black"/>
  <img src="https://img.shields.io/badge/TailwindCSS-3.4-38BDF8?style=flat-square&logo=tailwindcss&logoColor=white"/>
  <img src="https://img.shields.io/badge/TTS-Edge_TTS_(id--ID)-0078D4?style=flat-square&logo=microsoft&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
</p>

<p align="center">
  <b>Real-time AI-powered illegal parking detection with live dashboard, multi-zone ROI, and Indonesian TTS alerts.</b><br/>
  Detects, tracks, and alerts on vehicles parked in restricted zones using YOLOv8 object detection and BotSORT multi-object tracking.
</p>

> [!NOTE]
> **v1.0 — Prototype Release**
> Ini adalah versi pertama (v1.0) dari sistem deteksi parkir liar berbasis AI. Seluruh komponen (backend, frontend dashboard, Docker deployment) dikelola dalam satu repository (monorepo). Versi ini berfungsi sebagai **proof of concept** dan fondasi untuk pengembangan selanjutnya. Fitur, arsitektur, dan performa dapat berubah signifikan di versi mendatang.
>
> **Roadmap versi berikutnya:**
> - v1.1 — Optimasi performa model & tracking accuracy
> - v1.2 — Database logging & historical analytics
> - v2.0 — Multi-camera support, cloud deployment, notification system

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Vehicle Detection** | YOLOv8 detects car, bus, truck, motorcycle with 96px input resolution |
| 🔄 **Multi-Object Tracking** | BotSORT tracker with logical ID re-linking across occlusions (up to 80s) |
| 🗺️ **Multi-Zone ROI** | Draw multiple no-parking polygon zones via interactive dashboard |
| ⚠️ **3-Stage Alert System** | Monitoring → Warning (60s) → Violation (300s) |
| 🔺 **Safety Triangle Detection** | Secondary YOLOv8 model detects warning triangles (breakdown/emergency exception) |
| 🎨 **Vehicle Color Classification** | HSV-based color detection (12 colors) for precise vehicle identification |
| 🔊 **Indonesian TTS Alerts** | Edge-TTS with `id-ID-GadisNeural` voice, non-blocking playback via threading |
| 📡 **Real-time Dashboard** | React dashboard with MJPEG stream + WebSocket alerts |
| 📷 **Video & Live Camera** | Supports MP4/AVI/MOV/MKV files and live webcam/IP cameras |
| 👻 **Occlusion Handling** | Vehicles temporarily hidden behind other objects stay tracked |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (React + Vite)                   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ SetupWizard │  │ VideoMonitor │  │  AlertSidebar    │   │
│  │ (3-step     │  │ (MJPEG       │  │  (WebSocket      │   │
│  │  onboarding)│  │  stream)     │  │   real-time)     │   │
│  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘   │
└─────────┼────────────────┼───────────────────┼─────────────┘
          │ POST /configure │ GET /video_feed   │ WS /ws/alerts
          │ POST /upload    │                   │
          │ POST /set_roi   │                   │
          ▼                 ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                   BACKEND (FastAPI)                          │
│  ┌──────────────┐  ┌─────────────────────────────────────┐  │
│  │  SharedState │  │         VideoProcessor               │  │
│  │  (thread-    │←→│  ┌──────────┐  ┌─────────────────┐  │  │
│  │   safe)      │  │  │ YOLOv8   │  │  BotSORT        │  │  │
│  └──────────────┘  │  │ Vehicle  │  │  Tracker +      │  │  │
│                    │  │ Detection│  │  Re-linking      │  │  │
│                    │  └──────────┘  └─────────────────┘  │  │
│                    │  ┌──────────┐  ┌─────────────────┐  │  │
│                    │  │ YOLOv8   │  │  HSV Color      │  │  │
│                    │  │ Triangle │  │  Classifier     │  │  │
│                    │  │ Check    │  │  (12 colors)    │  │  │
│                    │  └──────────┘  └─────────────────┘  │  │
│                    │  ┌──────────────────────────────┐    │  │
│                    │  │  Edge-TTS + Pygame (Audio)   │    │  │
│                    │  └──────────────────────────────┘    │  │
│                    └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Detection Pipeline (per 1-second tick)

```
Frame → YOLO Track → Filter by ROI → Assignment Logic
     → Stationary Check (3s) → Parked State
     → 60s: Triangle Check → Warning
     → 300s: Color Classify → TTS Alert → Violation
```

---

## 🚀 Quick Start

### Option A — Standalone Script (no dashboard)

```bash
# 1. Clone & install
git clone https://github.com/YOUR_USERNAME/illegal-parking-detection.git
cd illegal-parking-detection
pip install -r requirements.txt

# 2. Place model files (see Models section)
# best.pt → vehicle detection
# triangle.pt → warning triangle detection

# 3. Run with environment check
python demo_quickstart.py --video input4.mp4
```

### Option B — Full Dashboard (recommended)

```bash
# Terminal 1: Backend
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd parking-dashboard
npm install
npm run dev
# Open http://localhost:5173
```

---

## 📋 Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.9+ | Tested on 3.10, 3.11 |
| Node.js | 18+ | For dashboard frontend |
| CUDA (optional) | 11.8+ | GPU acceleration for YOLO |
| RAM | 8GB+ | 16GB recommended |
| GPU VRAM | 4GB+ | Optional, runs on CPU |

---

## 🧩 Project Structure (Monorepo)

Repository ini menggunakan arsitektur **monorepo** — seluruh komponen (backend AI, frontend dashboard, konfigurasi Docker) berada dalam satu repository untuk mempermudah development dan deployment pada tahap prototyping.

```
illegal-parking-detection/
│
├── README.md                 # Dokumentasi utama (file ini)
├── requirements.txt          # Python dependencies
├── .gitignore                # File/folder yang tidak di-track git
├── LICENSE                   # MIT License
│
├── demo3_tts.py              # Standalone detection script (algoritma utama)
├── demo_quickstart.py        # Quick-start wrapper dengan environment check
├── best.pt                   # YOLOv8 vehicle detection model *
├── triangle.pt               # YOLOv8 warning triangle model *
│
├── backend/                  # FastAPI web backend (lihat backend/README.md)
│   ├── __init__.py
│   ├── main.py               # API endpoints & lifespan management
│   ├── processor.py          # VideoProcessor — inti deteksi & tracking
│   └── shared.py             # Thread-safe SharedState antar thread
│
├── parking-dashboard/        # React frontend (lihat parking-dashboard/README.md)
│   ├── src/
│   │   ├── App.jsx           # Root: state machine setup ↔ dashboard
│   │   ├── index.css         # Design system & animasi
│   │   ├── components/
│   │   │   ├── SetupWizard.jsx   # Onboarding wizard 3 langkah
│   │   │   ├── VideoMonitor.jsx  # MJPEG stream viewer + ROI editor
│   │   │   └── AlertSidebar.jsx  # Feed alert real-time
│   │   └── hooks/
│   │       └── useAlerts.js  # WebSocket hook dengan auto-reconnect
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.js
│
├── docker/                   # Docker deployment (lihat docker/README.md)
│   ├── backend/Dockerfile
│   ├── frontend/Dockerfile
│   ├── frontend/nginx.conf
│   ├── docker-compose.yml
│   └── .env.example
│
├── docs/                     # Dokumentasi tambahan & diagram
├── outputs/                  # Output video & TTS temp (gitignored)
└── uploads/                  # Video upload dari dashboard (gitignored)
```

> **\*** Model weights (`.pt`) tidak disertakan di repository karena ukurannya besar.  
> Download dari [Releases](../../releases) atau train sendiri menggunakan notebook yang tersedia.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload video file (MP4, AVI, MOV, MKV, WebM) |
| `POST` | `/configure` | Set source (video/camera) and ROI mode (manual/auto) |
| `GET` | `/first_frame` | Get first frame as JPEG for ROI drawing |
| `POST` | `/set_roi` | Set multi-polygon no-parking zones |
| `GET` | `/video_feed` | MJPEG video stream |
| `WS` | `/ws/alerts` | Real-time alert + stats WebSocket |
| `GET` | `/status` | Processor status + current stats |
| `GET` | `/docs` | Auto-generated FastAPI Swagger UI |

### ROI Payload Format

```json
POST /set_roi
{
  "rois": [
    [[0.1, 0.2], [0.5, 0.2], [0.5, 0.8], [0.1, 0.8]],
    [[0.6, 0.3], [0.9, 0.3], [0.9, 0.7], [0.6, 0.7]]
  ]
}
```
Coordinates are normalized `[0, 1]` relative to frame dimensions.

### Alert WebSocket Payload

```json
{
  "alerts": [
    {
      "id": 1,
      "lid": "LID-01",
      "type": "car",
      "color": "White",
      "zone": 1,
      "duration": 312,
      "status": "violation",
      "triangle": false
    }
  ],
  "stats": {
    "total": 3,
    "warning": 1,
    "violation": 1,
    "safe": 1
  }
}
```

---

## ⚙️ Configuration

Key parameters in `demo3_tts.py` / `backend/processor.py`:

```python
# ── Timing rules ─────────────────────────────────
PARKING_CHECK_SECONDS   = 60    # Seconds before warning
ILLEGAL_PARKING_SECONDS = 300   # Seconds before violation

# ── Detection ────────────────────────────────────
VEHICLE_CONF  = 0.25   # Vehicle detection confidence threshold
TRIANGLE_CONF = 0.20   # Triangle detection confidence threshold
VEHICLE_IMGSZ = 960    # Input resolution for vehicle model

# ── Occlusion handling ───────────────────────────
OCCLUSION_TTL_SECONDS = 80     # Max seconds to keep hidden vehicle record
RELINK_MAX_DISTANCE   = 350    # Max pixel distance for re-linking

# ── TTS ──────────────────────────────────────────
ENABLE_TTS      = True
EDGE_TTS_VOICE  = "id-ID-GadisNeural"
TTS_REPEAT_GAP  = 20           # Seconds between repeated alerts
```

---

## 🎯 Alert Status Logic

```
Vehicle enters ROI
  │
  ├─ 0s – 3s          → Stationary check (noise filter)
  ├─ 3s – 60s         → 🟡 MONITORING  
  ├─ 60s – 300s       → 🟠 WARNING (check for safety triangle)
  │     └─ Triangle detected → ✅ SAFE (emergency/breakdown)
  └─ > 300s (no triangle) → 🔴 VIOLATION + TTS alert
```

---

## 🧠 Technical Highlights

### Re-linking Algorithm
When a tracked vehicle gets a new ID from BotSORT (e.g., after partial occlusion), the system re-links the new detection to the existing `ParkedRecord` using a multi-pass scoring:
1. **Pass 1**: Prioritize hidden records with `parked_since` (long-term parked vehicles)
2. **Pass 2**: Fallback to recently-hidden non-parked records
3. **Score function**: `IoU × 500 − center_dist × 1.2 − foot_dist × 2.2 + parked_duration × 8`

### Color Classification
HSV-based pixel voting without ML overhead:
- Resize crop to 96×96
- Filter low-value pixels (shadow/dark)
- Classify each pixel into 12 color buckets
- Return majority color + confidence %

### Occlusion Handling
Vehicles hidden for up to 80 seconds maintain their full state (`ParkedRecord`) including parking duration, triangle status, and color. When they reappear, a grace period of 4 seconds prevents false motion resets.

---

## 📊 Model Information

| Model | Architecture | Task | Input Size |
|---|---|---|---|
| `best.pt` | YOLOv8 (custom) | Vehicle detection | 960px |
| `triangle.pt` | YOLOv8n (custom) | Warning triangle | 640px |

Both models were trained on custom datasets. See `VehicleDetection.ipynb` and `WarningTriangleYOLOv8.ipynb` for training details.

---

## 📸 Screenshots

> *Add screenshots here after deployment. Recommended: SetupWizard, Dashboard with active violations, AlertSidebar with mixed statuses.*

---

## 🛠️ Development

```bash
# Backend (hot reload)
uvicorn backend.main:app --reload --port 8000

# Frontend (hot reload)
cd parking-dashboard && npm run dev

# Standalone (no dashboard)
python demo_quickstart.py --video input4.mp4 --no-tts
```

---

## 🐳 Docker Deployment

Seluruh sistem bisa dijalankan via Docker Compose tanpa setup manual:

```bash
cd docker
cp .env.example .env
docker compose up --build

# Dashboard : http://localhost:5173
# API       : http://localhost:8000
# API docs  : http://localhost:8000/docs
```

Lihat [`docker/README.md`](docker/README.md) untuk detail konfigurasi, GPU support, dan troubleshooting.

---

## 📂 Dokumentasi Per Komponen

Setiap komponen utama memiliki README tersendiri dengan penjelasan detail:

| Komponen | Dokumentasi | Deskripsi |
|---|---|---|
| Backend API | [`backend/README.md`](backend/README.md) | Arsitektur FastAPI, modul processor & shared state, endpoint reference |
| Frontend Dashboard | [`parking-dashboard/README.md`](parking-dashboard/README.md) | Komponen React, flow UI, design system |
| Docker Setup | [`docker/README.md`](docker/README.md) | Docker Compose, volume mount, GPU support, TTS config |

---

## 🗓️ Versioning & Changelog

| Versi | Tanggal | Deskripsi |
|---|---|---|
| **v1.0** | 2025 | Prototype awal — deteksi + tracking + dashboard + TTS. Monorepo single deployment. |
| v1.1 | _Planned_ | Optimasi model accuracy, performance tuning |
| v1.2 | _Planned_ | Database logging, historical analytics |
| v2.0 | _Planned_ | Multi-camera, cloud deployment, push notification |

> Versi ini (v1.0) merupakan bagian dari penelitian skripsi di BINUS University.
> Arsitektur dan fitur dapat berubah signifikan di versi mendatang.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection framework
- [Microsoft Edge TTS](https://github.com/rany2/edge-tts) — Indonesian TTS voice synthesis
- [FastAPI](https://fastapi.tiangolo.com/) — High-performance async API framework
- [BotSORT](https://github.com/NirAharon/BoT-SORT) — Multi-object tracking algorithm
- [React](https://react.dev/) — Frontend UI library
- [Tailwind CSS](https://tailwindcss.com/) — Utility-first CSS framework
- [Vite](https://vitejs.dev/) — Next-generation frontend build tool

---

<p align="center">
  Built with ❤️ for smarter urban parking enforcement<br/>
  <sub>v1.0 Prototype — Skripsi BINUS University 2025</sub>
</p>
