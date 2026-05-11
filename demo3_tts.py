from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
import time
import asyncio
import threading
import uuid
import os

import cv2
import numpy as np
from ultralytics import YOLO
import pygame
import edge_tts

# =========================
# CONFIG
# =========================
VIDEO_PATH = "input4.mp4"
VEHICLE_MODEL_PATH = "best.pt"
TRIANGLE_MODEL_PATH = "triangle.pt"
OUTPUT_VIDEO_PATH = "outputs/illegal_parking_result.mp4"

# ===== SOURCE MODE =====
# USE_REALTIME_CLOCK = True

# ===== DISPLAY MODE =====
DISPLAY_VIDEO_TIME = True

# ===== ROI MODE =====
USE_INTERACTIVE_MULTI_ROI = True
MANUAL_MULTI_ROI = None

# ===== Timing rules =====
PARKING_CHECK_SECONDS = 60
ILLEGAL_PARKING_SECONDS = 300

# ===== Analysis tick =====
ANALYSIS_INTERVAL_SECONDS = 1

# ===== Detection / Tracking =====
VEHICLE_IMGSZ = 960
TRIANGLE_IMGSZ = 640
VEHICLE_CONF = 0.25
TRIANGLE_CONF = 0.20
IOU = 0.50
EXPAND_MARGIN = 0.18
ONLY_VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]
TRACKER_CONFIG = "botsort.yaml"

# ===== Occlusion / persistence =====
OCCLUSION_TTL_SECONDS = 80
RELINK_MAX_DISTANCE = 350
RELINK_FOOT_DISTANCE = 220
RELINK_IOU_THRESHOLD = 0.01

# ===== Strong lock to avoid creating a new blue record
# near an existing hidden parked record
HIDDEN_PARKED_LOCK_CENTER_DISTANCE = 200
HIDDEN_PARKED_LOCK_FOOT_DISTANCE = 120

# ===== Stationary filtering =====
STATIONARY_DISTANCE_THRESHOLD = 22.0
STATIONARY_MIN_SECONDS = 3

# ===== Movement reset protection =====
MOVING_RESET_SECONDS = 4
REAPPEAR_GRACE_SECONDS = 4

# ===== Color crop trim =====
COLOR_CROP_TRIM_BOTTOM = 0.28
COLOR_CROP_TRIM_TOP = 0.08
COLOR_CROP_TRIM_LEFT = 0.08
COLOR_CROP_TRIM_RIGHT = 0.08
HEURISTIC_SAMPLE_SIZE = 96

# ===== TTS =====
ENABLE_TTS = True
TTS_REPEAT_GAP = 20

# Edge TTS voice wanita Indonesia
EDGE_TTS_VOICE = "id-ID-GadisNeural"
EDGE_TTS_RATE = "+0%"
EDGE_TTS_VOLUME = "+0%"
EDGE_TTS_PITCH = "+0Hz"
TTS_TEMP_DIR = "outputs/tts_temp"


# =========================
# HELPERS
# =========================
def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def box_center(xyxy) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def foot_point(xyxy) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, float(y2))


def point_inside_poly(point_xy: Tuple[float, float], poly_points: List[Tuple[int, int]]) -> bool:
    poly = np.array(poly_points, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(point_xy[0]), float(point_xy[1])), False) >= 0


def get_anchor_points(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    cx = (x1 + x2) / 2.0
    return [
        ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        (cx, max(y1, y2 - 2)),
        (min(x2, x1 + 2), max(y1, y2 - 2)),
        (max(x1, x2 - 2), max(y1, y2 - 2)),
    ]


def find_roi_index_for_box(xyxy, roi_list: List[List[Tuple[int, int]]]) -> Optional[int]:
    anchors = get_anchor_points(xyxy)

    best_roi = None
    best_hits = 0

    for idx, roi in enumerate(roi_list):
        hits = 0
        for pt in anchors:
            if point_inside_poly(pt, roi):
                hits += 1

        if hits > best_hits:
            best_hits = hits
            best_roi = idx

    if best_hits >= 1:
        return best_roi
    return None


def expand_xyxy(xyxy, img_w: int, img_h: int, margin: float):
    x1, y1, x2, y2 = map(float, xyxy)
    bw = x2 - x1
    bh = y2 - y1
    dx = bw * margin
    dy = bh * margin
    ex1 = clamp(int(x1 - dx), 0, img_w - 1)
    ey1 = clamp(int(y1 - dy), 0, img_h - 1)
    ex2 = clamp(int(x2 + dx), 0, img_w - 1)
    ey2 = clamp(int(y2 + dy), 0, img_h - 1)
    return ex1, ey1, ex2, ey2


def euclidean(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def bbox_iou(boxA, boxB) -> float:
    ax1, ay1, ax2, ay2 = map(float, boxA)
    bx1, by1, bx2, by2 = map(float, boxB)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def draw_all_rois(img, roi_list: List[List[Tuple[int, int]]]):
    for idx, roi in enumerate(roi_list):
        poly = np.array([roi], dtype=np.int32)
        cv2.polylines(img, poly, isClosed=True, color=(255, 255, 0), thickness=2)
        x0, y0 = roi[0]
        cv2.putText(
            img,
            f"ROI {idx + 1} - No Parking Zone",
            (x0, max(20, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )


def draw_box(img, xyxy, label, color, thickness=2):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        ty = y1 - 8
        if ty < 25:
            ty = min(img.shape[0] - 10, y2 + 20)
        cv2.putText(
            img,
            label,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_dashed_box(img, xyxy, color, thickness=2, dash_len=12):
    x1, y1, x2, y2 = map(int, xyxy)

    def draw_dashed_line(p1, p2):
        dist = int(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if dist == 0:
            return
        for i in range(0, dist, dash_len * 2):
            a = i / dist
            b = min(i + dash_len, dist) / dist
            sx = int(p1[0] + (p2[0] - p1[0]) * a)
            sy = int(p1[1] + (p2[1] - p1[1]) * a)
            ex = int(p1[0] + (p2[0] - p1[0]) * b)
            ey = int(p1[1] + (p2[1] - p1[1]) * b)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)

    draw_dashed_line((x1, y1), (x2, y1))
    draw_dashed_line((x2, y1), (x2, y2))
    draw_dashed_line((x2, y2), (x1, y2))
    draw_dashed_line((x1, y2), (x1, y1))


# =========================
# COLOR CLASSIFICATION
# =========================
def crop_vehicle_body_for_color(bgr_crop: np.ndarray) -> np.ndarray:
    if bgr_crop is None or bgr_crop.size == 0:
        return bgr_crop

    h, w = bgr_crop.shape[:2]
    if h < 10 or w < 10:
        return bgr_crop

    top = int(h * COLOR_CROP_TRIM_TOP)
    bottom = int(h * (1.0 - COLOR_CROP_TRIM_BOTTOM))
    left = int(w * COLOR_CROP_TRIM_LEFT)
    right = int(w * (1.0 - COLOR_CROP_TRIM_RIGHT))

    top = clamp(top, 0, h - 1)
    bottom = clamp(bottom, top + 1, h)
    left = clamp(left, 0, w - 1)
    right = clamp(right, left + 1, w)

    return bgr_crop[top:bottom, left:right]


def classify_vehicle_color_hsv(bgr_crop: np.ndarray, sample_size: int = 96) -> Tuple[str, float]:
    if bgr_crop is None or bgr_crop.size == 0:
        return "unknown", 0.0

    h, w = bgr_crop.shape[:2]
    if h < 4 or w < 4:
        return "unknown", 0.0

    crop = cv2.resize(bgr_crop, (sample_size, sample_size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int32)
    S = hsv[:, :, 1].astype(np.int32)
    V = hsv[:, :, 2].astype(np.int32)

    Hf, Sf, Vf = H.reshape(-1), S.reshape(-1), V.reshape(-1)
    valid = Vf > 25
    Hf, Sf, Vf = Hf[valid], Sf[valid], Vf[valid]
    if Hf.size == 0:
        return "unknown", 0.0

    counts = {
        "black": 0, "white": 0, "gray": 0, "red": 0, "orange": 0,
        "yellow": 0, "brown": 0, "green": 0, "cyan": 0, "blue": 0,
        "purple": 0, "pink": 0
    }

    black = Vf < 60
    white = (Sf < 35) & (Vf >= 200)
    gray = (Sf < 35) & (Vf >= 60) & (Vf < 200)

    counts["black"] = int(np.sum(black))
    counts["white"] = int(np.sum(white))
    counts["gray"] = int(np.sum(gray))

    chroma = ~(black | white | gray)
    Hc, Sc, Vc = Hf[chroma], Sf[chroma], Vf[chroma]

    if Hc.size:
        brown = (Hc >= 10) & (Hc <= 25) & (Sc > 50) & (Vc < 160)
        counts["brown"] = int(np.sum(brown))
        counts["red"] = int(np.sum((((Hc <= 10) | (Hc >= 170)) & (~brown))))
        counts["orange"] = int(np.sum(((Hc >= 11) & (Hc <= 20) & (~brown))))
        counts["yellow"] = int(np.sum(((Hc >= 21) & (Hc <= 35) & (~brown))))
        counts["green"] = int(np.sum((Hc >= 36) & (Hc <= 85)))
        counts["cyan"] = int(np.sum((Hc >= 86) & (Hc <= 100)))
        counts["blue"] = int(np.sum((Hc >= 101) & (Hc <= 130)))
        counts["purple"] = int(np.sum((Hc >= 131) & (Hc <= 160)))
        counts["pink"] = int(np.sum((Hc >= 161) & (Hc <= 169)))

    total = sum(counts.values())
    if total <= 0:
        return "unknown", 0.0

    best_color, best_cnt = max(counts.items(), key=lambda kv: kv[1])
    pct = (best_cnt / total) * 100.0
    return best_color, float(pct)


# =========================
# ROI INTERACTIVE MULTI
# =========================
clicked_points = []


def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))


def get_multiple_rois_from_first_frame(frame) -> List[List[Tuple[int, int]]]:
    global clicked_points

    all_rois: List[List[Tuple[int, int]]] = []

    try:
        total_rois = int(input("Masukkan jumlah ROI polygon yang ingin dibuat: ").strip())
    except Exception:
        total_rois = 1

    total_rois = max(1, total_rois)

    window_name = "Set Multiple ROI"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    for roi_idx in range(total_rois):
        clicked_points = []

        while True:
            disp = frame.copy()

            for idx, roi in enumerate(all_rois):
                cv2.polylines(disp, [np.array(roi, dtype=np.int32)], True, (255, 255, 0), 2)
                x0, y0 = roi[0]
                cv2.putText(
                    disp,
                    f"ROI {idx + 1}",
                    (x0, max(20, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            for p in clicked_points:
                cv2.circle(disp, p, 4, (0, 255, 255), -1)

            if len(clicked_points) >= 2:
                cv2.polylines(
                    disp,
                    [np.array(clicked_points, dtype=np.int32)],
                    False,
                    (0, 255, 255),
                    2
                )

            cv2.putText(
                disp,
                f"Buat ROI {roi_idx + 1}/{total_rois} | Klik titik | ENTER=selesai | C=clear",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:
                if len(clicked_points) >= 3:
                    all_rois.append(clicked_points.copy())
                    break
            elif key in [ord("c"), ord("C")]:
                clicked_points = []

    cv2.destroyWindow(window_name)
    return all_rois


# =========================
# TRACK STATE
# =========================
@dataclass
class ParkedRecord:
    logical_id: int
    tracker_id: Optional[int]
    cls_name: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    roi_idx: int

    first_seen_time: int
    last_seen_time: int
    hidden_since: Optional[int] = None

    stationary_since: Optional[int] = None
    parked_since: Optional[int] = None

    moving_since: Optional[int] = None
    reappeared_at: Optional[int] = None

    triangle_checked: bool = False
    triangle_found: bool = False

    illegal_reported: bool = False
    last_alert_time: int = -9999

    color_name: str = "unknown"
    color_pct: float = 0.0

    prev_logic_center: Optional[Tuple[float, float]] = None
    parked_bbox: Optional[Tuple[int, int, int, int]] = None
    parked_center: Optional[Tuple[float, float]] = None


# =========================
# OPTIONAL TTS (EDGE TTS WANITA)
# =========================
tts_busy = False
tts_lock = threading.Lock()

if ENABLE_TTS:
    Path(TTS_TEMP_DIR).mkdir(parents=True, exist_ok=True)
    try:
        pygame.mixer.init()
    except Exception as e:
        print(f"[TTS] pygame mixer gagal init: {e}")


async def edge_tts_save_mp3(text: str, out_mp3: str):
    communicate = edge_tts.Communicate(
        text=text,
        voice=EDGE_TTS_VOICE,
        rate=EDGE_TTS_RATE,
        volume=EDGE_TTS_VOLUME,
        pitch=EDGE_TTS_PITCH,
    )
    await communicate.save(out_mp3)


def _speak_worker(message: str):
    global tts_busy
    temp_file = os.path.join(TTS_TEMP_DIR, f"tts_{uuid.uuid4().hex}.mp3")

    try:
        asyncio.run(edge_tts_save_mp3(message, temp_file))

        if not pygame.mixer.get_init():
            pygame.mixer.init()

        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    except Exception as e:
        print(f"[TTS ERROR] {e}")

    finally:
        try:
            pygame.mixer.music.unload()
        except Exception:
            pass

        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            pass

        with tts_lock:
            tts_busy = False


def speak_once(message: str):
    global tts_busy

    if not ENABLE_TTS:
        return

    with tts_lock:
        if tts_busy:
            return
        tts_busy = True

    threading.Thread(target=_speak_worker, args=(message,), daemon=True).start()


def bahasa_label(cls_name: str) -> str:
    mapping = {
        "car": "mobil",
        "truck": "truk",
        "bus": "bus",
        "motorcycle": "motor",
    }
    return mapping.get(cls_name.lower(), cls_name)


def warna_label(color_name: str) -> str:
    mapping = {
        "black": "hitam",
        "white": "putih",
        "gray": "abu-abu",
        "red": "merah",
        "orange": "oranye",
        "yellow": "kuning",
        "brown": "cokelat",
        "green": "hijau",
        "cyan": "biru muda",
        "blue": "biru",
        "purple": "ungu",
        "pink": "pink",
        "unknown": "warna tidak diketahui",
    }
    return mapping.get(color_name.lower(), color_name)


def posisi_label(cx: float, frame_w: int) -> str:
    if cx < frame_w / 3:
        return "kiri"
    elif cx < 2 * frame_w / 3:
        return "tengah"
    return "kanan"


def get_elapsed_seconds(rec: ParkedRecord, logic_second: int) -> int:
    if rec.parked_since is None:
        return 0
    return max(0, logic_second - rec.parked_since)


def choose_status(rec: ParkedRecord, logic_second: int):
    if rec.parked_since is None:
        return (255, 200, 0), "monitoring"

    duration = get_elapsed_seconds(rec, logic_second)

    if duration >= ILLEGAL_PARKING_SECONDS and not rec.triangle_found:
        return (0, 0, 255), "ILEGAL PARKING"
    if duration >= PARKING_CHECK_SECONDS:
        if rec.triangle_found:
            return (0, 255, 0), "aman / mogok (triangle terdeteksi)"
        return (0, 255, 255), "warning - tunggu sampai menit ke-5"
    return (255, 200, 0), "monitoring"


def get_required_moving_reset_seconds(rec: ParkedRecord, logic_second: int) -> int:
    if rec.parked_since is None:
        return MOVING_RESET_SECONDS

    parked_duration = get_elapsed_seconds(rec, logic_second)

    if parked_duration >= ILLEGAL_PARKING_SECONDS:
        return 12
    if parked_duration >= PARKING_CHECK_SECONDS:
        return 10
    return MOVING_RESET_SECONDS


def make_warning_text(rec: ParkedRecord, frame_w: int) -> str:
    jenis = bahasa_label(rec.cls_name)
    warna = warna_label(rec.color_name)

    return (
        f"Mohon untuk {jenis} dengan warna {warna} "
        f"untuk tidak parkir di situ atau akan terkena denda "
        f"karena bukan tempat parkir."
    )


def recover_roi_from_nearby_hidden_record(
    center: Tuple[float, float],
    cls_name: str,
    parked_records: Dict[int, ParkedRecord],
    logic_second: int
) -> Optional[int]:
    best_roi = None
    best_dist = 1e9

    for _, rec in parked_records.items():
        if rec.cls_name != cls_name:
            continue
        if rec.hidden_since is None:
            continue
        if logic_second - rec.hidden_since > OCCLUSION_TTL_SECONDS:
            continue

        dist = euclidean(center, rec.center)
        if dist < RELINK_MAX_DISTANCE and dist < best_dist:
            best_dist = dist
            best_roi = rec.roi_idx

    return best_roi


def get_best_hidden_match(
    det_bbox,
    det_center,
    det_cls_name: str,
    det_roi_idx: Optional[int],
    det_color: str,
    parked_records: Dict[int, ParkedRecord],
    logic_second: int,
    used_hidden_ids: set
) -> Optional[int]:
    det_foot = foot_point(det_bbox)

    def _eval(rec: ParkedRecord):
        if rec.parked_since is not None and rec.parked_bbox is not None:
            ref_bbox = rec.parked_bbox
            ref_center = rec.parked_center if rec.parked_center is not None else rec.center
        else:
            ref_bbox = rec.bbox
            ref_center = rec.center

        center_dist = euclidean(det_center, ref_center)
        foot_dist = euclidean(det_foot, foot_point(ref_bbox))
        iou_score = bbox_iou(det_bbox, ref_bbox)
        return ref_bbox, ref_center, center_dist, foot_dist, iou_score

    def _candidate_ok(rec, center_dist, foot_dist, iou_score):
        if center_dist > RELINK_MAX_DISTANCE:
            return False
        if rec.parked_since is not None:
            iou_ok = iou_score >= 0.30
            near_ok = center_dist <= 120 and foot_dist <= 140
            return iou_ok or near_ok
        else:
            if iou_score < RELINK_IOU_THRESHOLD and center_dist > (RELINK_MAX_DISTANCE * 0.65) and foot_dist > (RELINK_FOOT_DISTANCE * 0.65):
                return False
            return True

    def _score(rec, center_dist, foot_dist, iou_score):
        score = 0.0
        score += iou_score * 500.0
        score -= center_dist * 1.2
        score -= foot_dist * 2.2
        parked_duration = get_elapsed_seconds(rec, logic_second)
        if parked_duration >= ILLEGAL_PARKING_SECONDS:
            score += 600.0
        elif parked_duration >= PARKING_CHECK_SECONDS:
            score += 450.0
        elif rec.parked_since is not None:
            score += 300.0
        # Strongly prefer record yg sudah lama parked —
        # bobot tinggi agar record lama hampir selalu menang vs record baru
        # ketika spatial score berdekatan.
        score += parked_duration * 8.0
        if rec.color_name != "unknown":
            if rec.color_name == det_color:
                score += 200.0
            else:
                score -= 100.0
        return score

    # PASS 1: hanya record yg sudah parked (parked_since is not None).
    # Jika ada yg lolos filter, pilih yg skornya tertinggi. Record non-parked
    # tidak boleh "mencuri" klaim dari record yg sudah parked.
    best_id = None
    best_score = -1e18
    for lid, rec in parked_records.items():
        if lid in used_hidden_ids:
            continue
        if rec.hidden_since is None:
            continue
        if logic_second - rec.hidden_since > OCCLUSION_TTL_SECONDS:
            continue
        if rec.cls_name != det_cls_name:
            continue
        if det_roi_idx is not None and rec.roi_idx != det_roi_idx:
            continue
        if rec.parked_since is None:
            continue

        _, _, center_dist, foot_dist, iou_score = _eval(rec)
        if not _candidate_ok(rec, center_dist, foot_dist, iou_score):
            continue

        s = _score(rec, center_dist, foot_dist, iou_score)
        if s > best_score:
            best_score = s
            best_id = lid

    if best_id is not None:
        return best_id

    # PASS 2: record yg belum parked (kandidat "pendatang baru" yg masih fresh).
    for lid, rec in parked_records.items():
        if lid in used_hidden_ids:
            continue
        if rec.hidden_since is None:
            continue
        if logic_second - rec.hidden_since > OCCLUSION_TTL_SECONDS:
            continue
        if rec.cls_name != det_cls_name:
            continue
        if det_roi_idx is not None and rec.roi_idx != det_roi_idx:
            continue
        if rec.parked_since is not None:
            continue

        _, _, center_dist, foot_dist, iou_score = _eval(rec)
        if not _candidate_ok(rec, center_dist, foot_dist, iou_score):
            continue

        s = _score(rec, center_dist, foot_dist, iou_score)
        if s > best_score:
            best_score = s
            best_id = lid

    return best_id


def get_locked_hidden_parked_record(
    det_bbox,
    det_center,
    det_cls_name: str,
    det_roi_idx: int,
    det_color: str,
    parked_records: Dict[int, ParkedRecord],
    logic_second: int
) -> Optional[int]:
    det_foot = foot_point(det_bbox)

    best_id = None
    best_score = 1e18

    for lid, rec in parked_records.items():
        if rec.hidden_since is None:
            continue
        if logic_second - rec.hidden_since > OCCLUSION_TTL_SECONDS:
            continue
        if rec.cls_name != det_cls_name:
            continue
        if rec.roi_idx != det_roi_idx:
            continue
        if rec.parked_since is None:
            continue

        if rec.parked_bbox is not None:
            ref_bbox = rec.parked_bbox
            ref_center = rec.parked_center if rec.parked_center is not None else rec.center
        else:
            ref_bbox = rec.bbox
            ref_center = rec.center

        center_dist = euclidean(det_center, ref_center)
        foot_dist = euclidean(det_foot, foot_point(ref_bbox))

        iou_score = bbox_iou(det_bbox, ref_bbox)

        if iou_score < 0.30:
            continue

        if center_dist <= HIDDEN_PARKED_LOCK_CENTER_DISTANCE or foot_dist <= HIDDEN_PARKED_LOCK_FOOT_DISTANCE:
            score = center_dist + (foot_dist * 2.0) - (iou_score * 500.0)
            if score < best_score:
                best_score = score
                best_id = lid

    return best_id

# MAIN
# =========================
def run_video():
    video_path = Path(VIDEO_PATH)
    vehicle_model = YOLO(VEHICLE_MODEL_PATH)
    triangle_model = YOLO(TRIANGLE_MODEL_PATH)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(OUTPUT_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Output video 1 FPS karena kita sampling 1 frame per detik
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 1.0, (W, H))

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Video kosong / frame pertama gagal dibaca.")

    if USE_INTERACTIVE_MULTI_ROI:
        roi_polygons = get_multiple_rois_from_first_frame(first_frame)
    else:
        if MANUAL_MULTI_ROI is not None and len(MANUAL_MULTI_ROI) > 0:
            roi_polygons = MANUAL_MULTI_ROI
        else:
            raise ValueError("MANUAL_MULTI_ROI masih None. Isi ROI manual dulu atau aktifkan mode interactive.")

    parked_records: Dict[int, ParkedRecord] = {}
    tracker_to_logical: Dict[int, int] = {}
    next_logical_id = 1
    cached_triangle_results: Dict[int, List[Tuple[Tuple[int, int, int, int], str, float]]] = {}

    start_wall_time = time.time()
    last_logic_second = -1

    # Mulai dari detik 0 -> frame 0 (first_frame sudah dibaca di atas)
    current_frame = first_frame
    frame_index = 0
    tick_second = 0  # detik video yang sedang diproses

    while True:
        frame = current_frame.copy()
        vis = frame.copy()

        
        logic_second = tick_second

        video_second = tick_second

        draw_all_rois(vis, roi_polygons)

        results = vehicle_model.track(
            source=frame,
            imgsz=VEHICLE_IMGSZ,
            conf=VEHICLE_CONF,
            iou=IOU,
            persist=True,
            tracker=TRACKER_CONFIG,
            verbose=False,
        )

        tracked_now = []

        boxes = results[0].boxes
        names = results[0].names

        if boxes is not None and len(boxes) > 0:
            xyxy_list = boxes.xyxy.cpu().numpy().astype(int)
            cls_list = boxes.cls.cpu().numpy().astype(int)

            id_list = None
            if boxes.id is not None:
                id_list = boxes.id.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                tracker_id = int(id_list[i]) if id_list is not None else None
                xyxy = tuple(xyxy_list[i].tolist())
                cls_id = int(cls_list[i])
                cls_name = str(names.get(cls_id, cls_id)).lower()

                if tracker_id is None:
                    continue
                if cls_name not in ONLY_VEHICLE_CLASSES:
                    continue

                # cx, cy = box_center(xyxy)
                # roi_idx = find_roi_index_for_box(xyxy, roi_polygons)

                # if roi_idx is None:
                #     roi_idx = recover_roi_from_nearby_hidden_record(
                #         (cx, cy), cls_name, parked_records, logic_second
                #     )
                # if roi_idx is None:
                #     continue
                cx, cy = box_center(xyxy)
                roi_idx = find_roi_index_for_box(xyxy, roi_polygons)

                # HARUS benar-benar di dalam salah satu ROI.
                # Kalau tidak, abaikan sepenuhnya — jangan scan, jangan recovery.
                if roi_idx is None:
                    continue

                tracked_now.append({
                    "tracker_id": tracker_id,
                    "bbox": xyxy,
                    "center": (cx, cy),
                    "cls_name": cls_name,
                    "roi_idx": roi_idx,
                    "color": "unknown",
                })

        visible_logical_ids = set()
        used_hidden_ids = set()

        for det in tracked_now:
            tracker_id = det["tracker_id"]
            bbox = det["bbox"]
            center = det["center"]
            cls_name = det["cls_name"]
            roi_idx = det["roi_idx"]
            current_det_color = det["color"]

            logical_id = None

            # Cek tracker_to_logical, tapi verifikasi dulu.
            if tracker_id in tracker_to_logical:
                candidate_id = tracker_to_logical[tracker_id]
                if candidate_id in parked_records:
                    candidate_rec = parked_records[candidate_id]
                    candidate_ok = True

                    # Kalau candidate sudah parked, wajib overlap dgn snapshot.
                    if candidate_rec.parked_since is not None and candidate_rec.parked_bbox is not None:
                        iou_vs_parked = bbox_iou(bbox, candidate_rec.parked_bbox)
                        if iou_vs_parked < 0.30:
                            candidate_ok = False

                    # Selain itu, cek apakah ada PARKED record lain (hidden) yg lebih cocok.
                    # Hanya tolak candidate jika record lain punya IoU JAUH lebih baik
                    # DAN durasi parked lebih lama (record lama lebih prioritas).
                    if candidate_ok:
                        candidate_parked_iou = 0.0
                        if candidate_rec.parked_bbox is not None:
                            candidate_parked_iou = bbox_iou(bbox, candidate_rec.parked_bbox)
                        candidate_duration = get_elapsed_seconds(candidate_rec, logic_second)

                        best_other_lid = None
                        best_other_iou = 0.0
                        best_other_duration = 0

                        for other_lid, other_rec in parked_records.items():
                            if other_lid == candidate_id:
                                continue
                            if other_rec.parked_since is None:
                                continue
                            if other_rec.parked_bbox is None:
                                continue
                            if other_rec.hidden_since is None:
                                continue
                            if other_rec.cls_name != cls_name:
                                continue
                            if other_rec.roi_idx != roi_idx:
                                continue
                            if logic_second - other_rec.hidden_since > OCCLUSION_TTL_SECONDS:
                                continue
                            iou_other = bbox_iou(bbox, other_rec.parked_bbox)
                            other_dur = get_elapsed_seconds(other_rec, logic_second)
                            if iou_other > best_other_iou:
                                best_other_iou = iou_other
                                best_other_lid = other_lid
                                best_other_duration = other_dur

                        # Tolak candidate jika ada hidden parked record yg:
                        # - overlap cukup baik (IoU >= 0.20), DAN
                        # - sudah parked LEBIH LAMA dari candidate.
                        # Record dgn histori terpanjang harus menang.
                        if best_other_lid is not None and best_other_iou >= 0.20:
                            if best_other_duration > candidate_duration:
                                candidate_ok = False

                    if candidate_ok:
                        logical_id = candidate_id

            if logical_id is None:
                best_hidden_id = get_best_hidden_match(
                    det_bbox=bbox,
                    det_center=center,
                    det_cls_name=cls_name,
                    det_roi_idx=roi_idx,
                    det_color=current_det_color,
                    parked_records=parked_records,
                    logic_second=logic_second,
                    used_hidden_ids=used_hidden_ids
                )
                if best_hidden_id is not None:
                    logical_id = best_hidden_id
                    used_hidden_ids.add(best_hidden_id)

            if logical_id is None:
                locked_id = get_locked_hidden_parked_record(
                    det_bbox=bbox,
                    det_center=center,
                    det_cls_name=cls_name,
                    det_roi_idx=roi_idx,
                    det_color=current_det_color,
                    parked_records=parked_records,
                    logic_second=logic_second
                )
                if locked_id is not None:
                    logical_id = locked_id
                    used_hidden_ids.add(locked_id)

            # Sebelum membuat record baru, cek apakah sudah ada record VISIBLE
            # yg parked di lokasi yg sama (same type, same ROI, overlapping bbox).
            # Jika ada, reuse record tsb — ini artinya tracker memberi ID baru
            # tapi objek sebenarnya sama (misal kendaraan lewat menyebabkan
            # tracker kehilangan lalu re-assign ID baru pada kendaraan yg parkir).
            if logical_id is None:
                best_visible_lid = None
                best_visible_score = -1e18
                for v_lid in visible_logical_ids:
                    if v_lid not in parked_records:
                        continue
                    v_rec = parked_records[v_lid]
                    if v_rec.cls_name != cls_name:
                        continue
                    if v_rec.roi_idx != roi_idx:
                        continue
                    if v_rec.parked_since is None:
                        continue
                    # Cek overlap dengan parked snapshot
                    ref = v_rec.parked_bbox if v_rec.parked_bbox is not None else v_rec.bbox
                    iou_v = bbox_iou(bbox, ref)
                    if iou_v >= 0.30:
                        dur = get_elapsed_seconds(v_rec, logic_second)
                        v_score = iou_v * 500 + dur * 8.0
                        if v_score > best_visible_score:
                            best_visible_score = v_score
                            best_visible_lid = v_lid
                if best_visible_lid is not None:
                    logical_id = best_visible_lid

            if logical_id is None:
                logical_id = next_logical_id
                parked_records[logical_id] = ParkedRecord(
                    logical_id=logical_id,
                    tracker_id=tracker_id,
                    cls_name=cls_name,
                    bbox=bbox,
                    center=center,
                    roi_idx=roi_idx,
                    color_name=current_det_color,
                    first_seen_time=logic_second,
                    last_seen_time=logic_second,
                )
                next_logical_id += 1
            else:
                rec = parked_records[logical_id]
                was_hidden = rec.hidden_since is not None

                old_tracker = rec.tracker_id
                if old_tracker is not None and old_tracker in tracker_to_logical and old_tracker != tracker_id:
                    del tracker_to_logical[old_tracker]

                rec.tracker_id = tracker_id
                rec.cls_name = cls_name
                rec.last_seen_time = logic_second

                # Jika record sudah parked dan punya snapshot, hanya update bbox
                # jika deteksi baru cukup dekat (IoU tinggi) dengan snapshot.
                # Ini mencegah kendaraan lewat "menggeser" bounding box
                # dari kendaraan yang sebenarnya parkir diam.
                if rec.parked_since is not None and rec.parked_bbox is not None:
                    iou_vs_snapshot = bbox_iou(bbox, rec.parked_bbox)
                    if iou_vs_snapshot >= 0.35:
                        # Deteksi cocok dgn posisi parkir -> update bbox
                        rec.bbox = bbox
                        rec.center = center
                        rec.roi_idx = roi_idx
                        rec.color_name = current_det_color
                    # else: deteksi terlalu jauh dari snapshot -> keep bbox lama
                    # (kemungkinan bbox terkontaminasi kendaraan lewat)
                else:
                    rec.bbox = bbox
                    rec.center = center
                    rec.roi_idx = roi_idx
                    rec.color_name = current_det_color

                if was_hidden:
                    rec.reappeared_at = logic_second
                    rec.prev_logic_center = rec.center
                    rec.moving_since = None

                rec.hidden_since = None

            tracker_to_logical[tracker_id] = logical_id
            visible_logical_ids.add(logical_id)

        tracker_to_logical = {
            tid: lid for tid, lid in tracker_to_logical.items()
            if lid in parked_records
        }

        for lid, rec in parked_records.items():
            if lid not in visible_logical_ids:
                if rec.hidden_since is None:
                    rec.hidden_since = logic_second

        # ====== CONSOLIDATION: merge short-lived visible ke long-lived hidden ======
        # Jika ada record hidden yg sudah parked lama dan overlap dgn record
        # visible yg parked lebih singkat → transfer deteksi ke record lama,
        # hapus record baru. Ini menyelesaikan kasus dimana tracker memberi
        # ID baru pada kendaraan yg sebenarnya sudah terparkir lama.
        merge_pairs = []  # list of (victim_lid, winner_lid)
        for v_lid in list(visible_logical_ids):
            if v_lid not in parked_records:
                continue
            v_rec = parked_records[v_lid]
            if v_rec.parked_since is None:
                continue
            v_duration = get_elapsed_seconds(v_rec, logic_second)

            best_hidden_lid = None
            best_hidden_duration = v_duration  # hanya merge jika hidden punya durasi lebih panjang

            for h_lid, h_rec in parked_records.items():
                if h_lid == v_lid:
                    continue
                if h_rec.hidden_since is None:
                    continue
                if h_rec.parked_since is None:
                    continue
                if h_rec.cls_name != v_rec.cls_name:
                    continue
                if h_rec.roi_idx != v_rec.roi_idx:
                    continue
                if logic_second - h_rec.hidden_since > OCCLUSION_TTL_SECONDS:
                    continue

                h_duration = get_elapsed_seconds(h_rec, logic_second)
                if h_duration <= best_hidden_duration:
                    continue

                # Bandingkan posisi: pakai parked_bbox snapshot
                h_ref = h_rec.parked_bbox if h_rec.parked_bbox is not None else h_rec.bbox
                v_ref = v_rec.parked_bbox if v_rec.parked_bbox is not None else v_rec.bbox
                iou_val = bbox_iou(v_ref, h_ref)
                center_dist = euclidean(
                    v_rec.parked_center if v_rec.parked_center else v_rec.center,
                    h_rec.parked_center if h_rec.parked_center else h_rec.center
                )

                # Overlap cukup ATAU center dekat → kendaraan fisik yang sama
                if iou_val >= 0.20 or center_dist <= 150:
                    if h_duration > best_hidden_duration:
                        best_hidden_duration = h_duration
                        best_hidden_lid = h_lid

            if best_hidden_lid is not None:
                merge_pairs.append((v_lid, best_hidden_lid))

        for victim_lid, winner_lid in merge_pairs:
            if victim_lid not in parked_records or winner_lid not in parked_records:
                continue

            v_rec = parked_records[victim_lid]
            w_rec = parked_records[winner_lid]

            # Transfer bbox/tracker dari victim ke winner
            w_rec.bbox = v_rec.bbox
            w_rec.center = v_rec.center
            w_rec.tracker_id = v_rec.tracker_id
            w_rec.last_seen_time = logic_second
            w_rec.hidden_since = None
            w_rec.reappeared_at = logic_second
            w_rec.prev_logic_center = w_rec.parked_center if w_rec.parked_center else w_rec.center
            w_rec.moving_since = None

            # Update tracker_to_logical
            if v_rec.tracker_id is not None:
                tracker_to_logical[v_rec.tracker_id] = winner_lid

            # Update visible set
            visible_logical_ids.discard(victim_lid)
            visible_logical_ids.add(winner_lid)

            # Hapus victim
            if victim_lid in cached_triangle_results:
                del cached_triangle_results[victim_lid]
            del parked_records[victim_lid]

            # Clean up old tracker refs to victim
            stale_tids = [t for t, l in tracker_to_logical.items() if l == victim_lid]
            for t in stale_tids:
                del tracker_to_logical[t]

        # ====== ANALISIS TIAP DETIK (sekarang selalu jalan karena 1 tick = 1 detik) ======
        if logic_second != last_logic_second:
            to_delete = []

            for lid, rec in parked_records.items():
                if rec.hidden_since is not None:
                    if logic_second - rec.hidden_since > OCCLUSION_TTL_SECONDS:
                        if rec.tracker_id is not None and rec.tracker_id in tracker_to_logical:
                            del tracker_to_logical[rec.tracker_id]
                        if lid in cached_triangle_results:
                            del cached_triangle_results[lid]
                        to_delete.append(lid)
                        continue

                if rec.hidden_since is None:
                    if rec.prev_logic_center is None:
                        rec.prev_logic_center = rec.center

                    skip_motion_reset = False
                    if rec.reappeared_at is not None and (logic_second - rec.reappeared_at) <= REAPPEAR_GRACE_SECONDS:
                        skip_motion_reset = True

                    move_dist = euclidean(rec.center, rec.prev_logic_center)

                    if skip_motion_reset:
                        rec.prev_logic_center = rec.center
                        rec.moving_since = None
                        if rec.stationary_since is None:
                            rec.stationary_since = logic_second
                    else:
                        if move_dist <= STATIONARY_DISTANCE_THRESHOLD:
                            if rec.stationary_since is None:
                                rec.stationary_since = logic_second
                            rec.moving_since = None
                        else:
                            if rec.moving_since is None:
                                rec.moving_since = logic_second
                            rec.stationary_since = None

                            if rec.parked_since is not None:
                                moving_duration = logic_second - rec.moving_since
                                required_reset = get_required_moving_reset_seconds(rec, logic_second)
                                if moving_duration >= required_reset * 1.5:
                                    rec.parked_since = None
                                    rec.stationary_since = None
                                    rec.moving_since = None
                                    rec.triangle_checked = False
                                    rec.triangle_found = False
                                    rec.illegal_reported = False
                                    rec.color_name = "unknown"
                                    rec.color_pct = 0.0
                                    rec.last_alert_time = -9999
                                    rec.parked_bbox = None
                                    rec.parked_center = None
                                    if lid in cached_triangle_results:
                                        del cached_triangle_results[lid]

                        rec.prev_logic_center = rec.center

                    if rec.stationary_since is not None and rec.parked_since is None:
                        if (logic_second - rec.stationary_since) >= STATIONARY_MIN_SECONDS:
                            rec.parked_since = rec.stationary_since
                            # Snapshot posisi saat pertama kali "parked".
                            # Ini tidak ikut ter-update oleh kontaminasi tracker.
                            rec.parked_bbox = rec.bbox
                            rec.parked_center = rec.center

                parked_duration = get_elapsed_seconds(rec, logic_second)

                if (
                    rec.parked_since is not None
                    and parked_duration >= PARKING_CHECK_SECONDS
                    and not rec.triangle_checked
                    and rec.hidden_since is None
                ):
                    x1, y1, x2, y2 = rec.bbox
                    ex1, ey1, ex2, ey2 = expand_xyxy((x1, y1, x2, y2), W, H, EXPAND_MARGIN)
                    expanded_crop = frame[ey1:ey2, ex1:ex2]

                    triangle_draw_data = []

                    if expanded_crop.size > 0:
                        t_results = triangle_model.predict(
                            source=expanded_crop,
                            imgsz=TRIANGLE_IMGSZ,
                            conf=TRIANGLE_CONF,
                            iou=IOU,
                            verbose=False,
                        )
                        t_boxes = t_results[0].boxes
                        t_names = t_results[0].names

                        rec.triangle_found = (t_boxes is not None) and (len(t_boxes) > 0)

                        if t_boxes is not None and len(t_boxes) > 0:
                            for j in range(len(t_boxes)):
                                t_xyxy = t_boxes.xyxy[j].cpu().numpy().astype(int).tolist()
                                t_conf = float(t_boxes.conf[j].item())
                                t_cls = int(t_boxes.cls[j].item())
                                t_name = str(t_names.get(t_cls, t_cls))

                                tx1, ty1, tx2, ty2 = t_xyxy
                                gx1, gy1 = ex1 + tx1, ey1 + ty1
                                gx2, gy2 = ex1 + tx2, ey1 + ty2
                                triangle_draw_data.append(((gx1, gy1, gx2, gy2), t_name, t_conf))

                    cached_triangle_results[lid] = triangle_draw_data
                    rec.triangle_checked = True

                if rec.parked_since is not None and parked_duration >= ILLEGAL_PARKING_SECONDS and not rec.triangle_found:
                    if rec.color_name == "unknown" and rec.hidden_since is None:
                        x1, y1, x2, y2 = rec.bbox
                        vehicle_crop_full = frame[y1:y2, x1:x2]
                        vehicle_crop = crop_vehicle_body_for_color(vehicle_crop_full)
                        rec.color_name, rec.color_pct = classify_vehicle_color_hsv(vehicle_crop, HEURISTIC_SAMPLE_SIZE)

                    if (logic_second - rec.last_alert_time) >= TTS_REPEAT_GAP:
                        warning_text = make_warning_text(rec, W)
                        print(f"[PROC {logic_second:04d}s | VID {video_second:04d}s] {warning_text}")
                        if ENABLE_TTS:
                            speak_once(warning_text)
                        rec.last_alert_time = logic_second
                        rec.illegal_reported = True

            for lid in to_delete:
                del parked_records[lid]

            last_logic_second = logic_second

        # ====== DRAW GHOST (hidden) ======
        for lid, rec in parked_records.items():
            if rec.hidden_since is not None:
                parked_duration = get_elapsed_seconds(rec, logic_second)
                _, status_text = choose_status(rec, logic_second)
                x1, y1, x2, y2 = rec.bbox
                ghost_color = (150, 150, 150)
                draw_dashed_box(vis, rec.bbox, ghost_color, 2)

                ghost_label = (
                    f"LID {lid} | ROI {rec.roi_idx + 1} | {rec.cls_name} | "
                    f"{parked_duration}s | {status_text} | occluded {logic_second - rec.hidden_since}s"
                )

                ty = y2 + 18
                if ty > H - 10:
                    ty = max(20, y1 - 8)

                cv2.putText(vis, ghost_label, (x1, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, ghost_color, 2, cv2.LINE_AA)

        # ====== DRAW VISIBLE ======
        for lid, rec in parked_records.items():
            if rec.hidden_since is not None:
                continue
            parked_duration = get_elapsed_seconds(rec, logic_second)
            box_color, status_text = choose_status(rec, logic_second)
            label = f"LID {lid} | ROI {rec.roi_idx + 1} | {rec.cls_name} | {parked_duration}s | {status_text}"
            if rec.color_name != "unknown":
                label += f" | {rec.color_name} {rec.color_pct:.0f}%"
            draw_box(vis, rec.bbox, label, box_color, 2)

        for lid, rec in parked_records.items():
            if lid in cached_triangle_results:
                for tri_box, tri_name, tri_conf in cached_triangle_results[lid]:
                    gx1, gy1, gx2, gy2 = tri_box
                    cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                    cv2.putText(vis, f"{tri_name} {tri_conf:.2f}",
                                (gx1, max(20, gy1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        time_text = f"Video Time: {video_second}s" if DISPLAY_VIDEO_TIME else f"Process Time: {logic_second}s"
        cv2.putText(vis, time_text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(vis)
        cv2.imshow("Illegal Parking Detection", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        # ====== LOMPAT KE DETIK BERIKUTNYA ======
        tick_second += 1
        target_frame = int(round(tick_second * fps))
        if target_frame >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, current_frame = cap.read()
        if not ret:
            break
        frame_index = target_frame

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Selesai. Output video: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    run_video()