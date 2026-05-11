from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math

import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
VIDEO_PATH = "input1.mp4"
VEHICLE_MODEL_PATH = "vehicle.pt"
TRIANGLE_MODEL_PATH = "triangle.pt"
OUTPUT_VIDEO_PATH = "outputs/illegal_parking_result.mp4"

# ===== ROI MODE =====
# True  -> user klik ROI polygon satu per satu
# False -> pakai ROI manual dari kode
USE_INTERACTIVE_MULTI_ROI = True

# ===== ROI MANUAL (opsional) =====
# Kalau mau pakai ROI manual, ubah USE_INTERACTIVE_MULTI_ROI = False
# lalu uncomment bagian ini
# MANUAL_MULTI_ROI = [
#     [(180, 430), (720, 430), (980, 760), (110, 760)],
#     [(1050, 350), (1500, 360), (1600, 800), (1000, 780)],
# ]
MANUAL_MULTI_ROI = None

# ===== Timing rules =====
PARKING_CHECK_SECONDS = 60        # 1 menit
ILLEGAL_PARKING_SECONDS = 300     # 5 menit

# ===== Realtime processing =====
# Jalankan inferensi utama setiap N detik
PROCESS_EVERY_N_SECONDS = 1

# ===== Detection =====
VEHICLE_IMGSZ = 960
TRIANGLE_IMGSZ = 640
VEHICLE_CONF = 0.25
TRIANGLE_CONF = 0.20
IOU = 0.50
EXPAND_MARGIN = 0.18
ONLY_VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]

# ===== Tracking =====
MAX_MATCH_DISTANCE = 90
MAX_MISSING_SECONDS = 2.0
MIN_STATIONARY_IOU = 0.35

# ===== Color crop trim =====
COLOR_CROP_TRIM_BOTTOM = 0.28
COLOR_CROP_TRIM_TOP = 0.08
COLOR_CROP_TRIM_LEFT = 0.08
COLOR_CROP_TRIM_RIGHT = 0.08
HEURISTIC_SAMPLE_SIZE = 96

# ===== TTS =====
ENABLE_TTS = False
TTS_REPEAT_GAP = 20

# =========================
# HELPERS
# =========================
def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def box_center(xyxy) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_inside_poly(point_xy: Tuple[float, float], poly_points: List[Tuple[int, int]]) -> bool:
    poly = np.array(poly_points, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(point_xy[0]), float(point_xy[1])), False) >= 0


def find_roi_index_for_point(point_xy: Tuple[float, float], roi_list: List[List[Tuple[int, int]]]) -> Optional[int]:
    for idx, roi in enumerate(roi_list):
        if point_inside_poly(point_xy, roi):
            return idx
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


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def euclidean(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


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
        cv2.putText(
            img,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

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
# TRACK
# =========================
@dataclass
class TrackState:
    track_id: int
    cls_name: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    roi_idx: int
    enter_time: int
    last_seen_time: int
    first_seen_time: int
    triangle_checked: bool = False
    triangle_found: bool = False
    illegal_reported: bool = False
    last_tts_time: int = -9999
    color_name: str = "unknown"
    color_pct: float = 0.0
    last_status: str = "new"

# =========================
# OPTIONAL TTS
# =========================
tts_engine = None
if ENABLE_TTS:
    import pyttsx3
    tts_engine = pyttsx3.init()


def speak_once(message: str):
    if not ENABLE_TTS or tts_engine is None:
        return
    tts_engine.say(message)
    tts_engine.runAndWait()


def bahasa_label(cls_name: str) -> str:
    mapping = {
        "car": "mobil",
        "truck": "truk",
        "bus": "bus",
        "motorcycle": "motor",
    }
    return mapping.get(cls_name.lower(), cls_name)


def posisi_label(cx: float, frame_w: int) -> str:
    if cx < frame_w / 3:
        return "kiri"
    elif cx < 2 * frame_w / 3:
        return "tengah"
    return "kanan"

# =========================
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

    Path(OUTPUT_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (W, H))

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

    tracks: Dict[int, TrackState] = {}
    next_track_id = 1

    frame_index = 0
    current_frame = first_frame

    last_processed_second = -1
    cached_triangle_results: Dict[int, List[Tuple[Tuple[int, int, int, int], str, float]]] = {}

    while True:
        frame = current_frame.copy()
        vis = frame.copy()

        now_sec_float = frame_index / fps
        current_second = int(now_sec_float)

        draw_all_rois(vis, roi_polygons)

        run_detection_this_second = (
            current_second % PROCESS_EVERY_N_SECONDS == 0
            and current_second != last_processed_second
        )

        if run_detection_this_second:
            # ======================
            # Vehicle detection only once per second
            # ======================
            results = vehicle_model.predict(
                source=frame,
                imgsz=VEHICLE_IMGSZ,
                conf=VEHICLE_CONF,
                iou=IOU,
                verbose=False,
            )

            boxes = results[0].boxes
            names = results[0].names
            detections = []

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    cls_name = str(names.get(cls_id, cls_id)).lower()

                    if cls_name not in ONLY_VEHICLE_CLASSES:
                        continue

                    cx, cy = box_center(xyxy)
                    roi_idx = find_roi_index_for_point((cx, cy), roi_polygons)

                    if roi_idx is None:
                        continue

                    detections.append({
                        "bbox": tuple(xyxy),
                        "center": (cx, cy),
                        "cls_name": cls_name,
                        "conf": conf,
                        "roi_idx": roi_idx,
                    })

            # ======================
            # Tracker update once per second
            # ======================
            used_track_ids = set()
            used_det_ids = set()

            for det_idx, det in enumerate(detections):
                best_id = None
                best_score = 999999

                for tid, tr in tracks.items():
                    if tid in used_track_ids:
                        continue

                    if tr.roi_idx != det["roi_idx"]:
                        continue

                    if det["cls_name"] != tr.cls_name:
                        continue

                    dist = euclidean(det["center"], tr.center)
                    iou_val = iou_xyxy(det["bbox"], tr.bbox)

                    if dist < MAX_MATCH_DISTANCE or iou_val >= MIN_STATIONARY_IOU:
                        score = dist - (iou_val * 50.0)
                        if score < best_score:
                            best_score = score
                            best_id = tid

                if best_id is not None:
                    tr = tracks[best_id]
                    tr.bbox = det["bbox"]
                    tr.center = det["center"]
                    tr.cls_name = det["cls_name"]
                    tr.roi_idx = det["roi_idx"]
                    tr.last_seen_time = current_second
                    used_track_ids.add(best_id)
                    used_det_ids.add(det_idx)

            for det_idx, det in enumerate(detections):
                if det_idx in used_det_ids:
                    continue

                tracks[next_track_id] = TrackState(
                    track_id=next_track_id,
                    cls_name=det["cls_name"],
                    bbox=det["bbox"],
                    center=det["center"],
                    roi_idx=det["roi_idx"],
                    enter_time=current_second,
                    last_seen_time=current_second,
                    first_seen_time=current_second,
                )
                next_track_id += 1

            # ======================
            # Remove lost tracks once per second
            # ======================
            to_delete = []
            for tid, tr in tracks.items():
                if current_second - tr.last_seen_time > MAX_MISSING_SECONDS:
                    to_delete.append(tid)
            for tid in to_delete:
                if tid in cached_triangle_results:
                    del cached_triangle_results[tid]
                del tracks[tid]

            # ======================
            # Process tracks once per second
            # ======================
            for tid, tr in tracks.items():
                x1, y1, x2, y2 = tr.bbox
                duration = current_second - tr.enter_time

                # triangle check saat >= 1 menit
                if duration >= PARKING_CHECK_SECONDS and not tr.triangle_checked:
                    ex1, ey1, ex2, ey2 = expand_xyxy(tr.bbox, W, H, EXPAND_MARGIN)
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

                        tr.triangle_found = (t_boxes is not None) and (len(t_boxes) > 0)

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

                    cached_triangle_results[tid] = triangle_draw_data
                    tr.triangle_checked = True

                # illegal parking saat >= 5 menit dan tidak ada triangle
                if duration >= ILLEGAL_PARKING_SECONDS and not tr.triangle_found:
                    if tr.color_name == "unknown":
                        vehicle_crop_full = frame[y1:y2, x1:x2]
                        vehicle_crop = crop_vehicle_body_for_color(vehicle_crop_full)
                        tr.color_name, tr.color_pct = classify_vehicle_color_hsv(
                            vehicle_crop, HEURISTIC_SAMPLE_SIZE
                        )

                    if not tr.illegal_reported:
                        pos = posisi_label(tr.center[0], W)
                        jenis = bahasa_label(tr.cls_name)
                        warna = tr.color_name

                        warning_text = (
                            f"Halo {jenis} dengan posisi di {pos} dan warna {warna}, "
                            f"dilarang parkir di area tersebut karena termasuk daerah dilarang parkir, "
                            f"silakan pergi dari area tersebut atau akan kami denda."
                        )

                        print(warning_text)
                        tr.illegal_reported = True
                        tr.last_tts_time = current_second
                        speak_once(warning_text)

                    elif ENABLE_TTS and (current_second - tr.last_tts_time >= TTS_REPEAT_GAP):
                        pos = posisi_label(tr.center[0], W)
                        jenis = bahasa_label(tr.cls_name)
                        warna = tr.color_name
                        warning_text = (
                            f"Halo {jenis} dengan posisi di {pos} dan warna {warna}, "
                            f"dilarang parkir di area tersebut. Silakan pergi dari area tersebut."
                        )
                        tr.last_tts_time = current_second
                        speak_once(warning_text)

            last_processed_second = current_second

        # ======================
        # Draw using latest cached track state
        # ======================
        for tid, tr in tracks.items():
            duration = max(0, current_second - tr.enter_time)

            box_color = (255, 200, 0)
            status_text = "monitoring"

            if duration >= PARKING_CHECK_SECONDS:
                if tr.triangle_found:
                    box_color = (0, 255, 0)
                    status_text = "aman / mogok (triangle terdeteksi)"
                else:
                    box_color = (0, 255, 255)
                    status_text = "warning - tunggu sampai menit ke-5"

            if duration >= ILLEGAL_PARKING_SECONDS and not tr.triangle_found:
                box_color = (0, 0, 255)
                status_text = "ILEGAL PARKING"

            label = f"ID {tid} | ROI {tr.roi_idx + 1} | {tr.cls_name} | {duration}s | {status_text}"
            if tr.color_name != "unknown":
                label += f" | {tr.color_name} {tr.color_pct:.0f}%"

            draw_box(vis, tr.bbox, label, box_color, 2)

            if tid in cached_triangle_results:
                for tri_box, tri_name, tri_conf in cached_triangle_results[tid]:
                    gx1, gy1, gx2, gy2 = tri_box
                    cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                    cv2.putText(
                        vis,
                        f"{tri_name} {tri_conf:.2f}",
                        (gx1, max(20, gy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

        cv2.putText(
            vis,
            f"Time: {current_second}s",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(vis)
        cv2.imshow("Illegal Parking Detection", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        frame_index += 1
        ret, current_frame = cap.read()
        if not ret:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Selesai. Output video: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    run_video()