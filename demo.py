from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIG (UBAH DI SINI SAJA)
# =========================
IMAGE_NAME = "pickup.jpg"   # <-- GANTI NAMA GAMBAR DI SINI
VEHICLE_MODEL_NAME = "best.pt"
TRIANGLE_MODEL_NAME = "triangle.pt"
OUTPUT_DIR = "outputs"

# --- ROI RECT (kotak) ---
# None = FULL LAYAR
# contoh manual: "0,800,600,100"
ROI_CUSTOM = None

# --- ROI POLYGON (bebas, termasuk 5 sisi) ---
# Kalau ROI_POLY diisi (list titik), maka polygon dipakai dan ROI_CUSTOM diabaikan.
# Set None kalau tidak pakai polygon.
ROI_POLY = [(0,0)]
# contoh 5 sisi:
# ROI_POLY = [(120, 800), (600, 780), (650, 950), (300, 1020), (100, 950)]

# Parameter detection
VEHICLE_IMGSZ = 960
TRIANGLE_IMGSZ = 960
VEHICLE_CONF = 0.25
TRIANGLE_CONF = 0.20
IOU = 0.50
EXPAND_MARGIN = 0.10
ONLY_VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]

# Debug draw
DRAW_TRIANGLE_BOX = True   # bbox triangle digambar
DRAW_VEHICLE_BOX = True    # bbox vehicle digambar
DRAW_ROI = True            # ROI digambar


# -----------------------------
# Utils
# -----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def parse_roi(roi_str: str):
    """
    roi_str format: "x1,y1,x2,y2"
    """
    parts = roi_str.split(",")
    if len(parts) != 4:
        raise ValueError("ROI harus format: x1,y1,x2,y2")
    x1, y1, x2, y2 = map(int, parts)
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return x1, y1, x2, y2


def box_center(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy


def box_center_inside_roi(xyxy, roi_xyxy):
    rx1, ry1, rx2, ry2 = roi_xyxy
    cx, cy = box_center(xyxy)
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)


def point_inside_poly(point_xy, poly_points):
    """
    True jika point di dalam polygon (atau di tepi).
    """
    x, y = point_xy
    poly = np.array(poly_points, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def expand_xyxy(xyxy, img_w: int, img_h: int, margin: float):
    """
    Expand bbox by margin ratio of bbox size.
    margin=0.10 berarti bbox diperbesar 10% dari width/height masing-masing sisi.
    """
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


def draw_box(img, xyxy, label, thickness=2, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_roi_rect(img, roi_xyxy):
    x1, y1, x2, y2 = roi_xyxy
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(
        img,
        "ROI RECT (No Parking Zone)",
        (x1, max(0, y1 + 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_roi_poly(img, poly_points):
    poly = np.array([poly_points], dtype=np.int32)  # (1, N, 2)
    cv2.polylines(img, poly, isClosed=True, color=(255, 255, 0), thickness=2)
    x0, y0 = poly_points[0]
    cv2.putText(
        img,
        "ROI POLY (No Parking Zone)",
        (x0, max(0, y0 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )


# -----------------------------
# Core pipeline
# -----------------------------
def run_on_image(
    image_path: Path,
    vehicle_model_path: Path,
    triangle_model_path: Path,
    out_dir: Path,
    roi_rect_str: str | None,
    roi_poly_points: list[tuple[int, int]] | None,
    vehicle_imgsz: int,
    triangle_imgsz: int,
    vehicle_conf: float,
    triangle_conf: float,
    iou: float,
    expand_margin: float,
    only_vehicle_classes: list[str] | None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Gagal baca image: {image_path}")

    h, w = img.shape[:2]

    # ROI RECT: jika None -> FULL LAYAR
    if roi_rect_str:
        roi_rect = parse_roi(roi_rect_str)
    else:
        roi_rect = (0, 0, w - 1, h - 1)

    # Load models
    vehicle_model = YOLO(str(vehicle_model_path))
    triangle_model = YOLO(str(triangle_model_path))

    # 1) Detect vehicles
    v_results = vehicle_model.predict(
        source=img,
        imgsz=vehicle_imgsz,
        conf=vehicle_conf,
        iou=iou,
        verbose=False,
    )

    vis = img.copy()

    # Draw ROI (rect or poly)
    if DRAW_ROI:
        if roi_poly_points is not None and len(roi_poly_points) >= 3:
            draw_roi_poly(vis, roi_poly_points)
        else:
            draw_roi_rect(vis, roi_rect)

    boxes = v_results[0].boxes
    names = v_results[0].names

    messages = []
    detected_any = False

    if boxes is None or len(boxes) == 0:
        messages.append("No vehicle detected.")
    else:
        allow_set = None
        if only_vehicle_classes:
            allow_set = set([c.strip().lower() for c in only_vehicle_classes])

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            cls_name = str(names.get(cls_id, cls_id))

            # optional filter by class name
            if allow_set is not None and cls_name.lower() not in allow_set:
                continue

            # 2) Check inside ROI:
            # - If polygon exists -> use polygon check
            # - Else -> use rect check
            cx, cy = box_center(xyxy)
            if roi_poly_points is not None and len(roi_poly_points) >= 3:
                if not point_inside_poly((cx, cy), roi_poly_points):
                    continue
            else:
                if not box_center_inside_roi(xyxy, roi_rect):
                    continue

            detected_any = True

            # draw vehicle bbox
            if DRAW_VEHICLE_BOX:
                draw_box(vis, xyxy, f"{cls_name} {conf:.2f}", color=(0, 255, 0))

            # 3) expand bbox slightly (DO NOT draw expanded box)
            ex1, ey1, ex2, ey2 = expand_xyxy(xyxy, w, h, expand_margin)
            crop = img[ey1:ey2, ex1:ex2]
            if crop.size == 0:
                messages.append(f"[{i}] {cls_name}: crop empty, skip triangle check.")
                continue

            # 4) detect triangle inside crop
            t_results = triangle_model.predict(
                source=crop,
                imgsz=triangle_imgsz,
                conf=triangle_conf,
                iou=iou,
                verbose=False,
            )

            t_boxes = t_results[0].boxes
            triangle_found = (t_boxes is not None) and (len(t_boxes) > 0)

            if triangle_found:
                messages.append(f"[{i}] {cls_name}: ALLOW (warning triangle detected)")

                # draw triangle boxes mapped back to original
                if DRAW_TRIANGLE_BOX:
                    t_names = t_results[0].names
                    for j in range(len(t_boxes)):
                        t_xyxy = t_boxes.xyxy[j].cpu().numpy().tolist()
                        t_conf = float(t_boxes.conf[j].item())
                        t_cls = int(t_boxes.cls[j].item())
                        t_name = str(t_names.get(t_cls, t_cls))

                        tx1, ty1, tx2, ty2 = map(int, t_xyxy)
                        gx1, gy1 = ex1 + tx1, ey1 + ty1
                        gx2, gy2 = ex1 + tx2, ey1 + ty2

                        cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
                        cv2.putText(
                            vis,
                            f"{t_name} {t_conf:.2f}",
                            (gx1, max(0, gy1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
            else:
                messages.append(f"[{i}] {cls_name}: WARNING (NO warning triangle), will be warned if within 20 minutes it still stops and there is no warning triangle.")

    if not detected_any and (boxes is not None) and (len(boxes) > 0):
        messages.append("Vehicle(s) detected, but none inside ROI.")

    # Save outputs
    out_img = out_dir / f"{image_path.stem}_result.jpg"
    out_txt = out_dir / f"{image_path.stem}_messages.txt"

    cv2.imwrite(str(out_img), vis)
    out_txt.write_text("\n".join(messages), encoding="utf-8")

    print("\n".join(messages))
    print(f"\nSaved image: {out_img}")
    print(f"Saved log  : {out_txt}")


def main():
    script_dir = Path(__file__).resolve().parent

    image_path = script_dir / IMAGE_NAME
    vehicle_model_path = script_dir / VEHICLE_MODEL_NAME
    triangle_model_path = script_dir / TRIANGLE_MODEL_NAME
    out_dir = script_dir / OUTPUT_DIR

    run_on_image(
        image_path=image_path,
        vehicle_model_path=vehicle_model_path,
        triangle_model_path=triangle_model_path,
        out_dir=out_dir,
        roi_rect_str=ROI_CUSTOM,     # None = full layar
        roi_poly_points=ROI_POLY,    # kalau list titik -> polygon
        vehicle_imgsz=VEHICLE_IMGSZ,
        triangle_imgsz=TRIANGLE_IMGSZ,
        vehicle_conf=VEHICLE_CONF,
        triangle_conf=TRIANGLE_CONF,
        iou=IOU,
        expand_margin=EXPAND_MARGIN,
        only_vehicle_classes=ONLY_VEHICLE_CLASSES,
    )


if __name__ == "__main__":
    main()