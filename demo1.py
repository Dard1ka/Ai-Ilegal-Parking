"""
demo.py — Vehicle + ROI + Warning Triangle + (Optional) Vehicle Color Classification

✅ What this does:
1) Detect vehicle (vehicle.pt)
2) Keep only vehicles whose center point is inside ROI (polygon or rect)
3) Crop each vehicle bbox (for color classification) + crop expanded bbox (for triangle detection)
4) Detect warning triangle inside expanded crop (triangle.pt)
5) Draw on original image:
   - vehicle bbox + class/conf
   - color prediction + percent (if enabled and model exists OR heuristic fallback)
   - triangle bbox (if found)
   - ROI outline
6) Save outputs/{image_stem}_result.jpg and outputs/{image_stem}_messages.txt

📌 Color fix (IMPORTANT):
- For color classification, we DO NOT use full bbox anymore.
- We crop the "vehicle body region" by trimming bottom/top/left/right,
  so road/asphalt/wheels are less likely to dominate and bias to gray.

📌 How to use a trained color model (model_terakhir.pth):
- Put it here (recommended):
  gabungin_vehicle_and_segitiga/color_model/model_terakhir.pth
- Then ENABLE_COLOR_MODEL=True below.

IMPORTANT NOTE:
Because I don't know the exact architecture/checkpoint format in your notebook,
this script supports 3 modes:
A) TorchScript model (best): .pt or a scripted .pth -> will run directly.
B) Checkpoint dict containing:
   - "model" (TorchScript) OR
   - "state_dict" + "class_names" + optionally "arch" (not guaranteed)
   If format unknown, it will gracefully fallback.
C) No usable model -> fallback HSV heuristic (still gives color + %).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# CONFIG (EDIT HERE)
# -----------------------------
IMAGE_NAME = "images/t2.png"
VEHICLE_MODEL_NAME = "vehicle.pt"
TRIANGLE_MODEL_NAME = "triangle.pt"
OUTPUT_DIR = "outputs"

# ROI rectangle (x1,y1,x2,y2) — if None, full image
ROI_CUSTOM = None  # e.g. "0,800,600,100"

# ROI polygon — if set (>=3 points), polygon is used instead of ROI_CUSTOM
ROI_POLY = [(0, 800), (1000, 400), (1300, 400), (900, 1450), (0, 1450)]
# ROI_POLY = None  # uncomment to use ROI_CUSTOM instead

# Detection params
VEHICLE_IMGSZ = 960
TRIANGLE_IMGSZ = 960
VEHICLE_CONF = 0.25
TRIANGLE_CONF = 0.20
IOU = 0.50
EXPAND_MARGIN = 0.10
ONLY_VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]

# Draw toggles
DRAW_ROI = True
DRAW_VEHICLE_BOX = True
DRAW_TRIANGLE_BOX = True
DRAW_COLOR_LABEL = True

# -----------------------------
# COLOR MODEL SETTINGS
# -----------------------------
ENABLE_COLOR = True

# If you have a trained PyTorch color model:
ENABLE_COLOR_MODEL = True

# Put your model here:
# gabungin_vehicle_and_segitiga/color_model/model_terakhir.pth
COLOR_MODEL_PATH = "color_model/model_terakhir.pth"

# Input size used during training (from your notebook name: 128)
COLOR_INPUT_SIZE = 128

# If your model classes are fixed and you know them, put them here.
# If None, we try to read from checkpoint (class_names). If still unknown, fallback heuristic.
COLOR_CLASSES = None
# Example:
# COLOR_CLASSES = ["black", "blue", "brown", "green", "gray", "red", "white", "yellow"]

# Heuristic fallback quality
HEURISTIC_SAMPLE_SIZE = 96

# -----------------------------
# NEW: BODY CROP (exclude road/wheels)
# Tune these if needed.
# -----------------------------
COLOR_CROP_TRIM_BOTTOM = 0.28   # trim 28% bottom of bbox (road/wheels usually here)
COLOR_CROP_TRIM_TOP    = 0.08   # trim 8% top (sometimes windshield reflections)
COLOR_CROP_TRIM_LEFT   = 0.08
COLOR_CROP_TRIM_RIGHT  = 0.08


# -----------------------------
# Utils
# -----------------------------
def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def parse_roi(roi_str: str) -> Tuple[int, int, int, int]:
    parts = roi_str.split(",")
    if len(parts) != 4:
        raise ValueError("ROI_CUSTOM must be 'x1,y1,x2,y2'")
    x1, y1, x2, y2 = map(int, parts)
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return x1, y1, x2, y2


def box_center(xyxy: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, xyxy)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_center_inside_rect(xyxy: List[float], roi_xyxy: Tuple[int, int, int, int]) -> bool:
    rx1, ry1, rx2, ry2 = roi_xyxy
    cx, cy = box_center(xyxy)
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)


def point_inside_poly(point_xy: Tuple[float, float], poly_points: List[Tuple[int, int]]) -> bool:
    poly = np.array(poly_points, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(point_xy[0]), float(point_xy[1])), False) >= 0


def expand_xyxy(xyxy: List[float], img_w: int, img_h: int, margin: float) -> Tuple[int, int, int, int]:
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


def draw_box(img, xyxy, label: str, color=(0, 255, 0), thickness: int = 2):
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


def draw_roi_rect(img, roi_xyxy: Tuple[int, int, int, int]):
    x1, y1, x2, y2 = roi_xyxy
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(
        img,
        "ROI RECT (No Parking Zone)",
        (x1, min(img.shape[0] - 10, y1 + 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_roi_poly(img, poly_points: List[Tuple[int, int]]):
    poly = np.array([poly_points], dtype=np.int32)
    cv2.polylines(img, poly, isClosed=True, color=(255, 255, 0), thickness=2)
    x0, y0 = poly_points[0]
    cv2.putText(
        img,
        "ROI POLY (No Parking Zone)",
        (x0, max(20, y0 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )


# -----------------------------
# NEW: Crop vehicle body region for color classification
# -----------------------------
def crop_vehicle_body_for_color(bgr_crop: np.ndarray) -> np.ndarray:
    """
    Trim bbox edges so road/asphalt/wheels are reduced.
    Works best for common dashcam/CCTV where wheels+road occupy lower bbox.
    """
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


# -----------------------------
# Color classification
# -----------------------------
def classify_vehicle_color_hsv(bgr_crop: np.ndarray, sample_size: int = 96) -> Tuple[str, float]:
    """
    Heuristic HSV classifier.
    Returns: (color_name, percent_of_pixels_for_that_color)
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return "unknown", 0.0

    h, w = bgr_crop.shape[:2]
    if h < 4 or w < 4:
        return "unknown", 0.0

    crop = cv2.resize(bgr_crop, (sample_size, sample_size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int32)  # 0..179
    S = hsv[:, :, 1].astype(np.int32)  # 0..255
    V = hsv[:, :, 2].astype(np.int32)  # 0..255

    Hf, Sf, Vf = H.reshape(-1), S.reshape(-1), V.reshape(-1)

    # remove very dark pixels
    valid = Vf > 25
    Hf, Sf, Vf = Hf[valid], Sf[valid], Vf[valid]
    if Hf.size == 0:
        return "unknown", 0.0

    counts = {k: 0 for k in [
        "black", "white", "gray", "red", "orange", "yellow", "brown",
        "green", "cyan", "blue", "purple", "pink"
    ]}

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

        red = ((Hc <= 10) | (Hc >= 170)) & (~brown)
        orange = (Hc >= 11) & (Hc <= 20) & (~brown)
        yellow = (Hc >= 21) & (Hc <= 35) & (~brown)
        green = (Hc >= 36) & (Hc <= 85)
        cyan = (Hc >= 86) & (Hc <= 100)
        blue = (Hc >= 101) & (Hc <= 130)
        purple = (Hc >= 131) & (Hc <= 160)
        pink = (Hc >= 161) & (Hc <= 169)

        counts["red"] = int(np.sum(red))
        counts["orange"] = int(np.sum(orange))
        counts["yellow"] = int(np.sum(yellow))
        counts["green"] = int(np.sum(green))
        counts["cyan"] = int(np.sum(cyan))
        counts["blue"] = int(np.sum(blue))
        counts["purple"] = int(np.sum(purple))
        counts["pink"] = int(np.sum(pink))

    total = sum(counts.values())
    if total <= 0:
        return "unknown", 0.0

    best_color, best_cnt = max(counts.items(), key=lambda kv: kv[1])
    pct = (best_cnt / total) * 100.0
    return best_color, float(pct)


class ColorModelWrapper:
    """
    Tries to load your PyTorch color model safely.
    If it fails, you can still use HSV heuristic.
    """

    def __init__(self, model_path: Path, input_size: int = 128, class_names: Optional[List[str]] = None):
        self.model_path = model_path
        self.input_size = input_size
        self.class_names = class_names
        self.device = "cpu"
        self.model = None
        self.ok = False
        self.mode = "none"  # torchscript | checkpoint | none

    def load(self):
        if not self.model_path.exists():
            self.ok = False
            self.mode = "none"
            return

        try:
            import torch  # noqa
            # 1) Try TorchScript first
            try:
                self.model = torch.jit.load(str(self.model_path), map_location=self.device)
                self.model.eval()
                self.ok = True
                self.mode = "torchscript"
                return
            except Exception:
                pass

            # 2) Try checkpoint dict
            ckpt = torch.load(str(self.model_path), map_location=self.device)

            # If checkpoint contains a TorchScript model under 'model'
            if isinstance(ckpt, dict) and "model" in ckpt:
                m = ckpt["model"]
                if hasattr(m, "forward"):
                    self.model = m
                    self.model.eval()
                    self.ok = True
                    self.mode = "checkpoint:model"
                    if self.class_names is None and "class_names" in ckpt:
                        self.class_names = list(ckpt["class_names"])
                    return

            self.ok = False
            self.mode = "unknown_checkpoint"

        except Exception:
            self.ok = False
            self.mode = "load_error"

    def predict(self, bgr_crop: np.ndarray) -> Tuple[str, float]:
        """
        Returns (label, percent)
        """
        if not self.ok or self.model is None:
            return "unknown", 0.0

        try:
            import torch
            from PIL import Image

            rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((self.input_size, self.input_size))

            x = np.asarray(img).astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))  # CHW
            x = torch.from_numpy(x).unsqueeze(0)  # NCHW

            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                conf, idx = torch.max(probs, dim=1)

            idx_i = int(idx.item())
            conf_pct = float(conf.item() * 100.0)

            if self.class_names and 0 <= idx_i < len(self.class_names):
                label = self.class_names[idx_i]
            else:
                label = f"class_{idx_i}"

            return label, conf_pct

        except Exception:
            return "unknown", 0.0


# -----------------------------
# Core pipeline
# -----------------------------
def run_on_image(
    image_path: Path,
    vehicle_model_path: Path,
    triangle_model_path: Path,
    out_dir: Path,
    roi_rect_str: Optional[str],
    roi_poly_points: Optional[List[Tuple[int, int]]],
):
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    H, W = img.shape[:2]

    # ROI RECT fallback = full image
    if roi_rect_str:
        roi_rect = parse_roi(roi_rect_str)
    else:
        roi_rect = (0, 0, W - 1, H - 1)

    # Load YOLO models
    vehicle_model = YOLO(str(vehicle_model_path))
    triangle_model = YOLO(str(triangle_model_path))

    vis = img.copy()

    # draw ROI
    if DRAW_ROI:
        if roi_poly_points is not None and len(roi_poly_points) >= 3:
            draw_roi_poly(vis, roi_poly_points)
        else:
            draw_roi_rect(vis, roi_rect)

    # Detect vehicles
    v_results = vehicle_model.predict(
        source=img,
        imgsz=VEHICLE_IMGSZ,
        conf=VEHICLE_CONF,
        iou=IOU,
        verbose=False,
    )

    boxes = v_results[0].boxes
    names = v_results[0].names

    allow_set = set([c.strip().lower() for c in ONLY_VEHICLE_CLASSES]) if ONLY_VEHICLE_CLASSES else None

    messages: List[str] = []
    kept_any = False

    # injected global wrapper
    global COLOR_WRAPPER  # noqa
    color_wrapper = COLOR_WRAPPER

    if boxes is None or len(boxes) == 0:
        messages.append("No vehicle detected.")
    else:
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            cls_id = int(boxes.cls[i].item())
            det_conf = float(boxes.conf[i].item())
            cls_name = str(names.get(cls_id, cls_id))

            # class filter
            if allow_set is not None and cls_name.lower() not in allow_set:
                continue

            # ROI filter using center
            cx, cy = box_center(xyxy)
            if roi_poly_points is not None and len(roi_poly_points) >= 3:
                inside = point_inside_poly((cx, cy), roi_poly_points)
            else:
                inside = box_center_inside_rect(xyxy, roi_rect)

            if not inside:
                continue

            kept_any = True

            # Crop vehicle bbox for color (use original bbox)
            x1, y1, x2, y2 = map(int, xyxy)
            x1 = clamp(x1, 0, W - 1)
            x2 = clamp(x2, 0, W - 1)
            y1 = clamp(y1, 0, H - 1)
            y2 = clamp(y2, 0, H - 1)

            vcolor, vcolor_pct = ("unknown", 0.0)
            if ENABLE_COLOR and (x2 > x1 and y2 > y1):
                vehicle_crop_full = img[y1:y2, x1:x2]

                # ✅ NEW: use body crop to reduce road/wheels/asphalt influence
                vehicle_crop = crop_vehicle_body_for_color(vehicle_crop_full)

                # Try model if available; else heuristic
                if color_wrapper is not None and color_wrapper.ok:
                    vcolor, vcolor_pct = color_wrapper.predict(vehicle_crop)
                    if vcolor == "unknown" and vcolor_pct == 0.0:
                        vcolor, vcolor_pct = classify_vehicle_color_hsv(vehicle_crop, HEURISTIC_SAMPLE_SIZE)
                else:
                    vcolor, vcolor_pct = classify_vehicle_color_hsv(vehicle_crop, HEURISTIC_SAMPLE_SIZE)

            # Draw vehicle bbox + label (include color)
            if DRAW_VEHICLE_BOX:
                label = f"{cls_name} {det_conf:.2f}"
                if ENABLE_COLOR and DRAW_COLOR_LABEL:
                    label = f"{label} | {vcolor} {vcolor_pct:.0f}%"
                draw_box(vis, xyxy, label, color=(0, 239, 255), thickness=2)

            # Expanded crop for triangle detection
            ex1, ey1, ex2, ey2 = expand_xyxy(xyxy, W, H, EXPAND_MARGIN)
            expanded_crop = img[ey1:ey2, ex1:ex2]
            if expanded_crop.size == 0:
                messages.append(f"[{i}] {cls_name}: crop empty (skip triangle) | color={vcolor} {vcolor_pct:.0f}%")
                continue

            t_results = triangle_model.predict(
                source=expanded_crop,
                imgsz=TRIANGLE_IMGSZ,
                conf=TRIANGLE_CONF,
                iou=IOU,
                verbose=False,
            )

            t_boxes = t_results[0].boxes
            triangle_found = (t_boxes is not None) and (len(t_boxes) > 0)

            if triangle_found:
                messages.append(f"[{i}] {cls_name}: ALLOW (warning triangle detected) | color={vcolor} {vcolor_pct:.0f}%")

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
                messages.append(
                    f"[{i}] {cls_name}: WARNING (NO warning triangle) | color={vcolor} {vcolor_pct:.0f}% "
                    f"(warn if still stopped within 20 minutes)"
                )

    if not kept_any and boxes is not None and len(boxes) > 0:
        messages.append("Vehicles detected, but none inside ROI.")

    out_img = out_dir / f"{image_path.stem}_result.jpg"
    out_txt = out_dir / f"{image_path.stem}_messages.txt"

    cv2.imwrite(str(out_img), vis)
    out_txt.write_text("\n".join(messages), encoding="utf-8")

    print("\n".join(messages))
    print(f"\nSaved image: {out_img}")
    print(f"Saved log  : {out_txt}")


# Global for color wrapper injection
COLOR_WRAPPER = None


def main():
    script_dir = Path(__file__).resolve().parent

    image_path = script_dir / IMAGE_NAME
    vehicle_model_path = script_dir / VEHICLE_MODEL_NAME
    triangle_model_path = script_dir / TRIANGLE_MODEL_NAME
    out_dir = script_dir / OUTPUT_DIR

    # Load color wrapper globally (optional)
    global COLOR_WRAPPER  # noqa
    if ENABLE_COLOR and ENABLE_COLOR_MODEL:
        color_model_path = script_dir / COLOR_MODEL_PATH
        wrapper = ColorModelWrapper(
            model_path=color_model_path,
            input_size=COLOR_INPUT_SIZE,
            class_names=COLOR_CLASSES,
        )
        wrapper.load()

        if wrapper.ok:
            print(f"[ColorModel] Loaded OK ({wrapper.mode}): {color_model_path}")
            if wrapper.class_names:
                print(f"[ColorModel] classes: {wrapper.class_names}")
        else:
            print(f"[ColorModel] NOT usable ({wrapper.mode}). Will fallback to HSV heuristic.")
            print(f"[ColorModel] expected path: {color_model_path}")

        COLOR_WRAPPER = wrapper
    else:
        COLOR_WRAPPER = None

    run_on_image(
        image_path=image_path,
        vehicle_model_path=vehicle_model_path,
        triangle_model_path=triangle_model_path,
        out_dir=out_dir,
        roi_rect_str=ROI_CUSTOM,
        roi_poly_points=ROI_POLY,
    )


if __name__ == "__main__":
    main()