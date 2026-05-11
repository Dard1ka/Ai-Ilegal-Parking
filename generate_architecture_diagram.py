"""
Generate system architecture diagram for AI Illegal Parking thesis.
Output: D:\Thesis\docs\arsitektur_sistem.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

os.makedirs(r"D:\Thesis\docs", exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(18, 22))
ax.set_xlim(0, 18)
ax.set_ylim(0, 22)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Color palette ──
C_CAMERA    = '#4FC3F7'   # light blue
C_RPI       = '#E8F5E9'   # light green bg
C_RPI_BORDER= '#388E3C'   # green border
C_BACKEND   = '#FFF8E1'   # cream bg
C_BACKEND_B = '#F9A825'   # amber border
C_YOLO      = '#EF5350'   # red
C_BOTSORT   = '#AB47BC'   # purple
C_ROI       = '#26A69A'   # teal
C_TEMPORAL  = '#FF7043'   # deep orange
C_HSV       = '#5C6BC0'   # indigo
C_TTS       = '#66BB6A'   # green
C_STREAM    = '#42A5F5'   # blue
C_REACT     = '#29B6F6'   # light blue
C_SPEAKER   = '#FFA726'   # orange
C_TRIANGLE  = '#EC407A'   # pink
C_ARROW     = '#455A64'   # dark gray

def draw_box(x, y, w, h, color, border_color, text, fontsize=10, bold=False, alpha=1.0, text_color='white', radius=0.3):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.1,rounding_size={radius}",
                         facecolor=color, edgecolor=border_color,
                         linewidth=2, alpha=alpha, zorder=3)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color=text_color, zorder=4,
            wrap=True)
    return box

def draw_arrow(x1, y1, x2, y2, color=C_ARROW, style='->', lw=2, connectionstyle="arc3,rad=0"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color,
                            linewidth=lw, zorder=2,
                            connectionstyle=connectionstyle,
                            mutation_scale=15)
    ax.add_patch(arrow)

def draw_label(x, y, text, fontsize=8, color='#333333', ha='center'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize, color=color,
            zorder=5, style='italic')

# ══════════════════════════════════════════════════════════════
# 1. TITLE
# ══════════════════════════════════════════════════════════════
ax.text(9, 21.5, 'Arsitektur Sistem Deteksi Parkir Ilegal Real-Time',
        ha='center', va='center', fontsize=16, fontweight='bold', color='#1a1a1a')
ax.text(9, 21.1, 'Berbasis Computer Vision dan IoT',
        ha='center', va='center', fontsize=12, color='#555555')

# ══════════════════════════════════════════════════════════════
# 2. CAMERA
# ══════════════════════════════════════════════════════════════
draw_box(6.5, 19.8, 5, 0.9, C_CAMERA, '#0288D1', 'Kamera / CCTV\n(IP Camera / USB Camera)',
         fontsize=11, bold=True, text_color='white')

# Arrow: Camera → RPi
draw_arrow(9, 19.8, 9, 19.1, color=C_ARROW, lw=2.5)
draw_label(9.8, 19.45, 'Video Feed', fontsize=9)

# ══════════════════════════════════════════════════════════════
# 3. RASPBERRY PI 4 (big container)
# ══════════════════════════════════════════════════════════════
rpi_box = FancyBboxPatch((1, 3.8), 16, 15.2,
                          boxstyle="round,pad=0.2,rounding_size=0.4",
                          facecolor=C_RPI, edgecolor=C_RPI_BORDER,
                          linewidth=3, alpha=0.5, zorder=1)
ax.add_patch(rpi_box)
ax.text(9, 18.65, 'RASPBERRY PI 4  (Edge Device)', ha='center', va='center',
        fontsize=13, fontweight='bold', color=C_RPI_BORDER, zorder=4)

# ══════════════════════════════════════════════════════════════
# 4. FASTAPI BACKEND (inner container)
# ══════════════════════════════════════════════════════════════
backend_box = FancyBboxPatch((1.8, 4.2), 14.4, 14,
                              boxstyle="round,pad=0.15,rounding_size=0.3",
                              facecolor=C_BACKEND, edgecolor=C_BACKEND_B,
                              linewidth=2.5, alpha=0.6, zorder=1.5)
ax.add_patch(backend_box)
ax.text(9, 17.85, 'FastAPI Backend  (Python)', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#E65100', zorder=4)

# ══════════════════════════════════════════════════════════════
# 5. ROW 1: YOLO Vehicle + YOLO Triangle
# ══════════════════════════════════════════════════════════════
draw_box(3, 16.2, 4.5, 1.2, C_YOLO, '#C62828',
         'YOLOv8\nVehicle Detection\n(best.pt, imgsz=960)',
         fontsize=9, bold=True)

draw_box(10.5, 16.2, 4.5, 1.2, C_TRIANGLE, '#AD1457',
         'YOLOv8\nWarning Triangle\n(triangle.pt, imgsz=640)',
         fontsize=9, bold=True)

# Arrows down
draw_arrow(5.25, 16.2, 5.25, 15.5, color=C_ARROW, lw=2)
draw_arrow(12.75, 16.2, 12.75, 15.5, color=C_ARROW, lw=2)

# ══════════════════════════════════════════════════════════════
# 6. ROW 2: BoT-SORT + Context Check
# ══════════════════════════════════════════════════════════════
draw_box(3, 14.3, 4.5, 1.2, C_BOTSORT, '#7B1FA2',
         'BoT-SORT\nMulti-Object Tracking\n(botsort.yaml)',
         fontsize=9, bold=True)

draw_box(10.5, 14.3, 4.5, 1.2, '#F48FB1', '#AD1457',
         'Context-Aware Check\nAda segitiga pengaman?\n→ Mogok, bukan ilegal',
         fontsize=9, bold=True, text_color='#880E4F')

# Arrow down from BoT-SORT
draw_arrow(5.25, 14.3, 5.25, 13.6, color=C_ARROW, lw=2)
# Arrow from Context Check to ROI
draw_arrow(10.5, 14.9, 7.5, 13.2, color='#AD1457', lw=2, style='->', connectionstyle="arc3,rad=-0.2")

# ══════════════════════════════════════════════════════════════
# 7. ROW 3: ROI Check
# ══════════════════════════════════════════════════════════════
draw_box(3, 12.4, 4.5, 1.2, C_ROI, '#00796B',
         'ROI Check\n(Multi-Polygon Zone)\nNormalized Coordinates',
         fontsize=9, bold=True)

# Arrow down
draw_arrow(5.25, 12.4, 5.25, 11.7, color=C_ARROW, lw=2)

# ══════════════════════════════════════════════════════════════
# 8. ROW 4: Temporal Analysis (wider box, centered)
# ══════════════════════════════════════════════════════════════
draw_box(2.5, 10.2, 13, 1.4, C_TEMPORAL, '#D84315',
         'Temporal Analysis\n'
         '0-3s → Tidak dihitung  |  3-60s → Monitoring (kuning)  |  '
         '60-300s → Warning (oranye)  |  >300s → Violation (merah)',
         fontsize=9, bold=True)

# Arrows down: split to HSV and TTS
draw_arrow(6, 10.2, 5.25, 9.5, color=C_ARROW, lw=2)
draw_arrow(12, 10.2, 12.75, 9.5, color=C_ARROW, lw=2)

# ══════════════════════════════════════════════════════════════
# 9. ROW 5: HSV Color + Edge TTS
# ══════════════════════════════════════════════════════════════
draw_box(3, 8.3, 4.5, 1.2, C_HSV, '#303F9F',
         'HSV Color\nClassification\n(12 kategori warna)',
         fontsize=9, bold=True)

draw_box(10.5, 8.3, 4.5, 1.2, C_TTS, '#2E7D32',
         'Edge TTS\nText-to-Speech\n(id-ID-GadisNeural)',
         fontsize=9, bold=True)

# Arrow from HSV down
draw_arrow(5.25, 8.3, 5.25, 7.6, color=C_ARROW, lw=2)

# Arrow from TTS to Speaker (right side)
draw_arrow(15, 8.9, 16.2, 8.9, color=C_TTS, lw=2.5)

# ══════════════════════════════════════════════════════════════
# 10. SPEAKER (outside backend, inside RPi)
# ══════════════════════════════════════════════════════════════
draw_box(16.2, 8.2, 0.6, 1.4, C_SPEAKER, '#E65100', 'S\np\ne\na\nk\ne\nr',
         fontsize=7, bold=True, text_color='white')
ax.text(16.5, 7.8, 'Audio\nWarning', ha='center', va='top', fontsize=7,
        color='#E65100', fontweight='bold', zorder=4)

# ══════════════════════════════════════════════════════════════
# 11. ROW 6: MJPEG + WebSocket
# ══════════════════════════════════════════════════════════════
draw_box(3, 6.2, 12, 1.2, C_STREAM, '#1565C0',
         'Output Layer\n'
         'MJPEG Stream (/video_feed)    |    WebSocket (/ws/alerts)    |    REST API (/status)',
         fontsize=9, bold=True)

# Arrow down from stream layer
draw_arrow(9, 6.2, 9, 5.0, color=C_ARROW, lw=2.5)
draw_label(10.2, 5.6, 'HTTP / WebSocket\n(localhost:8000)', fontsize=9, color='#1565C0')

# ══════════════════════════════════════════════════════════════
# 12. REACT DASHBOARD
# ══════════════════════════════════════════════════════════════
draw_box(3.5, 0.8, 11, 2.8, '#E1F5FE', '#0277BD',
         '', fontsize=10, bold=True, text_color='#01579B')

ax.text(9, 3.25, 'React Dashboard  (Vite + Tailwind CSS)', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#01579B', zorder=4)

# Sub-items in dashboard
features = [
    (5.5, 2.3, 'Live Video\nMonitor'),
    (8, 2.3, 'Alert\nSidebar'),
    (10.5, 2.3, 'Real-Time\nStatistik'),
    (13, 2.3, 'ROI Drawing\nTool'),
]
for fx, fy, ftxt in features:
    draw_box(fx-1.0, fy-0.55, 2.0, 1.1, '#B3E5FC', '#0288D1', ftxt,
             fontsize=8, bold=True, text_color='#01579B', radius=0.2)

# Arrow from Dashboard back up (ROI drawing sends ROI to backend)
draw_arrow(13, 3.6, 13, 5.0, color='#0288D1', lw=1.5, style='->')
draw_label(14.2, 4.3, 'POST /set_roi', fontsize=8, color='#0288D1')

# ══════════════════════════════════════════════════════════════
# 13. Arrow from Camera into YOLO boxes
# ══════════════════════════════════════════════════════════════
draw_arrow(7, 19.1, 5.25, 17.4, color=C_ARROW, lw=2, connectionstyle="arc3,rad=0.15")
draw_arrow(11, 19.1, 12.75, 17.4, color=C_ARROW, lw=2, connectionstyle="arc3,rad=-0.15")

# ══════════════════════════════════════════════════════════════
# 14. Legend
# ══════════════════════════════════════════════════════════════
legend_items = [
    (C_YOLO, 'Object Detection (YOLO)'),
    (C_BOTSORT, 'Multi-Object Tracking'),
    (C_ROI, 'Region of Interest'),
    (C_TEMPORAL, 'Temporal Analysis'),
    (C_HSV, 'Color Classification'),
    (C_TTS, 'Text-to-Speech'),
    (C_STREAM, 'Streaming / API'),
]

for i, (color, label) in enumerate(legend_items):
    lx = 0.3
    ly = 3.2 - i * 0.32
    ax.add_patch(FancyBboxPatch((lx, ly - 0.1), 0.25, 0.2,
                                 boxstyle="round,pad=0.02", facecolor=color,
                                 edgecolor='none', zorder=5))
    ax.text(lx + 0.35, ly, label, va='center', fontsize=7, color='#333', zorder=5)

plt.tight_layout(pad=0.5)
plt.savefig(r'D:\Thesis\docs\arsitektur_sistem.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Diagram saved to D:\\Thesis\\docs\\arsitektur_sistem.png")
plt.close()
