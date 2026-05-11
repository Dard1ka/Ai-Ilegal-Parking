"""
Generate hardware/device diagram for AI Illegal Parking thesis.
Output: D:\Thesis\docs\diagram_alat.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os

os.makedirs(r"D:\Thesis\docs", exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(20, 12))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis('off')
fig.patch.set_facecolor('white')

# Color palette
C_PRIMARY   = '#1565C0'  # dark blue
C_RPI       = '#2E7D32'  # green
C_CAMERA    = '#0277BD'  # blue
C_SPEAKER   = '#E65100'  # orange
C_POWER     = '#C62828'  # red
C_NETWORK   = '#6A1B9A'  # purple
C_FRONTEND  = '#00838F'  # teal
C_ARROW     = '#455A64'  # dark gray
C_BG_LEFT   = '#E3F2FD'  # light blue bg
C_BG_CENTER = '#E8F5E9'  # light green bg
C_BG_RIGHT  = '#FFF3E0'  # light orange bg
C_BOX_BG    = '#FFFFFF'

def draw_device_box(x, y, w, h, color, label, sublabel="", icon_text="", fontsize=9):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.08,rounding_size=0.15",
                         facecolor='white', edgecolor=color,
                         linewidth=2.5, zorder=3)
    ax.add_patch(box)

    icon_circle = Circle((x + w/2, y + h*0.62), 0.35, facecolor=color,
                         edgecolor='white', linewidth=2, zorder=4, alpha=0.9)
    ax.add_patch(icon_circle)

    if icon_text:
        ax.text(x + w/2, y + h*0.62, icon_text, ha='center', va='center',
                fontsize=11, color='white', fontweight='bold', zorder=5,
                fontfamily='DejaVu Sans')

    ax.text(x + w/2, y + h*0.22, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='#1a1a1a', zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h*0.05, sublabel, ha='center', va='center',
                fontsize=7, color='#666666', zorder=4, style='italic')

def draw_arrow(x1, y1, x2, y2, color=C_ARROW, lw=2.5, style='->', bidirectional=False, connectionstyle="arc3,rad=0"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color,
                            linewidth=lw, zorder=2,
                            connectionstyle=connectionstyle,
                            mutation_scale=18)
    ax.add_patch(arrow)
    if bidirectional:
        arrow2 = FancyArrowPatch((x2, y2), (x1, y1),
                                arrowstyle='->', color=color,
                                linewidth=lw, zorder=2,
                                connectionstyle=connectionstyle,
                                mutation_scale=18)
        ax.add_patch(arrow2)

def draw_label_on_arrow(x, y, text, fontsize=8, color='#333', bg='white'):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=color, zorder=6, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=bg, edgecolor='#ccc',
                     alpha=0.9, linewidth=0.5))

# ══════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════
ax.text(10, 11.5, 'Diagram Alat — Sistem Deteksi Parkir Ilegal Real-Time',
        ha='center', va='center', fontsize=16, fontweight='bold', color='#1a1a1a')

# ══════════════════════════════════════════════════════════════
# BACKGROUND ZONES
# ══════════════════════════════════════════════════════════════
# Left zone - Input
zone_left = FancyBboxPatch((0.5, 1.5), 4.5, 9,
                            boxstyle="round,pad=0.2,rounding_size=0.3",
                            facecolor=C_BG_LEFT, edgecolor='#90CAF9',
                            linewidth=2, alpha=0.5, zorder=0)
ax.add_patch(zone_left)
ax.text(2.75, 10.2, 'INPUT', ha='center', va='center',
        fontsize=11, fontweight='bold', color=C_CAMERA, zorder=1)

# Center zone - Processing
zone_center = FancyBboxPatch((5.8, 1.5), 8.4, 9,
                              boxstyle="round,pad=0.2,rounding_size=0.3",
                              facecolor=C_BG_CENTER, edgecolor='#A5D6A7',
                              linewidth=2, alpha=0.5, zorder=0)
ax.add_patch(zone_center)
ax.text(10, 10.2, 'PROCESSING (Edge Device)', ha='center', va='center',
        fontsize=11, fontweight='bold', color=C_RPI, zorder=1)

# Right zone - Output
zone_right = FancyBboxPatch((15, 1.5), 4.5, 9,
                             boxstyle="round,pad=0.2,rounding_size=0.3",
                             facecolor=C_BG_RIGHT, edgecolor='#FFCC80',
                             linewidth=2, alpha=0.5, zorder=0)
ax.add_patch(zone_right)
ax.text(17.25, 10.2, 'OUTPUT', ha='center', va='center',
        fontsize=11, fontweight='bold', color=C_SPEAKER, zorder=1)

# ══════════════════════════════════════════════════════════════
# INPUT DEVICES (Left)
# ══════════════════════════════════════════════════════════════

# CCTV Camera
draw_device_box(1, 7.8, 3.5, 1.8, C_CAMERA, 'Kamera CCTV', 'IP Camera / USB Camera', 'CAM', fontsize=10)

# Video file (alternative input)
draw_device_box(1, 5.5, 3.5, 1.8, '#5E35B1', 'Video File', 'MP4 / AVI / MKV', 'VID', fontsize=10)

# Power supply
draw_device_box(1, 3.2, 3.5, 1.8, C_POWER, 'Power Supply', 'Adaptor 5V 3A USB-C', 'PWR', fontsize=10)

# ══════════════════════════════════════════════════════════════
# CENTRAL PROCESSING UNIT - Raspberry Pi 4
# ══════════════════════════════════════════════════════════════

# Main RPi box
rpi_main = FancyBboxPatch((6.5, 4.5), 7, 5,
                           boxstyle="round,pad=0.15,rounding_size=0.25",
                           facecolor='white', edgecolor=C_RPI,
                           linewidth=3.5, zorder=2.5)
ax.add_patch(rpi_main)

# RPi header
rpi_header = FancyBboxPatch((6.5, 8.7), 7, 0.8,
                              boxstyle="round,pad=0.05,rounding_size=0.15",
                              facecolor=C_RPI, edgecolor=C_RPI,
                              linewidth=2, zorder=3)
ax.add_patch(rpi_header)
ax.text(10, 9.1, 'Raspberry Pi 4 Model B (4GB RAM)', ha='center', va='center',
        fontsize=12, fontweight='bold', color='white', zorder=4)

# Software components inside RPi
sw_items = [
    (6.9, 7.6, 2.8, 0.85, '#EF5350', 'YOLOv8 Detection', 'Vehicle + Triangle'),
    (10.1, 7.6, 3.0, 0.85, '#AB47BC', 'BoT-SORT Tracking', 'Multi-Object Tracker'),
    (6.9, 6.4, 2.8, 0.85, '#FF7043', 'Temporal Analysis', '3s → 60s → 300s'),
    (10.1, 6.4, 3.0, 0.85, '#5C6BC0', 'HSV Color Classify', '12 Kategori Warna'),
    (6.9, 5.2, 2.8, 0.85, '#66BB6A', 'Edge TTS Engine', 'id-ID-GadisNeural'),
    (10.1, 5.2, 3.0, 0.85, '#42A5F5', 'FastAPI + WebSocket', 'Backend Server'),
]

for sx, sy, sw, sh, sc, slabel, ssub in sw_items:
    box = FancyBboxPatch((sx, sy), sw, sh,
                         boxstyle="round,pad=0.05,rounding_size=0.1",
                         facecolor=sc, edgecolor='white',
                         linewidth=1.5, alpha=0.9, zorder=3.5)
    ax.add_patch(box)
    ax.text(sx + sw/2, sy + sh*0.62, slabel, ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=4)
    ax.text(sx + sw/2, sy + sh*0.22, ssub, ha='center', va='center',
            fontsize=6.5, color='#ffffffcc', zorder=4)

# OS label
ax.text(10, 4.75, 'Raspberry Pi OS (64-bit) + Python 3.11',
        ha='center', va='center', fontsize=7.5, color='#555',
        style='italic', zorder=4)

# Ethernet/WiFi indicator
draw_device_box(7.5, 2.2, 2.2, 1.7, C_NETWORK, 'Ethernet', 'LAN / Wi-Fi', 'NET', fontsize=9)

draw_device_box(10.3, 2.2, 2.2, 1.7, '#00838F', 'MicroSD', '32GB+ Class 10', 'SD', fontsize=9)

# ══════════════════════════════════════════════════════════════
# OUTPUT DEVICES (Right)
# ══════════════════════════════════════════════════════════════

# Speaker
draw_device_box(15.5, 7.8, 3.5, 1.8, C_SPEAKER, 'Speaker Aktif', 'Audio Jack 3.5mm / USB', 'SPK', fontsize=10)

# React Dashboard / Monitor
draw_device_box(15.5, 5.5, 3.5, 1.8, C_FRONTEND, 'Monitor / PC', 'React Dashboard\n(Browser)', 'MON', fontsize=10)

# Mobile device (optional)
draw_device_box(15.5, 3.2, 3.5, 1.8, '#7B1FA2', 'Perangkat Mobile', 'Akses via Browser\n(Opsional)', 'MOB', fontsize=10)

# ══════════════════════════════════════════════════════════════
# ARROWS - Input to RPi
# ══════════════════════════════════════════════════════════════

# Camera → RPi
draw_arrow(4.5, 8.7, 6.5, 8.7, color=C_CAMERA, lw=3)
draw_label_on_arrow(5.5, 9.0, 'Video Feed\n(USB/RTSP)', fontsize=7, color=C_CAMERA)

# Video File → RPi
draw_arrow(4.5, 6.4, 6.5, 6.8, color='#5E35B1', lw=2, connectionstyle="arc3,rad=0.15")
draw_label_on_arrow(5.5, 6.9, 'Upload\n(/upload)', fontsize=7, color='#5E35B1')

# Power → RPi
draw_arrow(4.5, 4.1, 6.5, 4.8, color=C_POWER, lw=2, connectionstyle="arc3,rad=0.15")
draw_label_on_arrow(5.5, 4.2, '5V 3A\nUSB-C', fontsize=7, color=C_POWER)

# ══════════════════════════════════════════════════════════════
# ARROWS - RPi to Output
# ══════════════════════════════════════════════════════════════

# RPi → Speaker
draw_arrow(13.5, 8.7, 15.5, 8.7, color=C_SPEAKER, lw=3)
draw_label_on_arrow(14.5, 9.0, 'Audio TTS\n(3.5mm Jack)', fontsize=7, color=C_SPEAKER)

# RPi → Monitor (bidirectional)
draw_arrow(13.5, 6.8, 15.5, 6.4, color=C_FRONTEND, lw=2.5, connectionstyle="arc3,rad=-0.1")
draw_arrow(15.5, 6.0, 13.5, 6.2, color=C_FRONTEND, lw=1.5, style='->', connectionstyle="arc3,rad=-0.15")
draw_label_on_arrow(14.5, 6.85, 'MJPEG +\nWebSocket', fontsize=7, color=C_FRONTEND)
draw_label_on_arrow(14.5, 5.7, 'ROI Config', fontsize=6.5, color=C_FRONTEND)

# RPi → Mobile
draw_arrow(13.5, 5.2, 15.5, 4.3, color='#7B1FA2', lw=2, connectionstyle="arc3,rad=-0.15")
draw_label_on_arrow(14.5, 4.5, 'HTTP/WS\n(Wi-Fi)', fontsize=7, color='#7B1FA2')

# Network → RPi
draw_arrow(8.6, 3.9, 8.6, 4.5, color=C_NETWORK, lw=2)
draw_arrow(11.4, 3.9, 11.4, 4.5, color='#00838F', lw=2)

# ══════════════════════════════════════════════════════════════
# CAPTION
# ══════════════════════════════════════════════════════════════
ax.text(10, 0.8, 'Gambar 3.X  Diagram Alat Sistem Deteksi Parkir Ilegal',
        ha='center', va='center', fontsize=13, fontweight='bold', color='#333')

plt.tight_layout(pad=0.5)
plt.savefig(r'D:\Thesis\docs\diagram_alat.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Diagram saved to D:\\Thesis\\docs\\diagram_alat.png")
plt.close()
