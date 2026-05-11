"""
Generate Low-Fidelity Wireframe for AI Illegal Parking Dashboard.
Shows 3 screens: Setup Wizard, Main Dashboard, ROI Drawing.
Output: D:\Thesis\docs\wireframe_dashboard.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import os

os.makedirs(r"D:\Thesis\docs", exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(22, 14))
fig.patch.set_facecolor('white')

# Colors
C_BG      = '#FAFAFA'
C_BORDER  = '#333333'
C_GRAY    = '#999999'
C_LIGHT   = '#CCCCCC'
C_DARK    = '#222222'
C_FILL    = '#E8E8E8'
C_ACCENT  = '#666666'

def phone_frame(ax, title_text):
    """Draw a desktop browser wireframe frame."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')

    # Outer frame
    frame = FancyBboxPatch((0.2, 0.5), 9.6, 15,
                            boxstyle="round,pad=0,rounding_size=0.3",
                            facecolor='white', edgecolor=C_BORDER,
                            linewidth=2.5)
    ax.add_patch(frame)

    # Top bar (browser-like)
    bar = FancyBboxPatch((0.2, 14.5), 9.6, 1.0,
                          boxstyle="round,pad=0,rounding_size=0.3",
                          facecolor='#F0F0F0', edgecolor=C_BORDER,
                          linewidth=1.5)
    ax.add_patch(bar)

    # URL bar
    url = FancyBboxPatch((1.5, 14.7), 6.5, 0.5,
                          boxstyle="round,pad=0,rounding_size=0.15",
                          facecolor='white', edgecolor=C_LIGHT,
                          linewidth=1)
    ax.add_patch(url)
    ax.text(4.75, 14.95, 'localhost:5173', ha='center', va='center',
            fontsize=7, color=C_GRAY, fontfamily='monospace')

    # Browser dots
    for i, c in enumerate(['#FF5F57', '#FFBD2E', '#28C840']):
        ax.add_patch(plt.Circle((0.6 + i*0.35, 15.0), 0.1,
                                facecolor=c, edgecolor='none', zorder=5))

def draw_x_placeholder(ax, x, y, w, h, label=""):
    """Draw a wireframe image placeholder with X."""
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0,rounding_size=0.1",
                           facecolor=C_FILL, edgecolor=C_BORDER,
                           linewidth=1)
    ax.add_patch(rect)
    # X lines
    ax.plot([x, x+w], [y, y+h], color=C_LIGHT, linewidth=1, zorder=3)
    ax.plot([x, x+w], [y+h, y], color=C_LIGHT, linewidth=1, zorder=3)
    if label:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=8, color=C_GRAY, zorder=4)

def draw_button(ax, x, y, w, h, label, filled=False):
    """Draw a wireframe button."""
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0,rounding_size=0.1",
                           facecolor=C_DARK if filled else 'white',
                           edgecolor=C_BORDER, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=8, fontweight='bold',
            color='white' if filled else C_DARK, zorder=4)

def draw_text_line(ax, x, y, w, label="", fontsize=7):
    """Draw a wireframe text line."""
    ax.plot([x, x+w], [y, y], color=C_LIGHT, linewidth=2)
    if label:
        ax.text(x, y + 0.15, label, fontsize=fontsize, color=C_ACCENT)

def draw_input_box(ax, x, y, w, h, label=""):
    """Draw a wireframe input field."""
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0,rounding_size=0.08",
                           facecolor='white', edgecolor=C_BORDER,
                           linewidth=1)
    ax.add_patch(rect)
    if label:
        ax.text(x + 0.2, y + h/2, label, va='center',
                fontsize=7, color=C_GRAY)

# ══════════════════════════════════════════════════════════════
# SCREEN 1: Setup Wizard (Pilih Sumber)
# ══════════════════════════════════════════════════════════════
ax1 = axes[0]
phone_frame(ax1, "Setup Wizard")

# Logo + title
ax1.text(5, 14.0, 'Parking Guard', ha='center', fontsize=11,
         fontweight='bold', color=C_DARK)
ax1.text(5, 13.6, 'Setup Konfigurasi', ha='center', fontsize=7, color=C_GRAY)

# Step indicator (3 circles)
for i, (label, active) in enumerate([(1, True), (2, False), (3, False)]):
    cx = 3.5 + i * 1.5
    circle = plt.Circle((cx, 13.0), 0.25,
                         facecolor=C_DARK if active else 'white',
                         edgecolor=C_BORDER, linewidth=1.2, zorder=5)
    ax1.add_patch(circle)
    ax1.text(cx, 13.0, str(label), ha='center', va='center',
             fontsize=8, fontweight='bold',
             color='white' if active else C_DARK, zorder=6)
    if i < 2:
        ax1.plot([cx+0.35, cx+1.15], [13.0, 13.0],
                 color=C_LIGHT, linewidth=1.5)

ax1.text(3.5, 12.55, 'Sumber', ha='center', fontsize=6, color=C_GRAY)
ax1.text(5.0, 12.55, 'Zona', ha='center', fontsize=6, color=C_GRAY)
ax1.text(6.5, 12.55, 'Mulai', ha='center', fontsize=6, color=C_GRAY)

# Title
ax1.text(1.0, 11.8, 'Dari mana sumber videonya?', fontsize=10,
         fontweight='bold', color=C_DARK)

# Option 1: Upload Video
opt1 = FancyBboxPatch((0.8, 9.8), 8.4, 1.6,
                       boxstyle="round,pad=0,rounding_size=0.15",
                       facecolor='white', edgecolor=C_BORDER,
                       linewidth=1.5)
ax1.add_patch(opt1)
ax1.text(1.5, 10.9, 'Upload File Video', fontsize=9, fontweight='bold', color=C_DARK)
ax1.text(1.5, 10.4, 'MP4, AVI, MOV, MKV, WebM', fontsize=7, color=C_GRAY)
# Radio circle
ax1.add_patch(plt.Circle((8.5, 10.6), 0.2,
                          facecolor='white', edgecolor=C_BORDER, linewidth=1.5, zorder=5))

# Option 2: Camera
opt2 = FancyBboxPatch((0.8, 7.8), 8.4, 1.6,
                       boxstyle="round,pad=0,rounding_size=0.15",
                       facecolor='white', edgecolor=C_BORDER,
                       linewidth=1.5)
ax1.add_patch(opt2)
ax1.text(1.5, 8.9, 'Kamera Langsung', fontsize=9, fontweight='bold', color=C_DARK)
ax1.text(1.5, 8.4, 'Webcam atau kamera IP real-time', fontsize=7, color=C_GRAY)
# Radio
ax1.add_patch(plt.Circle((8.5, 8.6), 0.2,
                          facecolor='white', edgecolor=C_BORDER, linewidth=1.5, zorder=5))
# LIVE badge
badge = FancyBboxPatch((7.2, 9.05), 1.0, 0.3,
                        boxstyle="round,pad=0,rounding_size=0.08",
                        facecolor=C_FILL, edgecolor=C_GRAY, linewidth=0.8)
ax1.add_patch(badge)
ax1.text(7.7, 9.2, 'LIVE', ha='center', va='center', fontsize=6, color=C_ACCENT, fontweight='bold')

# Drag & drop area
drag = FancyBboxPatch((0.8, 5.5), 8.4, 1.8,
                       boxstyle="round,pad=0,rounding_size=0.15",
                       facecolor='#F8F8F8', edgecolor=C_GRAY,
                       linewidth=1, linestyle='dashed')
ax1.add_patch(drag)
ax1.text(5, 6.7, 'Drag & drop video di sini', ha='center', fontsize=8, color=C_GRAY)
ax1.text(5, 6.2, 'atau klik untuk browse', ha='center', fontsize=7, color=C_ACCENT)

# ROI Mode section
ax1.text(1.0, 4.8, 'Mode Zona Parkir:', fontsize=8, fontweight='bold', color=C_DARK)

# Mode options
opt3 = FancyBboxPatch((0.8, 3.4), 4.0, 1.2,
                       boxstyle="round,pad=0,rounding_size=0.1",
                       facecolor='white', edgecolor=C_BORDER, linewidth=1)
ax1.add_patch(opt3)
ax1.text(2.8, 4.15, 'Gambar Sendiri', ha='center', fontsize=8, fontweight='bold', color=C_DARK)
ax1.text(2.8, 3.75, 'Manual polygon', ha='center', fontsize=6.5, color=C_GRAY)

opt4 = FancyBboxPatch((5.2, 3.4), 4.0, 1.2,
                       boxstyle="round,pad=0,rounding_size=0.1",
                       facecolor='white', edgecolor=C_BORDER, linewidth=1)
ax1.add_patch(opt4)
ax1.text(7.2, 4.15, 'Otomatis', ha='center', fontsize=8, fontweight='bold', color=C_DARK)
ax1.text(7.2, 3.75, 'Seluruh frame', ha='center', fontsize=6.5, color=C_GRAY)

# Next button
draw_button(ax1, 0.8, 1.5, 8.4, 1.2, 'Lanjut  >', filled=True)

# Camera index
ax1.text(1.0, 1.1, 'Index Kamera:', fontsize=7, color=C_ACCENT)
for i in range(4):
    bx = 4.5 + i * 1.1
    rect = FancyBboxPatch((bx, 0.8), 0.8, 0.6,
                           boxstyle="round,pad=0,rounding_size=0.05",
                           facecolor=C_DARK if i == 0 else 'white',
                           edgecolor=C_BORDER, linewidth=1)
    ax1.add_patch(rect)
    ax1.text(bx + 0.4, 1.1, str(i), ha='center', va='center',
             fontsize=8, color='white' if i == 0 else C_DARK, fontweight='bold')


# ══════════════════════════════════════════════════════════════
# SCREEN 2: Main Dashboard
# ══════════════════════════════════════════════════════════════
ax2 = axes[1]
phone_frame(ax2, "Dashboard")

# Header bar
header = FancyBboxPatch((0.3, 13.5), 9.4, 0.9,
                          boxstyle="square,pad=0",
                          facecolor='#F5F5F5', edgecolor=C_LIGHT,
                          linewidth=1)
ax2.add_patch(header)
ax2.text(0.8, 14.05, 'Parking Guard', fontsize=9, fontweight='bold', color=C_DARK)
ax2.text(0.8, 13.75, 'Deteksi parkir ilegal real-time', fontsize=6, color=C_GRAY)

# Connection badge
conn = FancyBboxPatch((7.2, 13.75), 1.2, 0.4,
                       boxstyle="round,pad=0,rounding_size=0.1",
                       facecolor='white', edgecolor=C_GRAY, linewidth=0.8)
ax2.add_patch(conn)
ax2.add_patch(plt.Circle((7.45, 13.95), 0.06, facecolor=C_DARK, zorder=5))
ax2.text(7.85, 13.95, 'LIVE', ha='center', va='center', fontsize=6, fontweight='bold', color=C_DARK)

# Clock
ax2.text(9.3, 14.05, '14:32:05', ha='right', fontsize=7, fontfamily='monospace', color=C_DARK)
ax2.text(9.3, 13.75, 'Kamis, 8 Mei', ha='right', fontsize=6, color=C_GRAY)

# Settings button
gear = FancyBboxPatch((8.5, 13.75), 0.4, 0.4,
                       boxstyle="round,pad=0,rounding_size=0.08",
                       facecolor='white', edgecolor=C_GRAY, linewidth=0.8)
ax2.add_patch(gear)
ax2.text(8.7, 13.95, '*', ha='center', va='center', fontsize=10, color=C_DARK)

# ── 3 Metric Cards ──
metrics = [
    ('Total\nTerpantau', '05', 0.4),
    ('Warning\n(60-300s)', '02', 3.5),
    ('Violation\n(>300s)', '01', 6.6),
]
for label, val, mx in metrics:
    card = FancyBboxPatch((mx, 12.0), 2.8, 1.3,
                           boxstyle="round,pad=0,rounding_size=0.12",
                           facecolor='white', edgecolor=C_BORDER,
                           linewidth=1)
    ax2.add_patch(card)
    ax2.text(mx + 0.3, 12.85, val, fontsize=18, fontweight='bold',
             color=C_DARK, fontfamily='monospace')
    ax2.text(mx + 0.3, 12.25, label, fontsize=6, color=C_GRAY)

# ── Video Feed Area ──
ax2.text(0.5, 11.65, 'LIVE FEED', fontsize=7, fontfamily='monospace',
         fontweight='bold', color=C_ACCENT)

# Streaming badge
strbadge = FancyBboxPatch((2.5, 11.45), 1.5, 0.35,
                           boxstyle="round,pad=0,rounding_size=0.08",
                           facecolor='white', edgecolor=C_GRAY, linewidth=0.8)
ax2.add_patch(strbadge)
ax2.text(3.25, 11.62, 'Streaming', ha='center', fontsize=6, color=C_DARK)

# Update Zona button
uz = FancyBboxPatch((5.0, 11.45), 1.8, 0.35,
                     boxstyle="round,pad=0,rounding_size=0.08",
                     facecolor='white', edgecolor=C_GRAY, linewidth=0.8)
ax2.add_patch(uz)
ax2.text(5.9, 11.62, 'Update Zona', ha='center', fontsize=6, color=C_DARK)

# Video placeholder
draw_x_placeholder(ax2, 0.4, 5.2, 6.4, 6.0, 'MJPEG Video Stream\n(/video_feed)')

# Corner brackets on video
corners = [(0.5, 5.3), (0.5, 11.0), (6.5, 5.3), (6.5, 11.0)]

# ── Alert Sidebar ──
ax2.text(7.2, 11.65, 'EVENT LOG', fontsize=7, fontfamily='monospace',
         fontweight='bold', color=C_ACCENT)

# Alert card 1 - Violation
a1 = FancyBboxPatch((7.1, 9.8), 2.6, 1.5,
                      boxstyle="round,pad=0,rounding_size=0.1",
                      facecolor='white', edgecolor=C_BORDER, linewidth=1)
ax2.add_patch(a1)
# Left strip
ax2.add_patch(Rectangle((7.1, 9.8), 0.08, 1.5, facecolor=C_DARK, zorder=4))
ax2.text(7.5, 10.95, 'Mobil - Merah', fontsize=7, fontweight='bold', color=C_DARK)
# Status badge
sb1 = FancyBboxPatch((8.5, 10.8), 1.1, 0.35,
                      boxstyle="round,pad=0,rounding_size=0.08",
                      facecolor=C_FILL, edgecolor=C_GRAY, linewidth=0.5)
ax2.add_patch(sb1)
ax2.text(9.05, 10.97, 'Violation', ha='center', fontsize=5.5, fontweight='bold', color=C_DARK)
ax2.text(7.5, 10.55, 'ID: VH-042  |  Zona 1', fontsize=5.5, color=C_GRAY)
# Duration bar
ax2.add_patch(Rectangle((7.5, 10.1), 1.8, 0.15, facecolor=C_FILL, zorder=3))
ax2.add_patch(Rectangle((7.5, 10.1), 1.5, 0.15, facecolor=C_DARK, zorder=4))
ax2.text(9.5, 10.17, '6m 12s', ha='right', fontsize=5.5, fontweight='bold', color=C_DARK)

# Alert card 2 - Warning
a2 = FancyBboxPatch((7.1, 8.0), 2.6, 1.5,
                      boxstyle="round,pad=0,rounding_size=0.1",
                      facecolor='white', edgecolor=C_BORDER, linewidth=1)
ax2.add_patch(a2)
ax2.add_patch(Rectangle((7.1, 8.0), 0.08, 1.5, facecolor=C_GRAY, zorder=4))
ax2.text(7.5, 9.15, 'Motor - Hitam', fontsize=7, fontweight='bold', color=C_DARK)
sb2 = FancyBboxPatch((8.5, 9.0), 1.1, 0.35,
                      boxstyle="round,pad=0,rounding_size=0.08",
                      facecolor=C_FILL, edgecolor=C_GRAY, linewidth=0.5)
ax2.add_patch(sb2)
ax2.text(9.05, 9.17, 'Warning', ha='center', fontsize=5.5, fontweight='bold', color=C_DARK)
ax2.text(7.5, 8.75, 'ID: VH-038  |  Zona 2', fontsize=5.5, color=C_GRAY)
ax2.add_patch(Rectangle((7.5, 8.3), 1.8, 0.15, facecolor=C_FILL, zorder=3))
ax2.add_patch(Rectangle((7.5, 8.3), 0.7, 0.15, facecolor=C_GRAY, zorder=4))
ax2.text(9.5, 8.37, '2m 15s', ha='right', fontsize=5.5, fontweight='bold', color=C_DARK)

# Alert card 3 - Monitoring
a3 = FancyBboxPatch((7.1, 6.2), 2.6, 1.5,
                      boxstyle="round,pad=0,rounding_size=0.1",
                      facecolor='white', edgecolor=C_LIGHT, linewidth=1)
ax2.add_patch(a3)
ax2.add_patch(Rectangle((7.1, 6.2), 0.08, 1.5, facecolor=C_LIGHT, zorder=4))
ax2.text(7.5, 7.35, 'Truk - Putih', fontsize=7, fontweight='bold', color=C_DARK)
sb3 = FancyBboxPatch((8.3, 7.2), 1.3, 0.35,
                      boxstyle="round,pad=0,rounding_size=0.08",
                      facecolor=C_FILL, edgecolor=C_LIGHT, linewidth=0.5)
ax2.add_patch(sb3)
ax2.text(8.95, 7.37, 'Monitoring', ha='center', fontsize=5.5, color=C_GRAY)
ax2.text(7.5, 6.95, 'ID: VH-045  |  Zona 1', fontsize=5.5, color=C_GRAY)
ax2.add_patch(Rectangle((7.5, 6.5), 1.8, 0.15, facecolor=C_FILL, zorder=3))
ax2.add_patch(Rectangle((7.5, 6.5), 0.2, 0.15, facecolor=C_LIGHT, zorder=4))
ax2.text(9.5, 6.57, '18s', ha='right', fontsize=5.5, color=C_GRAY)

# Alert card 4 - Safe (with triangle)
a4 = FancyBboxPatch((7.1, 4.4), 2.6, 1.5,
                      boxstyle="round,pad=0,rounding_size=0.1",
                      facecolor='white', edgecolor=C_LIGHT, linewidth=1)
ax2.add_patch(a4)
ax2.add_patch(Rectangle((7.1, 4.4), 0.08, 1.5, facecolor=C_LIGHT, zorder=4))
ax2.text(7.5, 5.55, 'Mobil - Abu', fontsize=7, fontweight='bold', color=C_DARK)
sb4 = FancyBboxPatch((8.6, 5.4), 1.0, 0.35,
                      boxstyle="round,pad=0,rounding_size=0.08",
                      facecolor=C_FILL, edgecolor=C_LIGHT, linewidth=0.5)
ax2.add_patch(sb4)
ax2.text(9.1, 5.57, 'Aman', ha='center', fontsize=5.5, color=C_GRAY)
ax2.text(7.5, 5.15, 'ID: VH-033  |  Zona 1', fontsize=5.5, color=C_GRAY)
ax2.text(7.5, 4.8, 'Segitiga terdeteksi', fontsize=5.5, color=C_ACCENT, style='italic')

# Footer
ax2.plot([7.1, 9.7], [4.2, 4.2], color=C_LIGHT, linewidth=0.8)
ax2.text(7.2, 3.9, '5 kendaraan terpantau', fontsize=5.5, color=C_GRAY)
ax2.text(9.5, 3.9, 'Refresh 1s', ha='right', fontsize=5.5, color=C_GRAY)

# ── Bottom section labels ──
# Progress bar under event log
ax2.text(0.5, 4.8, 'Status distribution:', fontsize=6, color=C_GRAY)
ax2.add_patch(Rectangle((0.5, 4.4), 2.0, 0.2, facecolor=C_DARK, zorder=3))
ax2.add_patch(Rectangle((2.5, 4.4), 1.5, 0.2, facecolor=C_GRAY, zorder=3))
ax2.add_patch(Rectangle((4.0, 4.4), 1.0, 0.2, facecolor=C_LIGHT, zorder=3))

# Legend
ax2.add_patch(Rectangle((0.5, 3.8), 0.3, 0.2, facecolor=C_DARK))
ax2.text(1.0, 3.9, 'Violation', fontsize=5.5, color=C_DARK, va='center')
ax2.add_patch(Rectangle((2.5, 3.8), 0.3, 0.2, facecolor=C_GRAY))
ax2.text(3.0, 3.9, 'Warning', fontsize=5.5, color=C_DARK, va='center')
ax2.add_patch(Rectangle((4.5, 3.8), 0.3, 0.2, facecolor=C_LIGHT))
ax2.text(5.0, 3.9, 'Monitor', fontsize=5.5, color=C_DARK, va='center')

# Connection info
ax2.text(0.5, 3.2, 'WebSocket: ws://localhost:8000/ws/alerts', fontsize=5.5,
         fontfamily='monospace', color=C_GRAY)
ax2.text(0.5, 2.8, 'MJPEG: http://localhost:8000/video_feed', fontsize=5.5,
         fontfamily='monospace', color=C_GRAY)
ax2.text(0.5, 2.4, 'Auto-reconnect: 3s | Keep-alive ping: 25s', fontsize=5.5,
         fontfamily='monospace', color=C_GRAY)

# Tech stack label
ax2.text(5, 1.5, 'React 18 + Vite + Tailwind CSS', ha='center',
         fontsize=7, color=C_ACCENT, style='italic')


# ══════════════════════════════════════════════════════════════
# SCREEN 3: ROI Drawing
# ══════════════════════════════════════════════════════════════
ax3 = axes[2]
phone_frame(ax3, "ROI Drawing")

# Header
ax3.text(5, 14.0, 'Parking Guard', ha='center', fontsize=11,
         fontweight='bold', color=C_DARK)
ax3.text(5, 13.6, 'Gambar Zona Larangan Parkir', ha='center', fontsize=7, color=C_GRAY)

# Step indicator (step 3 active)
for i, (label, active) in enumerate([(1, False), (2, False), (3, True)]):
    cx = 3.5 + i * 1.5
    circle = plt.Circle((cx, 12.8), 0.25,
                         facecolor=C_DARK if (i < 2 or active) else 'white',
                         edgecolor=C_BORDER, linewidth=1.2, zorder=5)
    ax3.add_patch(circle)
    txt = 'V' if i < 2 else str(label)
    ax3.text(cx, 12.8, txt, ha='center', va='center',
             fontsize=8, fontweight='bold',
             color='white' if (i < 2 or active) else C_DARK, zorder=6)
    if i < 2:
        ax3.plot([cx+0.35, cx+1.15], [12.8, 12.8],
                 color=C_DARK, linewidth=1.5)

ax3.text(3.5, 12.35, 'Sumber', ha='center', fontsize=6, color=C_GRAY)
ax3.text(5.0, 12.35, 'Zona', ha='center', fontsize=6, color=C_GRAY)
ax3.text(6.5, 12.35, 'Mulai', ha='center', fontsize=6, color=C_GRAY)

# Instructions
ax3.text(1.0, 11.7, 'Klik titik-titik di frame untuk', fontsize=8,
         fontweight='bold', color=C_DARK)
ax3.text(1.0, 11.3, 'bikin polygon zona larangan parkir', fontsize=8,
         color=C_DARK)

# First frame with ROI polygon overlay
draw_x_placeholder(ax3, 0.6, 5.0, 8.8, 6.0, 'Frame Pertama Video\n(first_frame)')

# Draw polygon ROI overlay on the frame
poly_x = [2.0, 4.5, 7.5, 8.0, 5.0, 1.5]
poly_y = [7.5, 9.5, 9.0, 6.5, 5.5, 6.0]
poly_x.append(poly_x[0])
poly_y.append(poly_y[0])
ax3.plot(poly_x, poly_y, color=C_DARK, linewidth=2, linestyle='--', zorder=5)
ax3.fill(poly_x, poly_y, alpha=0.1, color=C_DARK, zorder=4)

# Points P1-P6
points = list(zip(poly_x[:-1], poly_y[:-1]))
for i, (px, py) in enumerate(points):
    ax3.add_patch(plt.Circle((px, py), 0.15, facecolor=C_DARK,
                              edgecolor='white', linewidth=1.5, zorder=6))
    ax3.text(px + 0.3, py + 0.2, f'P{i+1}', fontsize=6,
             fontweight='bold', color=C_DARK, zorder=6)

# Zone label in center
ax3.text(4.8, 7.5, 'Zona 1', ha='center', fontsize=9,
         fontweight='bold', color=C_ACCENT, zorder=6)

# Hint badge
hint = FancyBboxPatch((2.5, 10.5), 5.0, 0.45,
                       boxstyle="round,pad=0,rounding_size=0.1",
                       facecolor='white', edgecolor=C_GRAY,
                       linewidth=0.8, alpha=0.9, zorder=7)
ax3.add_patch(hint)
ax3.text(5.0, 10.72, 'Klik di frame untuk tambah titik', ha='center',
         fontsize=6.5, color=C_ACCENT, zorder=8)

# Zone counter badge
zc = FancyBboxPatch((7.5, 10.5), 1.7, 0.45,
                     boxstyle="round,pad=0,rounding_size=0.1",
                     facecolor=C_FILL, edgecolor=C_GRAY, linewidth=0.8, zorder=7)
ax3.add_patch(zc)
ax3.text(8.35, 10.72, '1 zona', ha='center', fontsize=6.5,
         fontweight='bold', color=C_DARK, zorder=8)

# Toolbar buttons
# Finish zone
draw_button(ax3, 0.6, 3.9, 4.5, 0.8, '+ Selesaikan Zona (6 titik)', filled=False)
# Undo
draw_button(ax3, 5.4, 3.9, 1.5, 0.8, 'Undo', filled=False)
# Clear
draw_button(ax3, 7.2, 3.9, 2.2, 0.8, 'Hapus', filled=False)

# Back button
draw_button(ax3, 0.6, 2.5, 2.5, 1.0, '< Balik', filled=False)
# Save & Start
draw_button(ax3, 3.4, 2.5, 6.0, 1.0, 'Simpan 1 Zona & Mulai', filled=True)

# Info text
ax3.text(5, 1.5, 'Mendukung multi-polygon ROI', ha='center',
         fontsize=7, color=C_ACCENT, style='italic')
ax3.text(5, 1.1, 'Koordinat dinormalisasi (0-1)', ha='center',
         fontsize=6, color=C_GRAY, fontfamily='monospace')


# ══════════════════════════════════════════════════════════════
# SCREEN LABELS
# ══════════════════════════════════════════════════════════════
ax1.text(5, -0.2, '(a) Setup Wizard\nPilih Sumber & Mode ROI', ha='center',
         fontsize=9, fontweight='bold', color=C_DARK)
ax2.text(5, -0.2, '(b) Main Dashboard\nMonitoring Real-Time', ha='center',
         fontsize=9, fontweight='bold', color=C_DARK)
ax3.text(5, -0.2, '(c) ROI Drawing\nGambar Zona Larangan', ha='center',
         fontsize=9, fontweight='bold', color=C_DARK)

# Main title
fig.suptitle('Low-Fidelity Wireframe — Parking Guard Dashboard',
             fontsize=16, fontweight='bold', y=0.98, color=C_DARK)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig(r'D:\Thesis\docs\wireframe_dashboard.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Wireframe saved to D:\\Thesis\\docs\\wireframe_dashboard.png")
plt.close()
