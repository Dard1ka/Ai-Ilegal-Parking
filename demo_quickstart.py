"""
demo_quickstart.py
══════════════════════════════════════════════════════════════
Quick-start script for Illegal Parking Detection System.

This is a thin wrapper around demo3_tts.py that validates
your environment before running the main pipeline.

Usage:
    python demo_quickstart.py
    python demo_quickstart.py --video path/to/video.mp4
    python demo_quickstart.py --video input4.mp4 --no-tts

Requirements:
    pip install -r requirements.txt
    Models: best.pt (vehicle), triangle.pt (warning sign)
"""

import sys
import argparse
import subprocess
from pathlib import Path


# ── ANSI colors ───────────────────────────────────────────────
GRN = "\033[92m"
YEL = "\033[93m"
RED = "\033[91m"
CYN = "\033[96m"
RST = "\033[0m"
BLD = "\033[1m"


def check(label: str, ok: bool, hint: str = ""):
    status = f"{GRN}✓{RST}" if ok else f"{RED}✗{RST}"
    print(f"  {status}  {label}")
    if not ok and hint:
        print(f"     {YEL}→ {hint}{RST}")
    return ok


def run_checks(video_path: str) -> bool:
    print(f"\n{BLD}{CYN}Illegal Parking Detection — Environment Check{RST}")
    print("─" * 50)

    all_ok = True

    # Python version
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 9
    all_ok &= check(f"Python {v.major}.{v.minor}.{v.micro}", ok,
                    "Requires Python 3.9+")

    # Package checks
    packages = {
        "cv2":          ("opencv-python", "pip install opencv-python"),
        "numpy":        ("numpy",         "pip install numpy"),
        "ultralytics":  ("ultralytics",   "pip install ultralytics"),
        "pygame":       ("pygame",        "pip install pygame"),
        "edge_tts":     ("edge-tts",      "pip install edge-tts"),
    }
    for mod, (pkg, install_cmd) in packages.items():
        try:
            __import__(mod)
            ok = True
        except ImportError:
            ok = False
            all_ok = False
        check(f"Package: {pkg}", ok, install_cmd)

    # Model files
    for model in ["best.pt", "triangle.pt"]:
        p = Path(model)
        ok = p.exists()
        all_ok &= check(f"Model: {model} ({p.stat().st_size // 1024 // 1024} MB)" if ok else f"Model: {model}",
                        ok, f"Download atau letakkan {model} di folder yang sama")

    # Video file
    vp = Path(video_path)
    ok = vp.exists()
    all_ok &= check(f"Video: {vp.name}", ok,
                    f"File tidak ditemukan: {video_path}")

    print("─" * 50)
    return all_ok


def patch_config(video_path: str, no_tts: bool):
    """Patch config di demo3_tts.py untuk video path dan TTS flag."""
    src = Path("demo3_tts.py").read_text(encoding="utf-8")

    src = src.replace(
        'VIDEO_PATH = "input4.mp4"',
        f'VIDEO_PATH = "{video_path}"'
    )
    if no_tts:
        src = src.replace("ENABLE_TTS = True", "ENABLE_TTS = False")

    patched = Path("_demo3_tts_run.py")
    patched.write_text(src, encoding="utf-8")
    return patched


def main():
    parser = argparse.ArgumentParser(
        description="Illegal Parking Detection — Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_quickstart.py
  python demo_quickstart.py --video input4.mp4
  python demo_quickstart.py --video recording.mp4 --no-tts
        """
    )
    parser.add_argument("--video",  default="input4.mp4",
                        help="Path to input video file (default: input4.mp4)")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable Text-to-Speech alerts")
    args = parser.parse_args()

    print(f"""
{BLD}╔══════════════════════════════════════════════════╗
║   Illegal Parking Detection System               ║
║   YOLOv8 + BotSORT + Edge-TTS + FastAPI          ║
╚══════════════════════════════════════════════════╝{RST}
""")

    # Validate environment
    if not run_checks(args.video):
        print(f"\n{RED}{BLD}Cek gagal. Perbaiki masalah di atas sebelum melanjutkan.{RST}\n")
        sys.exit(1)

    print(f"\n{GRN}{BLD}Semua cek passed!{RST}")
    print(f"\n{CYN}Memulai deteksi...{RST}")
    print(f"  Video : {args.video}")
    print(f"  TTS   : {'Disabled' if args.no_tts else 'Enabled (id-ID-GadisNeural)'}")
    print(f"\n{YEL}Tip:{RST} Setelah window terbuka, gambar ROI polygon dengan klik kiri.")
    print(f"     Tekan ENTER saat polygon selesai. ESC untuk keluar.\n")

    # Patch and run
    try:
        if args.video != "input4.mp4" or args.no_tts:
            patched = patch_config(args.video, args.no_tts)
            subprocess.run([sys.executable, str(patched)], check=True)
            patched.unlink(missing_ok=True)
        else:
            subprocess.run([sys.executable, "demo3_tts.py"], check=True)
    except KeyboardInterrupt:
        print(f"\n{YEL}Dihentikan oleh user.{RST}")
    except subprocess.CalledProcessError as e:
        print(f"\n{RED}Error saat menjalankan: {e}{RST}")
        sys.exit(1)


if __name__ == "__main__":
    main()
