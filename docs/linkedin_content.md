# LinkedIn Content — Illegal Parking Detection System

Semua teks di bawah ini siap copy-paste ke LinkedIn.
Pilih salah satu post sesuai timing (launch, follow-up, atau technical deep-dive).

---

## 📌 POST 1 — Project Launch Post (gunakan ini saat push ke GitHub)

**Tipe:** Activity Post  
**Timing:** Setelah repository public  
**Tone:** Casual tapi profesional

---

🅿️ **Selesai nulis thesis yang kayaknya bakal jadi salah satu project paling bangga yang pernah gue bikin.**

Gue baru nge-launch **Illegal Parking Detection System** — sistem AI yang bisa otomatis deteksi kendaraan yang parkir sembarangan secara real-time, lengkap dengan dashboard web dan peringatan suara Bahasa Indonesia.

**Apa yang bisa sistem ini lakuin:**
✅ Deteksi kendaraan (mobil, motor, bus, truk) pakai **YOLOv8**
✅ Tracking multi-objek dengan **BotSORT** + algoritma re-linking custom
✅ Monitor **beberapa zona larangan** sekaligus (gambar polygon di dashboard)
✅ Deteksi **segitiga pengaman** → kendaraan mogok dikecualikan dari pelanggaran
✅ Identifikasi warna kendaraan (12 warna via HSV classification)
✅ Peringatan suara otomatis pakai **Microsoft Edge TTS** bahasa Indonesia
✅ Dashboard real-time pakai **React + FastAPI + WebSocket**

**Timeline detection:**
🟡 60 detik → Peringatan  
🔴 300 detik → Pelanggaran + Alert suara

Yang paling challenging: bikin algoritma re-linking supaya kendaraan yang sempat tertutup objek lain (okluded) tetap terlacak dan hitungan parkir-nya nggak ke-reset.

Tech stack: **Python • YOLOv8 • OpenCV • FastAPI • React • TailwindCSS • Edge-TTS • BotSORT**

GitHub: [link repo]

#ComputerVision #MachineLearning #YOLOv8 #Python #React #FastAPI #ObjectDetection #SmartCity #AI #DeepLearning #Thesis #OpenCV

---

## 📌 POST 2 — Technical Deep Dive (1-2 minggu setelah launch)

**Tipe:** Activity Post — carousel atau text  
**Tone:** Teknikal, educational

---

🧠 **Breakdown teknikal sistem deteksi parkir liar yang baru gue build.**

Banyak yang nanya soal cara kerjanya. Here's the breakdown:

**The core problem:**  
YOLO bisa deteksi kendaraan, tapi kalau kendaraan ketutupan sebentar (misalnya ada mobil lewat), tracker-nya sering kasih ID baru. Padahal kendaraan yang parkir udah diem 4 menit — kalau ID-nya ganti, hitungan waktunya ke-reset ke 0. Fatal.

**Solusinya — Custom Re-linking Algorithm:**

Pas ada deteksi baru, sistem lakuin 3 pass sebelum bikin record baru:

1️⃣ **Pass 1** — Cek hidden records yang udah "parked" (durasi lebih panjang diprioritasin)  
2️⃣ **Pass 2** — Fallback ke hidden records yang baru masuk  
3️⃣ **Visible merge** — Kalau deteksi baru overlap dengan record yang udah keliatan

**Scoring function per kandidat:**
```
score = IoU × 500 - center_dist × 1.2 - foot_dist × 2.2 + parked_duration × 8
```

Record yang udah lama diparkir dapat bonus besar (`× 8`) — biar nggak "dicuri" sama deteksi lain yang posisinya kebetulan mirip.

**Hasilnya:** Kendaraan yang ketutupan sampai 80 detik tetap terlacak dengan ID yang sama dan hitungan parkir yang akurat.

Tech: YOLOv8 + BotSORT + custom Python re-linking + FastAPI + React

#ComputerVision #ObjectTracking #YOLOv8 #Python #Algorithm #BotSORT #DeepLearning

---

## 📌 POST 3 — Dashboard Showcase (bisa post bareng screenshot/demo video)

**Tipe:** Activity Post dengan media (screenshot/video)  
**Tone:** Visual, showcase

---

🖥️ **Dashboard real-time buat deteksi parkir liar — dari backend AI sampai frontend-nya.**

Setelah sistem deteksinya jadi, gue build juga dashboard web supaya bisa dipake tanpa buka terminal.

**Flow penggunaan:**
1. 📁 Upload video atau sambungin kamera langsung
2. ✏️ Gambar zona larangan parkir langsung di frame (bisa banyak zona)
3. 📡 Dashboard langsung streaming + nampilin alert real-time

**Tech yang dipake:**
- **Backend:** FastAPI + WebSocket + MJPEG streaming
- **Frontend:** React 18 + Vite + TailwindCSS
- **Communication:** WebSocket untuk alert, MJPEG untuk video stream

Fitur yang gue suka: kalau user mau ganti zona pantau, bisa langsung gambar ulang polygon dari dashboard tanpa restart sistem.

[Tambahkan screenshot/GIF di sini]

GitHub: [link repo]

#WebDev #React #FastAPI #Python #Dashboard #ComputerVision #FullStack #TailwindCSS

---

## 📌 LINKEDIN PROFILE — Projects Section

**Tambahkan di LinkedIn Profile → "Projects"**

---

**Project Name:**  
Illegal Parking Detection System

**Project URL:**  
[GitHub repo link]

**Date:**  
2024 – 2025

**Description (copy this):**

> AI-powered real-time illegal parking detection system built as undergraduate thesis project.
>
> The system uses YOLOv8 for vehicle and warning triangle detection, BotSORT for multi-object tracking, and a custom re-linking algorithm to maintain tracking continuity across occlusions (up to 80 seconds).
>
> Key technical achievements:
> • Custom re-linking algorithm preventing duration reset when tracker reassigns IDs
> • 3-stage alert system: Monitoring → Warning (60s) → Violation (300s) with automatic exception for emergency vehicles (triangle detection)
> • HSV-based vehicle color classification (12 colors, no additional ML model needed)
> • Non-blocking Indonesian TTS alerts via Microsoft Edge-TTS + pygame threading
> • Real-time web dashboard: React + FastAPI + WebSocket + MJPEG streaming
>
> Built with: Python, YOLOv8, OpenCV, BotSORT, FastAPI, React, TailwindCSS, Edge-TTS

**Skills to tag:**
Python, Computer Vision, Object Detection, Deep Learning, FastAPI, React.js, OpenCV, YOLOv8, Machine Learning, Full-Stack Development

---

## 📌 TIPS POSTING

### Do's ✅
- Post di **Selasa/Rabu/Kamis pagi** (09.00–11.00 WIB) — engagement tertinggi
- Attach **screenshot atau video demo** di Post 3 — post dengan media dapat reach lebih tinggi
- Reply ke **setiap komentar** dalam 1 jam pertama — ini sinyal engagement yang boost distribusi
- Tag dosen pembimbing kalau beliau ada di LinkedIn (minta permission dulu)
- Gunakan bahasa campuran Indonesia-English — audience lokal lebih relate

### Don'ts ❌
- Jangan semua 3 post dalam satu hari
- Jangan terlalu banyak hashtag (10-15 sudah cukup, LinkedIn bukan Instagram)
- Jangan cuma paste link repo tanpa context — tulis insight/cerita di baliknya

### Timing yang disarankan:
| Post | Kapan |
|---|---|
| Post 1 (Launch) | Hari repo public |
| Post 2 (Technical) | 1–2 minggu setelah Post 1 |
| Post 3 (Dashboard) | 1 minggu setelah Post 2 |
| Projects section | Update bersamaan dengan Post 1 |

---

## 📌 ENGAGEMENT HACK

Setelah post, langsung:
1. Share post ke story LinkedIn kamu
2. Kirim link ke teman-teman yang relevan (minta mereka like/comment)
3. Comment sendiri di post kamu dengan info tambahan (extend thread)

Algoritma LinkedIn mengutamakan post yang dapat interaksi dalam 1 jam pertama.
