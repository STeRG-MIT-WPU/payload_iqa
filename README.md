# CM5 Two-Coral Screening Pipeline

Real-time image screening pipeline for a **Raspberry Pi Compute Module 5**
with **two Google Coral USB Edge TPUs** and a **USB camera**, plus a **PyQt5
dashboard** to run and monitor the whole thing live. Every part of the
pipeline can also be run and tested from the command line without any GUI.

```
                        ┌────────────────────────────────────────────┐
USB camera ─ 1 frame ──>│ data/captures/  (timestamped JPEG)         │
             every 2 s  └───────────────┬────────────────────────────┘
                                        │
                     ┌──────────────────▼──────────────────┐
                     │  CORAL 1  (usb:0)                   │
                     │  1. IQA score (MOS scale)           │
                     │     score < 60        ──> REJECT    │
                     │  2. Cloud coverage %                │
                     │     coverage > 70 %   ──> REJECT    │
                     │     60–70 %           ──> CROP to   │
                     │        least-cloudy window, keep    │
                     │     < 60 %            ──> PASS      │
                     └──────────────────┬──────────────────┘
                                        │  (only survivors)
                     ┌──────────────────▼──────────────────┐
                     │  CORAL 2  (usb:1)                   │
                     │  3. Face detection                  │
                     │     boxes + scores drawn onto a     │
                     │     *_faces.jpg copy                │
                     └──────────────────┬──────────────────┘
                                        │
             ┌──────────────────────────┴───────────────────────┐
             │ data/passed/    image + annotated *_faces.jpg    │
             │ data/rejected/  image, reason logged             │
             │ data/results.csv  one row per final verdict      │
             └──────────────────────────────────────────────────┘
```

Capture, Coral 1 and Coral 2 run as **three separate OS processes** connected
by bounded queues, so the camera keeps capturing while Coral 1 screens one
image and Coral 2 detects faces in another — all three stages genuinely run
simultaneously. IQA and cloud segmentation share Coral 1: both compiled
models are under 8 MB, so they stay resident on the TPU together.

---

## Repository layout

```
ai_pipeline/
├── run_dashboard.py        PyQt5 dashboard entry point
├── run_headless.py         full pipeline in the terminal, no GUI
├── requirements.txt
│
├── models/
│   ├── iqa/
│   │   ├── iqa_lite0_int8_io_edgetpu.tflite   Edge TPU model (int8 in/out)
│   │   ├── iqa_lite0_int8.tflite              CPU fallback (float32 input)
│   │   └── norm.json                          MOS range + preprocessing
│   ├── cloud_seg/
│   │   ├── cloudseg_int8_edgetpu.tflite       Edge TPU model
│   │   └── cloudseg_int8.tflite               CPU fallback
│   └── face/
│       ├── face_detector_int8_edgetpu.tflite  Edge TPU model
│       ├── face_detector_int8.tflite          CPU fallback
│       └── model_meta.json                    anchors / strides / variances
│
├── pipeline/               the pipeline itself (no GUI code in here)
│   ├── config.py           all paths, thresholds, devices, intervals
│   ├── engines.py          IQAScorer / CloudDetector / FaceDetector
│   │                       + per-stage CLI (python -m pipeline.engines ...)
│   ├── imaging.py          image load/save/resize/annotate helpers
│   ├── camera.py           capture process (USB camera or image folder)
│   ├── workers.py          Coral 1 and Coral 2 worker processes
│   ├── orchestrator.py     Pipeline class: start/stop, event stream, CSV log
│   └── monitor.py          host CPU/RAM sampling (psutil)
│
├── dashboard/
│   ├── main_window.py      the dashboard window
│   └── widgets.py          image panes, sparklines, resource meters
│
├── data/                   runtime output (gitignored)
│   ├── captures/           frames currently in flight
│   ├── passed/             accepted images + *_faces.jpg annotated copies
│   ├── rejected/           rejected images
│   └── results.csv         verdict log
│
└── test-images/            10 satellite test scenes (used by folder mode)
```

---

## The three models

| Stage | Model | Input | Output | Runs on |
|---|---|---|---|---|
| IQA | `iqa_lite0_int8_io_edgetpu.tflite` (EfficientNet-Lite0, int8 in/out) | 224×224 RGB, `(x−127)/128` | quality score, rescaled to MOS via `norm.json` | Coral 1 |
| Cloud segmentation | `cloudseg_int8_edgetpu.tflite` | 4-channel (R,G,B,NIR) in [0,1]; NIR synthesized for RGB inputs (`nir_mode`: mean/red/zero) | per-pixel cloud mask → coverage % | Coral 1 |
| Face detection | `face_detector_int8_edgetpu.tflite` (SSD-style) | 320×320 RGB uint8 | anchor deltas + logits, decoded + NMS on the CPU | Coral 2 |

The engine wrappers read each tensor's dtype and quantization parameters at
runtime, so the same code drives the int8-io Edge TPU models and the
float-input CPU fallbacks. Edge-TPU-compiled `.tflite` files **cannot** run
without a Coral attached — that is why the CPU variants are kept in the repo:
they are what makes development on a machine without Corals possible.

---

## Installation

### On the CM5 (deployment)

```bash
# Edge TPU runtime (one-time)
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update && sudo apt install libedgetpu1-std python3-pyqt5

# Python deps
pip3 install -r requirements.txt
pip3 install tflite-runtime
```

Plug in **both Corals before starting**. Enumeration order decides identity:
the first TPU is `usb:0` (Coral 1 → IQA + cloud), the second `usb:1`
(Coral 2 → faces). If you want a specific physical dongle to be Coral 1,
plug it in first, or change `coral1_device` / `coral2_device` in
`pipeline/config.py`.

> Tip: `libedgetpu1-max` clocks the TPUs higher but they run hot; `-std` is
> the safe default.

### On a dev machine (Windows/Linux/macOS, no Corals needed)

```bash
pip install -r requirements.txt
pip install ai-edge-litert     # TFLite runtime for modern Python versions
```

Everything then runs in CPU mode (`--cpu` flag, or untick "Use Edge TPUs" in
the dashboard). A laptop webcam works as the camera source.

---

## Running — dashboard

```bash
python3 run_dashboard.py
```

Controls along the top:

- **▶ Run / ■ Stop** — starts/stops the whole pipeline (camera + both workers).
- **Source** — `camera` (USB webcam) or `folder (test-images)` to replay the
  bundled satellite scenes.
- **Use Edge TPUs** — untick to force CPU models.
- **Interval s** — seconds between captures (default 2).

What you see while it runs:

- **Latest capture** and **latest face detection** previews (left).
- **All images** tab — one row per capture: IQA score, cloud %, face count,
  per-stage latency in ms, and a color-coded verdict
  (`PASSED` / `CROPPED+PASSED` / `REJECTED (iqa)` / `REJECTED (cloud)`).
- **Rejected** tab — thumbnail, which model rejected it, and the exact reason
  (e.g. `IQA 58.8 < 60`). Click an entry to view the image full-size.
- **Log** tab — engine startup messages (which device each model landed on),
  crop notifications, errors.
- **Right panel** — session counters, CPU (total + per core), RAM, and one
  load meter per Coral.

**About the TPU meters:** Coral Edge TPUs expose no hardware utilization
counter, so the meters show the **inference duty cycle** — the share of
wall-clock time each TPU spends inside `invoke()`, reported once a second by
the worker that owns it. 0 % = idle, 100 % = back-to-back inference.

In folder mode the pipeline stops itself after the last image; in camera mode
it runs until you press Stop.

---

## Running — without the GUI

### Full pipeline in the terminal

```bash
python3 run_headless.py                       # camera + both Corals (CM5)
python3 run_headless.py --cpu                 # camera, CPU models
python3 run_headless.py --source folder --cpu # test images, exits when done
python3 run_headless.py --source folder --input /path/to/images --loop
python3 run_headless.py --duration 60         # stop after 60 s
python3 run_headless.py --interval 0.5        # capture faster than 2 s
```

Console output mirrors the dashboard, e.g.:

```
[CAPTURE] img_20260719_174828_343.jpg
[IQA    ] img_...jpg: score 63.37 (4 ms) -> ok
[CLOUD  ] img_...jpg: coverage 0.1% (5 ms) -> pass
[FACE   ] img_...jpg: 1 face(s) (7 ms)
[PASS   ] img_...jpg: IQA 63.4, cloud 0.1%, 1 face(s) -> data/passed/...
[resmon ] CPU 14%  RAM 76% (11.9/15.6 GB)  |  coral1[usb:0] 41%  |  coral2[usb:1] 12%

summary: 10 captured | 3 rejected by IQA | 2 rejected by cloud | 5 passed | 2 faces
```

Thresholds can be overridden per run: `--iqa-min`, `--cloud-reject`,
`--cloud-crop`, `--face-threshold`.

### Single model on a single image

The fastest way to sanity-check one model or one Coral:

```bash
python3 -m pipeline.engines iqa   --image photo.jpg --device usb:0
python3 -m pipeline.engines cloud --image scene.tif --device usb:0
python3 -m pipeline.engines face  --image people.jpg --device usb:1 --save out.jpg
python3 -m pipeline.engines all   --image photo.jpg           # CPU, all 3 stages
```

Omit `--device` to run on the CPU. `face --save` writes an annotated copy.

---

## Configuration

Defaults live in `pipeline/config.py` (`Config` dataclass); anything the CLI
or dashboard exposes just overrides them per run.

| Setting | Default | Meaning |
|---|---|---|
| `capture_interval` | `2.0` s | time between camera captures |
| `frame_width/height` | 1280×720 | requested camera resolution (MJPG) |
| `iqa_min` | `60.0` | reject below this IQA (MOS) score |
| `cloud_reject` | `70.0` % | reject above this cloud coverage |
| `cloud_crop` | `60.0` % | between crop and reject → crop to least-cloudy window |
| `face_threshold` | `0.5` | face detector confidence cutoff |
| `nir_mode` | `mean` | synthesized NIR band: `mean` / `red` / `zero` |
| `coral1_device` | `usb:0` | TPU for IQA + cloud segmentation |
| `coral2_device` | `usb:1` | TPU for face detection |
| `cpu_fallback` | `True` | drop to CPU models if a TPU is missing |
| `queue_size` | `16` | bound on inter-stage queues |

---

## Outputs

- **`data/passed/`** — accepted images. Cropped ones carry a `_cropped`
  suffix. Each accepted image also gets a `*_faces.jpg` annotated copy with
  red face boxes and confidence scores.
- **`data/rejected/`** — images rejected by IQA or cloud coverage.
- **`data/results.csv`** — one row per final verdict:

  ```
  timestamp, image, result, stage, reason, iqa, cloud, faces, path
  2026-07-19 17:47:08, na_coast_01_...jpg, passed,   face,  ,               63.37, 0.1, 1, ...
  2026-07-19 17:47:08, na_coast_02_...jpg, rejected, iqa,   IQA 58.8 < 60,  58.80,    ,  , ...
  ```

- **`data/captures/`** — only holds frames currently in flight; every image
  ends up in `passed/` or `rejected/`.

---

## Architecture notes

- **Processes, not threads.** Python threads would serialize the NumPy/TFLite
  work behind the GIL and stutter the GUI. Each stage is a
  `multiprocessing.Process` (spawn context, identical behavior on Windows and
  the CM5); the GUI process never runs inference.
- **Event stream.** Workers publish typed events (`capture`, `iqa`, `cloud`,
  `face`, `rejected`, `passed`, `util`, `status`, `error`, `done`) onto a
  shared queue. `Pipeline.poll_events()` drains it; the dashboard and the
  headless runner are just two different consumers of the same stream, which
  is what makes GUI-free operation a first-class mode rather than an
  afterthought.
- **Backpressure.** Inter-stage queues are bounded (16). If Coral 1 falls
  behind, the capture process blocks instead of eating RAM.
- **End-of-source handling.** In folder mode the capture process sends a
  sentinel that flows through both workers; Coral 2 turns it into a `done`
  event so headless runs exit and the dashboard stops itself.
- **Windows quirk.** Previewing an image (QPixmap) can hold its file open for
  a moment, which makes renames fail on Windows; workers therefore use a
  retrying move (`_safe_move`). Irrelevant on the CM5, harmless everywhere.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Edge TPU delegate failed for usb:0` | Coral not plugged in / not enumerated. Check `lsusb` (look for `18d1:9302` Google Inc., or `1a6e:089a` before first inference). Reconnect and restart. On Linux make sure the `apex`/udev rules from `libedgetpu1-std` are installed and you're in the `plugdev` group. |
| Pipeline silently runs on CPU | That's `cpu_fallback` doing its job — the Log tab / `[status]` lines say exactly which device each model landed on. Set `cpu_fallback=False` in `config.py` to hard-fail instead. |
| Only one Coral detected | Both TPUs on one USB hub can starve for power; use separate ports or a powered hub. |
| `could not open camera 0` | Wrong index (`--camera-index 1`), or another app is holding the camera. On Linux check `v4l2-ctl --list-devices`. |
| No TFLite interpreter found | `pip install tflite-runtime` (CM5) or `pip install ai-edge-litert` (dev machine). |
| IQA stage fails in CPU mode | Don't delete `models/iqa/iqa_lite0_int8.tflite` — the Edge TPU model cannot run on CPU. |
| Slow first inference | Normal: the first `invoke()` uploads model parameters to the TPU. |

---

## Provenance

- IQA: EfficientNet-Lite0 trained for MOS prediction, QAT + Edge TPU
  compiled (`norm.json` holds the MOS range and preprocessing).
- Cloud segmentation: QAT U-Net-style segmenter, trained on 4-band
  (R,G,B,NIR) satellite data.
- Face detection: SSD-style detector, anchors/decoding described by
  `model_meta.json`.
- Pre-restructure history (training weights, the original single-Coral
  two-stage script) lives in git history — see commit `fe2eff8`.
