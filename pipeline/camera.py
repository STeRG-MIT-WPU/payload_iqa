"""Capture stage: USB camera (1 frame every N seconds) or a folder of images.

Runs as its own process. Every captured/ingested frame is written to
data/captures/ as a timestamped JPEG and its path is pushed to the Coral-1
queue. In folder mode a ``None`` sentinel is sent when the folder is
exhausted (unless loop_folder is set), which lets headless runs terminate
by themselves.
"""
import queue
import sys
import time
from pathlib import Path

from . import config
from .imaging import load_image, save_image, list_images


def _emit(event_q, **ev):
    ev.setdefault("t", time.time())
    try:
        event_q.put_nowait(ev)
    except queue.Full:
        pass


def _stamp_name(prefix="img"):
    now = time.time()
    ms = int((now % 1) * 1000)
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(now))}_{ms:03d}.jpg"


def _enqueue(out_q, event_q, stop_evt, item):
    """Blocking put that stays responsive to the stop event."""
    while not stop_evt.is_set():
        try:
            out_q.put(item, timeout=0.25)
            return True
        except queue.Full:
            continue
    return False


def capture_worker(cfg, out_q, event_q, stop_evt):
    try:
        if cfg.source == "folder":
            _folder_loop(cfg, out_q, event_q, stop_evt)
        else:
            _camera_loop(cfg, out_q, event_q, stop_evt)
    except Exception as e:
        _emit(event_q, type="error", stage="capture", msg=str(e))


def _camera_loop(cfg, out_q, event_q, stop_evt):
    import cv2

    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    cap = cv2.VideoCapture(cfg.camera_index, backend)
    if not cap.isOpened():
        _emit(event_q, type="error", stage="capture",
              msg=f"could not open camera {cfg.camera_index}")
        return

    # MJPG keeps USB webcams at full fps; buffer of 1 = always the newest frame
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _emit(event_q, type="status", stage="capture",
          msg=f"camera {cfg.camera_index} open at {w}x{h}, "
              f"1 frame every {cfg.capture_interval:g}s")

    next_shot = time.monotonic()
    try:
        while not stop_evt.is_set():
            ok, frame = cap.read()   # keep draining so the buffer stays fresh
            if not ok:
                _emit(event_q, type="error", stage="capture",
                      msg="camera read failed")
                time.sleep(0.5)
                continue
            if time.monotonic() < next_shot:
                continue
            next_shot = time.monotonic() + cfg.capture_interval

            name = _stamp_name()
            path = config.CAPTURES_DIR / name
            cv2.imwrite(str(path), frame)
            _emit(event_q, type="capture", image=name, path=str(path))
            if not _enqueue(out_q, event_q, stop_evt,
                            {"name": name, "path": str(path)}):
                break
    finally:
        cap.release()


def _folder_loop(cfg, out_q, event_q, stop_evt):
    files = list_images(cfg.input_folder)
    if not files:
        _emit(event_q, type="error", stage="capture",
              msg=f"no images in {cfg.input_folder}")
        out_q.put(None)
        return
    _emit(event_q, type="status", stage="capture",
          msg=f"feeding {len(files)} images from {cfg.input_folder} "
              f"every {cfg.capture_interval:g}s"
              + (" (looping)" if cfg.loop_folder else ""))

    i = 0
    while not stop_evt.is_set():
        src = files[i % len(files)]
        # normalize everything (incl. 16-bit TIFs) to a uint8 JPEG "capture"
        name = _stamp_name(prefix=src.stem)
        path = config.CAPTURES_DIR / name
        save_image(path, load_image(src))
        _emit(event_q, type="capture", image=name, path=str(path), source=str(src))
        if not _enqueue(out_q, event_q, stop_evt, {"name": name, "path": str(path)}):
            return

        i += 1
        if i >= len(files) and not cfg.loop_folder:
            break
        deadline = time.monotonic() + cfg.capture_interval
        while time.monotonic() < deadline and not stop_evt.is_set():
            time.sleep(0.05)

    if not stop_evt.is_set():
        _emit(event_q, type="status", stage="capture", msg="source exhausted")
        out_q.put(None)   # sentinel: no more images
