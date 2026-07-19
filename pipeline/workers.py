"""The two Coral worker processes.

coral1_worker: IQA + cloud segmentation, both interpreters resident on the
               first Edge TPU (each model is < 8 MB so they share it well).
coral2_worker: face detection on the second Edge TPU.

Each worker reports per-inference latency and, once a second, its TPU duty
cycle (fraction of wall-clock time spent inside invoke()) — the closest
thing to a "TPU utilization %" the Coral exposes.
"""
import queue
import shutil
import time
from pathlib import Path

from . import config
from .engines import (EngineError, IQAScorer, CloudDetector, FaceDetector,
                      find_best_crop, crop_image)
from .imaging import load_image, save_image, annotate_faces


def _emit(event_q, **ev):
    ev.setdefault("t", time.time())
    try:
        event_q.put_nowait(ev)
    except queue.Full:
        pass


def _safe_move(src, dst, retries=8, delay=0.1):
    """shutil.move that tolerates a reader briefly holding the file open
    (on Windows the dashboard's preview can block a rename for a moment)."""
    for i in range(retries):
        try:
            shutil.move(src, dst)
            return
        except (PermissionError, OSError):
            if i == retries - 1:
                break
            time.sleep(delay)
    shutil.copy2(src, dst)   # last resort: copy, then best-effort cleanup
    try:
        Path(src).unlink()
    except OSError:
        pass


def _build(event_q, stage, factory_tpu, factory_cpu, device, cpu_fallback):
    """Build an engine on the requested TPU, falling back to CPU if allowed."""
    if device:
        try:
            eng = factory_tpu()
            _emit(event_q, type="status", stage=stage,
                  msg=f"{stage}: Edge TPU {device} ready")
            return eng, device
        except EngineError as e:
            if not cpu_fallback:
                raise
            _emit(event_q, type="status", stage=stage,
                  msg=f"{stage}: {e} -- falling back to CPU")
    eng = factory_cpu()
    _emit(event_q, type="status", stage=stage, msg=f"{stage}: running on CPU")
    return eng, None


class _DutyMeter:
    """Rolls engine busy-time into a once-a-second duty-cycle event."""

    def __init__(self, event_q, tpu_name, device, engines):
        self.event_q = event_q
        self.tpu_name = tpu_name
        self.device = device or "cpu"
        self.engines = engines
        self.t0 = time.monotonic()

    def tick(self):
        now = time.monotonic()
        elapsed = now - self.t0
        if elapsed < 1.0:
            return
        busy = sum(e.take_busy_seconds() for e in self.engines)
        _emit(self.event_q, type="util", tpu=self.tpu_name, device=self.device,
              duty=min(100.0, 100.0 * busy / elapsed))
        self.t0 = now


def coral1_worker(cfg, in_q, out_q, event_q, stop_evt):
    """IQA + cloud segmentation on Coral 1."""
    try:
        dev = cfg.coral1_device if cfg.use_edgetpu else None
        iqa, dev_used = _build(
            event_q, "iqa",
            lambda: IQAScorer(device=dev, num_threads=cfg.num_threads),
            lambda: IQAScorer(device=None, num_threads=cfg.num_threads),
            dev, cfg.cpu_fallback)
        cloud, _ = _build(
            event_q, "cloud",
            lambda: CloudDetector(device=dev, nir_mode=cfg.nir_mode,
                                  num_threads=cfg.num_threads),
            lambda: CloudDetector(device=None, nir_mode=cfg.nir_mode,
                                  num_threads=cfg.num_threads),
            dev, cfg.cpu_fallback)
    except Exception as e:
        _emit(event_q, type="error", stage="coral1", msg=f"engine init failed: {e}")
        out_q.put(None)
        return

    meter = _DutyMeter(event_q, "coral1", dev_used, [iqa, cloud])
    while not stop_evt.is_set():
        meter.tick()
        try:
            item = in_q.get(timeout=0.25)
        except queue.Empty:
            continue
        if item is None:                    # end of source: pass it downstream
            out_q.put(None)
            break
        try:
            _process_coral1(cfg, item, out_q, event_q, stop_evt, iqa, cloud)
        except Exception as e:
            _emit(event_q, type="error", stage="coral1",
                  msg=f"{item.get('name', '?')}: {e}")


def _process_coral1(cfg, item, out_q, event_q, stop_evt, iqa, cloud):
    name, path = item["name"], item["path"]
    img = load_image(path)

    # --- stage 1: IQA -------------------------------------------------------
    score = iqa.score(img)
    ok = score >= cfg.iqa_min
    _emit(event_q, type="iqa", image=name, score=score, ms=iqa.last_ms, ok=ok)
    if not ok:
        dst = config.REJECTED_DIR / name
        _safe_move(path, dst)
        _emit(event_q, type="rejected", image=name, stage="iqa",
              reason=f"IQA {score:.1f} < {cfg.iqa_min:.0f}",
              iqa=score, cloud=None, path=str(dst))
        return

    # --- stage 2: cloud coverage -------------------------------------------
    cov, mask = cloud.coverage(img)
    if cov > cfg.cloud_reject:
        action = "reject"
    elif cov >= cfg.cloud_crop:
        action = "crop"
    else:
        action = "pass"
    _emit(event_q, type="cloud", image=name, coverage=cov, ms=cloud.last_ms,
          action=action)

    if action == "reject":
        dst = config.REJECTED_DIR / name
        _safe_move(path, dst)
        _emit(event_q, type="rejected", image=name, stage="cloud",
              reason=f"cloud {cov:.1f}% > {cfg.cloud_reject:.0f}%",
              iqa=score, cloud=cov, path=str(dst))
        return

    meta = {"name": name, "path": path, "iqa": score, "cloud": cov,
            "cropped": False}
    if action == "crop":
        box, crop_cov = find_best_crop(mask)
        crop_path = config.CAPTURES_DIR / (Path(name).stem + "_cropped.jpg")
        save_image(crop_path, crop_image(img, box))
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass   # a preview may hold the original briefly; not fatal
        meta.update(name=crop_path.name, path=str(crop_path),
                    cropped=True, crop_cov=crop_cov)
        _emit(event_q, type="status", stage="coral1",
              msg=f"{name}: cropped to least-cloudy window "
                  f"({crop_cov:.1f}% cloud in crop)")

    # forward to Coral 2 (blocking put, responsive to stop)
    while not stop_evt.is_set():
        try:
            out_q.put(meta, timeout=0.25)
            return
        except queue.Full:
            continue


def coral2_worker(cfg, in_q, event_q, stop_evt):
    """Face detection on Coral 2."""
    try:
        dev = cfg.coral2_device if cfg.use_edgetpu else None
        face, dev_used = _build(
            event_q, "face",
            lambda: FaceDetector(device=dev, num_threads=cfg.num_threads),
            lambda: FaceDetector(device=None, num_threads=cfg.num_threads),
            dev, cfg.cpu_fallback)
    except Exception as e:
        _emit(event_q, type="error", stage="coral2", msg=f"engine init failed: {e}")
        return

    meter = _DutyMeter(event_q, "coral2", dev_used, [face])
    while not stop_evt.is_set():
        meter.tick()
        try:
            item = in_q.get(timeout=0.25)
        except queue.Empty:
            continue
        if item is None:                    # end of source
            _emit(event_q, type="done")
            break
        try:
            _process_coral2(cfg, item, event_q, face)
        except Exception as e:
            _emit(event_q, type="error", stage="coral2",
                  msg=f"{item.get('name', '?')}: {e}")


def _process_coral2(cfg, item, event_q, face):
    name, path = item["name"], item["path"]
    img = load_image(path)

    boxes, scores = face.detect(img, cfg.face_threshold)
    annotated = config.PASSED_DIR / (name.rsplit(".", 1)[0] + "_faces.jpg")
    annotate_faces(img, boxes, scores, annotated)

    dst = config.PASSED_DIR / name
    _safe_move(path, dst)
    _emit(event_q, type="face", image=name, count=len(boxes), ms=face.last_ms,
          scores=[float(s) for s in scores], annotated=str(annotated))
    _emit(event_q, type="passed", image=name, path=str(dst),
          annotated=str(annotated), iqa=item.get("iqa"), cloud=item.get("cloud"),
          cropped=item.get("cropped", False), faces=len(boxes))
