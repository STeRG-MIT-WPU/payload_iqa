"""Model engines: IQA, cloud segmentation, face detection.

Each engine wraps one TFLite interpreter, optionally delegated to a specific
Edge TPU ("usb:0", "usb:1"). With ``device=None`` the CPU int8 models are used
instead, so the whole pipeline can be exercised on a dev machine without
Corals or a camera.

Standalone testing (no GUI needed):

    python -m pipeline.engines iqa   --image photo.jpg
    python -m pipeline.engines cloud --image scene.tif
    python -m pipeline.engines face  --image group.jpg --save annotated.jpg
    python -m pipeline.engines all   --image photo.jpg

Add ``--device usb:0`` (or usb:1) to run on a Coral; default is CPU.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import config
from .imaging import load_image, resize, annotate_faces

# --- TFLite interpreter: tflite-runtime (CM5) -> LiteRT -> TensorFlow -------
load_delegate = None
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter, load_delegate
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
            from tensorflow.lite.experimental import load_delegate
        except ImportError:
            sys.exit("No TFLite interpreter found. Run: pip install tflite-runtime "
                     "(CM5) or ai-edge-litert")

EDGETPU_LIB = {
    "win32": "edgetpu.dll",
    "darwin": "libedgetpu.1.dylib",
}.get(sys.platform, "libedgetpu.so.1")


class EngineError(RuntimeError):
    pass


def make_interpreter(model_path, device=None, num_threads=4):
    """Build an allocated interpreter, on a specific Edge TPU if requested."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise EngineError(f"model not found: {model_path}")
    delegates = []
    if device:
        if load_delegate is None:
            raise EngineError("this TFLite runtime has no load_delegate; "
                              "cannot use the Edge TPU")
        try:
            delegates = [load_delegate(EDGETPU_LIB, {"device": device})]
        except Exception as e:
            raise EngineError(f"Edge TPU delegate failed for {device!r}: {e}")
    interp = Interpreter(model_path=str(model_path),
                         experimental_delegates=delegates,
                         num_threads=num_threads)
    interp.allocate_tensors()
    return interp


class _Engine:
    """Shared invoke timing: last latency + accumulated busy time, which the
    workers turn into a TPU duty-cycle figure."""

    def __init__(self):
        self.last_ms = 0.0
        self._busy_s = 0.0

    def _invoke(self):
        t0 = time.perf_counter()
        self.interp.invoke()
        dt = time.perf_counter() - t0
        self.last_ms = dt * 1000.0
        self._busy_s += dt

    def take_busy_seconds(self):
        b, self._busy_s = self._busy_s, 0.0
        return b

    def _set_input(self, x_float):
        """Feed a float32 array, quantizing if the input tensor is int8/uint8."""
        d = self.inp
        if d["dtype"] == np.float32:
            self.interp.set_tensor(d["index"], x_float[None])
        else:
            scale, zp = d["quantization"]
            info = np.iinfo(d["dtype"])
            q = np.clip(np.round(x_float / scale + zp), info.min, info.max)
            self.interp.set_tensor(d["index"], q.astype(d["dtype"])[None])

    def _get_output(self, detail):
        t = self.interp.get_tensor(detail["index"])
        if detail["dtype"] != np.float32:
            scale, zp = detail["quantization"]
            t = (t.astype(np.float32) - zp) * scale
        return t


class IQAScorer(_Engine):
    """Stage 1: image quality score on the MOS scale."""

    def __init__(self, device=None, num_threads=4,
                 model_path=None, norm_path=config.IQA_NORM):
        super().__init__()
        if model_path is None:
            model_path = config.IQA_MODEL_EDGETPU if device else config.IQA_MODEL_CPU
        with open(norm_path) as f:
            norm = json.load(f)
        self.mos_min = norm["MOS_MIN"]
        self.mos_max = norm["MOS_MAX"]
        self.input_size = norm.get("input_size", 224)
        self.interp = make_interpreter(model_path, device, num_threads)
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def score(self, img):
        """Return the IQA score (MOS scale) for a HxWx3 uint8 RGB image."""
        x = resize(img, self.input_size).astype(np.float32)
        x = (x - 127.0) / 128.0
        self._set_input(x)
        self._invoke()
        y = float(self._get_output(self.out).reshape(-1)[0])
        return y * (self.mos_max - self.mos_min) + self.mos_min


class CloudDetector(_Engine):
    """Stage 2: cloud coverage %. The model wants 4 channels (R,G,B,NIR) in
    [0,1]; for RGB inputs a NIR band is synthesized (nir_mode)."""

    def __init__(self, device=None, nir_mode="mean", num_threads=4,
                 model_path=None):
        super().__init__()
        if model_path is None:
            model_path = config.CLOUD_MODEL_EDGETPU if device else config.CLOUD_MODEL_CPU
        self.nir_mode = nir_mode
        self.interp = make_interpreter(model_path, device, num_threads)
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.size = int(self.inp["shape"][1])

    def _make_nir(self, rgb01):
        if self.nir_mode == "mean":
            return rgb01.mean(axis=-1)
        if self.nir_mode == "red":
            return rgb01[..., 0]
        if self.nir_mode == "zero":
            return np.zeros(rgb01.shape[:2], dtype=np.float32)
        raise ValueError(f"unknown nir_mode {self.nir_mode!r}")

    def coverage(self, img):
        """Return (coverage_percent, mask) for a HxWx3 uint8 RGB image."""
        rgb01 = resize(img, self.size).astype(np.float32) / 255.0
        x = np.dstack([rgb01, self._make_nir(rgb01)])
        self._set_input(x)
        self._invoke()
        raw = self._get_output(self.out)[0, ..., 0]
        mask = raw > 0.5
        return 100.0 * float(mask.mean()), mask


def find_best_crop(mask, scales=(0.75, 0.6, 0.5), accept_pct=40.0, stride=8):
    """Least-cloudy crop window in a boolean cloud mask.

    Returns ((y0, x0, y1, x1), coverage_pct) with fractional coordinates:
    the LARGEST window under accept_pct, else the least cloudy one found.
    """
    h, w = mask.shape
    ii = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    fallback = None
    for s in scales:
        wh, ww = int(round(h * s)), int(round(w * s))
        ys = np.arange(0, h - wh + 1, stride)
        xs = np.arange(0, w - ww + 1, stride)
        sums = (ii[ys + wh][:, xs + ww] - ii[ys + wh][:, xs]
                - ii[ys][:, xs + ww] + ii[ys][:, xs])
        iy, ix = np.unravel_index(np.argmin(sums), sums.shape)
        y0, x0 = int(ys[iy]), int(xs[ix])
        pct = 100.0 * sums[iy, ix] / (wh * ww)
        box = (y0 / h, x0 / w, (y0 + wh) / h, (x0 + ww) / w)
        if pct <= accept_pct:
            return box, pct
        if fallback is None or pct < fallback[1]:
            fallback = (box, pct)
    return fallback


def crop_image(img, box):
    """Crop a HxWxC image to a fractional (y0, x0, y1, x1) box."""
    h, w = img.shape[:2]
    y0, x0, y1, x1 = box
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


class FaceDetector(_Engine):
    """Stage 3: SSD-style face detector (anchors + NMS decoded on the CPU)."""

    def __init__(self, device=None, num_threads=4,
                 model_path=None, meta_path=config.FACE_META):
        super().__init__()
        if model_path is None:
            model_path = config.FACE_MODEL_EDGETPU if device else config.FACE_MODEL_CPU
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.size = self.meta["input_size"]
        self.anchors = self._generate_anchors()
        self.interp = make_interpreter(model_path, device, num_threads)
        self.inp = self.interp.get_input_details()[0]
        self.outs = self.interp.get_output_details()

    def _generate_anchors(self):
        out = []
        for stride, grid, sizes in zip(self.meta["strides"], self.meta["grids"],
                                       self.meta["anchor_sizes"]):
            for gy in range(grid):
                for gx in range(grid):
                    cx = (gx + 0.5) * stride
                    cy = (gy + 0.5) * stride
                    for s in sizes:
                        out.append([cx, cy, float(s), float(s)])
        return np.array(out, dtype=np.float32) / self.meta["input_size"]

    @staticmethod
    def _decode_boxes(pred, anchors, variances):
        v0, v1 = variances
        cx = anchors[:, 0] + pred[:, 0] * v0 * anchors[:, 2]
        cy = anchors[:, 1] + pred[:, 1] * v0 * anchors[:, 3]
        w = anchors[:, 2] * np.exp(pred[:, 2] * v1)
        h = anchors[:, 3] * np.exp(pred[:, 3] * v1)
        return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)

    @staticmethod
    def _nms(boxes, scores, iou_thresh=0.4, top_k=100):
        idx = np.argsort(-scores)[:top_k * 4]
        keep = []
        while idx.size > 0:
            i = idx[0]
            keep.append(i)
            if len(keep) >= top_k:
                break
            rest = idx[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = ((boxes[rest, 2] - boxes[rest, 0])
                      * (boxes[rest, 3] - boxes[rest, 1]))
            iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
            idx = rest[iou < iou_thresh]
        return np.array(keep, dtype=np.int64)

    def detect(self, img, threshold=0.5):
        """Detect faces in a HxWx3 uint8 RGB image.

        Returns (boxes, scores): boxes are normalized (x1, y1, x2, y2).
        """
        resized = resize(img, self.size).astype(np.uint8)
        self.interp.set_tensor(self.inp["index"], resized[None])
        self._invoke()
        cls, box = None, None
        for od in self.outs:
            t = self._get_output(od)[0]
            if t.shape[-1] == 1:
                cls = t
            else:
                box = t
        scores = 1.0 / (1.0 + np.exp(-cls[:, 0]))
        sel = scores >= threshold
        if not np.any(sel):
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        boxes = self._decode_boxes(box[sel], self.anchors[sel],
                                   self.meta["variances"])
        scores = scores[sel]
        keep = self._nms(np.clip(boxes, 0, 1), scores)
        return np.clip(boxes[keep], 0, 1), scores[keep]


# --- standalone per-stage testing -------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["iqa", "cloud", "face", "all"])
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--device", default=None,
                    help="Edge TPU device, e.g. usb:0 (default: CPU)")
    ap.add_argument("--threshold", type=float, default=0.5, help="face threshold")
    ap.add_argument("--save", default=None, help="save annotated faces here")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    img = load_image(args.image)
    print(f"{args.image}: {img.shape[1]}x{img.shape[0]}  "
          f"[{'EdgeTPU ' + args.device if args.device else 'CPU'}]")

    if args.stage in ("iqa", "all"):
        eng = IQAScorer(device=args.device, num_threads=args.threads)
        s = eng.score(img)
        print(f"IQA score: {s:.2f}  ({eng.last_ms:.1f} ms)")

    if args.stage in ("cloud", "all"):
        eng = CloudDetector(device=args.device, num_threads=args.threads)
        cov, mask = eng.coverage(img)
        print(f"cloud coverage: {cov:.1f}%  ({eng.last_ms:.1f} ms)")
        if cov > 40.0:
            box, pct = find_best_crop(mask)
            print(f"least-cloudy window: {pct:.1f}% cloud at "
                  f"({box[0]:.2f},{box[1]:.2f})-({box[2]:.2f},{box[3]:.2f})")

    if args.stage in ("face", "all"):
        eng = FaceDetector(device=args.device, num_threads=args.threads)
        boxes, scores = eng.detect(img, args.threshold)
        print(f"faces: {len(boxes)}  ({eng.last_ms:.1f} ms)")
        H, W = img.shape[:2]
        for b, s in zip(boxes, scores):
            print(f"  score={s:.2f} box=({b[0]*W:.0f},{b[1]*H:.0f},"
                  f"{b[2]*W:.0f},{b[3]*H:.0f})")
        if args.save:
            annotate_faces(img, boxes, scores, args.save)
            print(f"annotated -> {args.save}")


if __name__ == "__main__":
    main()
