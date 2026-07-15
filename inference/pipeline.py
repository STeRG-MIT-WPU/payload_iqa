#!/usr/bin/env python3
"""Master two-stage image screening pipeline (self-contained).

Designed for the Raspberry Pi Compute Module 5. Pull the repo, install the
dependencies, and run this one script:

    pip3 install numpy pillow tifffile imagecodecs tflite-runtime
    python3 inference/pipeline.py

Stage 1 - IQA (models/iqa_model/iqa_lite0_int8.tflite):
    score < IQA_MIN            -> reject
Stage 2 - cloud coverage (models/cloud_segmentation_model/cloudseg_int8.tflite):
    coverage > CLOUD_REJECT    -> reject
    CLOUD_CROP..CLOUD_REJECT   -> crop to the least-cloudy window, save crop
    coverage < CLOUD_CROP      -> save as-is

Outputs:
    output/passed/    accepted images (cropped ones get a _cropped suffix)
                      + results.txt with per-image scores
    output/rejected/  rejected images + results.txt with scores and reason
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

# --- TFLite interpreter: tflite-runtime (RPi) -> LiteRT -> TensorFlow ------
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter

ROOT = Path(__file__).resolve().parents[1]
IQA_MODEL = ROOT / "models" / "iqa_model" / "iqa_lite0_int8.tflite"
IQA_NORM = ROOT / "models" / "iqa_model" / "norm.json"
CLOUD_MODEL = ROOT / "models" / "cloud_segmentation_model" / "cloudseg_int8.tflite"

IQA_MIN = 60.0       # reject below this IQA score
CLOUD_REJECT = 70.0  # reject above this cloud coverage %
CLOUD_CROP = 60.0    # crop when coverage is in [CLOUD_CROP, CLOUD_REJECT]

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


# --- image I/O --------------------------------------------------------------

def load_image(path):
    """Load an image as a HxWx3 uint8 RGB numpy array.

    TIFFs are read with tifffile (Pillow's decoder fails on the tiled LZW
    GeoTIFF test scenes); everything else goes through Pillow.
    """
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        img = tifffile.imread(str(path))
    else:
        from PIL import Image

        img = np.array(Image.open(path).convert("RGB"))

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    return np.ascontiguousarray(img.astype(np.uint8))


def save_image(path, img):
    """Save a HxWx3 uint8 array; format is chosen from the extension."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        tifffile.imwrite(str(path), img)
    else:
        from PIL import Image

        Image.fromarray(img).save(path)


def resize(img, size):
    """Bilinear resize of a HxWxC uint8 array to (size, size)."""
    from PIL import Image

    return np.array(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def list_images(folder):
    return sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# --- stage 1: IQA -----------------------------------------------------------

class IQAScorer:
    def __init__(self, model_path=IQA_MODEL, norm_path=IQA_NORM, num_threads=4):
        with open(norm_path) as f:
            norm = json.load(f)
        self.mos_min = norm["MOS_MIN"]
        self.mos_max = norm["MOS_MAX"]
        self.input_size = norm.get("input_size", 224)

        self.interp = Interpreter(model_path=str(model_path), num_threads=num_threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def score(self, img):
        """Return the IQA score on the MOS scale for a HxWx3 uint8 image."""
        x = resize(img, self.input_size).astype(np.float32)
        x = (x - 127.0) / 128.0
        self.interp.set_tensor(self.inp["index"], x[None])
        self.interp.invoke()
        y = float(self.interp.get_tensor(self.out["index"])[0, 0])
        return y * (self.mos_max - self.mos_min) + self.mos_min


# --- stage 2: cloud coverage + crop ----------------------------------------

class CloudDetector:
    """Cloud segmentation. The model expects 4 channels (R,G,B,NIR) in [0,1];
    for RGB-only inputs a NIR band is synthesized (nir_mode)."""

    def __init__(self, model_path=CLOUD_MODEL, nir_mode="mean", num_threads=4):
        self.nir_mode = nir_mode
        self.interp = Interpreter(model_path=str(model_path), num_threads=num_threads)
        self.interp.allocate_tensors()
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
        """Return (coverage_percent, mask) for a HxWx3 uint8 RGB image.

        mask is a size x size boolean array (True = cloud).
        """
        rgb01 = resize(img, self.size).astype(np.float32) / 255.0
        x = np.dstack([rgb01, self._make_nir(rgb01)])

        scale, zp = self.inp["quantization"]
        q = np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)
        self.interp.set_tensor(self.inp["index"], q[None])
        self.interp.invoke()
        raw = self.interp.get_tensor(self.out["index"])[0, ..., 0]

        oscale, ozp = self.out["quantization"]
        mask = (raw.astype(np.float32) - ozp) * oscale > 0.5
        return 100.0 * float(mask.mean()), mask


def find_best_crop(mask, scales=(0.75, 0.6, 0.5), accept_pct=40.0, stride=8):
    """Find the least-cloudy crop window in a boolean cloud mask.

    Slides windows of several sizes (fractions of the full frame) over an
    integral image of the mask. Returns the LARGEST window whose cloud
    coverage is below accept_pct; if none qualifies, the overall least
    cloudy window found. Result: ((y0, x0, y1, x1), coverage_pct) with
    coordinates as fractions of the frame, directly applicable to the
    full-resolution image.
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


# --- pipeline ---------------------------------------------------------------

def run(input_dir, output_dir, nir_mode="mean", num_threads=4):
    passed_dir = Path(output_dir) / "passed"
    rejected_dir = Path(output_dir) / "rejected"
    passed_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    iqa = IQAScorer(num_threads=num_threads)
    cloud = CloudDetector(nir_mode=nir_mode, num_threads=num_threads)

    passed_log, rejected_log = [], []
    images = list_images(input_dir)
    print(f"Processing {len(images)} images from {input_dir}\n")

    for path in images:
        t0 = time.perf_counter()
        img = load_image(path)

        iqa_score = iqa.score(img)
        if iqa_score < IQA_MIN:
            shutil.copy2(path, rejected_dir / path.name)
            rejected_log.append(
                f"{path.name} | IQA {iqa_score:.2f} | cloud -    | "
                f"rejected: IQA score below {IQA_MIN:.0f}")
            print(f"[REJECT] {path.name}: IQA {iqa_score:.2f} < {IQA_MIN:.0f}")
            continue

        cov, mask = cloud.coverage(img)
        dt = (time.perf_counter() - t0) * 1000

        if cov > CLOUD_REJECT:
            shutil.copy2(path, rejected_dir / path.name)
            rejected_log.append(
                f"{path.name} | IQA {iqa_score:.2f} | cloud {cov:5.1f}% | "
                f"rejected: cloud coverage above {CLOUD_REJECT:.0f}%")
            print(f"[REJECT] {path.name}: IQA {iqa_score:.2f}, "
                  f"cloud {cov:.1f}% > {CLOUD_REJECT:.0f}%  ({dt:.0f} ms)")
        elif cov >= CLOUD_CROP:
            box, crop_cov = find_best_crop(mask)
            out_name = f"{path.stem}_cropped{path.suffix}"
            save_image(passed_dir / out_name, crop_image(img, box))
            passed_log.append(
                f"{out_name} | IQA {iqa_score:.2f} | cloud {cov:5.1f}% | "
                f"cropped to least-cloudy window ({crop_cov:.1f}% cloud in crop)")
            print(f"[CROP]   {path.name}: IQA {iqa_score:.2f}, cloud {cov:.1f}% "
                  f"-> crop {crop_cov:.1f}%  ({dt:.0f} ms)")
        else:
            shutil.copy2(path, passed_dir / path.name)
            passed_log.append(
                f"{path.name} | IQA {iqa_score:.2f} | cloud {cov:5.1f}% | passed")
            print(f"[PASS]   {path.name}: IQA {iqa_score:.2f}, "
                  f"cloud {cov:.1f}%  ({dt:.0f} ms)")

    header = "filename | IQA score | cloud coverage | result"
    (passed_dir / "results.txt").write_text(
        "\n".join([header, "-" * len(header)] + passed_log) + "\n")
    (rejected_dir / "results.txt").write_text(
        "\n".join([header, "-" * len(header)] + rejected_log) + "\n")

    print(f"\nDone: {len(passed_log)} passed, {len(rejected_log)} rejected.")
    print(f"Results in {passed_dir} and {rejected_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(ROOT / "test-images"))
    ap.add_argument("--output", default=str(ROOT / "output"))
    ap.add_argument("--nir-mode", default="mean", choices=["mean", "red", "zero"],
                    help="how to synthesize the NIR band for RGB-only inputs")
    ap.add_argument("--threads", type=int, default=4,
                    help="interpreter threads (CM5 has 4 cores)")
    args = ap.parse_args()
    run(args.input, args.output, nir_mode=args.nir_mode, num_threads=args.threads)


if __name__ == "__main__":
    main()
