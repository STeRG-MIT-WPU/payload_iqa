# 13a) On-device inference script (runs on the machine hosting the Coral;
# needs only pycoral + numpy, no TensorFlow)
"""Cloud detection + coverage % on the Coral EdgeTPU.
Usage: python3 infer_coral.py --model cloudseg_int8_edgetpu.tflite --input scene.npy
scene.npy: [H,W,4] float32 (R,G,B,NIR) scaled to [0,1] (uint16 TIF / 65535)."""
import argparse, time
import numpy as np
from pycoral.utils.edgetpu import make_interpreter

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--input', required=True)
ap.add_argument('--detect-threshold', type=float, default=1.0)
args = ap.parse_args()

interp = make_interpreter(args.model)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
size = int(inp['shape'][1])

img = np.load(args.input).astype(np.float32)
if img.shape[:2] != (size, size):  # nearest-neighbor resize, numpy only
    ys = (np.arange(size) * img.shape[0] / size).astype(int)
    xs = (np.arange(size) * img.shape[1] / size).astype(int)
    img = img[ys][:, xs]

scale, zp = inp['quantization']
q = np.clip(np.round(img / scale + zp), -128, 127).astype(np.int8)

t0 = time.perf_counter()
interp.set_tensor(inp['index'], q[None])
interp.invoke()
raw = interp.get_tensor(out['index'])[0, ..., 0]
dt = (time.perf_counter() - t0) * 1000

oscale, ozp = out['quantization']
mask = (raw.astype(np.float32) - ozp) * oscale > 0.5
pct = 100.0 * mask.mean()
print(f"clouds: {'yes' if pct > args.detect_threshold else 'no'}, "
      f"coverage: {pct:.1f}%  ({dt:.1f} ms)")
