# Payload image screening pipeline

One self-contained script: `pipeline.py`. Two-stage screening of captured
scenes:

1. **IQA** (`models/iqa_model/iqa_lite0_int8.tflite`) — predicts an image
   quality score on the MOS scale (denormalized via `norm.json`).
   Score **< 60 → reject**.
2. **Cloud coverage** (`models/cloud_segmentation_model/cloudseg_int8.tflite`)
   — segments clouds and computes coverage %.
   - **> 70% → reject**
   - **60–70% → crop** to the least-cloudy window (integral-image search
     over the segmentation mask), save the crop
   - **< 60% → pass** unchanged

## Folder layout

```
payload_iqa/
├── test-images/   input scenes (.tif/.png/.jpg)
├── models/        tflite models + norm.json
├── inference/     pipeline.py (this folder)
└── output/        created by the script
    ├── passed/    accepted images (+ *_cropped) + results.txt
    └── rejected/  rejected images + results.txt (score + reason)
```

## Run on the RPi Compute Module 5 (64-bit Raspberry Pi OS)

```bash
sudo apt update && sudo apt install -y python3-pip
pip3 install numpy pillow tifffile imagecodecs tflite-runtime

cd payload_iqa
python3 inference/pipeline.py
```

If `tflite-runtime` is unavailable for your Python version, use
`pip3 install ai-edge-litert` instead — the script tries both automatically
(and falls back to full TensorFlow as a last resort).

Options:

```bash
python3 inference/pipeline.py --input /path/to/imgs --output /path/to/out \
                              --threads 4 --nir-mode mean
```

Thresholds are constants at the top of `pipeline.py`:
`IQA_MIN = 60`, `CLOUD_REJECT = 70`, `CLOUD_CROP = 60`.

## Notes

- **NIR band**: the cloud model expects 4 channels (R,G,B,NIR). The test
  scenes are RGB-only, so a NIR band is synthesized (`--nir-mode`:
  `mean` of RGB [default], `red`, or `zero`). When the payload delivers a
  real NIR band, feed it through instead — see `CloudDetector.coverage`.
- **TIFF reading** uses `tifffile`, because Pillow's decoder crashes on
  the tiled LZW GeoTIFF test scenes.
- Passed/rejected originals are copied unchanged (GeoTIFF tags preserved);
  cropped outputs are re-encoded and lose geo-referencing.
- `cloudseg_int8_edgetpu.tflite` is for a Coral EdgeTPU accelerator only
  (see `models/cloud_segmentation_model/infer_coral.py`); the plain CPU
  int8 model is what the CM5 pipeline uses.
