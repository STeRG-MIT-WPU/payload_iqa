"""Central configuration for the two-Coral screening pipeline.

Flow (on the Raspberry Pi CM5 with two USB Coral Edge TPUs):

    USB camera --(1 frame / 2 s)--> data/captures/
        -> Coral 1: IQA            (reject if score < IQA_MIN)
        -> Coral 1: cloud coverage (reject if > CLOUD_REJECT, crop if in
                                    [CLOUD_CROP, CLOUD_REJECT])
        -> Coral 2: face detection (annotated result saved)
        -> data/passed/  or  data/rejected/

Everything below can be overridden per-run through the ``Config`` dataclass
(the headless CLI and the dashboard both expose the important knobs).
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- model files ------------------------------------------------------------
MODELS_DIR = ROOT / "models"
IQA_MODEL_EDGETPU = MODELS_DIR / "iqa" / "iqa_lite0_int8_io_edgetpu.tflite"
IQA_MODEL_CPU = MODELS_DIR / "iqa" / "iqa_lite0_int8.tflite"
IQA_NORM = MODELS_DIR / "iqa" / "norm.json"
CLOUD_MODEL_EDGETPU = MODELS_DIR / "cloud_seg" / "cloudseg_int8_edgetpu.tflite"
CLOUD_MODEL_CPU = MODELS_DIR / "cloud_seg" / "cloudseg_int8.tflite"
FACE_MODEL_EDGETPU = MODELS_DIR / "face" / "face_detector_int8_edgetpu.tflite"
FACE_MODEL_CPU = MODELS_DIR / "face" / "face_detector_int8.tflite"
FACE_META = MODELS_DIR / "face" / "model_meta.json"

# --- data folders -----------------------------------------------------------
DATA_DIR = ROOT / "data"
CAPTURES_DIR = DATA_DIR / "captures"
PASSED_DIR = DATA_DIR / "passed"
REJECTED_DIR = DATA_DIR / "rejected"
RESULTS_CSV = DATA_DIR / "results.csv"


@dataclass
class Config:
    # capture source
    source: str = "camera"          # "camera" (USB webcam) or "folder"
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    capture_interval: float = 2.0   # seconds between captures
    input_folder: str = str(ROOT / "test-images")   # used when source == "folder"
    loop_folder: bool = False       # keep re-feeding the folder forever

    # devices: Coral 1 runs IQA + cloud segmentation, Coral 2 runs faces.
    # use_edgetpu=False forces everything onto the CPU (dev machines).
    use_edgetpu: bool = True
    coral1_device: str = "usb:0"
    coral2_device: str = "usb:1"
    cpu_fallback: bool = True       # fall back to CPU models if a TPU is missing
    num_threads: int = 4            # CM5 has 4 cores

    # stage thresholds
    iqa_min: float = 60.0           # reject below this IQA (MOS) score
    cloud_reject: float = 70.0      # reject above this cloud coverage %
    cloud_crop: float = 60.0        # crop when coverage in [cloud_crop, cloud_reject]
    face_threshold: float = 0.5     # face detector confidence
    nir_mode: str = "mean"          # synthesized NIR band for the cloud model

    # queue sizes (bounded so a stalled stage can't eat all the RAM)
    queue_size: int = 16

    def ensure_dirs(self):
        for d in (CAPTURES_DIR, PASSED_DIR, REJECTED_DIR):
            d.mkdir(parents=True, exist_ok=True)
