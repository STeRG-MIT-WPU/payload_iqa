"""Image I/O helpers shared by every stage (numpy + Pillow only)."""
from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def load_image(path):
    """Load an image as a HxWx3 uint8 RGB numpy array.

    TIFFs go through tifffile (Pillow fails on the tiled LZW GeoTIFF test
    scenes); everything else through Pillow.
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
    """Save a HxWx3 uint8 RGB array; format chosen from the extension."""
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


def annotate_faces(img, boxes, scores, out_path):
    """Draw normalized face boxes + scores on a copy of img, save to out_path."""
    from PIL import Image, ImageDraw

    pil = Image.fromarray(img)
    W, H = pil.size
    d = ImageDraw.Draw(pil)
    for b, s in zip(boxes, scores):
        x1, y1, x2, y2 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=max(2, W // 300))
        d.text((x1, max(0, y1 - 12)), f"{s:.2f}", fill=(255, 0, 0))
    pil.save(out_path)
