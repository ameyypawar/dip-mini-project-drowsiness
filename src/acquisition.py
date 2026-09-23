"""Task 2: Image Acquisition, Representation, and Image Fundamentals.

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057).

Covers sub-tasks 2.1-2.7 of the assignment brief as a set of small, composable
functions rather than one monolithic script:

- 2.1 ``library_versions`` -- verify/print installed library versions.
- 2.3 ``load_image`` -- read an image from the dataset.
- 2.4 ``image_properties`` / ``pixel_at`` -- shape, dtype, pixel access.
- 2.5 ``to_colour_spaces`` -- Grayscale / RGB / HSV views.
- 2.6 ``sample_image`` / ``quantize`` / ``quantization_metrics`` -- spatial
  and intensity resolution reduction, with PSNR/SSIM against the original.
- 2.7 ``save_formats`` / ``compression_ratio`` -- lossless vs. lossy file
  formats, measured on disk.

**Dataset note (drives the colour-space section, 2.5):** every source image
in ``data/samples/`` is a single-channel near-infrared eye crop (see
``data/samples/README.md`` -- MRL Eye Dataset subset). There is no true
colour information to convert. ``to_colour_spaces`` still produces RGB/HSV
views as the assignment requires, but does so honestly: RGB is the grayscale
plane broadcast to 3 equal channels, and HSV's Hue/Saturation planes are
therefore degenerate (Saturation ~0 everywhere, Hue undefined/0) -- only the
Value plane carries real information, and it is numerically identical to the
source grayscale image. See ``docs/task2_report.html`` sub-task 2.5 for the
figure that makes this visible rather than hiding it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ---------------------------------------------------------------------------
# 2.1 Environment setup / verification
# ---------------------------------------------------------------------------

def library_versions() -> dict[str, str]:
    """Return installed version strings for every library the assignment
    requires (Python, OpenCV, NumPy, Matplotlib, Pillow, scikit-image).

    Importing each module here (rather than trusting ``requirements.txt``)
    is itself the verification step: if any library is missing or broken,
    this call raises ``ImportError`` instead of silently reporting a stale
    version string.
    """
    import matplotlib
    import PIL
    import skimage

    return {
        "Python": sys.version.split()[0],
        "OpenCV (cv2)": cv2.__version__,
        "NumPy": np.__version__,
        "Matplotlib": matplotlib.__version__,
        "Pillow (PIL)": PIL.__version__,
        "scikit-image": skimage.__version__,
    }


# ---------------------------------------------------------------------------
# 2.3 Image acquisition
# ---------------------------------------------------------------------------

def load_image(path: str | Path, mode: str = "unchanged") -> np.ndarray:
    """Read an image from the dataset.

    mode: 'unchanged' (native channel count, e.g. single-channel for this
    dataset), 'gray' (force single-channel), or 'color' (force 3-channel
    BGR -- OpenCV's default promotion of a grayscale PNG, see module note).
    """
    flags = {
        "unchanged": cv2.IMREAD_UNCHANGED,
        "gray": cv2.IMREAD_GRAYSCALE,
        "color": cv2.IMREAD_COLOR,
    }
    if mode not in flags:
        raise ValueError(f"mode must be one of {list(flags)}, got {mode!r}")
    img = cv2.imread(str(path), flags[mode])
    if img is None:
        raise FileNotFoundError(path)
    return img


# ---------------------------------------------------------------------------
# 2.4 Image representation
# ---------------------------------------------------------------------------

def image_properties(img: np.ndarray) -> dict[str, Any]:
    """Height, width, channel count, total pixel count, dtype and in-memory
    size (bytes) of an already-loaded image array."""
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    return {
        "height": h,
        "width": w,
        "channels": channels,
        "total_pixels": h * w,
        "dtype": str(img.dtype),
        "size_bytes": int(img.nbytes),
    }


def pixel_at(img: np.ndarray, y: int, x: int) -> Any:
    """Pixel value at a user-specified (row=y, col=x) coordinate.

    The coordinate is a parameter (not hardcoded): callers -- including the
    CLI/report script -- pass whichever (y, x) they want inspected. Raises
    ``IndexError`` with the image bounds if the coordinate falls outside
    the image, rather than silently wrapping/clamping it.
    """
    h, w = img.shape[:2]
    if not (0 <= y < h and 0 <= x < w):
        raise IndexError(f"({y}, {x}) out of bounds for a {h}x{w} image")
    return img[y, x]


# ---------------------------------------------------------------------------
# 2.5 Colour space conversion
# ---------------------------------------------------------------------------

def to_colour_spaces(img: np.ndarray) -> dict[str, np.ndarray]:
    """Grayscale / RGB / HSV views of ``img``, plus the HSV planes split out
    individually (h, s, v) so the degenerate Hue/Saturation planes are
    directly visible rather than hidden inside a composite HSV array.

    ``img`` may already be single-channel (this dataset's native form) or
    3-channel (e.g. loaded with ``mode='color'``, in which case OpenCV has
    already broadcast the single grayscale plane to B=G=R).
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    return {"gray": gray, "rgb": rgb, "hsv": hsv, "h": h, "s": s, "v": v}


# ---------------------------------------------------------------------------
# 2.6 Sampling and quantization
# ---------------------------------------------------------------------------

def sample_image(img: np.ndarray, scale: float) -> np.ndarray:
    """Spatial re-sampling by ``scale`` (1.0 = 100%, 0.5 = 50%, 0.25 = 25%).

    Downsampling (`scale < 1`) uses ``INTER_AREA`` (area-weighted, the
    correct anti-aliased choice for shrinking). Upsampling back to display
    size for a *reduced* sample is the caller's job (via ``INTER_NEAREST``,
    see report) so pixelation is shown honestly instead of being smoothed
    away by a second resampling step.
    """
    h, w = img.shape[:2]
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def quantize(img: np.ndarray, levels: int) -> np.ndarray:
    """Uniform intensity quantization to ``levels`` gray levels (<=256).

    Each of the 256 input intensities is bucketed into one of ``levels``
    equal-width bins and mapped to that bin's midpoint, e.g. levels=32 maps
    every pixel to one of 32 output values spaced ~8 apart. levels=256 is
    the identity mapping (no quantization loss beyond the midpoint-rounding
    already implicit in 8-bit storage).
    """
    if not (1 < levels <= 256):
        raise ValueError("levels must be in (1, 256]")
    step = 256.0 / levels
    q = np.floor(img.astype(np.float64) / step) * step + step / 2.0
    return np.clip(q, 0, 255).astype(np.uint8)


def quantization_metrics(original: np.ndarray, quantized: np.ndarray) -> dict[str, float]:
    """PSNR (dB) and SSIM of a quantized image against the original.

    PSNR is +inf when ``quantized`` is pixel-identical to ``original``
    (levels=256): reported as-is (skimage returns ``inf``), not clipped.
    """
    psnr = peak_signal_noise_ratio(original, quantized, data_range=255)
    ssim = structural_similarity(original, quantized, data_range=255)
    return {"psnr_db": float(psnr), "ssim": float(ssim)}


# ---------------------------------------------------------------------------
# 2.7 File formats
# ---------------------------------------------------------------------------

def save_formats(
    img: np.ndarray,
    out_dir: str | Path,
    basename: str,
    jpeg_quality: int = 95,
) -> dict[str, dict[str, Any]]:
    """Save ``img`` as BMP, PNG (lossless, max compression) and JPEG
    (lossy, ``jpeg_quality``). Returns, per format, the path written and the
    actual file size in bytes measured on disk (not estimated).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs: dict[str, tuple[str, list[int]]] = {
        "bmp": (f"{basename}.bmp", []),
        "png": (f"{basename}.png", [cv2.IMWRITE_PNG_COMPRESSION, 9]),
        "jpg": (f"{basename}.jpg", [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]),
    }
    results: dict[str, dict[str, Any]] = {}
    for fmt, (fname, params) in specs.items():
        path = out_dir / fname
        ok = cv2.imwrite(str(path), img, params)
        if not ok:
            raise IOError(f"cv2.imwrite failed for {path}")
        results[fmt] = {"path": str(path), "bytes": path.stat().st_size}
    return results


def compression_ratio(raw_bytes: int, file_bytes: int) -> float:
    """Ratio of raw (uncompressed, 1 byte/pixel) size to on-disk file size.

    >1 means the file is smaller than the raw pixel data (real compression);
    <1 means the file is *larger* than the raw pixel data (e.g. BMP's
    header/row-padding overhead on very small images, see report).
    """
    return raw_bytes / file_bytes
