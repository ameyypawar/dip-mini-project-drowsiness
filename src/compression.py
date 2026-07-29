"""Week-8 image compression: JPEG rate/quality study + the headline
accuracy-vs-bitrate experiment.

The question this module answers: an in-vehicle camera transmitting eye
crops to a fleet server has limited bandwidth -- how far can a frame be
JPEG-compressed before the Week-6 eye-state classifier
(`src.classify.predict_eye_state`, RBF-SVM on HOG+LBP, test accuracy
0.8965) degrades? Three independent codecs are compared, cheapest/most
realistic first:

  1. Library JPEG (`cv2.imencode`/`imdecode`) -- the real codec, used for
     the headline rate-accuracy sweep since it is what an actual camera
     pipeline would use.
  2. A from-scratch block-wise 8x8 DCT + scaled-standard-luminance-table
     quantization + inverse DCT codec (`manual_dct_reconstruct`) -- shows
     *what JPEG actually does* rather than treating it as a black box.
     No entropy coding stage (no Huffman/arithmetic coding, no zig-zag
     RLE), so its "rate" is reported as a zeroth-order Shannon-entropy
     *estimate* over the quantized coefficient stream, not a real byte
     count -- this is stated explicitly wherever it is used, never
     conflated with a measured library-JPEG byte count.
  3. Full-image top-k DCT coefficient truncation
     (`src.transforms.dct2`/`dct_energy_compaction`, reused from Week 5,
     not reimplemented) -- the cleanest illustration of energy compaction:
     keep only the top-left k x k low-frequency block of one whole-image
     DCT and see how quality degrades as k shrinks.

All three operate on grayscale `np.ndarray` (uint8) images. The rate-
accuracy sweep operates on the *raw, native-resolution* MRL eye crop (the
frame as a camera would actually capture and transmit it, before the
Week-2 `preprocess_pipeline` resize/enhance) -- see
docs/week8_compression.md for why that distinction matters (a real system
compresses the full captured frame, not a pre-resized 64x64 crop).
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.transforms import dct2, dct_energy_compaction

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKING_SET_CSV = REPO_ROOT / "data" / "working_set.csv"

# JPEG quality levels for every sweep in this module. ~13 levels across the
# 5-95 range asked for -- dense at the low end (where accuracy actually
# starts moving) and sparse at the high end (where nothing changes).
QUALITY_LEVELS: tuple[int, ...] = (5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 95)

# Standard JPEG (ITU-T T.81 Annex K, Table K.1) luminance quantization
# table -- the same table libjpeg/OpenCV scale internally per quality
# factor. Used both for reference and as the base table
# `manual_dct_reconstruct` scales itself, so the manual codec's rate
# knob matches the real codec's.
STD_LUMA_QTABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Library JPEG codec (cv2.imencode/imdecode)
# ---------------------------------------------------------------------------

def jpeg_encode_decode(img: np.ndarray, quality: int) -> tuple[np.ndarray, int]:
    """Encode `img` (grayscale uint8) as JPEG at `quality` (0-100), decode
    back. Returns (decoded_image, n_bytes) -- `n_bytes` is the real encoded
    byte count (`len(buf)`), the ground truth for bpp/KB-per-frame.
    """
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed at quality={quality}")
    decoded = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return decoded, int(buf.size)


def bits_per_pixel(n_bytes: int, shape: tuple[int, int]) -> float:
    """Encoded bits per original pixel: n_bytes*8 / (H*W)."""
    h, w = shape[:2]
    return (n_bytes * 8) / (h * w)


def image_quality_metrics(reference: np.ndarray, test: np.ndarray) -> dict:
    """PSNR (dB) and SSIM of `test` vs `reference`, both uint8 grayscale,
    same shape. `data_range=255` fixes the dynamic range explicitly rather
    than inferring it from the (possibly near-constant) image content.
    """
    psnr = float(peak_signal_noise_ratio(reference, test, data_range=255))
    ssim = float(structural_similarity(reference, test, data_range=255))
    return {"psnr": psnr, "ssim": ssim}


@dataclass
class QualityPoint:
    quality: int
    n_bytes: int
    bpp: float
    psnr: float
    ssim: float
    decoded: np.ndarray


def jpeg_quality_sweep(img: np.ndarray, qualities: Sequence[int] = QUALITY_LEVELS) -> list[QualityPoint]:
    """Encode/decode `img` at every quality in `qualities`, measuring bytes,
    bpp, PSNR, SSIM. One image in, one point per quality out -- the
    per-image building block for both the notebook's PSNR/SSIM-vs-quality
    figure and (aggregated over the test split) the headline rate-accuracy
    experiment.
    """
    out = []
    for q in qualities:
        decoded, n_bytes = jpeg_encode_decode(img, q)
        metrics = image_quality_metrics(img, decoded)
        out.append(QualityPoint(
            quality=q, n_bytes=n_bytes, bpp=bits_per_pixel(n_bytes, img.shape),
            psnr=metrics["psnr"], ssim=metrics["ssim"], decoded=decoded,
        ))
    return out


# ---------------------------------------------------------------------------
# Manual block-wise 8x8 DCT + quantization (from-scratch JPEG core)
# ---------------------------------------------------------------------------

def scale_quant_table(quality: int, base_table: np.ndarray = STD_LUMA_QTABLE) -> np.ndarray:
    """Scale `base_table` for `quality` (1-100) using the standard
    libjpeg/IJG scaling formula, so the manual codec's quantization
    strength is directly comparable to `cv2.imencode`'s quality parameter:

        scale = 5000/quality        if quality < 50
        scale = 200 - 2*quality     otherwise
        table = clip(floor((base_table*scale + 50) / 100), 1, 255)

    quality=50 reproduces `base_table` unchanged (scale=100); quality<50
    coarsens it (larger divisors -> more coefficients quantized to 0);
    quality>50 refines it (smaller divisors, up to /1 at quality=100).
    """
    q = min(max(int(quality), 1), 100)
    scale = 5000 / q if q < 50 else 200 - 2 * q
    table = np.floor((base_table * scale + 50) / 100)
    return np.clip(table, 1, 255)


def _pad_to_block_multiple(img: np.ndarray, block: int = 8) -> tuple[np.ndarray, tuple[int, int]]:
    """Replicate-pad `img` on the bottom/right so both dims are multiples
    of `block`. Returns (padded, original_shape) so the caller can crop
    back after reconstruction.
    """
    h, w = img.shape[:2]
    pad_h = (-h) % block
    pad_w = (-w) % block
    if pad_h == 0 and pad_w == 0:
        return img, (h, w)
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    return padded, (h, w)


def manual_dct_reconstruct(img: np.ndarray, quality: int, base_table: np.ndarray = STD_LUMA_QTABLE) -> np.ndarray:
    """From-scratch JPEG-core codec: level-shift -> per-8x8-block DCT
    (`src.transforms.dct2`) -> divide-and-round by a `quality`-scaled
    standard quantization table -> dequantize (multiply back) -> per-block
    inverse DCT -> level-shift back -> clip to uint8.

    This is exactly what JPEG's DCT/quantization stage does; the only
    thing missing versus a real JPEG encoder is the entropy-coding stage
    (zig-zag reorder + run-length + Huffman/arithmetic coding), which is
    a lossless bookkeeping step that does not affect reconstructed image
    quality -- only the final byte count, which `manual_bit_estimate`
    approximates separately.
    """
    padded, (h, w) = _pad_to_block_multiple(img, 8)
    ph, pw = padded.shape[:2]
    qtable = scale_quant_table(quality, base_table)

    shifted = padded.astype(np.float64) - 128.0
    recon = np.zeros_like(shifted)

    for y in range(0, ph, 8):
        for x in range(0, pw, 8):
            block = shifted[y:y + 8, x:x + 8]
            coef = dct2(block)
            quantized = np.round(coef / qtable)
            dequantized = quantized * qtable
            recon[y:y + 8, x:x + 8] = cv2.idct(dequantized.astype(np.float32))

    recon = np.clip(recon + 128.0, 0, 255).astype(np.uint8)
    return recon[:h, :w]


def manual_bit_estimate(img: np.ndarray, quality: int, base_table: np.ndarray = STD_LUMA_QTABLE) -> float:
    """Zeroth-order Shannon-entropy estimate of bits-per-pixel for the
    manual codec's quantized-coefficient stream: quantize every 8x8 block
    (as `manual_dct_reconstruct` does), pool all quantized coefficient
    values dataset-wide, and compute `-sum(p*log2(p))` over their value
    histogram, times total-coefficients / n_pixels.

    This is an *estimate* of what an ideal entropy coder (Huffman or
    arithmetic) could achieve on this exact coefficient stream -- not a
    measured byte count like `jpeg_encode_decode`'s. It ignores JPEG's
    actual DC-prediction and AC run-length structure, so treat it as a
    lower-bound/ballpark for comparing manual-vs-library rate, not as a
    real encoded size.
    """
    padded, (h, w) = _pad_to_block_multiple(img, 8)
    ph, pw = padded.shape[:2]
    qtable = scale_quant_table(quality, base_table)
    shifted = padded.astype(np.float64) - 128.0

    values: list[float] = []
    for y in range(0, ph, 8):
        for x in range(0, pw, 8):
            block = shifted[y:y + 8, x:x + 8]
            coef = dct2(block)
            quantized = np.round(coef / qtable)
            values.append(quantized.ravel())

    all_vals = np.concatenate(values)
    _, counts = np.unique(all_vals, return_counts=True)
    probs = counts / counts.sum()
    entropy_bits_per_coef = float(-(probs * np.log2(probs)).sum())
    return entropy_bits_per_coef * all_vals.size / (h * w)


# ---------------------------------------------------------------------------
# Full-image top-k DCT coefficient truncation (Week-5 energy compaction)
# ---------------------------------------------------------------------------

def reconstruct_from_topk(img: np.ndarray, k: int) -> np.ndarray:
    """Reconstruct `img` keeping only the top-left k x k block of its
    *whole-image* 2D DCT (`src.transforms.dct2`, reused not reimplemented)
    and zeroing every other coefficient, then inverse-DCT. Illustrates
    energy compaction directly: low k already captures most structure
    because natural-image energy concentrates in low frequencies.
    """
    coef = dct2(img)
    k = min(k, coef.shape[0], coef.shape[1])
    truncated = np.zeros_like(coef)
    truncated[:k, :k] = coef[:k, :k]
    recon = cv2.idct(truncated)
    return np.clip(recon, 0, 255).astype(np.uint8)


@dataclass
class TopKPoint:
    k: int
    energy_fraction: float
    psnr: float
    ssim: float
    reconstructed: np.ndarray


def topk_reconstruction_sweep(img: np.ndarray, ks: Sequence[int]) -> list[TopKPoint]:
    """`reconstruct_from_topk` + `dct_energy_compaction` (Week 5) + image
    quality metrics, for every k in `ks`.
    """
    out = []
    for k in ks:
        recon = reconstruct_from_topk(img, k)
        frac = dct_energy_compaction(img, k)
        metrics = image_quality_metrics(img, recon)
        out.append(TopKPoint(k=k, energy_fraction=frac, psnr=metrics["psnr"], ssim=metrics["ssim"], reconstructed=recon))
    return out


# ---------------------------------------------------------------------------
# Headline experiment: accuracy vs bitrate over the test split
# ---------------------------------------------------------------------------

def load_test_split(working_set_csv: str | Path = WORKING_SET_CSV) -> list[dict]:
    """All `split == 'test'` rows of `data/working_set.csv`, in file order
    (1140 rows) -- the same split `src.classify.evaluate_on_test` reports
    0.8965 test accuracy on, so sweep numbers here are directly comparable.
    """
    with open(working_set_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["split"] == "test"]


@dataclass
class RatePoint:
    quality: Optional[int]  # None = uncompressed baseline
    n: int
    accuracy: float
    mean_bytes: float
    mean_bpp: float
    mean_kb: float
    psnr_mean: float
    ssim_mean: float


def rate_accuracy_sweep(
    test_rows: list[dict],
    qualities: Sequence[int] = QUALITY_LEVELS,
    repo_root: str | Path = REPO_ROOT,
    include_baseline: bool = True,
) -> list[RatePoint]:
    """For each quality in `qualities` (plus an uncompressed baseline if
    `include_baseline`): JPEG-compress every test-split raw eye crop at
    that quality, decode, run `src.classify.predict_eye_state` on the
    decoded crop, and aggregate accuracy + bytes/bpp + PSNR/SSIM (vs the
    original raw crop) over all `len(test_rows)` images.

    Compression operates on the *raw, native-resolution* source crop
    (`row['source_relpath']`), not the 64x64 `preprocess_pipeline` output
    -- `predict_eye_state` applies `preprocess_pipeline` internally, so
    this mirrors a real camera pipeline: capture -> compress -> transmit
    -> decode -> resize/enhance -> classify.
    """
    from src.classify import predict_eye_state

    root = Path(repo_root)
    y_true = np.array([int(r["eye_state"]) for r in test_rows])

    raw_imgs = []
    for r in test_rows:
        img = cv2.imread(str(root / r["source_relpath"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not read image: {root / r['source_relpath']}")
        raw_imgs.append(img)

    results: list[RatePoint] = []

    if include_baseline:
        preds = np.array([1 if predict_eye_state(img)[0] == "open" else 0 for img in raw_imgs])
        acc = float((preds == y_true).mean())
        results.append(RatePoint(
            quality=None, n=len(test_rows), accuracy=acc,
            mean_bytes=float("nan"), mean_bpp=float("nan"), mean_kb=float("nan"),
            psnr_mean=float("inf"), ssim_mean=1.0,
        ))
        print(f"  baseline (uncompressed): accuracy={acc:.4f}")

    for q in qualities:
        t0 = time.perf_counter()
        n_bytes_list = []
        psnr_list = []
        ssim_list = []
        preds = np.empty(len(raw_imgs), dtype=int)
        for i, img in enumerate(raw_imgs):
            decoded, n_bytes = jpeg_encode_decode(img, q)
            n_bytes_list.append(n_bytes)
            metrics = image_quality_metrics(img, decoded)
            psnr_list.append(metrics["psnr"])
            ssim_list.append(metrics["ssim"])
            label, _ = predict_eye_state(decoded)
            preds[i] = 1 if label == "open" else 0

        acc = float((preds == y_true).mean())
        bpp_list = [bits_per_pixel(nb, img.shape) for nb, img in zip(n_bytes_list, raw_imgs)]
        results.append(RatePoint(
            quality=q, n=len(test_rows), accuracy=acc,
            mean_bytes=float(np.mean(n_bytes_list)),
            mean_bpp=float(np.mean(bpp_list)),
            mean_kb=float(np.mean(n_bytes_list)) / 1024.0,
            psnr_mean=float(np.mean([p for p in psnr_list if np.isfinite(p)])),
            ssim_mean=float(np.mean(ssim_list)),
        ))
        print(f"  quality={q:3d} accuracy={acc:.4f} mean_bpp={results[-1].mean_bpp:.3f} "
              f"mean_KB={results[-1].mean_kb:.3f} psnr={results[-1].psnr_mean:.2f} "
              f"ssim={results[-1].ssim_mean:.4f} ({time.perf_counter() - t0:.1f}s)")

    return results


def find_knee_point(results: list[RatePoint], tolerance: float = 0.01) -> Optional[RatePoint]:
    """Lowest-bitrate quality level whose accuracy stays within `tolerance`
    absolute of the uncompressed baseline (the first `quality=None` entry
    in `results`, if present, else the max-quality entry). Sweeps
    `results` from lowest to highest quality (results are expected in the
    same order `rate_accuracy_sweep` returns, baseline first then
    ascending quality) and returns the first quality point that meets the
    tolerance -- i.e. the cheapest (lowest-bitrate) safe operating point.
    Returns None if no quality level meets the tolerance.
    """
    baseline = next((r for r in results if r.quality is None), None)
    baseline_acc = baseline.accuracy if baseline is not None else max(r.accuracy for r in results)

    quality_points = sorted((r for r in results if r.quality is not None), key=lambda r: r.quality)
    for r in quality_points:
        if baseline_acc - r.accuracy <= tolerance:
            return r
    return None


def bandwidth_kb_per_s(mean_kb_per_frame: float, fps: float = 30.0) -> float:
    """KB/s a video stream needs at `fps` frames/sec if every frame costs
    `mean_kb_per_frame` KB -- the concrete engineering payoff number.
    """
    return mean_kb_per_frame * fps


def main() -> None:
    test_rows = load_test_split()
    print(f"loaded {len(test_rows)} test-split rows")
    print("\n=== rate-accuracy sweep (library JPEG) ===")
    results = rate_accuracy_sweep(test_rows)

    knee = find_knee_point(results, tolerance=0.01)
    if knee is not None:
        bw = bandwidth_kb_per_s(knee.mean_kb)
        print(f"\nknee point: quality={knee.quality} bpp={knee.mean_bpp:.3f} "
              f"KB/frame={knee.mean_kb:.3f} accuracy={knee.accuracy:.4f} "
              f"psnr={knee.psnr_mean:.2f} ssim={knee.ssim_mean:.4f}")
        print(f"bandwidth @30fps: {bw:.1f} KB/s")
    else:
        print("\nno quality level in the sweep stayed within tolerance of baseline")


if __name__ == "__main__":
    main()
