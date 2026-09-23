#!/usr/bin/env python3
"""Generate all Task 2 (Image Acquisition, Representation, and Image
Fundamentals) figures, CSVs, and outputs/ artifacts.

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057).

Writes:
  results/task2_library_versions.png   (2.1)
  results/task2_acquisition.png        (2.3)
  outputs/task2_acquired_image.png     (2.3 -- required save-to-outputs/)
  results/task2_properties.png         (2.4)
  results/task2_properties.csv         (2.4)
  results/task2_colour_spaces.png      (2.5)
  results/task2_sampling.png           (2.6a)
  results/task2_quantization.png       (2.6b)
  results/task2_quantization_metrics.csv (2.6b)
  outputs/task2_sample.bmp/.png/.jpg   (2.7 -- saved formats)
  results/task2_file_formats.png       (2.7)
  results/task2_file_formats.csv       (2.7)

Uses src.acquisition exclusively for the actual DIP logic; this script only
does layout/plotting/CSV writing. No pandas; CSV via stdlib csv.

Run: ./.venv/bin/python scripts/make_task2_figures.py [--y Y --x X]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import acquisition as acq  # noqa: E402

RESULTS_DIR = ROOT / "results"
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Same canonical demo image used across Tasks 3/4/6/7 (good lighting, open
# eye, no glasses) -- kept consistent so this report is directly comparable
# to the others rather than introducing a new, unexplained sample.
IMAGE_PATH = ROOT / "data/samples/alert/s0031_00366_1_0_1_0_1_02.png"
IMAGE_LABEL = "s0031_00366_1_0_1_0_1_02 (alert / open eye, good lighting)"

DISPLAY_MIN = 320  # px, INTER_NEAREST upscale target so small-crop pixelation stays honest


def nearest_upscale(img: np.ndarray, min_side: int = DISPLAY_MIN) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(1, min_side // max(1, min(h, w)))
    if scale <= 1:
        return img
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def savefig(fig, name: str):
    out = RESULTS_DIR / name
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def imshow_gray(ax, img, title=""):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=9)
    ax.axis("off")


# ---------------------------------------------------------------------------
# 2.1 Library versions
# ---------------------------------------------------------------------------

def fig_library_versions():
    versions = acq.library_versions()
    for k, v in versions.items():
        print(f"  {k}: {v}")

    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    ax.axis("off")
    rows = [[k, v] for k, v in versions.items()]
    table = ax.table(
        cellText=rows,
        colLabels=["Library", "Version"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("2.1 Environment verification -- installed library versions", fontsize=10, pad=14)
    savefig(fig, "task2_library_versions.png")
    return versions


# ---------------------------------------------------------------------------
# 2.3 Image acquisition
# ---------------------------------------------------------------------------

def fig_acquisition(img: np.ndarray):
    # Required deliverable: save the acquired image into outputs/.
    out_path = OUTPUTS_DIR / "task2_acquired_image.png"
    cv2.imwrite(str(out_path), img)
    print(f"wrote {out_path}")

    fig, ax = plt.subplots(figsize=(4, 4))
    imshow_gray(ax, img, f"Acquired image\n{IMAGE_LABEL}\n{img.shape[1]}x{img.shape[0]} px")
    savefig(fig, "task2_acquisition.png")


# ---------------------------------------------------------------------------
# 2.4 Image representation
# ---------------------------------------------------------------------------

def fig_properties(img: np.ndarray, y: int, x: int):
    props = acq.image_properties(img)
    pixel_val = acq.pixel_at(img, y, x)
    props["pixel_coord_yx"] = f"({y}, {x})"
    props["pixel_value"] = int(pixel_val)

    with open(RESULTS_DIR / "task2_properties.csv", "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["property", "value"])
        for k, v in props.items():
            w_.writerow([k, v])
    print(f"wrote {RESULTS_DIR / 'task2_properties.csv'}")

    disp = cv2.cvtColor(nearest_upscale(img), cv2.COLOR_GRAY2RGB)
    scale = disp.shape[0] // img.shape[0]
    cy, cx = y * scale, x * scale
    cv2.drawMarker(disp, (cx, cy), (255, 0, 0), markerType=cv2.MARKER_CROSS,
                    markerSize=18, thickness=2)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    axes[0].imshow(disp)
    axes[0].set_title(f"Marked coordinate (y={y}, x={x})", fontsize=9)
    axes[0].axis("off")

    axes[1].axis("off")
    lines = [
        f"Height          : {props['height']} px",
        f"Width           : {props['width']} px",
        f"Channels        : {props['channels']}",
        f"Total pixels    : {props['total_pixels']}",
        f"Data type       : {props['dtype']}",
        f"In-memory size  : {props['size_bytes']} bytes",
        "",
        f"Pixel at (y={y}, x={x}) : {pixel_val}",
    ]
    axes[1].text(0.02, 0.95, "\n".join(lines), va="top", ha="left",
                 fontsize=11, family="monospace", transform=axes[1].transAxes)
    fig.suptitle("2.4 Image representation", fontsize=11)
    savefig(fig, "task2_properties.png")
    return props


# ---------------------------------------------------------------------------
# 2.5 Colour space conversion
# ---------------------------------------------------------------------------

def fig_colour_spaces(img: np.ndarray):
    cs = acq.to_colour_spaces(img)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
    imshow_gray(axes[0, 0], cs["gray"], "Grayscale (native)")
    axes[0, 1].imshow(cs["rgb"], interpolation="nearest")
    axes[0, 1].set_title("RGB (grayscale promoted to 3ch)", fontsize=9)
    axes[0, 1].axis("off")
    axes[0, 2].imshow(cv2.cvtColor(cs["hsv"], cv2.COLOR_HSV2RGB), interpolation="nearest")
    axes[0, 2].set_title("HSV (rendered as RGB)", fontsize=9)
    axes[0, 2].axis("off")

    h_im = axes[1, 0].imshow(cs["h"], cmap="hsv", vmin=0, vmax=179, interpolation="nearest")
    axes[1, 0].set_title(f"H plane (range {cs['h'].min()}-{cs['h'].max()})", fontsize=9)
    axes[1, 0].axis("off")
    fig.colorbar(h_im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    s_im = axes[1, 1].imshow(cs["s"], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[1, 1].set_title(f"S plane (range {cs['s'].min()}-{cs['s'].max()})", fontsize=9)
    axes[1, 1].axis("off")
    fig.colorbar(s_im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    v_im = axes[1, 2].imshow(cs["v"], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[1, 2].set_title(f"V plane (range {cs['v'].min()}-{cs['v'].max()})", fontsize=9)
    axes[1, 2].axis("off")
    fig.colorbar(v_im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    v_matches_gray = bool(np.array_equal(cs["v"], cs["gray"]))
    fig.suptitle(
        "2.5 Colour space conversion -- source is single-channel IR, "
        f"so S is constant 0 and V == grayscale (verified equal: {v_matches_gray})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, "task2_colour_spaces.png")
    return {
        "s_min": int(cs["s"].min()), "s_max": int(cs["s"].max()),
        "h_min": int(cs["h"].min()), "h_max": int(cs["h"].max()),
        "v_equals_gray": v_matches_gray,
    }


# ---------------------------------------------------------------------------
# 2.6a Sampling
# ---------------------------------------------------------------------------

def fig_sampling(img: np.ndarray):
    scales = [1.0, 0.5, 0.25]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4))
    for ax, scale in zip(axes, scales):
        sampled = acq.sample_image(img, scale)
        disp = nearest_upscale(sampled, DISPLAY_MIN)
        imshow_gray(ax, disp,
                    f"{int(scale*100)}%  ({sampled.shape[1]}x{sampled.shape[0]} px)\n"
                    f"shown via INTER_NEAREST upscale")
    fig.suptitle("2.6 Sampling -- 100% / 50% / 25% of original spatial resolution", fontsize=11)
    savefig(fig, "task2_sampling.png")


# ---------------------------------------------------------------------------
# 2.6b Quantization
# ---------------------------------------------------------------------------

def fig_quantization(img: np.ndarray):
    levels_list = [256, 128, 64, 32]
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for col, levels in enumerate(levels_list):
        q = acq.quantize(img, levels)
        metrics = acq.quantization_metrics(img, q)
        n_unique = int(len(np.unique(q)))
        rows.append({"levels": levels, **metrics, "unique_values": n_unique})

        psnr_str = "inf" if np.isinf(metrics["psnr_db"]) else f"{metrics['psnr_db']:.2f}"
        imshow_gray(axes[0, col], q,
                    f"{levels} levels\nPSNR={psnr_str} dB  SSIM={metrics['ssim']:.4f}")

        crop = q[40:72, 40:72]  # 32x32 region spanning iris/sclera boundary
        zoom = nearest_upscale(crop, 256)
        imshow_gray(axes[1, col], zoom, "zoomed crop (nearest)")

    with open(RESULTS_DIR / "task2_quantization_metrics.csv", "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["levels", "psnr_db", "ssim", "unique_values"])
        for r in rows:
            w_.writerow([r["levels"], r["psnr_db"], r["ssim"], r["unique_values"]])
    print(f"wrote {RESULTS_DIR / 'task2_quantization_metrics.csv'}")

    fig.suptitle("2.6 Quantization -- 256 / 128 / 64 / 32 gray levels, full image + zoomed crop", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig(fig, "task2_quantization.png")
    return rows


# ---------------------------------------------------------------------------
# 2.7 File formats
# ---------------------------------------------------------------------------

def fig_file_formats(img: np.ndarray):
    saved = acq.save_formats(img, OUTPUTS_DIR, "task2_sample", jpeg_quality=95)
    raw_bytes = img.nbytes

    # Bit-identity check for the lossless formats (required deliverable).
    bmp_back = cv2.imread(saved["bmp"]["path"], cv2.IMREAD_GRAYSCALE)
    png_back = cv2.imread(saved["png"]["path"], cv2.IMREAD_GRAYSCALE)
    jpg_back = cv2.imread(saved["jpg"]["path"], cv2.IMREAD_GRAYSCALE)
    bmp_identical = bool(np.array_equal(bmp_back, img))
    png_identical = bool(np.array_equal(png_back, img))
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
    jpg_psnr = float(psnr(img, jpg_back, data_range=255))
    jpg_ssim = float(ssim(img, jpg_back, data_range=255))
    print(f"  BMP bit-identical to original: {bmp_identical}")
    print(f"  PNG bit-identical to original: {png_identical}")
    print(f"  JPEG(95) PSNR={jpg_psnr:.2f} dB  SSIM={jpg_ssim:.4f}")

    # Extra illustrative low-quality JPEG (not part of the 3 required saves)
    # purely to make compression artifacts visible in the figure.
    low_path = OUTPUTS_DIR / "task2_sample_q10.jpg"
    cv2.imwrite(str(low_path), img, [cv2.IMWRITE_JPEG_QUALITY, 10])
    jpg10_back = cv2.imread(str(low_path), cv2.IMREAD_GRAYSCALE)
    jpg10_bytes = low_path.stat().st_size
    jpg10_psnr = float(psnr(img, jpg10_back, data_range=255))
    jpg10_ssim = float(ssim(img, jpg10_back, data_range=255))
    print(f"  JPEG(10, illustrative) bytes={jpg10_bytes} PSNR={jpg10_psnr:.2f} dB SSIM={jpg10_ssim:.4f}")

    rows = [
        {"format": "BMP", "bytes": saved["bmp"]["bytes"],
         "compression_ratio": acq.compression_ratio(raw_bytes, saved["bmp"]["bytes"]),
         "psnr_db": float("inf"), "ssim": 1.0, "bit_identical": bmp_identical},
        {"format": "PNG", "bytes": saved["png"]["bytes"],
         "compression_ratio": acq.compression_ratio(raw_bytes, saved["png"]["bytes"]),
         "psnr_db": float("inf"), "ssim": 1.0, "bit_identical": png_identical},
        {"format": "JPEG (q=95)", "bytes": saved["jpg"]["bytes"],
         "compression_ratio": acq.compression_ratio(raw_bytes, saved["jpg"]["bytes"]),
         "psnr_db": jpg_psnr, "ssim": jpg_ssim, "bit_identical": False},
        {"format": "JPEG (q=10, illustrative)", "bytes": jpg10_bytes,
         "compression_ratio": acq.compression_ratio(raw_bytes, jpg10_bytes),
         "psnr_db": jpg10_psnr, "ssim": jpg10_ssim, "bit_identical": False},
    ]
    with open(RESULTS_DIR / "task2_file_formats.csv", "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["format", "bytes", "compression_ratio", "psnr_db", "ssim", "bit_identical"])
        for r in rows:
            w_.writerow([r["format"], r["bytes"], f"{r['compression_ratio']:.4f}",
                         r["psnr_db"], f"{r['ssim']:.4f}", r["bit_identical"]])
    print(f"wrote {RESULTS_DIR / 'task2_file_formats.csv'}")

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.3, 1])

    ax_bar = fig.add_subplot(gs[0, :])
    labels = [r["format"] for r in rows]
    sizes = [r["bytes"] for r in rows]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#dd8452"]
    ax_bar.bar(labels, sizes, color=colors)
    ax_bar.axhline(raw_bytes, color="black", linestyle="--", linewidth=1,
                    label=f"raw pixel data ({raw_bytes} bytes)")
    ax_bar.set_ylabel("File size (bytes)")
    ax_bar.set_title("On-disk file size, measured")
    ax_bar.tick_params(axis="x", rotation=10)
    ax_bar.legend(fontsize=8)
    for i, s in enumerate(sizes):
        ax_bar.text(i, s, str(s), ha="center", va="bottom", fontsize=8)

    imgs_disp = [nearest_upscale(x, 256) for x in
                 (img, bmp_back, png_back, jpg_back, jpg10_back)]
    titles = ["original", "BMP", "PNG", "JPEG q95", "JPEG q10"]
    for col, (im, t) in enumerate(zip(imgs_disp, titles)):
        ax = fig.add_subplot(gs[1, col])
        imshow_gray(ax, im, t)

    fig.suptitle("2.7 File formats -- size comparison and decoded-image check", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, "task2_file_formats.png")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--y", type=int, default=None, help="row coordinate for pixel_at (default: image center)")
    parser.add_argument("--x", type=int, default=None, help="column coordinate for pixel_at (default: image center)")
    args = parser.parse_args()

    print("== 2.1 Environment verification ==")
    fig_library_versions()

    print("== 2.3 Image acquisition ==")
    img = acq.load_image(IMAGE_PATH, mode="gray")
    fig_acquisition(img)

    print("== 2.4 Image representation ==")
    h, w = img.shape[:2]
    y = args.y if args.y is not None else h // 2
    x = args.x if args.x is not None else w // 2
    fig_properties(img, y, x)

    print("== 2.5 Colour space conversion ==")
    fig_colour_spaces(img)

    print("== 2.6 Sampling ==")
    fig_sampling(img)

    print("== 2.6 Quantization ==")
    fig_quantization(img)

    print("== 2.7 File formats ==")
    fig_file_formats(img)

    print("Done.")


if __name__ == "__main__":
    main()
