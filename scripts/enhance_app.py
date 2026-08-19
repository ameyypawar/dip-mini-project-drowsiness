#!/usr/bin/env python3
"""Task 3 integration app: menu-driven CLI for src.enhancement.

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057).

Two modes:
  1. Non-interactive (scriptable / testable without a TTY): pass --image and
     --technique (plus any technique-specific flags) and it loads, applies,
     prints before/after stats, and saves to --out, then exits. This is the
     mode used for automated testing (no input() call is reached).
     Example:
       ./.venv/bin/python scripts/enhance_app.py \\
         --image data/samples/alert/s0031_00366_1_0_1_0_1_02.png \\
         --technique gamma --gamma 1.5 \\
         --out results/task3_cli_demo_gamma.png

  2. Interactive menu (no --technique given, and stdin is a TTY): prompts
     for an image path, a technique from a numbered menu, its parameters,
     then a save path.

`--list-techniques` prints the technique menu and exits.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import enhancement as enh  # noqa: E402

MENU = [
    ("negative", "Image negative"),
    ("log", "Log transformation"),
    ("gamma", "Gamma (power-law) correction"),
    ("contrast_stretch", "Contrast stretching (min-max / percentile)"),
    ("hist_eq", "Histogram equalization (OpenCV)"),
    ("hist_eq_manual", "Histogram equalization (own CDF implementation)"),
    ("add_constant", "Add constant (brightness boost)"),
    ("add", "Add two images (weighted blend / overlay)"),
    ("subtract", "Subtract two images (change / motion detection)"),
    ("average", "Average N noisy copies (noise reduction demo)"),
]


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def apply(technique: str, img: np.ndarray, args) -> np.ndarray:
    if technique == "negative":
        return enh.negative(img)
    if technique == "log":
        return enh.log_transform(img, c=args.c)
    if technique == "gamma":
        return enh.gamma_correction(img, gamma=args.gamma, c=args.c or 1.0)
    if technique == "contrast_stretch":
        return enh.contrast_stretch(
            img, r_min=args.r_min, r_max=args.r_max,
            low_pct=args.low_pct, high_pct=args.high_pct,
        )
    if technique == "hist_eq":
        return enh.equalize_histogram_cv(img)
    if technique == "hist_eq_manual":
        return enh.equalize_histogram_manual(img)
    if technique == "add_constant":
        return enh.add_constant(img, value=args.value)
    if technique == "add":
        img2 = load_gray(args.image2)
        return enh.add_images(img, img2, alpha=args.alpha)
    if technique == "subtract":
        img2 = load_gray(args.image2)
        return enh.subtract_images(img, img2)
    if technique == "average":
        copies = enh.make_noisy_copies(img, n=args.n_noisy, sigma=args.noise_sigma, seed=args.seed)
        return enh.average_images(copies)
    raise ValueError(f"unknown technique: {technique}")


def print_stats(label: str, img: np.ndarray):
    print(f"  {label}: mean(brightness)={enh.mean_intensity(img):.2f}  "
          f"std(contrast)={enh.rms_contrast(img):.2f}  "
          f"noise_proxy={enh.noise_proxy(img):.2f}")


def run_noninteractive(args) -> int:
    img = load_gray(args.image)
    print(f"Loaded {args.image} ({img.shape[1]}x{img.shape[0]})")
    out = apply(args.technique, img, args)
    print(f"Applied technique: {args.technique}")
    print_stats("original", img)
    print_stats("output  ", out)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, out)
        print(f"Saved: {args.out}")
    return 0


def run_interactive() -> int:
    print("=== Task 3 Enhancement App (interactive) ===")
    image_path = input("Image path: ").strip()
    img = load_gray(image_path)
    print(f"Loaded {image_path} ({img.shape[1]}x{img.shape[0]})")

    print("\nTechniques:")
    for i, (key, desc) in enumerate(MENU, start=1):
        print(f"  {i}. {desc} [{key}]")
    choice = input("Pick a technique number: ").strip()
    try:
        technique = MENU[int(choice) - 1][0]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return 1

    class Args:
        pass
    args = Args()
    args.c = None
    args.gamma = 1.0
    args.r_min = None
    args.r_max = None
    args.low_pct = 2.0
    args.high_pct = 98.0
    args.value = 40
    args.image2 = None
    args.alpha = 0.5
    args.n_noisy = 5
    args.noise_sigma = 20.0
    args.seed = 0

    if technique == "gamma":
        args.gamma = float(input("gamma (e.g. 0.5 brighten, 1.5 darken) [1.0]: ") or 1.0)
    elif technique == "contrast_stretch":
        low = input("low percentile [2.0]: ").strip()
        high = input("high percentile [98.0]: ").strip()
        args.low_pct = float(low) if low else 2.0
        args.high_pct = float(high) if high else 98.0
    elif technique == "add_constant":
        args.value = int(input("value to add (can be negative) [40]: ") or 40)
    elif technique in ("add", "subtract"):
        args.image2 = input("second image path: ").strip()
        if technique == "add":
            args.alpha = float(input("alpha (weight on first image) [0.5]: ") or 0.5)
    elif technique == "average":
        args.n_noisy = int(input("number of noisy copies to average [5]: ") or 5)
        args.noise_sigma = float(input("Gaussian noise sigma [20.0]: ") or 20.0)

    out = apply(technique, img, args)
    print_stats("original", img)
    print_stats("output  ", out)

    save_path = input("Save output to (path, blank to skip): ").strip()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, out)
        print(f"Saved: {save_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Task 3 image-enhancement CLI (src.enhancement).")
    p.add_argument("--image", help="input image path")
    p.add_argument("--image2", help="second image path (for add/subtract)")
    p.add_argument("--technique", choices=[k for k, _ in MENU], help="technique to apply")
    p.add_argument("--out", help="output image path")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--c", type=float, default=None, help="log/gamma scale constant")
    p.add_argument("--r-min", type=float, default=None, dest="r_min")
    p.add_argument("--r-max", type=float, default=None, dest="r_max")
    p.add_argument("--low-pct", type=float, default=2.0, dest="low_pct")
    p.add_argument("--high-pct", type=float, default=98.0, dest="high_pct")
    p.add_argument("--value", type=int, default=40, help="add_constant offset")
    p.add_argument("--alpha", type=float, default=0.5, help="add blend weight")
    p.add_argument("--n-noisy", type=int, default=5, dest="n_noisy")
    p.add_argument("--noise-sigma", type=float, default=20.0, dest="noise_sigma")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--list-techniques", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.list_techniques:
        for key, desc in MENU:
            print(f"{key}: {desc}")
        return 0

    if args.image and args.technique:
        return run_noninteractive(args)

    if not sys.stdin.isatty():
        print("No --image/--technique given and stdin is not a TTY -- "
              "refusing to block on input(). Pass --image and --technique "
              "(see --help) or run interactively from a real terminal.",
              file=sys.stderr)
        return 2

    return run_interactive()


if __name__ == "__main__":
    raise SystemExit(main())
