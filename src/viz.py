"""Week-1 visualization: a contact-sheet grid of sample eye images.

Images in data/samples/ are variable-size (73 distinct dimensions across a
200-image sample of this dataset), so every tile MUST be resized to a
common size before being placed in the grid.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.dataset import EyeSample, load_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "samples" / "manifest.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "week1_sample_grid.png"

ALERT_BORDER = (0, 170, 0)   # green: eyeState == 1 (open)
DROWSY_BORDER = (200, 0, 0)  # red:   eyeState == 0 (closed)
BORDER_PX = 4
CAPTION_H = 16


def contact_sheet(samples: list[EyeSample], cols: int = 8, tile: tuple[int, int] = (100, 100)) -> Image.Image:
    """Build an 8xN contact sheet: each tile resized to `tile`, bordered
    green (alert / eyeState=1) or red (drowsy / eyeState=0), with a small
    caption strip (subject id + eye state) beneath each tile.
    """
    n = len(samples)
    rows = math.ceil(n / cols)
    tile_w, tile_h = tile
    cell_w = tile_w + 2 * BORDER_PX
    cell_h = tile_h + 2 * BORDER_PX + CAPTION_H

    canvas = Image.new("RGB", (cell_w * cols, cell_h * rows), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, sample in enumerate(samples):
        row, col = divmod(idx, cols)
        img = Image.open(sample.path).convert("RGB").resize(tile, Image.BILINEAR)
        border_color = ALERT_BORDER if sample.is_open else DROWSY_BORDER
        bordered = ImageOps.expand(img, border=BORDER_PX, fill=border_color)

        x0 = col * cell_w
        y0 = row * cell_h
        canvas.paste(bordered, (x0, y0))

        caption = f"{sample.subject_id} {'open' if sample.is_open else 'closed'}"
        draw.text((x0 + BORDER_PX, y0 + tile_h + 2 * BORDER_PX), caption, fill=(0, 0, 0), font=font)

    return canvas


def main() -> None:
    samples = load_manifest(DEFAULT_MANIFEST)
    sheet = contact_sheet(samples, cols=8, tile=(100, 100))
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(DEFAULT_OUTPUT)
    print(f"Wrote {DEFAULT_OUTPUT} ({sheet.size[0]}x{sheet.size[1]}, {len(samples)} tiles)")


if __name__ == "__main__":
    main()
