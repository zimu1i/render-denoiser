"""
Build the visual comparison used in the README.

    ./venv/bin/python figures.py --run runs/with_aux --views 3

Numbers say a denoiser works; a picture says how it fails. Blur artifacts, lost
texture and over-smoothed edges are obvious on sight and invisible in a PSNR
average, so the figure is part of the evidence rather than decoration.

Every panel is tonemapped identically. Comparing images under different display
transforms would be meaningless, and it is an easy mistake to make when one of
them came out of a different tool.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from denoiser import CACHE_PATH, RAW_DIR, ROOT
from denoiser.data import find_views, load_cache, load_exr, psnr, split_views, tonemap
from denoiser.evaluate import gaussian_blur, load_model, predict, select_blur_sigma
from denoiser.model import select_device

LABEL_HEIGHT = 22


def to_image(linear: np.ndarray) -> Image.Image:
    return Image.fromarray((tonemap(linear) * 255).astype(np.uint8))


def crop_region(array: np.ndarray, box) -> np.ndarray:
    top, left, size = box
    return array[top : top + size, left : left + size]


def build_figure(run_dir: Path, count: int, out_path: Path, zoom: int | None = None):
    device = select_device()
    cache = load_cache(CACHE_PATH)
    views = find_views(RAW_DIR)
    position = {view: index for index, view in enumerate(views)}
    train_views, test_views = split_views(views)

    sigma, _ = select_blur_sigma(cache, [position[v] for v in train_views])
    model, in_channels = load_model(run_dir, device)

    chosen = test_views[:count]
    rows = []
    for view in chosen:
        row = np.asarray(cache[position[view]], np.float32)
        noisy, reference = row[..., :9], row[..., 9:]

        panels = [
            ("4 spp input", noisy[..., :3]),
            (f"gaussian blur", gaussian_blur(noisy[..., :3], sigma)),
            ("learned denoiser", predict(model, noisy, in_channels, device)),
        ]
        oidn_path = Path(view) / "oidn.exr"
        if oidn_path.exists():
            panels.append(("OpenImageDenoise", load_exr(oidn_path)["beauty"]))
        panels.append(("512 spp reference", reference))

        if zoom:
            box = (reference.shape[0] // 2 - zoom // 2, reference.shape[1] // 2 - zoom // 2, zoom)
            panels = [(name, crop_region(image, box)) for name, image in panels]

        rows.append([(name, image, psnr(image, panels[-1][1])) for name, image in panels])

    height, width = rows[0][0][1].shape[:2]
    columns = len(rows[0])
    canvas = Image.new("RGB", (width * columns, (height + LABEL_HEIGHT) * len(rows)), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)

    for r, row in enumerate(rows):
        top = r * (height + LABEL_HEIGHT)
        for c, (name, image, score) in enumerate(row):
            canvas.paste(to_image(image), (c * width, top + LABEL_HEIGHT))
            # The reference scores infinity against itself, which is noise on a
            # label rather than information.
            caption = name if not np.isfinite(score) else f"{name}  {score:.1f} dB"
            draw.text((c * width + 5, top + 5), caption, fill=(235, 235, 235))

    canvas.save(out_path)
    return out_path, canvas.size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(ROOT / "runs" / "with_aux"))
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--out", default=str(ROOT / "comparison.png"))
    parser.add_argument(
        "--zoom",
        type=int,
        default=None,
        help="centre crop of this many pixels, to show detail rather than thumbnails",
    )
    args = parser.parse_args()

    path, size = build_figure(Path(args.run), args.views, Path(args.out), args.zoom)
    print(f"wrote {path} ({size[0]}x{size[1]})")


if __name__ == "__main__":
    main()
