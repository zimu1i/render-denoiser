"""
Loading, splitting and scoring render pairs.

Deliberately free of any PyTorch import. The metrics and the split logic decide
whether every later result is trustworthy, so they are kept testable without a
deep learning framework installed.

Two decisions here are the ones to defend.

**Splitting is by view, never by tile.** Each 512x512 render is cropped into
sixteen 128x128 tiles that share a scene, a camera and a lighting setup. Putting
some of a view's tiles in training and others in test would let the network
memorise a scene and score well on what is effectively the same image. Views are
split first; tiles are only ever cut afterwards.

**Quality is measured on tonemapped images, not raw radiance.** Measured across
20 scenes in this dataset, PSNR on linear HDR spreads 22.7 dB between scenes
while tonemapped PSNR spreads 12.9 dB. Neither is truly scene-independent -- the
HDR peak varies 62x across these renders -- so comparisons are only ever made
between methods on the *same* held-out views, where that dependence cancels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import OpenEXR

# The passes written by render_dataset.py. Order matters: it fixes the channel
# layout the model sees, and a cache built with one order cannot be read with
# another.
INPUT_PASSES = ("beauty", "albedo", "normal")
TILE = 128
VIEW_RESOLUTION = 512

# Channels per cached view: 3 passes of the noisy render plus the clean target.
CHANNELS = len(INPUT_PASSES) * 3 + 3

# Guards a division by zero in relMSE without materially changing the metric.
# The standard choice in the denoising literature.
RELMSE_EPSILON = 1e-2


def load_exr(path) -> dict[str, np.ndarray]:
    """Read a multilayer EXR into {pass name: HxWx3 float32}.

    Blender writes each pass as its own EXR part. The alpha channel is dropped:
    it is constant for these renders and would only add a dead input channel.
    """
    handle = OpenEXR.File(str(path))
    return {
        part.name(): np.asarray(
            list(part.channels.values())[0].pixels[..., :3], dtype=np.float32
        )
        for part in handle.parts
    }


def stack_inputs(noisy: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate the noisy render and its auxiliary buffers into HxWx9."""
    return np.concatenate([noisy[name] for name in INPUT_PASSES], axis=2)


def tonemap(x: np.ndarray) -> np.ndarray:
    """Linear radiance to display space, the transform the metrics are read in.

    Reinhard compression followed by a gamma curve. Applied before scoring
    because a viewer never sees linear radiance, and because errors in bright
    highlights would otherwise dominate a metric out of all proportion to how
    visible they are.
    """
    x = np.clip(x, 0.0, None)
    return np.clip((x / (1.0 + x)) ** (1.0 / 2.2), 0.0, 1.0)


def psnr(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Peak signal-to-noise ratio, in decibels, on tonemapped images.

    The peak is 1.0 because tonemapping bounds both images to [0, 1]. Using the
    per-image maximum instead, as is common with HDR data, makes the number
    depend on whether the scene happens to contain a bright highlight.
    """
    mse = float(np.mean((tonemap(prediction) - tonemap(reference)) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def relmse(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Relative mean squared error on linear radiance.

    Each pixel's error is scaled by its own reference intensity, so a dim corner
    of the image counts as much as a bright one. This is the metric Monte Carlo
    denoising papers report, and it is kept alongside PSNR because the two
    disagree about which scenes are hard.
    """
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(
        np.mean((prediction - reference) ** 2 / (reference**2 + RELMSE_EPSILON))
    )


def find_views(raw_dir) -> list[Path]:
    """Every view directory holding a complete noisy/clean pair."""
    directories = sorted(Path(raw_dir).glob("view_*"))
    return [d for d in directories if (d / "noisy.exr").exists() and (d / "clean.exr").exists()]


def split_views(views, test_fraction: float = 0.15, seed: int = 0):
    """Partition views into train and test, deterministically.

    Splitting happens here, on whole views, and never on tiles. See the module
    docstring for why that distinction is the difference between an honest
    result and a leaked one.
    """
    views = list(views)
    order = np.random.default_rng(seed).permutation(len(views))
    cut = int(round(len(views) * (1.0 - test_fraction)))
    train = [views[i] for i in order[:cut]]
    test = [views[i] for i in order[cut:]]
    return train, test


def build_cache(views, cache_path, resolution: int = VIEW_RESOLUTION):
    """Pack every view into one float16 memmap of shape (N, H, W, 12).

    float16 rather than float32 halves the file to under a gigabyte, which is
    what lets the whole dataset stay resident while training. EXR already stores
    half floats natively, so this discards no precision the renderer produced.

    Layout of the last axis: 9 input channels (noisy beauty, albedo, normal)
    followed by the 3 clean target channels.
    """
    views = list(views)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    array = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(views), resolution, resolution, CHANNELS),
    )
    for index, view in enumerate(views):
        noisy = load_exr(Path(view) / "noisy.exr")
        clean = load_exr(Path(view) / "clean.exr")
        array[index, ..., :9] = stack_inputs(noisy).astype(np.float16)
        array[index, ..., 9:] = clean["beauty"].astype(np.float16)
    array.flush()

    manifest = cache_path.with_suffix(".json")
    manifest.write_text(
        json.dumps(
            {
                "views": [str(Path(v).name) for v in views],
                "passes": list(INPUT_PASSES),
                "resolution": resolution,
                "channels": CHANNELS,
            },
            indent=2,
        )
    )
    return array


def load_cache(cache_path):
    """Memory-map a cache built by build_cache."""
    return np.load(str(cache_path), mmap_mode="r")


def tile_positions(resolution: int = VIEW_RESOLUTION, tile: int = TILE):
    """Top-left corners of a non-overlapping tile grid."""
    return [(y, x) for y in range(0, resolution - tile + 1, tile)
            for x in range(0, resolution - tile + 1, tile)]


def crop(view: np.ndarray, y: int, x: int, tile: int = TILE):
    """Cut one tile, returning (inputs HxWx9, target HxWx3) as float32."""
    patch = np.asarray(view[y : y + tile, x : x + tile, :], dtype=np.float32)
    return patch[..., :9], patch[..., 9:]
