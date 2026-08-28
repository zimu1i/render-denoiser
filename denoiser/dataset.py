"""
PyTorch datasets over the cached renders.

Training and evaluation sample the cache differently, and the difference is
deliberate.

**Training takes random crops.** Every epoch sees different 128x128 windows, so
255 views yield far more than 255 distinct examples and the network never learns
a fixed tile grid. Random cropping is the augmentation here; flips and rotations
are not used, because a denoiser should not be encouraged to treat a rotated
version of a scene as new information about light transport.

**Evaluation walks a fixed grid.** Scores have to be comparable between runs, so
the test set is deterministic: the same sixteen tiles of the same 45 views,
every time.

The split itself is done on whole views before either class sees the data, so no
scene can appear on both sides.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from denoiser.data import CHANNELS, TILE, tile_positions

# Channels 0:9 are the network's input, 9:12 the target.
INPUT_SLICE = slice(0, 9)
TARGET_SLICE = slice(9, CHANNELS)

# The ablation: with the auxiliary buffers, or with the noisy render alone.
RADIANCE_ONLY = slice(0, 3)


def to_tensor(patch: np.ndarray) -> torch.Tensor:
    """HWC numpy to CHW float32 tensor."""
    return torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1), dtype=np.float32))


class RandomTiles(Dataset):
    """Random crops from the training views."""

    def __init__(self, cache, indices, tile: int = TILE, per_view: int = 16, seed: int = 0):
        self.cache = cache
        self.indices = list(indices)
        self.tile = tile
        self.per_view = per_view
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.indices) * self.per_view

    def set_epoch(self, epoch: int) -> None:
        """Re-seed cropping so each epoch sees new windows but stays repeatable."""
        self.epoch = epoch

    def __getitem__(self, item: int):
        view = self.indices[item // self.per_view]
        # Seeded per (epoch, item) so a run can be reproduced exactly while
        # still showing different crops from one epoch to the next.
        rng = np.random.default_rng((self.seed, self.epoch, item))
        limit = self.cache.shape[1] - self.tile
        y, x = int(rng.integers(0, limit + 1)), int(rng.integers(0, limit + 1))

        patch = np.asarray(self.cache[view, y : y + self.tile, x : x + self.tile, :], np.float32)
        return to_tensor(patch[..., INPUT_SLICE]), to_tensor(patch[..., TARGET_SLICE])


class GridTiles(Dataset):
    """Every tile of every held-out view, in a fixed order."""

    def __init__(self, cache, indices, tile: int = TILE):
        self.cache = cache
        self.tile = tile
        self.positions = tile_positions(cache.shape[1], tile)
        self.items = [(v, y, x) for v in indices for (y, x) in self.positions]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int):
        view, y, x = self.items[item]
        patch = np.asarray(self.cache[view, y : y + self.tile, x : x + self.tile, :], np.float32)
        return to_tensor(patch[..., INPUT_SLICE]), to_tensor(patch[..., TARGET_SLICE])


class FullViews(Dataset):
    """Whole 512x512 views, for scoring images rather than tiles.

    Reported metrics come from here. A per-tile average would quietly weight
    every scene by how many tiles it happens to contribute, and would miss
    artifacts that only appear across tile boundaries.
    """

    def __init__(self, cache, indices):
        self.cache = cache
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        view = np.asarray(self.cache[self.indices[item]], np.float32)
        return to_tensor(view[..., INPUT_SLICE]), to_tensor(view[..., TARGET_SLICE])


def drop_auxiliary(batch: torch.Tensor) -> torch.Tensor:
    """Keep only the noisy radiance, for the no-buffers arm of the ablation."""
    return batch[:, RADIANCE_ONLY]
