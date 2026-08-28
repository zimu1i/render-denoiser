"""Tests for the network and the datasets that feed it.

All offline and on tiny synthetic tensors: these check shapes, invariants and
the log-space handling, not whether the model denoises well. Whether it denoises
well is what the evaluation script measures.
"""

import numpy as np
import pytest
import torch

from denoiser.data import CHANNELS
from denoiser.dataset import FullViews, GridTiles, RandomTiles, drop_auxiliary, to_tensor
from denoiser.model import UNet, from_log, prepare_inputs, to_log


@pytest.fixture
def cache():
    """A stand-in for the memmapped dataset: 6 views at 64x64."""
    rng = np.random.default_rng(0)
    return rng.random((6, 64, 64, CHANNELS), dtype=np.float32) * 4.0


class TestLogSpace:
    def test_round_trips(self):
        x = torch.tensor([0.0, 0.5, 1.0, 10.0, 45.0])

        assert torch.allclose(from_log(to_log(x)), x, atol=1e-4)

    def test_zero_maps_to_zero(self):
        assert to_log(torch.zeros(3)).abs().max() == 0.0

    def test_is_monotonic(self):
        x = torch.linspace(0, 50, 100)

        assert torch.all(torch.diff(to_log(x)) > 0)

    def test_compresses_the_dynamic_range(self):
        """The reason it exists: raw radiance spans 0 to ~45 in this dataset,
        which a few bright pixels would otherwise dominate."""
        raw = torch.tensor([1.0, 45.0])
        compressed = to_log(raw)

        assert (raw[1] / raw[0]) > 40
        assert (compressed[1] / compressed[0]) < 6

    def test_negative_radiance_is_clamped(self):
        # Cycles emits small negatives from filter ringing.
        assert to_log(torch.tensor([-1.0]))[0] == 0.0


class TestPrepareInputs:
    def test_compresses_radiance_but_not_the_buffers(self):
        """Albedo and normals are already well scaled, and a log would destroy
        the sign that makes a normal meaningful."""
        x = torch.ones(1, 9, 8, 8) * 3.0

        prepared = prepare_inputs(x)

        assert torch.allclose(prepared[:, :3], torch.log1p(torch.tensor(3.0)))
        assert torch.allclose(prepared[:, 3:], torch.tensor(3.0))

    def test_handles_radiance_only_input(self):
        # The ablation arm passes 3 channels, not 9.
        prepared = prepare_inputs(torch.ones(1, 3, 8, 8) * 3.0)

        assert prepared.shape == (1, 3, 8, 8)


class TestUNet:
    def test_output_shape_matches_the_target(self):
        model = UNet(in_channels=9, base=8)

        assert model(torch.rand(2, 9, 64, 64)).shape == (2, 3, 64, 64)

    def test_accepts_the_ablation_channel_count(self):
        model = UNet(in_channels=3, base=8)

        assert model(torch.rand(2, 3, 64, 64)).shape == (2, 3, 64, 64)

    def test_predicts_a_residual_on_the_noisy_input(self):
        """With zeroed weights the network must return its input unchanged.

        The noisy render is already an unbiased estimate, so the identity is a
        sensible starting point and the network only learns the correction.
        """
        model = UNet(in_channels=9, base=8)
        torch.nn.init.zeros_(model.head.weight)
        torch.nn.init.zeros_(model.head.bias)

        noisy = torch.rand(1, 9, 32, 32) * 5
        output = model(noisy)

        assert torch.allclose(output, to_log(noisy[:, :3]), atol=1e-5)

    def test_is_differentiable(self):
        model = UNet(in_channels=9, base=8)
        loss = model(torch.rand(1, 9, 32, 32)).mean()
        loss.backward()

        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_size_is_modest(self):
        """Small enough to train on a laptop GPU in an evening."""
        assert UNet(in_channels=9, base=32).parameter_count() < 5_000_000

    def test_handles_a_non_square_tile(self):
        # Dimensions must stay divisible by 8 for three pooling levels.
        assert UNet(in_channels=9, base=8)(torch.rand(1, 9, 32, 64)).shape == (1, 3, 32, 64)


class TestRandomTiles:
    def test_length_counts_crops_not_views(self, cache):
        data = RandomTiles(cache, [0, 1, 2], tile=32, per_view=4)

        assert len(data) == 12

    def test_returns_split_input_and_target(self, cache):
        inputs, target = RandomTiles(cache, [0], tile=32, per_view=1)[0]

        assert inputs.shape == (9, 32, 32)
        assert target.shape == (3, 32, 32)

    def test_crops_stay_inside_the_view(self, cache):
        data = RandomTiles(cache, [0], tile=32, per_view=50)

        for i in range(len(data)):
            inputs, _ = data[i]
            assert inputs.shape == (9, 32, 32)

    def test_is_reproducible_within_an_epoch(self, cache):
        a = RandomTiles(cache, [0, 1], tile=32, per_view=4, seed=7)
        b = RandomTiles(cache, [0, 1], tile=32, per_view=4, seed=7)

        assert torch.equal(a[3][0], b[3][0])

    def test_a_new_epoch_gives_new_crops(self, cache):
        """Otherwise every epoch trains on an identical fixed tile grid."""
        data = RandomTiles(cache, [0, 1], tile=32, per_view=4, seed=7)
        first = data[3][0].clone()
        data.set_epoch(1)

        assert not torch.equal(first, data[3][0])


class TestGridTiles:
    def test_covers_every_tile_of_every_view(self, cache):
        data = GridTiles(cache, [0, 1, 2], tile=32)

        assert len(data) == 3 * 4  # 64x64 view, 32px tiles -> 4 per view

    def test_is_deterministic(self, cache):
        a, b = GridTiles(cache, [0], tile=32), GridTiles(cache, [0], tile=32)

        assert torch.equal(a[2][0], b[2][0])


class TestFullViews:
    def test_returns_whole_views(self, cache):
        inputs, target = FullViews(cache, [0, 1])[0]

        assert inputs.shape == (9, 64, 64)
        assert target.shape == (3, 64, 64)

    def test_one_item_per_view(self, cache):
        assert len(FullViews(cache, [0, 1, 2])) == 3


class TestDropAuxiliary:
    def test_keeps_only_the_noisy_radiance(self):
        batch = torch.arange(9 * 4, dtype=torch.float32).reshape(1, 9, 2, 2)

        reduced = drop_auxiliary(batch)

        assert reduced.shape == (1, 3, 2, 2)
        assert torch.equal(reduced, batch[:, :3])


class TestToTensor:
    def test_moves_channels_first(self):
        assert to_tensor(np.zeros((8, 8, 9), np.float32)).shape == (9, 8, 8)

    def test_casts_float16_to_float32(self):
        """The cache is float16 to fit in memory; the model needs float32."""
        assert to_tensor(np.zeros((8, 8, 3), np.float16)).dtype == torch.float32
