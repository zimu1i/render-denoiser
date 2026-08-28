"""Tests for loading, splitting and scoring.

The metrics and the split decide whether every later number is trustworthy, so
they are tested on synthetic arrays where the right answer is known exactly. The
tests marked slow touch the real rendered dataset.
"""

import numpy as np
import pytest

from denoiser import RAW_DIR
from denoiser.data import (
    CHANNELS,
    INPUT_PASSES,
    build_cache,
    crop,
    find_views,
    load_exr,
    psnr,
    relmse,
    split_views,
    stack_inputs,
    tile_positions,
    tonemap,
)


class TestTonemap:
    def test_maps_into_the_unit_interval(self):
        x = np.array([0.0, 0.5, 1.0, 10.0, 1000.0], dtype=np.float32)

        assert tonemap(x).min() >= 0.0
        assert tonemap(x).max() <= 1.0

    def test_is_monotonic(self):
        """Brighter radiance must stay brighter after the transform."""
        x = np.linspace(0, 50, 200, dtype=np.float32)

        assert np.all(np.diff(tonemap(x)) >= 0)

    def test_clamps_negative_radiance(self):
        # Cycles can emit small negatives from filter ringing.
        assert tonemap(np.array([-1.0], dtype=np.float32))[0] == 0.0

    def test_compresses_the_highlights(self):
        """A 10x difference in radiance must not remain a 10x difference on
        screen, or bright pixels dominate every metric."""
        dim, bright = tonemap(np.array([1.0])), tonemap(np.array([10.0]))

        assert bright / dim < 2.0


class TestPsnr:
    def test_identical_images_are_infinite(self):
        x = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)

        assert psnr(x, x) == float("inf")

    def test_more_error_scores_lower(self):
        rng = np.random.default_rng(0)
        ref = rng.random((32, 32, 3)).astype(np.float32)
        small = ref + rng.normal(0, 0.01, ref.shape).astype(np.float32)
        large = ref + rng.normal(0, 0.10, ref.shape).astype(np.float32)

        assert psnr(small, ref) > psnr(large, ref)

    def test_does_not_depend_on_a_scene_having_a_bright_highlight(self):
        """The failure of HDR PSNR that motivated tonemapping first.

        Two images with the same visible error should score the same whether or
        not one of them contains a single very bright pixel.
        """
        rng = np.random.default_rng(0)
        ref = rng.random((32, 32, 3)).astype(np.float32)
        noisy = ref + rng.normal(0, 0.05, ref.shape).astype(np.float32)

        ref_spike, noisy_spike = ref.copy(), noisy.copy()
        ref_spike[0, 0] = 50.0
        noisy_spike[0, 0] = 50.0

        assert psnr(noisy, ref) == pytest.approx(psnr(noisy_spike, ref_spike), abs=0.5)


class TestRelmse:
    def test_identical_images_score_zero(self):
        x = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32) + 1.0

        assert relmse(x, x) == pytest.approx(0.0)

    def test_weights_dim_and_bright_regions_comparably(self):
        """Plain MSE is dominated by bright pixels; relMSE is why it is here.

        The same *proportional* error in a dim region and a bright one should
        contribute similarly.
        """
        dim_ref = np.full((16, 16, 3), 1.0, dtype=np.float32)
        bright_ref = np.full((16, 16, 3), 20.0, dtype=np.float32)

        dim_err = relmse(dim_ref * 1.1, dim_ref)
        bright_err = relmse(bright_ref * 1.1, bright_ref)

        assert bright_err == pytest.approx(dim_err, rel=0.3)

    def test_more_error_scores_higher(self):
        ref = np.full((16, 16, 3), 2.0, dtype=np.float32)

        assert relmse(ref * 1.5, ref) > relmse(ref * 1.05, ref)


class TestSplitViews:
    def test_partitions_without_overlap(self):
        views = [f"view_{i:03d}" for i in range(100)]

        train, test = split_views(views, test_fraction=0.2)

        assert len(train) == 80 and len(test) == 20
        assert not set(train) & set(test)
        assert set(train) | set(test) == set(views)

    def test_is_deterministic(self):
        views = [f"view_{i:03d}" for i in range(50)]

        assert split_views(views, seed=1) == split_views(views, seed=1)

    def test_a_different_seed_gives_a_different_split(self):
        views = [f"view_{i:03d}" for i in range(50)]

        assert split_views(views, seed=1)[1] != split_views(views, seed=2)[1]

    def test_shuffles_rather_than_taking_a_contiguous_block(self):
        """Views are rendered in sequence, so a contiguous tail could correlate
        with whatever the random scene generator happened to be doing late in
        the run."""
        views = [f"view_{i:03d}" for i in range(100)]

        _, test = split_views(views, test_fraction=0.2)

        assert test != views[80:]

    def test_empty(self):
        assert split_views([]) == ([], [])


class TestTiling:
    def test_covers_the_view_without_overlap(self):
        positions = tile_positions(resolution=512, tile=128)

        assert len(positions) == 16
        assert len(set(positions)) == 16

    def test_stays_inside_the_view(self):
        for y, x in tile_positions(resolution=512, tile=128):
            assert 0 <= y <= 512 - 128 and 0 <= x <= 512 - 128

    def test_crop_splits_inputs_from_target(self):
        view = np.arange(512 * 512 * CHANNELS, dtype=np.float32).reshape(512, 512, CHANNELS)

        inputs, target = crop(view, 0, 0, tile=128)

        assert inputs.shape == (128, 128, 9)
        assert target.shape == (128, 128, 3)
        assert np.array_equal(inputs, view[:128, :128, :9])
        assert np.array_equal(target, view[:128, :128, 9:])

    def test_crop_returns_float32_from_a_float16_cache(self):
        view = np.zeros((256, 256, CHANNELS), dtype=np.float16)

        inputs, target = crop(view, 0, 0, tile=128)

        assert inputs.dtype == np.float32 and target.dtype == np.float32


class TestStackInputs:
    def test_orders_channels_as_declared(self):
        passes = {
            name: np.full((4, 4, 3), i, dtype=np.float32)
            for i, name in enumerate(INPUT_PASSES)
        }

        stacked = stack_inputs(passes)

        assert stacked.shape == (4, 4, 9)
        for i in range(len(INPUT_PASSES)):
            assert np.all(stacked[..., i * 3 : (i + 1) * 3] == i)


@pytest.mark.slow
class TestRealRenders:
    @pytest.fixture(scope="class")
    def views(self):
        found = find_views(RAW_DIR)
        if not found:
            pytest.skip("no rendered views; run render_dataset.py first")
        return found

    def test_dataset_is_complete(self, views):
        assert len(views) == 300

    def test_every_pass_is_present_and_finite(self, views):
        passes = load_exr(views[0] / "noisy.exr")

        assert set(INPUT_PASSES).issubset(passes)
        for name in INPUT_PASSES:
            assert np.isfinite(passes[name]).all(), name

    def test_the_target_is_genuinely_less_noisy_than_the_input(self, views):
        noisy = load_exr(views[0] / "noisy.exr")["beauty"]
        clean = load_exr(views[0] / "clean.exr")["beauty"]

        variance = lambda x: np.diff(x.mean(axis=2), axis=1).var()

        assert variance(noisy) > 3 * variance(clean)

    def test_auxiliary_buffers_are_not_blank(self, views):
        """A silently unconnected socket would write zeros and quietly remove
        the network's most useful input."""
        passes = load_exr(views[0] / "noisy.exr")

        assert passes["albedo"].std() > 0.01
        assert passes["normal"].std() > 0.01

    def test_cache_round_trips(self, views, tmp_path):
        cache = build_cache(views[:3], tmp_path / "cache.npy")

        assert cache.shape == (3, 512, 512, CHANNELS)
        assert cache.dtype == np.float16
        assert (tmp_path / "cache.json").exists()

        original = load_exr(views[0] / "clean.exr")["beauty"]
        assert np.allclose(cache[0, ..., 9:], original, atol=0.05)
