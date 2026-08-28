"""
A small U-Net for denoising path-traced renders.

Two things about this network are specific to the problem rather than generic
image-to-image plumbing.

**It works in log space.** Radiance in these renders spans 0 to about 45, and a
handful of pixels carry values hundreds of times the image mean. Feeding that in
raw lets a few bright pixels dominate every gradient. `log(1 + x)` compresses the
range while leaving zero at zero and staying monotonic, so the network sees a
well-conditioned signal and the ordering of brightnesses is preserved.

**Only the radiance channels are compressed.** Albedo already lies in [0, 1] and
normals in [-1, 1]; both are well scaled, and taking a log of a normal would
destroy the sign that makes it meaningful.

The channel count is a constructor argument because the central experiment is an
ablation: 9 channels with the auxiliary buffers, 3 without. Everything else about
the two runs stays identical, so the difference in score is attributable to the
buffers rather than to a change in capacity.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Channel layout produced by data.stack_inputs.
RADIANCE_CHANNELS = 3  # the noisy beauty pass; albedo and normal follow
FULL_INPUT_CHANNELS = 9
OUTPUT_CHANNELS = 3


def to_log(x: torch.Tensor) -> torch.Tensor:
    """Compress radiance for the network. Monotonic, and log(0) stays 0."""
    return torch.log1p(torch.clamp(x, min=0.0))


def from_log(x: torch.Tensor) -> torch.Tensor:
    """Undo to_log, returning linear radiance."""
    return torch.expm1(torch.clamp(x, min=0.0))


def prepare_inputs(x: torch.Tensor) -> torch.Tensor:
    """Log-compress the radiance channels, leave the buffers untouched.

    Expects NCHW with the layout from data.stack_inputs: beauty, albedo, normal.
    """
    radiance = to_log(x[:, :RADIANCE_CHANNELS])
    if x.shape[1] <= RADIANCE_CHANNELS:
        return radiance
    return torch.cat([radiance, x[:, RADIANCE_CHANNELS:]], dim=1)


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, the standard U-Net unit."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Encoder/decoder with skip connections.

    Skips are what make this architecture right for denoising specifically. The
    deep layers see enough context to tell noise from texture, while the skip
    connections carry high-resolution detail straight across, so sharp edges
    survive rather than being reconstructed from a bottleneck.
    """

    def __init__(self, in_channels: int = FULL_INPUT_CHANNELS, base: int = 32,
                 input_mean=None, input_std=None):
        super().__init__()
        self.in_channels = in_channels

        # Per-channel standardisation of the network's input.
        #
        # Measured on this dataset, the albedo and normal buffers carry 1.8x and
        # 2.2x the variance of the log-radiance channels. Whichever inputs have
        # the largest variance dominate the first layer's activations, so
        # unnormalised buffers drown out the signal the network is meant to be
        # correcting. Defaults are a no-op so checkpoints saved before this
        # existed still load.
        mean = torch.zeros(in_channels) if input_mean is None else torch.as_tensor(input_mean)
        std = torch.ones(in_channels) if input_std is None else torch.as_tensor(input_std)
        self.register_buffer("input_mean", mean.view(1, -1, 1, 1).float())
        self.register_buffer("input_std", std.view(1, -1, 1, 1).float().clamp(min=1e-6))

        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.bottleneck = ConvBlock(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.head = nn.Conv2d(base, OUTPUT_CHANNELS, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Takes linear radiance, returns a prediction in log space.

        The output stays in log space because that is where the loss is
        computed; callers convert with from_log to get an image.
        """
        # Named separately from the argument: `prepared` is already in log
        # space, and the residual below must add that, not re-compress it.
        prepared = prepare_inputs(x)
        # Standardise for the encoder only. The residual below is added back in
        # unnormalised log space, which is where the loss lives.
        e1 = self.enc1((prepared - self.input_mean) / self.input_std)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # A residual on the noisy input. The identity is already a decent
        # answer -- the noisy render is unbiased, just high variance -- so the
        # network only has to learn the correction, which converges faster and
        # cannot lose the input entirely early in training.
        return self.head(d1) + prepared[:, :RADIANCE_CHANNELS]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def compute_input_stats(cache, indices, in_channels: int = FULL_INPUT_CHANNELS):
    """Per-channel mean and standard deviation of the prepared inputs.

    Computed on training views only. Deriving normalisation statistics from the
    held-out set would leak information about it into the model.
    """
    import numpy as np

    sample = np.asarray(cache[sorted(indices)[:40], ..., :in_channels], np.float32)
    prepared = prepare_inputs(torch.from_numpy(sample.transpose(0, 3, 1, 2)))
    return (
        prepared.mean(dim=(0, 2, 3)).tolist(),
        prepared.std(dim=(0, 2, 3)).tolist(),
    )


def select_device() -> torch.device:
    """Prefer Apple's GPU, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
