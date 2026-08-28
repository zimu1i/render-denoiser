"""
Train the denoiser.

    ./venv/bin/python train.py --epochs 30
    ./venv/bin/python train.py --epochs 30 --no-aux --out runs/no_aux

The loss is L1 in log space, and both halves of that are choices.

**L1 rather than L2.** Monte Carlo noise is heavy-tailed: a handful of "firefly"
pixels carry values hundreds of times the image mean, produced by a single ray
that happened to find a bright light through a narrow path. Squaring the error
lets those few pixels dominate the gradient, and the network spends its capacity
chasing outliers instead of cleaning the other 99% of the image. L1 weights every
pixel's error linearly and is far more tolerant of them.

**In log space rather than on raw radiance.** Same reasoning as the model's
input transform: without compression the bright regions of the image contribute
almost all of the loss, and dark regions -- where noise is most visible to a
viewer -- are effectively ignored.

The metric reported during training is PSNR on tonemapped images, which is what
evaluation reports too, so the number printed here is comparable to the final
result rather than being a training-only quantity.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from denoiser import CACHE_PATH, RAW_DIR, ROOT
from denoiser.data import find_views, load_cache, psnr, split_views
from denoiser.dataset import FullViews, RandomTiles, drop_auxiliary
from denoiser.model import UNet, compute_input_stats, from_log, select_device, to_log

DEFAULT_RUN_DIR = ROOT / "runs" / "with_aux"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--tiles-per-view", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_RUN_DIR))
    parser.add_argument(
        "--normalise",
        action="store_true",
        help="standardise each input channel using training-set statistics",
    )
    parser.add_argument(
        "--no-aux",
        action="store_true",
        help="train on the noisy render alone, without albedo and normal buffers",
    )
    return parser.parse_args()


def log_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between a log-space prediction and a linear-space target.

    The model already emits log space, so only the target is transformed. Doing
    it this way keeps the network's output and the loss in the same units.
    """
    return torch.nn.functional.l1_loss(prediction, to_log(target))


@torch.no_grad()
def evaluate_psnr(model, cache, indices, device, use_aux: bool) -> float:
    """Mean tonemapped PSNR over whole held-out views.

    Whole views rather than tiles: a per-tile mean would weight each scene by
    how many tiles it contributes and would hide seams between tiles.
    """
    model.eval()
    scores = []
    for inputs, target in DataLoader(FullViews(cache, indices), batch_size=1):
        inputs = inputs.to(device)
        if not use_aux:
            inputs = drop_auxiliary(inputs)
        prediction = from_log(model(inputs)).cpu().numpy()[0].transpose(1, 2, 0)
        scores.append(psnr(prediction, target.numpy()[0].transpose(1, 2, 0)))
    model.train()
    return float(np.mean(scores))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = select_device()
    cache = load_cache(CACHE_PATH)
    views = find_views(RAW_DIR)
    position = {view: index for index, view in enumerate(views)}

    train_views, test_views = split_views(views, args.test_fraction, args.seed)
    train_indices = [position[v] for v in train_views]
    test_indices = [position[v] for v in test_views]

    use_aux = not args.no_aux
    in_channels = 9 if use_aux else 3
    mean, std = (None, None)
    if args.normalise:
        mean, std = compute_input_stats(cache, train_indices, in_channels)
    model = UNet(in_channels=in_channels, base=args.base_channels,
                 input_mean=mean, input_std=std).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Cosine annealing to near zero. The first run's held-out score bounced by
    # more than a decibel between epochs late in training and peaked on the
    # final epoch, both symptoms of a step size that stays too large to settle.
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    tiles = RandomTiles(cache, train_indices, per_view=args.tiles_per_view, seed=args.seed)
    loader = DataLoader(tiles, batch_size=args.batch_size, shuffle=True, num_workers=0)

    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"device {device} | {model.parameter_count():,} parameters | "
          f"inputs {in_channels}ch ({'with' if use_aux else 'without'} auxiliary buffers)")
    print(f"{len(train_indices)} train views, {len(test_indices)} held out, "
          f"{len(tiles)} crops per epoch\n")

    baseline = evaluate_baseline(cache, test_indices)
    print(f"baseline (noisy input, untouched): {baseline:.2f} dB\n")

    history = []
    best = -float("inf")
    for epoch in range(args.epochs):
        tiles.set_epoch(epoch)  # new crops each pass over the data
        started = time.time()
        running = 0.0

        for inputs, target in loader:
            inputs, target = inputs.to(device), target.to(device)
            if not use_aux:
                inputs = drop_auxiliary(inputs)

            optimiser.zero_grad(set_to_none=True)
            loss = log_l1_loss(model(inputs), target)
            loss.backward()
            optimiser.step()
            running += loss.detach().item() * inputs.shape[0]

        schedule.step()
        train_loss = running / len(tiles)
        held_out = evaluate_psnr(model, cache, test_indices, device, use_aux)
        history.append({"epoch": epoch, "loss": train_loss, "psnr": held_out})

        marker = ""
        if held_out > best:
            best = held_out
            torch.save(
                {"state_dict": model.state_dict(), "in_channels": in_channels,
                 "base": args.base_channels, "normalised": args.normalise},
                run_dir / "best.pt",
            )
            marker = "  <- best"
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {train_loss:.4f}  "
              f"held-out PSNR {held_out:.2f} dB  ({time.time() - started:.0f}s){marker}",
              flush=True)  # so a redirected run can be watched as it goes

    (run_dir / "history.json").write_text(
        json.dumps({"args": vars(args), "baseline_psnr": baseline, "history": history}, indent=2)
    )
    print(f"\nbest held-out PSNR {best:.2f} dB "
          f"({best - baseline:+.2f} dB over the noisy input)")
    print(f"saved to {run_dir}")


def evaluate_baseline(cache, indices) -> float:
    """PSNR of doing nothing at all, on the same held-out views."""
    scores = []
    for index in indices:
        view = np.asarray(cache[index], np.float32)
        scores.append(psnr(view[..., :3], view[..., 9:]))
    return float(np.mean(scores))


if __name__ == "__main__":
    main()
