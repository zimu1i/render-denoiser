"""
Score a trained denoiser against baselines on the held-out views.

    ./venv/bin/python evaluate.py --run runs/with_aux

Every method is scored on the same 45 views, which is the only thing that makes
the comparison meaningful. Measured on this dataset, PSNR varies by more than
12 dB between scenes depending on their brightness and content, so an absolute
score means little; a difference between two methods on identical scenes means
a lot.

The baselines are chosen to bracket the result:

**Do nothing.** The noisy input, untouched. Any method that fails to beat this
is actively harmful, and it is surprisingly easy to build one -- a denoiser that
blurs too eagerly scores below the input it was given.

**Gaussian blur.** The naive answer to "remove noise". It sets the bar that
separates a learned denoiser from ordinary smoothing. Beating it convincingly is
the minimum evidence that the network learned something about light transport
rather than just low-pass filtering.

**OpenImageDenoise**, when available. Intel's production denoiser, which Blender
ships and which is trained on far more data than this project uses. Matching it
is not the expectation; measuring the gap honestly is the point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from denoiser import CACHE_PATH, RAW_DIR, ROOT
from denoiser.data import find_views, load_cache, psnr, relmse, split_views, tonemap
from denoiser.model import UNet, from_log, select_device


def gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Separable Gaussian blur, the naive denoising baseline.

    Written out rather than pulled from scipy to keep the dependency list to
    numpy, torch and the EXR reader.
    """
    kernel = gaussian_kernel(sigma)
    radius = len(kernel) // 2
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")

    blurred = np.zeros_like(image)
    for offset, weight in enumerate(kernel):
        blurred += weight * padded[offset : offset + image.shape[0], radius:-radius or None]
    padded = np.pad(blurred, ((0, 0), (radius, radius), (0, 0)), mode="reflect")
    result = np.zeros_like(image)
    for offset, weight in enumerate(kernel):
        result += weight * padded[:, offset : offset + image.shape[1]]
    return result


def select_blur_sigma(cache, train_indices, candidates=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)):
    """Pick the blur strength on TRAINING views, never on the held-out ones.

    The blur baseline has one free parameter, and choosing it by looking at test
    scores would be fitting a baseline to the test set -- the same error that
    would invalidate the model's own numbers. Sigma is selected here on training
    data and then applied unchanged to the held-out views, which is the same
    discipline the network is held to.
    """
    best_sigma, best_psnr = candidates[0], -float("inf")
    for sigma in candidates:
        scores = []
        for index in train_indices:
            row = np.asarray(cache[index], np.float32)
            scores.append(psnr(gaussian_blur(row[..., :3], sigma), row[..., 9:]))
        mean = float(np.mean(scores))
        if mean > best_psnr:
            best_sigma, best_psnr = sigma, mean
    return best_sigma, best_psnr


def load_model(run_dir: Path, device):
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model = UNet(in_channels=checkpoint["in_channels"], base=checkpoint["base"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["in_channels"]


@torch.no_grad()
def predict(model, inputs: np.ndarray, in_channels: int, device) -> np.ndarray:
    """Run the network on one HxWx9 view, returning linear radiance."""
    tensor = torch.from_numpy(inputs.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    if in_channels == 3:
        tensor = tensor[:, :3]
    return from_log(model(tensor)).cpu().numpy()[0].transpose(1, 2, 0)


def score(prediction: np.ndarray, reference: np.ndarray) -> dict:
    return {"psnr": psnr(prediction, reference), "relmse": relmse(prediction, reference)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(ROOT / "runs" / "with_aux"))
    parser.add_argument("--sigma", type=float, default=None,
                        help="blur strength; selected on training views if omitted")
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = select_device()
    cache = load_cache(CACHE_PATH)
    views = find_views(RAW_DIR)
    position = {view: index for index, view in enumerate(views)}
    train_views, test_views = split_views(views, args.test_fraction, args.seed)

    sigma = args.sigma
    if sigma is None:
        train_indices = [position[v] for v in train_views]
        sigma, train_psnr = select_blur_sigma(cache, train_indices)
        print(f"blur baseline: sigma {sigma} selected on {len(train_indices)} "
              f"training views ({train_psnr:.2f} dB there)\n")

    run_dir = Path(args.run)
    model, in_channels = load_model(run_dir, device)

    results = {name: [] for name in ("noisy", "blur", "model", "oidn")}
    for view in test_views:
        row = np.asarray(cache[position[view]], np.float32)
        noisy, reference = row[..., :9], row[..., 9:]

        results["noisy"].append(score(noisy[..., :3], reference))
        results["blur"].append(score(gaussian_blur(noisy[..., :3], sigma), reference))
        results["model"].append(score(predict(model, noisy, in_channels, device), reference))

        # Produced separately by oidn_baseline.py, which runs Blender's denoiser
        # over the same saved buffers. Absent unless that has been run.
        oidn_path = Path(view) / "oidn.exr"
        if oidn_path.exists():
            from denoiser.data import load_exr

            results["oidn"].append(score(load_exr(oidn_path)["beauty"], reference))

    print(f"{len(test_views)} held-out views, none seen during training\n")
    header = f"{'method':<28}{'PSNR':>9}{'vs input':>11}{'relMSE':>11}"
    print(header)
    print("-" * len(header))

    baseline_psnr = float(np.mean([r["psnr"] for r in results["noisy"]]))
    labels = {
        "noisy": "noisy input (do nothing)",
        "blur": f"gaussian blur (sigma {sigma})",
        "model": "learned denoiser",
        "oidn": "OpenImageDenoise",
    }
    summary = {}
    for key, label in labels.items():
        if not results[key]:
            continue
        mean_psnr = float(np.mean([r["psnr"] for r in results[key]]))
        mean_relmse = float(np.mean([r["relmse"] for r in results[key]]))
        summary[key] = {"psnr": mean_psnr, "relmse": mean_relmse}
        delta = "" if key == "noisy" else f"{mean_psnr - baseline_psnr:+.2f} dB"
        print(f"{label:<28}{mean_psnr:>8.2f} dB{delta:>11}{mean_relmse:>11.4f}")

    if "model" in summary:
        gain = summary["model"]["psnr"] - baseline_psnr
        # Monte Carlo error falls as 1/N, so 10*log10(N) dB per N-fold increase
        # in samples. Inverting that turns a PSNR gain into the sample count it
        # substitutes for, which is the number a graphics person cares about.
        equivalent = 10 ** (gain / 10)
        print(f"\nthe model's {gain:+.2f} dB is worth roughly {equivalent:.1f}x the samples,")
        print(f"so a 4-sample render is made to look like a ~{4 * equivalent:.0f}-sample one.")

    (run_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten to {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
