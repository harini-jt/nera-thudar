"""
Step-by-step visualisation of the intermediate QUANT representations.

Run from the project root:
    python -m scripts.visualize_quant_flow --dataset Covid3MonthDiscrete
    python -m scripts.visualize_quant_flow --dataset Covid3MonthDiscrete --split test --sample 3
    python -m scripts.visualize_quant_flow --dataset Covid3MonthDiscrete --sample 0 --channel 0 --save figures/quant_flow.png

This visualises the exact operations in Quant._generate_representations:
    X
    X.diff()
    F.pad(first_diff, (2, 2), mode="replicate")
    F.avg_pool1d(..., kernel_size=5, stride=1)
    torch.diff(X, n=2)
    torch.fft.rfft(X).abs()
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data_loader import load_dataset


def _get_sample_tensor(X, sample_index: int, channel_index: int) -> torch.Tensor:
    """Return one sample as a float32 tensor shaped (1, 1, timepoints)."""
    X = np.asarray(X)

    if X.ndim == 2:
        # (cases, timepoints) -> (cases, 1, timepoints)
        X = X[:, np.newaxis, :]
    elif X.ndim != 3:
        raise ValueError(
            f"Expected X with 2 or 3 dimensions; received shape {X.shape}"
        )

    if sample_index < 0 or sample_index >= X.shape[0]:
        raise IndexError(
            f"Sample index {sample_index} is out of range "
            f"(valid: 0 to {X.shape[0] - 1})."
        )

    if channel_index < 0 or channel_index >= X.shape[1]:
        raise IndexError(
            f"Channel index {channel_index} is out of range "
            f"(valid: 0 to {X.shape[1] - 1})."
        )

    # Keep the batch and channel dimensions, matching Quant._to_tensor().
    sample = X[sample_index : sample_index + 1, channel_index : channel_index + 1, :]
    return torch.as_tensor(sample, dtype=torch.float32)


def _print_values(name: str, tensor: torch.Tensor, max_values: int = 100):
    """Print shape and values, truncating very long arrays for readability."""
    values = tensor.detach().cpu().numpy().reshape(-1)
    print(f"\n{name}")
    print(f"  shape: {tuple(tensor.shape)}")
    if len(values) > max_values:
        print(f"  values (first {max_values} of {len(values)}):")
        print(np.array2string(values[:max_values], precision=4, separator=", "))
        print("  ...")
    else:
        print("  values:")
        print(np.array2string(values, precision=4, separator=", "))


def visualize_quant_flow(
    X,
    label,
    dataset_name="Dataset",
    split_name="train",
    sample_index=0,
    channel_index=0,
    save_path=None,
):
    """
    Plot and print every intermediate array in QUANT's representation pipeline.

    This function does not fit or transform the data with Quant. It explicitly
    reproduces the operations in Quant._generate_representations so each
    intermediate result can be inspected.
    """
    sample = _get_sample_tensor(X, sample_index, channel_index)

    # These operations intentionally mirror src/quant.py exactly.
    first_diff = sample.diff()
    first_diff_padded = F.pad(first_diff, (2, 2), mode="replicate")
    first_diff_smoothed = F.avg_pool1d(
        first_diff_padded, kernel_size=5, stride=1
    )
    second_diff = torch.diff(sample, n=2)
    series_fft = torch.fft.rfft(sample).abs()

    stages = [
        ("X — original signal", sample, "Time point", "Value"),
        ("first_diff = X.diff()", first_diff, "Difference index", "Change"),
        (
            "first_diff_padded = F.pad(..., (2, 2), mode='replicate')",
            first_diff_padded,
            "Padded difference index",
            "Change",
        ),
        (
            "first_diff_smoothed = F.avg_pool1d(..., kernel_size=5)",
            first_diff_smoothed,
            "Smoothed difference index",
            "Average change",
        ),
        (
            "second_diff = torch.diff(X, n=2)",
            second_diff,
            "Second-difference index",
            "Change in change",
        ),
        (
            "series_fft = torch.fft.rfft(X).abs()",
            series_fft,
            "Frequency bin",
            "Magnitude",
        ),
    ]

    print("=" * 80)
    print(
        f"QUANT representation flow | {dataset_name} | {split_name} "
        f"| sample {sample_index} | channel {channel_index} | label {label}"
    )
    print("=" * 80)

    for title, tensor, _, _ in stages:
        _print_values(title, tensor)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.ravel()

    for ax, (title, tensor, xlabel, ylabel) in zip(axes, stages):
        values = tensor.detach().cpu().numpy().reshape(-1)
        x_values = np.arange(len(values))

        ax.plot(x_values, values, marker="o", markersize=3, linewidth=1.2)
        ax.set_title(f"{title}\nshape={tuple(tensor.shape)}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{dataset_name} — {split_name} sample {sample_index} "
        f"(class: {label})\nQUANT intermediate representations",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")

    plt.show()
    return {
        "X": sample,
        "first_diff": first_diff,
        "first_diff_padded": first_diff_padded,
        "first_diff_smoothed": first_diff_smoothed,
        "second_diff": second_diff,
        "series_fft": series_fft,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Visualise each intermediate step in QUANT's representation pipeline."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name/folder")
    parser.add_argument("--resample", type=int, default=0)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--sample", type=int, default=0, help="Sample index")
    parser.add_argument("--channel", type=int, default=0, help="Channel index")
    parser.add_argument("--save", help="Optional figure path")
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_dataset(
        args.dataset, args.resample
    )

    if args.split == "train":
        X, y = X_train, y_train
    else:
        X, y = X_test, y_test

    print(f"Dataset shape: {np.asarray(X).shape}")
    print(f"Number of samples: {len(X)}")
    print(f"Classes: {np.unique(y)}")

    visualize_quant_flow(
        X,
        label=y[args.sample],
        dataset_name=args.dataset,
        split_name=args.split,
        sample_index=args.sample,
        channel_index=args.channel,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
