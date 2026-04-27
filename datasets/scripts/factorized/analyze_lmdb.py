"""
Analyze CatFlow training set LMDB files.
Shows data count and what fields/labels are in the data.
Writes the first 10 samples to catflow_analysis.txt.
"""

import sys
from pathlib import Path

# Numpy compatibility shim (data pickled with numpy>=1.25, redirects _core -> core)
import numpy, numpy.core
if not hasattr(numpy, '_core'):
    numpy._core = numpy.core
    import pkgutil
    for _, modname, _ in pkgutil.walk_packages(numpy.core.__path__, prefix='numpy.core.', onerror=lambda x: None):
        alias = modname.replace('numpy.core', 'numpy._core')
        if modname in sys.modules:
            sys.modules[alias] = sys.modules[modname]

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from tqdm import tqdm
from adsflow.utils.lmdb_loader import LazyLmdbDataset


class AxisStats:
    """Per-axis running mean/std/min/max accumulator for (..., 3) tensors."""

    def __init__(self):
        self.sum = torch.zeros(3, dtype=torch.float64)
        self.sum_sq = torch.zeros(3, dtype=torch.float64)
        self.min = torch.full((3,), float("inf"), dtype=torch.float64)
        self.max = torch.full((3,), float("-inf"), dtype=torch.float64)
        self.count = 0

    def update(self, x: torch.Tensor):
        x = x.to(torch.float64).reshape(-1, 3)
        self.sum += x.sum(dim=0)
        self.sum_sq += x.pow(2).sum(dim=0)
        self.min = torch.minimum(self.min, x.amin(dim=0))
        self.max = torch.maximum(self.max, x.amax(dim=0))
        self.count += x.shape[0]

    def summary(self, label: str, axis_names=("x", "y", "z"), entry_label="entries") -> str:
        mean = self.sum / max(self.count, 1)
        var = (self.sum_sq / max(self.count, 1)) - mean.pow(2)
        std = var.clamp_min(0).sqrt()
        lines = [
            "=" * 70,
            f"{label} statistics (across full dataset)",
            "=" * 70,
            f"Total {entry_label}: {self.count}",
        ]
        for i, name in enumerate(axis_names):
            lines.append(
                f"  {name:>5s}: "
                f"mean={mean[i].item():>14.8f}  "
                f"std={std[i].item():>14.8f}  "
                f"min={self.min[i].item():>12.6f}  "
                f"max={self.max[i].item():>12.6f}"
            )
        return "\n".join(lines)


def format_sample(idx: int, sample) -> str:
    """Format a single PyG Data sample as a readable string."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"Sample {idx}")
    lines.append(f"{'=' * 70}")

    for key in sorted(sample.keys()):
        value = sample[key]
        if isinstance(value, torch.Tensor):
            if value.numel() <= 12:
                lines.append(f"  {key:25s}: Tensor{tuple(value.shape)}  {value.tolist()}")
            else:
                preview = value.flatten()[:5].tolist()
                lines.append(f"  {key:25s}: Tensor{tuple(value.shape)}  [{', '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in preview)}, ...]")
        else:
            lines.append(f"  {key:25s}: {value}")

    return "\n".join(lines)


def analyze_lmdb(dataset_path: str, n_samples: int = 10, out_file: str = "catflow_analysis.txt"):
    """
    Analyze LMDB dataset — print field schema and first n_samples to a txt file.

    Args:
        dataset_path: Path to the LMDB dataset file or directory.
        n_samples:    Number of samples to dump (default: 10).
        out_file:     Output text file path (default: catflow_analysis.txt).
    """
    config = {"src": dataset_path}
    dataset = LazyLmdbDataset(config)

    header = "\n".join([
        "=" * 70,
        "CatFlow LMDB Analysis",
        f"Dataset path : {dataset_path}",
        f"Total samples: {len(dataset)}",
        f"Dumping first: {min(n_samples, len(dataset))} samples",
        "=" * 70,
    ])

    print(header)

    out_path = Path(__file__).parent / out_file

    ads_pos_stats = AxisStats()
    pos_stats = AxisStats()
    lengths_stats = AxisStats()
    angles_stats = AxisStats()
    total = len(dataset)

    with open(out_path, "w") as f:
        f.write(header + "\n\n")

        for i in tqdm(range(total), desc="Scanning samples"):
            sample = dataset[i]

            ads_pos_stats.update(sample.ads_pos)
            pos_stats.update(sample.pos)
            lengths_stats.update(sample.lengths_scaled)
            angles_stats.update(sample.angles_radians)

            if i < n_samples:
                block = format_sample(i, sample)
                print(block)
                f.write(block + "\n\n")

        footer = "\n\n".join([
            ads_pos_stats.summary("Adsorbate Cartesian positions (ads_pos)", entry_label="adsorbate atoms"),
            pos_stats.summary("Primitive-cell positions (pos)", entry_label="atoms"),
            lengths_stats.summary("Lattice lengths_scaled", axis_names=("a", "b", "c"), entry_label="samples"),
            angles_stats.summary("Lattice angles_radians", axis_names=("α", "β", "γ"), entry_label="samples"),
        ])
        print("\n" + footer)
        f.write("\n" + footer + "\n")

    print(f"\nWrote {min(n_samples, total)} samples + dataset stats across {total} graphs → {out_path}")


if __name__ == "__main__":
    train_path = "/workspace/datasets/catflow/train/dataset_processed.lmdb"
    analyze_lmdb(train_path, n_samples=10, out_file="catflow_analysis.txt")