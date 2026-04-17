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
from adsflow.utils.lmdb_loader import LazyLmdbDataset


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
    with open(out_path, "w") as f:
        f.write(header + "\n\n")

        for i in range(min(n_samples, len(dataset))):
            block = format_sample(i, dataset[i])
            print(block)
            f.write(block + "\n\n")

    print(f"\nWrote {min(n_samples, len(dataset))} samples → {out_path}")


if __name__ == "__main__":
    train_path = "/workspace/datasets/catflow/dataset/train/dataset_processed.lmdb"
    analyze_lmdb(train_path, n_samples=10, out_file="catflow_analysis.txt")