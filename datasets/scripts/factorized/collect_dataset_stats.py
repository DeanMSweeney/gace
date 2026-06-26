"""
Analyze CatFlow training set LMDB files.
Shows data count and what fields/labels are in the data.
Writes the first 10 samples to catflow_analysis.txt.
"""

import sys
from pathlib import Path
import argparse

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
from gace.utils.lmdb_loader import LazyLmdbDataset


class ScalarStats:
    """Running mean/std/min/max accumulator for scalar values."""

    def __init__(self):
        self.sum    = 0.0
        self.sum_sq = 0.0
        self.min    = float("inf")
        self.max    = float("-inf")
        self.count  = 0

    def update(self, x: float):
        self.sum    += x
        self.sum_sq += x * x
        self.min     = min(self.min, x)
        self.max     = max(self.max, x)
        self.count  += 1

    def summary(self, label: str) -> str:
        mean = self.sum / max(self.count, 1)
        var  = max(0.0, self.sum_sq / max(self.count, 1) - mean ** 2)
        std  = var ** 0.5
        return "\n".join([
            "=" * 70,
            f"{label} statistics (across full dataset)",
            "=" * 70,
            f"Total samples: {self.count}",
            f"  mean={mean:>14.8f}  std={std:>14.8f}  "
            f"min={self.min:>12.6f}  max={self.max:>12.6f}",
        ])


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

    binding_energy_stats = ScalarStats()
    ads_pos_stats = AxisStats()
    pos_stats = AxisStats()
    lengths_stats = AxisStats()
    angles_stats = AxisStats()
    total = len(dataset)

    prim_counts: dict[int, int] = {}
    ads_counts:  dict[int, int] = {}
    prim_element_counts: dict[int, int] = {}
    max_prim, max_prim_idx = 0, -1
    max_ads,  max_ads_idx  = 0, -1
    min_prim, min_prim_idx = float("inf"), -1
    min_ads,  min_ads_idx  = float("inf"), -1

    with open(out_path, "w") as f:
        f.write(header + "\n\n")

        for i in tqdm(range(total), desc="Scanning samples"):
            sample = dataset[i]

            binding_energy_stats.update(float(sample.ads_binding_energy))
            ads_pos_stats.update(sample.ads_pos)
            pos_stats.update(sample.pos)
            lengths_stats.update(sample.lengths_scaled)
            angles_stats.update(sample.angles_radians)

            n_prim = int(sample.prim_natoms)
            n_ads  = int(sample.ads_atomic_numbers.shape[0])
            prim_counts[n_prim] = prim_counts.get(n_prim, 0) + 1
            ads_counts[n_ads]   = ads_counts.get(n_ads,  0) + 1
            for z in set(int(z) for z in sample.atomic_numbers.tolist()):
                prim_element_counts[z] = prim_element_counts.get(z, 0) + 1
            if n_prim > max_prim:
                max_prim, max_prim_idx = n_prim, i
            if n_prim < min_prim:
                min_prim, min_prim_idx = n_prim, i
            if n_ads > max_ads:
                max_ads, max_ads_idx = n_ads, i
            if n_ads < min_ads:
                min_ads, min_ads_idx = n_ads, i

            if i < n_samples:
                block = format_sample(i, sample)
                print(block)
                f.write(block + "\n\n")

        def _count_table(counts: dict[int, int], label: str, max_val: int, max_idx: int, min_val: int, min_idx: int) -> str:
            lines = [
                "=" * 70,
                label,
                "=" * 70,
                f"  max = {max_val}  (sample idx {max_idx})",
                f"  min = {min_val}  (sample idx {min_idx})",
                "  Distribution:",
            ]
            for val in sorted(counts):
                lines.append(f"    {val:>4} atoms: {counts[val]:>6} samples")
            return "\n".join(lines)

        atom_count_summary = "\n\n".join([
            _count_table(prim_counts, "Primitive-cell atom counts (prim_natoms)", max_prim, max_prim_idx, min_prim, min_prim_idx),
            _count_table(ads_counts,  "Adsorbate atom counts (n_ads)",            max_ads,  max_ads_idx,  min_ads,  min_ads_idx),
        ])

        _SYMBOLS = [
            "", "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
            "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
            "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
            "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
            "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
            "Fr","Ra","Ac","Th","Pa","U","Np","Pu",
        ]

        def _element_table(counts: dict[int, int]) -> str:
            lines = [
                "=" * 70,
                "Primitive-cell element presence (graphs containing element)",
                "=" * 70,
                f"  Unique elements: {len(counts)}",
                "  Count by element (sorted by count desc):",
            ]
            for z, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
                sym = _SYMBOLS[z] if z < len(_SYMBOLS) else "?"
                lines.append(f"    Z={z:>3}  {sym:<4}  {cnt:>10} graphs")
            return "\n".join(lines)

        footer = "\n\n".join([
            atom_count_summary,
            _element_table(prim_element_counts),
            binding_energy_stats.summary("Adsorption binding energy (ads_binding_energy)"),
            ads_pos_stats.summary("Adsorbate Cartesian positions (ads_pos)", entry_label="adsorbate atoms"),
            pos_stats.summary("Primitive-cell positions (pos)", entry_label="atoms"),
            lengths_stats.summary("Lattice lengths_scaled", axis_names=("a", "b", "c"), entry_label="samples"),
            angles_stats.summary("Lattice angles_radians", axis_names=("α", "β", "γ"), entry_label="samples"),
        ])
        print("\n" + footer)
        f.write("\n" + footer + "\n")

    print(f"\nWrote {min(n_samples, total)} samples + dataset stats across {total} graphs → {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Collect dataset statistics including atom count, atom types, ...etc."
    )
    parser.add_argument("--src",     required=True,                   help="Input LMDB file path")
    parser.add_argument("--dst",     default="dataset_stats.txt",                    help="Output LMDB file path")
    args = parser.parse_args()

    analyze_lmdb(args.src, n_samples=10, out_file=args.dst)


if __name__ == "__main__":
    main()