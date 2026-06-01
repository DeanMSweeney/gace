"""Evaluate uniqueness and compositional diversity of generated structures.

Usage:
    python scripts/evaluate_gen.py \
        --gen_strucs ./gen_strucs \
        --num_workers 8
"""

import argparse
import glob
import logging
import os
import time
from multiprocessing import Pool

import numpy as np
from ase.io import read
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-worker loaders
# ---------------------------------------------------------------------------

def _load_slab_structure(path):
    """Load a .vasp file and return a pymatgen Structure of the slab only (tags != 2)."""
    try:
        atoms = read(path)
        slab = atoms[atoms.get_tags() != 2]
        if len(slab) == 0:
            return None
        return AseAtomsAdaptor().get_structure(slab)
    except Exception:
        return None


def _load_comp_fingerprint(path):
    """Load a .vasp file and return a Magpie composition fingerprint vector."""
    # Featurizer instantiated here to avoid multiprocessing pickling issues.
    from matminer.featurizers.composition import ElementProperty
    featurizer = ElementProperty.from_preset("magpie")
    try:
        atoms = read(path)
        struct = AseAtomsAdaptor().get_structure(atoms)
        return np.array(featurizer.featurize(struct.composition))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_uniqueness(vasp_files: list[str], num_workers: int) -> dict:
    log.info(f"Uniqueness: loading {len(vasp_files)} structures with {num_workers} workers...")
    t0 = time.time()
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(_load_slab_structure, vasp_files),
                            total=len(vasp_files), desc="Loading slabs"))
    structures = [s for s in results if s is not None]
    load_time = time.time() - t0
    log.info(f"Loaded {len(structures)}/{len(vasp_files)} structures ({load_time:.1f}s)")

    log.info("Running StructureMatcher (this is the slow step)...")
    t1 = time.time()
    groups = StructureMatcher().group_structures(structures)
    match_time = time.time() - t1

    return {
        "total": len(structures),
        "unique": len(groups),
        "uniqueness": len(groups) / len(structures) if structures else 0.0,
        "load_time": load_time,
        "match_time": match_time,
    }


def compute_diversity(vasp_files: list[str], num_workers: int) -> dict:
    log.info(f"Diversity: featurizing {len(vasp_files)} structures with {num_workers} workers...")
    t0 = time.time()
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(_load_comp_fingerprint, vasp_files),
                            total=len(vasp_files), desc="Featurizing"))
    fps = [r for r in results if r is not None]
    extract_time = time.time() - t0
    log.info(f"Featurized {len(fps)}/{len(vasp_files)} structures ({extract_time:.1f}s)")

    t1 = time.time()
    X = StandardScaler().fit_transform(np.array(fps))
    diversity = pdist(X, metric="euclidean").mean()
    calc_time = time.time() - t1

    return {
        "total": len(fps),
        "comp_diversity": diversity,
        "extract_time": extract_time,
        "calc_time": calc_time,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate uniqueness and compositional diversity of generated structures.")
    p.add_argument("--gen_strucs", required=True, help="Directory containing generated .vasp files")
    p.add_argument("--num_workers", type=int, default=8, help="Parallel workers for file loading")
    return p.parse_args()


def main():
    args = parse_args()

    vasp_files = sorted(glob.glob(os.path.join(args.gen_strucs, "*.vasp")))
    if not vasp_files:
        log.error(f"No .vasp files found in {args.gen_strucs}")
        return

    log.info(f"Found {len(vasp_files)} .vasp files in {args.gen_strucs}")

    u = compute_uniqueness(vasp_files, args.num_workers)
    d = compute_diversity(vasp_files, args.num_workers)

    print()
    print("=" * 42)
    print("Uniqueness")
    print(f"  Total loaded  : {u['total']}")
    print(f"  Unique groups : {u['unique']}")
    print(f"  Score         : {u['uniqueness']:.4f}")
    print(f"  Load time     : {u['load_time']:.1f}s")
    print(f"  Match time    : {u['match_time']:.1f}s")
    print("-" * 42)
    print("Compositional Diversity (Magpie)")
    print(f"  Valid samples : {d['total']}")
    print(f"  Score         : {d['comp_diversity']:.4f}")
    print(f"  Extract time  : {d['extract_time']:.1f}s")
    print(f"  Calc time     : {d['calc_time']:.1f}s")
    print("=" * 42)


if __name__ == "__main__":
    main()
