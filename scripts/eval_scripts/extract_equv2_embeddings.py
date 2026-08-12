#!/usr/bin/env python3
"""Extract rotation-invariant EquiformerV2 embeddings for a directory of structures.

Step 1 of the structural-novelty analysis. Every .traj is pushed through the
pretrained EquiformerV2-31M-S2EF-OC20-All+MD model; a forward hook on the final
layer norm (immediately before the energy/force heads) captures the per-atom
spherical-harmonic features. Only the l=0 channels are kept — those are the
rotation-invariant ones, and flattening the full irreps tensor instead would make
every downstream distance depend on how the structure happens to be oriented.
The per-atom l=0 vectors are pooled (mean and max, both saved) into one vector
per structure.

--pool_atoms chooses which atoms enter that pooling. The default, ``all``, uses
the whole system. ``slab`` pools only ``tags != 2``, which drops the adsorbate's
own features so that binding site, height and orientation stop moving the
embedding. The adsorbate stays in the forward pass in both modes: this checkpoint
was trained on adsorbate+slab systems only, so feeding it a bare slab would put
the surface atoms in a neighborhood the model never saw. Embeddings from the two
modes are not comparable, so the mode is recorded in the .npz.

Give it one or more directories of .traj files; everything found is embedded
into a single embeddings.npz (plus a matching .csv) in the current directory.

Usage:
    python extract_equv2_embeddings.py \
        --traj_dir for_uma/1_threshold for_uma/2_threshold for_uma/3_threshold \
        --adsorbate NO2 [--pool_atoms slab]

    # inspect the module tree to confirm the extraction point, then exit
    python extract_equv2_embeddings.py --print_model
"""

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
import ase.io
from fairchem.core import OCPCalculator
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.preprocessing import AtomsToGraphs
from torch_geometric.data import Batch as PygBatch
from tqdm import tqdm
import time
from multiprocessing import Pool


# gace.models must NOT be imported: equiformerv2_{oc20,flow,forge}.py each call
# @registry.register_model("equiformer_v2"), which would overwrite fairchem's own
# entry and make OCPCalculator build the wrong class from the checkpoint config.
from gace.utils.relaxation import Silent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Above this the l=0 slice is wrong and every downstream distance is meaningless.
ROTATION_TOL = 1e-3

def _load_full_atoms(path: str) -> tuple[str, "Atoms | None"]:
    """Worker: load full atoms (with adsorbate) and return (path, atoms)."""
    try:
        return path, ase.io.read(path)
    except Exception:
        return path, None

def load_atoms_parallel(files: list[str], num_workers: int) -> dict[str, "Atoms | None"]:
    """Load all structure files in parallel. Returns path → Atoms mapping."""
    log.info(f"Loading {len(files)} structures ({num_workers} workers)...")
    t0 = time.time()
    with Pool(processes=num_workers) as pool:
        pairs = list(tqdm(pool.imap(_load_full_atoms, files),
                          total=len(files), desc="Loading structures"))
    log.info(f"Loaded in {time.time() - t0:.1f}s")
    return dict(pairs)


def load_model(model_name: str, local_cache: str, device: str, seed: int):
    """Return the bare EquiformerV2 nn.Module from the pretrained checkpoint."""
    checkpoint_path = model_name_to_local_file(model_name, local_cache=local_cache)
    logging.info(f"Checkpoint: {checkpoint_path}")

    with Silent():
        calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=(device == "cpu"),
            seed=seed,
        )

    # calculator.trainer.model may be wrapped in OCPDataParallel / DDP
    model = calculator.trainer.model
    while hasattr(model, "module"):
        model = model.module

    return model.to(device).eval()


def invariant_row_offsets(model) -> list[int]:
    """Row indices of the l=0 coefficients, one per spherical-harmonic resolution.

    The coefficient axis stacks each resolution's (l_i + 1)**2 rows back to back, and
    the l=0 row of a resolution sits at that block's starting offset.
    """
    lmax_list = getattr(model, "lmax_list", None)
    if not lmax_list:
        logging.warning("Model has no lmax_list; assuming a single resolution (l=0 at row 0)")
        return [0]

    offsets, offset = [], 0
    for lmax in lmax_list:
        offsets.append(offset)
        offset += (int(lmax) + 1) ** 2
    return offsets


class InvariantEmbedder:
    """Hooks a submodule and pools its l=0 channels into one vector per structure."""

    def __init__(self, model, layer: str, device: str):
        self.model = model
        self.device = device
        self.offsets = invariant_row_offsets(model=model)
        self._captured = None

        submodule = model.get_submodule(layer)
        logging.info(f"Hooking '{layer}' -> {type(submodule).__name__}")
        submodule.register_forward_hook(self._capture)

        # r_edges=False: the model builds its own graph with its own max_radius /
        # max_neighbors, matching how relaxation.batch_relax_w_mlff feeds it.
        self.a2g = AtomsToGraphs(
            r_energy=False, r_forces=False, r_distances=False, r_edges=False, r_fixed=True
        )

    def _capture(self, module, inputs, output):
        self._captured = getattr(output, "embedding", output)

    def _forward(self, atoms_list: list[Atoms]):
        """Run one batch; return per-atom l=0 features, atom -> structure index, and tags."""
        batch = PygBatch.from_data_list(
            [self.a2g.convert(atoms, sid=i) for i, atoms in enumerate(atoms_list)]
        )
        if getattr(batch, "pbc", None) is None:
            batch.pbc = torch.tensor([[True, True, True]] * len(atoms_list))
        batch = batch.to(self.device)

        self._captured = None
        with torch.no_grad():
            self.model(batch)
        if self._captured is None:
            raise RuntimeError("Forward hook did not fire — wrong --layer for this model?")

        # [n_atoms, n_coefficients, channels] -> [n_atoms, n_res * channels]
        features = self._captured
        return (features[:, self.offsets, :].reshape(features.shape[0], -1),
                batch.batch, batch.tags)

    def embed(self, atoms_list: list[Atoms], batch_size: int, quiet: bool = False,
              slab_only: bool = False):
        """Return (mean-pooled, max-pooled) embeddings, one row per input structure.

        ``slab_only`` restricts the pooling to ``tags != 2``. The adsorbate is still
        part of the forward pass — dropping it from the input would hand the model a
        bare slab, which is not what the S2EF-OC20 checkpoint was trained on — it just
        does not enter the pooled vector, so binding pose stops moving the embedding.
        """
        pooled_mean, pooled_max = [], []

        starts = range(0, len(atoms_list), batch_size)
        for start in tqdm(starts, desc="Embedding", disable=quiet):
            chunk = atoms_list[start:start + batch_size]
            per_atom, atom_to_struct, tags = self._forward(atoms_list=chunk)

            for i in range(len(chunk)):
                selected = atom_to_struct == i
                if slab_only:
                    selected = selected & (tags != 2)
                vectors = per_atom[selected]
                pooled_mean.append(vectors.mean(dim=0).detach().cpu().numpy())
                pooled_max.append(vectors.amax(dim=0).detach().cpu().numpy())

        return (
            np.asarray(pooled_mean, dtype=np.float32),
            np.asarray(pooled_max, dtype=np.float32),
        )


def check_rotation_invariance(embedder: InvariantEmbedder, atoms_list: list[Atoms],
                              n_structures: int, n_rotations: int, seed: int,
                              slab_only: bool = False) -> None:
    """Re-embed a few structures under random SO(3) rotations and log the deviation."""
    rng = np.random.default_rng(seed)
    sample = atoms_list[:n_structures]
    if not sample:
        return

    reference, _ = embedder.embed(atoms_list=sample, batch_size=len(sample), quiet=True,
                                  slab_only=slab_only)
    worst = 0.0

    for _ in range(n_rotations):
        rotated = []
        for atoms in sample:
            copy = atoms.copy()
            axis = rng.normal(size=3)
            # rotate_cell=True: rotating positions alone would make it a different
            # periodic system, and the check would be measuring nothing.
            copy.rotate(rng.uniform(0.0, 360.0), axis / np.linalg.norm(axis), rotate_cell=True)
            rotated.append(copy)

        candidate, _ = embedder.embed(atoms_list=rotated, batch_size=len(rotated), quiet=True,
                                      slab_only=slab_only)
        worst = max(worst, float(np.abs(candidate - reference).max()))

    logging.info(f"Rotation invariance: max |delta| = {worst:.3e} "
                 f"over {n_rotations} rotations of {len(sample)} structures "
                 f"(mean |embedding| = {float(np.abs(reference).mean()):.3e})")


def require_adsorbate_tags(loaded: list[tuple[str, Atoms]]) -> None:
    """Abort unless every structure has tagged adsorbate atoms.

    A .traj that lost its tags reads back as all-zero, and ``tags != 2`` then selects
    every atom — slab pooling would quietly return full-system embeddings instead of
    failing, and nothing downstream could tell the difference.
    """
    untagged = [path for path, atoms in loaded if not (atoms.get_tags() == 2).any()]
    if untagged:
        listed = "\n  ".join(untagged[:10])
        more = f"\n  ... and {len(untagged) - 10} more" if len(untagged) > 10 else ""
        raise SystemExit(
            f"--pool_atoms slab needs tags==2 to mark the adsorbate, but "
            f"{len(untagged)} / {len(loaded)} structures have none:\n  {listed}{more}"
        )


def gather_structures(directories: list[Path], workers: int) -> list[tuple[str, Atoms]]:
    """Load the .traj files across all given directories. Returns (path, atoms) pairs."""
    files = []
    for directory in directories:
        found = sorted(str(p) for p in directory.glob("*.traj"))
        logging.info(f"Found {len(found)} .traj files in {directory}")
        files.extend(found)

    if not files:
        raise SystemExit(f"No .traj files found in {', '.join(str(d) for d in directories)}")
    atoms_by_path = load_atoms_parallel(files=files, num_workers=workers)
    loaded = [(path, atoms) for path, atoms in atoms_by_path.items() if atoms is not None]

    n_unreadable = len(atoms_by_path) - len(loaded)
    logging.info(f"{len(loaded)} structures loaded ({n_unreadable} unreadable)")
    return loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traj_dir", type=Path, nargs="+", default=None,
                        help="One or more directories of .traj files; all are embedded together")
    parser.add_argument("--adsorbate", type=str, default="",
                        help="Adsorbate name for this run, e.g. NO2; stored in the output")
    parser.add_argument("--mlff_checkpoint", type=str, default="EquiformerV2-31M-S2EF-OC20-All+MD",
                        help="fairchem model name (default matches relax_by_ads.py)")
    parser.add_argument("--local_cache", type=str, default="./mlff_checkpoint_cache",
                        help="Checkpoint download cache")
    parser.add_argument("--layer", type=str, default="norm",
                        help="Submodule to hook: the final layer norm before the output heads")
    parser.add_argument("--print_model", action="store_true",
                        help="Print the module tree and exit, to confirm the extraction point")
    parser.add_argument("--pool_atoms", choices=["all", "slab"], default="slab",
                        help="Pool over every atom (default) or only the slab (tags != 2). "
                             "The adsorbate is in the forward pass either way; 'slab' only "
                             "keeps it out of the pooled vector. Not comparable across modes.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--check_rotations", type=int, default=3,
                        help="Random rotations for the invariance check (0 to skip)")
    parser.add_argument("--check_structures", type=int, default=2,
                        help="Structures used in the invariance check")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help="Workers for parallel structure loading")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = load_model(
        model_name=args.mlff_checkpoint,
        local_cache=args.local_cache,
        device=args.device,
        seed=args.seed,
    )

    if args.print_model:
        print(model)
        return

    # Not argparse-required so --print_model works on its own
    if args.traj_dir is None:
        raise SystemExit("Nothing to do: pass --traj_dir.")

    loaded = gather_structures(directories=args.traj_dir, workers=args.num_workers)
    if not loaded:
        raise SystemExit("No structures survived loading.")

    slab_only = args.pool_atoms == "slab"
    if slab_only:
        require_adsorbate_tags(loaded=loaded)

    paths = [path for path, _ in loaded]
    atoms_list = [atoms for _, atoms in loaded]

    embedder = InvariantEmbedder(model=model, layer=args.layer, device=args.device)

    if args.check_rotations > 0:
        check_rotation_invariance(
            embedder=embedder,
            atoms_list=atoms_list,
            n_structures=args.check_structures,
            n_rotations=args.check_rotations,
            seed=args.seed,
            slab_only=slab_only,
        )

    emb_mean, emb_max = embedder.embed(
        atoms_list=atoms_list,
        batch_size=args.batch_size,
        slab_only=slab_only,
    )
    logging.info(f"Embedded {emb_mean.shape[0]} structures into {emb_mean.shape[1]} dimensions "
                 f"(pooling over {'slab atoms only' if slab_only else 'all atoms'})")

    natoms = [len(atoms) for atoms in atoms_list]

    if slab_only: 
        out_path = Path(f"slab_embeddings.npz")
    else: 
        out_path = Path(f"all_embeddings.npz")

    np.savez(
        out_path,
        emb_mean=emb_mean,
        emb_max=emb_max,
        path=np.array(paths),
        natoms=np.array(natoms, dtype=np.int64),
        adsorbate=np.array(args.adsorbate),
        traj_dir=np.array([str(d) for d in args.traj_dir]),
        pool_atoms=np.array(args.pool_atoms),
    )

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "natoms"])
        writer.writerows(zip(paths, natoms))

    logging.info(f"Wrote {out_path} and {csv_path}")


if __name__ == "__main__":
    main()
