"""
Collect OC20 samples grouped by adsorbate id, writing each as a .traj file.

Every LMDB entry's ``sid`` is resolved to an adsorbate id through
oc20_mapping.pkl (same lookup as scripts/evaluation/list_trajs_by_ads_id.py),
and the structure is written to ``<out_dir>/ads_<id>/``. Pass --ads_ids to keep
only certain adsorbates; omit it to collect every species in the LMDB.

Adsorbate atoms are identified by ``tags == 2`` (slab/surface = other tags).

Each saved Atoms object carries its source LMDB index, sid, adsorbate id and
adsorbate formula in ``atoms.info`` so downstream code can trace it back.

The IS2RE energies (``y_relaxed``, and ``y_init`` when present) are stored in
``atoms.info``, and the one matching the written geometry is attached as a
``SinglePointCalculator`` result so ``atoms.get_potential_energy()`` works after
reading the .traj back.

--positions selects which geometry is written. ``relaxed`` (the default) uses
``pos_relaxed``, which is what you want when comparing against MLFF-relaxed
generated structures — mixing relaxed and unrelaxed geometries puts a
relaxation-state offset into any downstream structural comparison.

Work is split across processes; each worker opens its own LMDB handle and writes
its matched .traj files independently. The sid -> ads_id table is built once in
the parent and inherited by the workers, so the pickle is read from disk once.

Usage:
    python collect_cats_by_ads_id.py \
        --lmdb /path/to/data_refined.lmdb \
        --mapping /path/to/oc20_mapping.pkl \
        --out_dir ./ads_strucs \
        [--ads_ids 0 71 75] [--positions relaxed] [--limit N] [--workers W]
"""

import sys
import pickle
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

# Numpy compatibility shim (data pickled with numpy>=1.25)
import numpy, numpy.core  # noqa: E401
if not hasattr(numpy, "_core"):
    numpy._core = numpy.core
    import pkgutil
    for _, modname, _ in pkgutil.walk_packages(
        numpy.core.__path__, prefix="numpy.core.", onerror=lambda x: None
    ):
        alias = modname.replace("numpy.core", "numpy._core")
        if modname in sys.modules:
            sys.modules[alias] = sys.modules[modname]

import lmdb
import numpy as np
from ase import Atoms
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator
from tqdm import tqdm


def _to_float(x) -> float | None:
    """Convert a scalar tensor / array / number to a python float, or None."""
    try:
        return float(_to_np(x).reshape(-1)[0])
    except (AttributeError, TypeError, ValueError, IndexError):
        return None


def _to_np(x):
    """Convert tensor / numpy array / list to plain numpy float64 array."""
    try:
        return np.array(x.cpu().numpy(), dtype=np.float64)
    except AttributeError:
        return np.array(x, dtype=np.float64)


def _key_to_sid(key) -> int | None:
    """Invert the OC20 mapping key convention: 'random{sid}', or a raw int / str sid."""
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        digits = key[len("random"):] if key.startswith("random") else key
        if digits.isdigit():
            return int(digits)
    return None


def load_mapping_tables(mapping_path: Path, keep_ads_ids: set[int] | None):
    """Build sid -> ads_id and ads_id -> formula tables from oc20_mapping.pkl.

    Inverts the per-sid lookup of list_trajs_by_ads_id.py into a plain int -> int
    dict so the workers carry a compact table instead of the full mapping. When
    --ads_ids is given the table is pruned to those adsorbates only, which is a
    large memory saving over the ~1M-entry full mapping.
    """
    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)

    ads_id_by_sid: dict[int, int] = {}
    formula_by_ads_id: dict[int, str] = {}

    for key, value in mapping.items():
        sid = _key_to_sid(key)
        if sid is None:
            continue
        try:
            ads_id = int(value["ads_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if keep_ads_ids is not None and ads_id not in keep_ads_ids:
            continue
        ads_id_by_sid[sid] = ads_id
        # ads_symbols depends only on ads_id, so this stays tiny (~82 adsorbates)
        if ads_id not in formula_by_ads_id:
            formula_by_ads_id[ads_id] = str(value.get("ads_symbols", ""))

    return ads_id_by_sid, formula_by_ads_id


def build_atoms(data, idx: int, sid: int, ads_id: int, formula: str,
                positions: str) -> Atoms | None:
    """Return an ``ase.Atoms`` for this structure, or None if fields are missing."""
    try:
        tags    = _to_np(data.tags).astype(int).flatten()
        numbers = _to_np(data.atomic_numbers).astype(int).flatten()
        cell    = _to_np(data.cell).reshape(3, 3)
        pos     = _to_np(data.pos_relaxed if positions == "relaxed" else data.pos)
    except AttributeError:
        return None

    if not (tags == 2).any():
        return None

    # IS2RE entries carry y_relaxed (relaxed adsorption energy); y_init is only
    # present in some builds, so it stays optional.
    y_relaxed = _to_float(getattr(data, "y_relaxed", None))
    y_init = _to_float(getattr(data, "y_init", None))

    atoms = Atoms(numbers=numbers, positions=pos, cell=cell, pbc=True)
    atoms.set_tags(tags)
    atoms.info["lmdb_index"] = idx
    atoms.info["sid"] = sid
    atoms.info["ads_id"] = ads_id
    atoms.info["ads_symbols"] = formula
    atoms.info["positions"] = positions
    if y_relaxed is not None:
        atoms.info["y_relaxed"] = y_relaxed
    if y_init is not None:
        atoms.info["y_init"] = y_init

    # Attach the energy belonging to the geometry actually written, so
    # atoms.get_potential_energy() on the reloaded .traj is not mixing states.
    energy = y_relaxed if positions == "relaxed" else y_init
    if energy is not None:
        atoms.calc = SinglePointCalculator(atoms, energy=energy)
    return atoms


# Globals initialised once per worker process.
_ENV = None
_OUT_DIR = None
_ADS_ID_BY_SID = None
_FORMULA_BY_ADS_ID = None
_POSITIONS = None
_CREATED_DIRS: set[int] = set()


def _init_worker(lmdb_path: str, out_dir: str, ads_id_by_sid: dict, formula_by_ads_id: dict,
                 positions: str) -> None:
    global _ENV, _OUT_DIR, _ADS_ID_BY_SID, _FORMULA_BY_ADS_ID, _POSITIONS
    _OUT_DIR = Path(out_dir)
    _ADS_ID_BY_SID = ads_id_by_sid
    _FORMULA_BY_ADS_ID = formula_by_ads_id
    _POSITIONS = positions
    _ENV = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )


def _process_chunk(index_range: tuple[int, int]) -> dict:
    """Process [start, stop) of LMDB indices, writing matched .traj files."""
    start, stop = index_range
    counts: dict = {"_missing": 0, "_no_sid": 0, "_unmapped": 0, "_unbuildable": 0,
                    "_no_energy": 0, "by_ads": {}}

    with _ENV.begin() as txn:
        for idx in range(start, stop):
            raw = txn.get(str(idx).encode("ascii"))
            if raw is None:
                counts["_missing"] += 1
                continue

            data = pickle.loads(raw)
            try:
                sid = int(data.sid)
            except (AttributeError, TypeError, ValueError):
                counts["_no_sid"] += 1
                continue

            ads_id = _ADS_ID_BY_SID.get(sid)
            if ads_id is None:
                # Either not in the mapping, or filtered out by --ads_ids
                counts["_unmapped"] += 1
                continue

            atoms = build_atoms(
                data=data,
                idx=idx,
                sid=sid,
                ads_id=ads_id,
                formula=_FORMULA_BY_ADS_ID.get(ads_id, ""),
                positions=_POSITIONS,
            )
            if atoms is None:
                counts["_unbuildable"] += 1
                continue
            if atoms.calc is None:
                counts["_no_energy"] += 1

            ads_dir = _OUT_DIR / f"ads_{ads_id}"
            if ads_id not in _CREATED_DIRS:
                ads_dir.mkdir(parents=True, exist_ok=True)
                _CREATED_DIRS.add(ads_id)

            write(str(ads_dir / f"{idx:08d}_sid{sid}.traj"), atoms)
            counts["by_ads"][ads_id] = counts["by_ads"].get(ads_id, 0) + 1

    return counts


def run(lmdb_path: str, mapping_path: Path, out_dir: Path, keep_ads_ids: set[int] | None,
        positions: str, limit: int | None, workers: int) -> None:
    print(f"Loading adsorbate mapping from {mapping_path} ...")
    ads_id_by_sid, formula_by_ads_id = load_mapping_tables(
        mapping_path=mapping_path,
        keep_ads_ids=keep_ads_ids,
    )
    if not ads_id_by_sid:
        raise SystemExit("No sids resolved from the mapping — check --mapping and --ads_ids.")
    print(f"  {len(ads_id_by_sid)} sids across {len(formula_by_ads_id)} adsorbate ids")

    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    with env.begin() as txn:
        length_entry = txn.get(b"length")
        n_total = pickle.loads(length_entry) if length_entry is not None else env.stat()["entries"]
    env.close()

    n_scan = n_total if limit is None else min(limit, n_total)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split the index range into one chunk per worker (a few chunks per worker
    # keeps the tqdm bar responsive and load roughly balanced).
    n_chunks = max(workers * 4, 1)
    bounds = np.linspace(0, n_scan, n_chunks + 1, dtype=int)
    chunks = [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_chunks) if bounds[i + 1] > bounds[i]]

    scope = "all adsorbates" if keep_ads_ids is None else f"ads_ids {sorted(keep_ads_ids)}"
    print(f"Scanning {n_scan} / {n_total} entries from {lmdb_path}  "
          f"({scope}, {positions} positions, {workers} workers)")

    totals: dict[int, int] = {}
    n_missing = n_no_sid = n_unmapped = n_unbuildable = n_no_energy = 0
    with Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(lmdb_path, str(out_dir), ads_id_by_sid, formula_by_ads_id, positions),
    ) as pool:
        for counts in tqdm(
            pool.imap_unordered(_process_chunk, chunks),
            total=len(chunks),
            desc="Chunks",
            ncols=100,
        ):
            n_missing += counts["_missing"]
            n_no_sid += counts["_no_sid"]
            n_unmapped += counts["_unmapped"]
            n_unbuildable += counts["_unbuildable"]
            n_no_energy += counts["_no_energy"]
            for ads_id, c in counts["by_ads"].items():
                totals[ads_id] = totals.get(ads_id, 0) + c

    n_matched = sum(totals.values())
    print(f"\nMatched structures : {n_matched}")
    print(f"  missing entries       : {n_missing}")
    print(f"  entries without sid   : {n_no_sid}")
    print(f"  sid not kept / mapped : {n_unmapped}")
    print(f"  could not build atoms : {n_unbuildable}"
          + ("   (missing pos_relaxed?)" if positions == "relaxed" and n_unbuildable else ""))
    print(f"  written without energy: {n_no_energy}"
          + (f"   (no y_{'relaxed' if positions == 'relaxed' else 'init'} field)" if n_no_energy else ""))
    for ads_id in sorted(totals):
        print(f"  ads_{ads_id:<4} {formula_by_ads_id.get(ads_id, ''):<12} : {totals[ads_id]}")
    print(f"\nSaved {n_matched} .traj files → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect OC20 samples grouped by adsorbate id, one .traj per structure."
    )
    parser.add_argument(
        "--lmdb",
        default="/datasets/oc20/full/is2res_train_val_test_lmdbs/data/is2re/all/train/data.lmdb",
        help="Path to OC20 LMDB file",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        required=True,
        help="Path to oc20_mapping.pkl (sid -> adsorbate metadata)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).parent,
        help="Root output directory; structures go in <out_dir>/ads_<id>/",
    )
    parser.add_argument(
        "--ads_ids",
        type=int,
        nargs="+",
        default=None,
        help="Adsorbate ids to keep, e.g. 0 71 75 (default: every species in the LMDB)",
    )
    parser.add_argument(
        "--positions",
        choices=["relaxed", "initial"],
        default="relaxed",
        help="Write pos_relaxed or pos (default: relaxed)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of structures to scan (default: all)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count(),
        help="Number of worker processes (default: all CPUs)",
    )
    args = parser.parse_args()

    run(
        lmdb_path=args.lmdb,
        mapping_path=args.mapping,
        out_dir=args.out_dir,
        keep_ads_ids=set(args.ads_ids) if args.ads_ids is not None else None,
        positions=args.positions,
        limit=args.limit,
        workers=max(args.num_workers, 1),
    )


if __name__ == "__main__":
    main()
