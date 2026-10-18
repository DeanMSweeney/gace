"""Set up and submit OC20-style VASP slab + adslab relaxations for each .traj file.

Run from a directory containing INCAR and VASP.sh:
    python optimize_structures.py --traj_dir path/to/trajs --out_dir path/to/runs
"""
import argparse
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read, write
from tqdm import tqdm

POTCAR_SCRIPT = "/home/dsweeney3/scripts/potcar.sh"


def oc20_kpoints(atoms):
    # Matches fairchem calculate_surface_k_points (infinity norm, multiplier 40)
    cell = atoms.get_cell()
    kx = max(1, int(round(40 / np.linalg.norm(cell[0], ord=np.inf))))
    ky = max(1, int(round(40 / np.linalg.norm(cell[1], ord=np.inf))))
    return kx, ky


def prep_atoms(atoms):
    # Enforce right-handed cell, as fairchem does before writing inputs
    if np.dot(np.cross(atoms.cell[0], atoms.cell[1]), atoms.cell[2]) < 0:
        atoms.set_cell(atoms.cell[[1, 0, 2], :])
    # Fix subsurface atoms (tag 0) if no constraints came with the traj
    if not atoms.constraints:
        atoms.set_constraint(FixAtoms(mask=atoms.get_tags() == 0))
    return atoms


def sbatch_with_retry(run_dir, job_name, retries=5, wait=10):
    for _ in range(retries):
        result = subprocess.run(["sbatch", f"--job-name={job_name}", "VASP.sh"], cwd=run_dir,
                                stdout=subprocess.DEVNULL)
        if result.returncode == 0:
            (run_dir / ".submitted").touch()
            return
        time.sleep(wait)
    raise RuntimeError(f"sbatch failed {retries} times for {run_dir}")


def queued_dirs():
    out = subprocess.run(["squeue", "--me", "-h", "-o", "%Z"], capture_output=True, text=True, check=True)
    return {Path(p).resolve() for p in out.stdout.split()}


def already_handled(run_dir, queued):
    return ((run_dir / ".submitted").exists() or (run_dir / "OUTCAR").exists()
            or run_dir.resolve() in queued)


def setup_and_submit(atoms, run_dir, launch_dir, submit, job_name, queued):
    if already_handled(run_dir, queued):
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    write(run_dir / "POSCAR", atoms, format="vasp", direct=True, sort=True)
    for name in ("INCAR", "VASP.sh"):
        shutil.copy(launch_dir / name, run_dir / name)
    kx, ky = oc20_kpoints(atoms)
    (run_dir / "KPOINTS").write_text(f"Automatic mesh\n0\nMonkhorst-Pack\n{kx} {ky} 1\n0 0 0\n")
    subprocess.run(["bash", POTCAR_SCRIPT], cwd=run_dir, check=True, stdout=subprocess.DEVNULL)
    if submit:
        sbatch_with_retry(run_dir, job_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", required=True)
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--frame", type=int, default=-1, help="traj frame to use")
    parser.add_argument("--no_submit", action="store_true", help="write inputs only")
    args = parser.parse_args()

    launch_dir = Path.cwd()
    out_dir = Path(args.out_dir)
    queued = queued_dirs()

    for traj in tqdm(sorted(Path(args.traj_dir).glob("*.traj")), unit="system"):
        adslab = prep_atoms(read(traj, index=args.frame))
        slab = adslab[adslab.get_tags() != 2]
        system_dir = out_dir / traj.stem
        setup_and_submit(slab, system_dir / "slab", launch_dir, not args.no_submit, traj.stem, queued)
        setup_and_submit(adslab, system_dir / "adslab", launch_dir, not args.no_submit, traj.stem, queued)


if __name__ == "__main__":
    main()
