"""
MLFF Structure Relaxation Script

Loads structures from an LMDB dataset and relaxes them using
EquiformerV2 as a machine-learned force field via ASE.

Requirements:
    - fairchem-core>=1.10
    - ase
    - torch
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS, LBFGS, FIRE
from ase.io import write as ase_write

from torch_geometric.loader import DataLoader
from torch.utils.data import Subset # Add this import!

from fairchem.core import OCPCalculator

from adsflow.utils.lmdb_loader import LazyLmdbDataset
from adsflow.utils.eval_metrics import ADwT

import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_to_atoms(data) -> Atoms:
    """
    Convert a PyTorch Geometric Data object to an ASE Atoms object.
    Assumes batch_size=1 (single structure from DataLoader).

    Args:
        data: PyG Data object with pos, atomic_numbers, cell, fixed, tags

    Returns:
        ASE Atoms object with positions, cell, and constraints
    """
    pos = data.pos.cpu().numpy()
    atomic_numbers = data.atomic_numbers.cpu().numpy()
    fixed = data.fixed.cpu().numpy()
    tags = data.tags.cpu().numpy()
    cell = data.cell.squeeze(0).cpu().numpy()

    atoms = Atoms(
        positions=pos,
        numbers=atomic_numbers,
        cell=cell,
        pbc=True,
        tags=tags,
    )

    if fixed.any():
        atoms.set_constraint(FixAtoms(mask=fixed.astype(bool)))

    return atoms


def relax_structure(
    atoms: Atoms,
    calculator: OCPCalculator,
    fmax: float = 0.05,
    steps: int = 200,
    optimizer: str = "BFGS",
) -> tuple[Atoms, bool, int]:
    """
    Relax an atomic structure using the MLFF calculator.

    Args:
        atoms: ASE Atoms object to relax
        calculator: OCP calculator with MLFF model
        fmax: Force convergence criterion (eV/A)
        steps: Maximum optimization steps
        optimizer: Optimizer type ("BFGS", "LBFGS", or "FIRE")

    Returns:
        Tuple of (relaxed_atoms, converged, num_steps)
    """
    atoms = atoms.copy()
    atoms.calc = calculator

    # Select optimizer
    optimizers = {
        "BFGS": BFGS,
        "LBFGS": LBFGS,
        "FIRE": FIRE,
    }
    opt_class = optimizers.get(optimizer, BFGS)

    # Run optimization
    opt = opt_class(atoms, logfile=None)
    converged = opt.run(fmax=fmax, steps=steps)

    return atoms, converged, opt.nsteps


def compute_metrics(data, pos_relaxed: np.ndarray, threshold: float = 0.5) -> tuple:
    """
    Compute relaxation metrics using ADwT from eval_metrics.py.
    Assumes batch_size=1 (single structure from DataLoader).

    Args:
        data: PyG Data object with pos_relaxed (ground truth), cell, fixed, tags
        pos_relaxed: MLFF-relaxed positions as numpy array
        threshold: Distance threshold for DwT metric (Angstroms)

    Returns:
        Tuple of (mae, dwt, adwt, ads_mae, ads_dwt, ads_adwt)
    """
    data = data.clone().cpu()
    data.pos_generated = torch.from_numpy(pos_relaxed).float()
    data.batch = torch.zeros(len(data.pos), dtype=torch.long)

    return ADwT(data=data, pos_predicted=data.pos_generated, threshold=threshold)


def compute_initial_metrics(data, threshold: float = 0.5) -> tuple:
    """
    Compute metrics for initial (unrelaxed) structure as baseline.
    Assumes batch_size=1 (single structure from DataLoader).

    Args:
        data: PyG Data object
        threshold: Distance threshold for DwT metric (Angstroms)

    Returns:
        Tuple of (mae, dwt, adwt, ads_mae, ads_dwt, ads_adwt)
    """
    data = data.clone().cpu()
    data.pos_generated = data.pos.clone()
    data.batch = torch.zeros(len(data.pos), dtype=torch.long)

    return ADwT(data=data, pos_predicted=data.pos_generated, threshold=threshold)

def main():

    set_seed(42)
    parser = argparse.ArgumentParser(
        description="Relax structures using EquiformerV2 MLFF"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to LMDB dataset directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="EquiformerV2-153M-S2EF-OC20-All+MD",
        help="Model checkpoint name or path (default: EquiformerV2-153M-S2EF-OC20-All+MD)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./mlff_relaxed",
        help="Directory to save relaxed structures",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Force convergence criterion in eV/A (default: 0.05)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Maximum optimization steps (default: 200)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="BFGS",
        choices=["BFGS", "LBFGS", "FIRE"],
        help="ASE optimizer to use (default: BFGS)",
    )

    parser.add_argument(
        "--save_structures",
        default=False,
        action="store_true",
        help="Save relaxed structures to POSCAR files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="Distance threshold for DwT/ADwT metrics in Angstroms (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on (default: cuda if available)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    dataset = LazyLmdbDataset({"src": args.data_path})
    dataset = Subset(dataset, indices=range(2))

    num_samples = len(dataset)
    print(f"Processing {num_samples} structures")

    # Create DataLoader with batch_size=1
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Initialize MLFF calculator
    print(f"Loading model: {args.checkpoint}")
    calculator = OCPCalculator(
        model_name=args.checkpoint,
        local_cache="./ocp_checkpoints",
        cpu=(args.device == "cpu"),
    )

    # Storage for metrics as lists
    generated_mae_list = []
    generated_dwt_list = []
    generated_adwt_list = []
    generated_ads_mae_list = []
    generated_ads_dwt_list = []
    generated_ads_adwt_list = []
    initial_mae_list = []
    initial_dwt_list = []
    initial_adwt_list = []
    initial_ads_mae_list = []
    initial_ads_dwt_list = []
    initial_ads_adwt_list = []
    converged_list = []
    num_steps_list = []
    sid_list = []

    # Process each structure (batch_size=1)
    for i, data in enumerate(tqdm(dataloader, desc="Relaxing structures")):
        # Convert to ASE Atoms
        atoms_initial = data_to_atoms(data)

        # Relax with MLFF
        try:
            atoms_relaxed, converged, num_steps = relax_structure(
                atoms=atoms_initial,
                calculator=calculator,
                fmax=args.fmax,
                steps=args.steps,
                optimizer=args.optimizer,
            )

            # Compute metrics if target available
            if hasattr(data, 'pos_relaxed'):
                pos_relaxed = atoms_relaxed.get_positions()

                # Compute metrics for MLFF-relaxed structure
                (
                    generated_mae,
                    generated_dwt,
                    generated_adwt,
                    generated_ads_mae,
                    generated_ads_dwt,
                    generated_ads_adwt,
                ) = compute_metrics(data=data, pos_relaxed=pos_relaxed, threshold=args.threshold)

                generated_mae_list.append(generated_mae)
                generated_dwt_list.append(generated_dwt)
                generated_adwt_list.append(generated_adwt)
                generated_ads_mae_list.append(generated_ads_mae)
                generated_ads_dwt_list.append(generated_ads_dwt)
                generated_ads_adwt_list.append(generated_ads_adwt)

                # Compute baseline metrics for initial structure
                (
                    initial_mae,
                    initial_dwt,
                    initial_adwt,
                    initial_ads_mae,
                    initial_ads_dwt,
                    initial_ads_adwt,
                ) = compute_initial_metrics(data=data, threshold=args.threshold)

                initial_mae_list.append(initial_mae)
                initial_dwt_list.append(initial_dwt)
                initial_adwt_list.append(initial_adwt)
                initial_ads_mae_list.append(initial_ads_mae)
                initial_ads_dwt_list.append(initial_ads_dwt)
                initial_ads_adwt_list.append(initial_ads_adwt)

            converged_list.append(converged)
            num_steps_list.append(num_steps)
            sid_list.append(data.sid.item() if hasattr(data, 'sid') else i)

            # Save structures if requested
            if args.save_structures:
                sid = data.sid.item() if hasattr(data, 'sid') else i
                ase_write(output_dir / f"relaxed_{sid}.vasp", atoms_relaxed, format="vasp")

        except Exception as e:
            print(f"Error relaxing structure {i}: {e}")
            converged_list.append(False)
            num_steps_list.append(-1)
            sid_list.append(data.sid.item() if hasattr(data, 'sid') else i)

    # Save metrics as numpy arrays
    print("\nSaving metrics...")
    np.save(output_dir / "generated_mae.npy", np.array(generated_mae_list))
    np.save(output_dir / "generated_dwt.npy", np.array(generated_dwt_list))
    np.save(output_dir / "generated_adwt.npy", np.array(generated_adwt_list))
    np.save(output_dir / "generated_ads_mae.npy", np.array(generated_ads_mae_list))
    np.save(output_dir / "generated_ads_dwt.npy", np.array(generated_ads_dwt_list))
    np.save(output_dir / "generated_ads_adwt.npy", np.array(generated_ads_adwt_list))
    np.save(output_dir / "initial_mae.npy", np.array(initial_mae_list))
    np.save(output_dir / "initial_dwt.npy", np.array(initial_dwt_list))
    np.save(output_dir / "initial_adwt.npy", np.array(initial_adwt_list))
    np.save(output_dir / "initial_ads_mae.npy", np.array(initial_ads_mae_list))
    np.save(output_dir / "initial_ads_dwt.npy", np.array(initial_ads_dwt_list))
    np.save(output_dir / "initial_ads_adwt.npy", np.array(initial_ads_adwt_list))
    np.save(output_dir / "converged.npy", np.array(converged_list))
    np.save(output_dir / "num_steps.npy", np.array(num_steps_list))
    np.save(output_dir / "sid.npy", np.array(sid_list))

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total structures: {num_samples}")
    print(f"Converged: {sum(converged_list)} ({100*sum(converged_list)/len(converged_list):.1f}%)")
    print(f"Average steps: {np.mean([s for s in num_steps_list if s > 0]):.1f}")

    if generated_mae_list:
        print(f"\nAll Atoms (threshold={args.threshold} A):")
        print(f"  MAE  - MLFF: {np.mean(generated_mae_list):.4f} A | Initial: {np.mean(initial_mae_list):.4f} A")
        print(f"  DwT  - MLFF: {np.mean(generated_dwt_list):.1f}% | Initial: {np.mean(initial_dwt_list):.1f}%")
        print(f"  ADwT - MLFF: {np.mean(generated_adwt_list):.1f}% | Initial: {np.mean(initial_adwt_list):.1f}%")

        print(f"\nAdsorbate Atoms Only:")
        print(f"  MAE  - MLFF: {np.mean(generated_ads_mae_list):.4f} A | Initial: {np.mean(initial_ads_mae_list):.4f} A")
        print(f"  DwT  - MLFF: {np.mean(generated_ads_dwt_list):.1f}% | Initial: {np.mean(initial_ads_dwt_list):.1f}%")
        print(f"  ADwT - MLFF: {np.mean(generated_ads_adwt_list):.1f}% | Initial: {np.mean(initial_ads_adwt_list):.1f}%")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
