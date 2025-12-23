import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from adsflow.utils.data_builder import select_by_force
   
def main(traj_root: str, sid_root: str, bins: List[int], force_bins: List[Tuple[float, float]], struc_tol: float = 0.005, seed: Optional[int] = None):
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    c1 = 0
    c2_total, c2_bin1, c2_bin2, c2_bin3, c2_bin4 = 0, 0, 0, 0, 0
    c3 = 0
    idx = 0

    # Store selected frames for each sid
    selected_data = []

    # iterate and compile list of all sids in validation and test sets
    test_val_sids = set()
    for sid_path in glob.glob(f"{sid_root}/*.parquet"):
        df_sid = pd.read_parquet(sid_path)
        test_val_sids.update(df_sid["sid"].to_numpy())

    # iterate through all trajectory files in trajectory root
    for traj_path in tqdm(glob.glob(f"{traj_root}/*.parquet")): # 82 adsorbates considered in OC20

        # iterate through each tracecjtory
        df = pd.read_parquet(traj_path)

        if idx == 0:
            print(df.head())

        # Group by sid
        grouped = df.groupby("sid")

        for sid, group_df in grouped:
            if sid not in test_val_sids:
                # energy, forces, positions
                dE = group_df["dE"].to_numpy()
                Fmax = group_df["Fmax"].to_numpy()
                dSmax = group_df["pos_max"].to_numpy()
                frame_idx = group_df["frame_idx"].to_numpy()
                
                
                # Use force-based binning selection
                selected_indices, bin_labels = select_by_force(
                    dE=dE,
                    Fmax=Fmax,
                    dSmax=dSmax,
                    frame_idx=frame_idx,
                    bin_count=bins,
                    bins=force_bins,
                    struc_tol=struc_tol,
                    seed=seed
                )
                
                c1 += 1
                # Store selected frame data
                if len(selected_indices) > 0:
                    selected_data.append({
                        'sid': sid,
                        'selected_frames': frame_idx[selected_indices].tolist(),
                        'dE': dE[selected_indices].tolist(),
                        'Fmax': Fmax[selected_indices].tolist(),
                        'dSmax': dSmax[selected_indices].tolist(),
                        'bin': bin_labels.tolist()
                    })
                    
                    c2_total += len(frame_idx[selected_indices])
                    c2_bin1 += np.sum(bin_labels == 1)
                    c2_bin2 += np.sum(bin_labels == 2)
                    c2_bin3 += np.sum(bin_labels == 3)
                    c2_bin4 += np.sum(bin_labels == 4)
                    
                c3 += len(frame_idx)


        idx += 1

    # Save selected frames to output file
    if selected_data:
        df_output = pd.DataFrame(selected_data)
        print(df_output.head())
        os.makedirs(traj_root / "selected", exist_ok=True)
        output_path = traj_root / "selected" / "selected_frames.parquet"
        df_output.to_parquet(output_path, index=False)
        print(f"Saved {len(selected_data)} selected frame groups to {output_path}")

    print("==============================")
    print("Dataset Statistics 📈")
    print("==============================")
    print(f"Total trajectories processed: {c1}  with {c2_total} out of {c3} frames selected")
    print(f"Bin 1 count: {c2_bin1}")
    print(f"Bin 2 count: {c2_bin2}")
    print(f"Bin 3 count: {c2_bin3}")
    print(f"Bin 4 count: {c2_bin4}")
    
        
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_root", type=str, required=True, help="Directory containing per-adsorbate *.tar files")
    ap.add_argument("--sid_root", type=str, required=True, help="Path to val/test system ids (sids)")
    ap.add_argument("--bins", type=int, nargs='+', required=True, help="Number of frames in each bin")
    ap.add_argument("--force_bins", type=float, nargs='+', default=[0.0, 0.05, 0.05, 0.10, 0.10, 0.30, 0.30, 0.60],
                    help="Force bin edges as pairs (lower, upper). Default: 0.0 0.05 0.05 0.10 0.10 0.30 0.30 0.60")
    ap.add_argument("--struc_tol", type=float, default=0.005, help="Minimum dSmax difference threshold (default: 0.005)")
    ap.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    traj_root = Path(args.traj_root)
    sid_root = Path(args.sid_root)

    # Create output directory if needed
    num_workers = args.num_workers

    # Parse force_bins into list of tuples
    force_bins = [(args.force_bins[i], args.force_bins[i+1]) for i in range(0, len(args.force_bins), 2)]

    main(traj_root=traj_root, sid_root=sid_root, bins=args.bins, force_bins=force_bins, struc_tol=args.struc_tol, seed=args.seed)
