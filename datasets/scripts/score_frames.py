import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

def boltzman(energy, beta: float=1.0):
    return (np.exp(-1 * beta * energy/1000))

downselect = True
compute_spearman = True
def main(traj_root: str):
    m1 = []
    m2 = []
    c1 = 0 
    c2 = 0 
    for idx in tqdm(range(1)):
        
        file_path = traj_root / f"{idx}.parquet"
        if os.path.exists(file_path): 
            df = pd.read_parquet(file_path)
            
            if idx == 0: 
                print(df.head())
            
            # Group by sid
            grouped = df.groupby("sid")


            for sid, group_df in grouped:
                # energy, forces, positions
                dE = group_df["dE"].to_numpy()
                dF = group_df["Fmax"].to_numpy()
                dS = group_df["pos_rmse"].to_numpy()
                dSmax = group_df["pos_max"].to_numpy()

                # Normalize dE, dF, dS using MinMaxScaler
                """
                scaler = MinMaxScaler()
                dE = scaler.fit_transform(dE.reshape(-1, 1)).flatten()
                dF = scaler.fit_transform(dF.reshape(-1, 1)).flatten()
                dS = scaler.fit_transform(dS.reshape(-1, 1)).flatten()
                """
               
                # frame count
                frame_idx = group_df["frame_idx"].to_numpy()
                nframes = group_df["nframes"].to_numpy()
                frame_ratio = (frame_idx + 1) / nframes

                if downselect:
                    idx_sort = np.argsort(dF)
                    topk = 100
                    idx_topk = idx_sort[:topk]

                    dE = dE[idx_topk]
                    dF = dF[idx_topk]
                    dS = dS[idx_topk]
                    dSmax = dSmax[idx_topk]
                    frame_ratio = frame_ratio[idx_topk]
                    print(f"Energy, {c2}: {dE}")
                    print(f"Force {c2}: {dF}")
                    print(f"Pos {c2}: {dS}")
                    print(f"Pos-max {c2}: {dSmax}")
                    c2 += 1

            if compute_spearman:
                energy_pos= spearmanr(dE, dSmax)
                force_pos = spearmanr(dF, dSmax)

                m1.append(energy_pos)
                m2.append(force_pos)

                c1 += dE.size
     

        else:
            print(f"{file_path} does not exist")
    
    if compute_spearman: 
        m1 = np.array(m1).mean()
        m2 = np.array(m2).mean()
        print(f"Spearman Correlation \n ===================") 
        print(f"Energy & Position: {m1} \nForces and Position: {m2}")
    
    print(f"Total samples processed: {c1} \nNo. of unique systems: {c2}")
        
if __name__ == "__main__":
    ap = argparse.ArgumentParser() 
    ap.add_argument("--traj_root", type=str, required=True, help="Directory containing per-adsorbate *.tar files")
    ap.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers")
    args = ap.parse_args()

    traj_root = Path(args.traj_root)
    out_root = traj_root          
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    num_workers = args.num_workers
    
    main(traj_root=traj_root)
