import os 
import pandas as pd 
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm 
import argparse


def boltzman(energy, beta: float=1.0):
    return (np.exp(-1 * beta * energy/1000))

downselect = True
def compute_spearman(traj_root: str):
    m1 = []
    m2 = []
    c1 = 0 
    c2 = 0 
    for idx in tqdm(range(0, 82)):
        
        if os.path.exists(f"{traj_root}{idx}.parquet"): 
            df = pd.read_parquet(f"{idx}.parquet")
            
            if idx ==0: 
                print(df.head())
            
            # energy, forces, positions 
            dE = df["dE"].to_numpy()
            dF = df["dFmax"].to_numpy()
            dS = df["pos_mae"].to_numpy()
            sid = np.unique(df["sid"].to_numpy())
        
            # frame count 
            frame_idx = df["frame_idx"].to_numpy()
            nframes = df["nframes"].to_numpy()
            frame_ratio = (frame_idx + 1) / nframes
            
            if downselect: 
                idx = np.argsort(dE)
                topk = 1000000
                idx_topk = idx[:topk]
                
                dE = dE[idx_topk]
                dF = dF[idx_topk]
                dS = dS[idx_topk]
                frame_ratio = frame_ratio[idx_topk]
                
            energy_pos= spearmanr(dE, dS)
            force_pos = spearmanr(dF, dS)
            
            m1.append(energy_pos)
            m2.append(force_pos)
            c1 += dE.size
            c2 += sid.size

        else:
            continue 

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
    
    compute_spearman(traj_root=traj_root)
