#!/usr/bin/env python3
"""
plot_test_metrics.py

Reads the saved numpy arrays from test_epoch_end (in same directory)
and plots:
  1. test_loss vs test_rmse
  2. test_rmse vs test_null_rmse
Also saves a CSV with all metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def main():
    # current directory
    here = os.path.dirname(os.path.abspath(__file__))

    # Try combined file first
    npz_path = os.path.join(here, "test_metrics.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        test_loss = data["loss"]
        test_rmse = data["rmse"]
        test_null_rmse = data["null_rmse"]
        test_max_loss = data["max_loss"]
    else:
        # fallback to individual npy files
        test_loss = np.load(os.path.join(here, "test_loss.npy"))
        test_rmse = np.load(os.path.join(here, "test_rmse.npy"))
        test_null_rmse = np.load(os.path.join(here, "test_null_rmse.npy"))
        test_max_loss = np.load(os.path.join(here, "test_max_loss.npy"))

    # 1. Plot test_loss vs test_rmse
    plt.figure(figsize=(6, 5), dpi=150)
    plt.scatter(test_loss, test_rmse, alpha=0.7, s=20)
    plt.xlabel("Test Loss")
    plt.ylabel("Test Struct RMSE (Å)")
    plt.title("Test Loss vs Struct RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(here, "loss_vs_rmse.png"), bbox_inches="tight")
    plt.close()

    # 2. Plot test_rmse vs null_test_rmse
    plt.figure(figsize=(6, 5), dpi=150)
    plt.scatter(test_null_rmse, test_rmse, alpha=0.7, s=20)
    plt.xlabel("Null Struct RMSE (Å)")
    plt.ylabel("Struct RMSE (Å)")
    plt.title("Struct RMSE vs Null Struct RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(here, "rmse_vs_null_rmse.png"), bbox_inches="tight")
    plt.close()

    # 3. Save CSV with all arrays
    df = pd.DataFrame({
        "loss": test_loss,
        "rmse": test_rmse,
        "null_rmse": test_null_rmse,
        "max_loss": test_max_loss,
    })
    csv_path = os.path.join(here, "test_metrics.csv")
    df.to_csv(csv_path, index=False)

    print("Saved plots: loss_vs_rmse.png, rmse_vs_null_rmse.png")
    print(f"Saved CSV: {csv_path}")

if __name__ == "__main__":
    main()
