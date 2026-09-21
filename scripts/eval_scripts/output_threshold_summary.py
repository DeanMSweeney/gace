#!/usr/bin/env python3
import pandas as pd
import sys

def main(csv_path, out_path="threshold_summary.txt"):
    df = pd.read_csv(csv_path)

    valid = df[df["is_valid"] == 1.0]
    n_valid = len(valid)
    n_total = len(df)

    thresholds = ["within_1T", "within_2T", "within_3T", "within_5T", "within_10T"]

    lines = []
    lines.append(f"Total samples: {n_total}")
    lines.append(f"Valid samples (is_valid == True): {n_valid}")
    lines.append("")
    lines.append("Threshold counts (out of valid samples):")
    for t in thresholds:
        count = (valid[t] == 1.0).sum()
        pct = 100 * count / n_valid if n_valid else 0
        lines.append(f"  {t}: {count}/{n_valid} ({pct:.1f}%)")

    summary = "\n".join(lines)
    print(summary)

    with open(out_path, "w") as f:
        f.write(summary + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python threshold_summary.py <csv_path> [out_path]")
        sys.exit(1)
    main(*sys.argv[1:])
