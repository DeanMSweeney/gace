"""Summarize relaxation results.

Computes statistics on binding energy differences and reports the top-5 most
common metal elements across relaxed structures.

Usage:
    python summarize_rlxd.py \
        --rlxd_strucs ./rlxd_strucs
"""

import argparse
import csv
import statistics

NON_METALS = {"H", "O", "N", "C"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rlxd_strucs", default="./rlxd_strucs", help="Directory containing output CSVs")
    return p.parse_args()


def main():
    args = parse_args()

    be_csv = f"{args.rlxd_strucs}/binding_energies.csv"
    ec_csv = f"{args.rlxd_strucs}/element_counts.csv"

    # --- Binding energy diff stats ---
    diffs = []
    total = 0
    with open(be_csv, newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            val = row["binding_E_diff"].strip()
            if val:
                diffs.append(float(val))

    print(f"Binding energy difference stats  ({len(diffs)}/{total} structures relaxed)")
    print(f"  mean   : {statistics.mean(diffs):.4f} eV")
    print(f"  median : {statistics.median(diffs):.4f} eV")
    print(f"  std    : {statistics.stdev(diffs):.4f} eV")
    print(f"  min    : {min(diffs):.4f} eV")
    print(f"  max    : {max(diffs):.4f} eV")

    # --- Top-5 metal elements ---
    metals = []
    with open(ec_csv, newline="") as f:
        for row in csv.DictReader(f):
            el = row["element"].strip()
            if el not in NON_METALS:
                metals.append((el, int(row["count"])))

    print(f"\nTop-5 most common metal elements")
    for i, (el, count) in enumerate(metals[:5], 1):
        print(f"  {i}. {el:4s}  {count} structures")


if __name__ == "__main__":
    main()
