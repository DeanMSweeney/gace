import os
import csv
import argparse
import pandas as pd
from tqdm import tqdm

def count_files_in_dirs(directory, output_csv):
    directory = os.path.realpath(directory)

    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return

    subfolders = [
        item for item in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, item))
    ]

    rows = []
    for item in tqdm(subfolders, desc="Scanning folders"):
        item_path = os.path.join(directory, item)
        file_count = len([
            f for f in os.listdir(item_path)
            if os.path.isfile(os.path.join(item_path, f))
        ])
        rows.append({"folder": item, "path": item_path, "file_count": file_count})

    if not rows:
        print("No folders found.")
        return

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["folder", "path", "file_count"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to '{output_csv}'\n")

    df = pd.read_csv(output_csv)
    print(df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count files in each subfolder and save to CSV.")
    parser.add_argument("--directory", default="/datasets/oc20/dense/trajs", help="Path to the directory to scan")
    parser.add_argument("--output", default="file_counts.csv", help="Output CSV filename (default: file_counts.csv)")
    args = parser.parse_args()
    count_files_in_dirs(args.directory, args.output)