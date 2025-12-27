# LMDB Dataset Builder

This guide explains how to build an LMDB dataset from selected frames.

## Overview

The workflow consists of three main scripts:

1. **gather_trajs.py** - Extract metrics from trajectory tar files
2. **select_frames.py** - Select frames based on force binning strategy
3. **build_lmdb_from_parquet.py** - Build LMDB dataset from selected frames

## Step-by-Step Usage

### Step 1: Gather Trajectory Metrics (if not already done)

```bash
python datasets/scripts/gather_trajs.py \
    --in_root /path/to/tar/files \
    --rows_per_shard 2000000 \
    --num_workers 16
```

This creates metrics parquet files in structure:
```
/path/to/tar/files/
├── 0.tar
├── 0/
│   └── metrics/
│       ├── part-00000.parquet
│       └── part-00001.parquet
├── 1.tar
├── 1/
│   └── metrics/
│       └── part-00000.parquet
...
```

### Step 2: Select Frames (if not already done)

```bash
python datasets/scripts/select_frames.py \
    --traj_parquets /path/to/tar/files/0/metrics \
    --sid_root /path/to/sid/parquets \
    --bins 6 4 1 1 \
    --force_bins 0.0 0.05 0.05 0.10 0.10 0.30 0.30 0.60 \
    --struc_tol 0.05 \
    --seed 42
```

This creates:
```
/path/to/tar/files/0/metrics/selected/selected_frames.parquet
```

### Step 3: Build LMDB Dataset

```bash
python datasets/scripts/build_lmdb_from_parquet.py \
    --parquet_file /path/to/selected/selected_frames.parquet \
    --tar_root /path/to/tar/files \
    --metrics_root /path/to/tar/files \
    --out_dir /path/to/output/lmdb \
    --entries_per_shard 10000
```

**Arguments:**
- `--parquet_file`: Path to selected_frames.parquet (output from select_frames.py)
- `--tar_root`: Directory containing the *.tar files with trajectories
- `--metrics_root`: Directory containing trajectory metrics (for sid->adsorbate mapping). Defaults to tar_root.
- `--out_dir`: Output directory for LMDB files
- `--entries_per_shard`: Number of entries per LMDB shard (default: 10,000)

**Output structure:**
```
/path/to/output/lmdb/
├── data_0000.lmdb
├── data_0001.lmdb
├── data_0002.lmdb
...
```

### Step 4: Inspect LMDB Dataset

To verify the LMDB dataset was created correctly:

```bash
python datasets/scripts/inspect_lmdb.py \
    --lmdb_dir /path/to/output/lmdb \
    --num_samples 5
```

This will print:
- Number of LMDB files
- Number of entries per file
- Sample data from each file showing structure

## LMDB Data Format

Each entry in the LMDB contains:

```python
{
    'pos': np.ndarray,           # (N, 3) - atomic positions of current frame
    'pos_relaxed': np.ndarray,   # (N, 3) - atomic positions of relaxed frame
    'cell': np.ndarray,          # (3, 3) - unit cell
    'atomic_numbers': np.ndarray,# (N,) - atomic numbers
    'natoms': int,               # number of atoms
    'force': np.ndarray,         # (N, 3) - forces on atoms
    'fixed': np.ndarray,         # (N,) - boolean mask for fixed atoms
    'tags': np.ndarray,          # (N,) - atom tags
    'y': float,                  # energy of current frame
    'y_init': float,             # energy of initial frame (frame 0)
    'y_relaxed': float,          # energy of relaxed frame (last frame)
    'sid': int,                  # system ID
    'fid': int,                  # frame index in trajectory
}
```

**Note:** edge_index and cell_offsets are not stored in the LMDB. These can be computed on-the-fly during training using your model's graph construction method.

## Using the LMDB Dataset

To load the dataset in your training code:

```python
from adsflow.utils.lmdb_dataset import LazyLmdbDataset
from torch_geometric.loader import DataLoader

config = {
    "src": "/path/to/output/lmdb",
    # Optional transforms, key_mapping, etc.
}

dataset = LazyLmdbDataset(config)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    print(batch.pos.shape)
    print(batch.y.shape)
    print(batch.force.shape)
    # ... your training code
```

## Example: Full Pipeline for Mini Dataset

```bash
# Assuming you have tar files in datasets/mini/tests/0/

# Step 1: Already done (metrics exist)

# Step 2: Select frames
python datasets/scripts/select_frames.py \
    --traj_parquets datasets/mini/tests/0/metrics \
    --sid_root datasets/sid_parquets \
    --bins 6 4 1 1 \
    --force_bins 0.0 0.05 0.05 0.10 0.10 0.30 0.30 0.60 \
    --struc_tol 0.05 \
    --seed 42

# Step 3: Build LMDB
python datasets/scripts/build_lmdb_from_parquet.py \
    --parquet_file datasets/mini/tests/0/metrics/selected/selected_frames.parquet \
    --tar_root datasets/mini/tests/0 \
    --out_dir datasets/mini/tests/0/lmdb \
    --entries_per_shard 1000

# Step 4: Inspect
python datasets/scripts/inspect_lmdb.py \
    --lmdb_dir datasets/mini/tests/0/lmdb \
    --num_samples 3
```

## Notes

- The script processes tar files sequentially by default
- Memory usage is controlled by `entries_per_shard` (lower = more shards, less memory)
- The LMDB files are compatible with the existing `LazyLmdbDataset` class
- Edge indices and cell offsets are not stored - these should be computed on-the-fly during training using your model's graph construction method
