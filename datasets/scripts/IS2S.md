# Dataset Builders

Documentation for IS2S and IS2RS dataset preparation scripts.

---

## IS2S Dataset Builder

Build an LMDB dataset for IS2S training using a three-step pipeline.

## Workflow

Run these scripts in order:

1. **gather_trajs.py** - Extract metrics from trajectory tar files
2. **select_frames.py** - Select frames using force binning
3. **write_is2s-large.py** - Build LMDB dataset from selected frames

## Step 1: Gather Trajectory Metrics

Extract per-frame metrics (energy, forces, positions) from trajectory tar files:

```bash
python datasets/scripts/IS2S/gather_trajs.py \
    --in_root /path/to/tar/files \
    --rows_per_shard 2000000 \
    --num_workers 16
```

**Output:** Creates `<adsorbate>/metrics/part-*.parquet` files for each tar file.

## Step 2: Select Frames

Select frames from trajectories using force-based binning:

```bash
python datasets/scripts/IS2S/select_frames.py \
    --traj_parquets /path/to/metrics \
    --sid_root /path/to/sid/parquets \
    --bins 6 4 1 1 \
    --force_bins 0.0 0.05 0.05 0.10 0.10 0.30 0.30 0.60 \
    --struc_tol 0.05 \
    --seed 42
```

**Key arguments:**
- `--bins`: Number of frames to select per force bin
- `--force_bins`: Force bin edges as pairs (lower, upper)
- `--struc_tol`: Minimum structural difference threshold

**Output:** Creates `selected/selected_frames.parquet` with frame selections.

## Step 3: Build LMDB Dataset

Build LMDB dataset from selected frames:

```bash
python datasets/scripts/IS2S/write_is2s-large.py \
    --parquet_file /path/to/selected/selected_frames.parquet \
    --tar_root /path/to/tar/files \
    --out_dir /path/to/output/lmdb \
    --entries_per_shard 10000 \
    --num_workers 16
```

**Key arguments:**
- `--parquet_file`: Output from select_frames.py
- `--tar_root`: Directory containing trajectory *.tar files
- `--entries_per_shard`: Frames per LMDB file (controls memory usage)

**Output:** Creates `data_*.lmdb` files in the output directory.

---

## IS2RS Dataset Builder

Precompute velocity metrics for IS2RE datasets using optimal transport.

### Workflow

The IS2RS pipeline adds velocity information to existing LMDB datasets:

**write_vel.py** - Precompute and add velocity vectors to LMDB data

### Precompute Velocities

Add optimal transport-based velocity vectors to an existing IS2RE LMDB dataset:

```bash
python datasets/scripts/IS2RS/write_vel.py
```

**Configuration:** Edit the script to set paths:
- `src_path`: Input LMDB dataset path
- `dst_path`: Output LMDB dataset path (with velocity)
- `batch_size`: Number of samples per batch
- `commit_every`: Commit frequency for LMDB writes

**Key features:**
- Uses optimal transport to compute velocity fields
- Creates new LMDB with velocity stored per sample
- Preserves all original data attributes
- Configurable batch processing and commit intervals

**Output:** Creates a new LMDB file with precomputed `velocity` attribute for each sample.
