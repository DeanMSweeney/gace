# GACE

**Generative Adsorbate Catalyst Engineer** — a conditional flow matching model for generating catalyst slab + adsorbate structures de novo.

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-792EE5?style=flat&logo=lightning&logoColor=white)
![Config](https://img.shields.io/badge/Config-gray?style=flat)
![Hydra](https://img.shields.io/badge/Hydra-89CFF0?style=flat)

## Overview

GACE uses conditional flow matching (CFM) to model the joint distribution over catalyst primitive-cell geometry (fractional coordinates, atom types, lattice parameters, supercell matrix) and adsorbate geometry (center-of-mass, relative positions). It learns a velocity field that transports from Gaussian noise to real DFT-relaxed structures, conditioned on binding energy and spacegroup. Hungarian-alignment-based optimal transport is applied at training time to handle permutation invariance.

### Key Features

- **Multi-modality flow matching**: simultaneous flows over positions, atom types (simplex corruption), lattice lengths/angles, supercell matrix M, and adsorbate geometry
- **Transformer backbone**: `TransformerModule` processes primitive-cell and adsorbate tokens as separate sequences with a shared trunk and modality-specific output heads
- **Endpoint & velocity matching**: configurable loss mode (`endpoint` or `flow`)
- **ODE-based sampling**: torchdiffeq integration with dopri5, euler, rk4, or adaptive_heun solvers
- **MLFF relaxation**: optional post-generation relaxation via FairChem OCPCalculator
- **PyTorch Lightning**: multi-GPU/multi-node training with bf16 mixed precision
- **WandB integration**: experiment tracking

## Installation

### Using Docker (Recommended)

```bash
# AMD64 (CUDA 12.4)
docker build -f Dockerfile.amd64 -t gace:amd64 .
docker run --gpus all -it gace:amd64

# ARM64 (Apple Silicon)
docker build -f Dockerfile.arm64 -t gace:arm64 .
docker run -it gace:arm64
```

### Local Installation

1. **Clone the repository**:
```bash
git clone <repo-url>
cd gace
```

2. **Create a conda environment**:
```bash
conda create -n gace python=3.10
conda activate gace
```

3. **Install PyTorch with CUDA support**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

4. **Install PyTorch Geometric dependencies**:
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

5. **Install fairchem-core**:
```bash
pip install --no-deps fairchem-core==1.10.0
```

6. **Install remaining dependencies**:
```bash
pip install -r requirements.txt
```

7. **Install GACE in editable mode**:
```bash
pip install -e .
```

## Usage

All configuration is passed via Hydra CLI overrides to `scripts/train.py`. There are no YAML config files — `model_cfg` is the only required argument and is a Hydra-instantiable model config.

### Train

```bash
python scripts/train.py \
  model_cfg=<hydra-instantiable-model-config> \
  mode=train \
  oc20_train=/path/to/train.lmdb \
  oc20_val=/path/to/val.lmdb \
  variant=forge \
  x_matching=endpoint
```

### Debug mode (small subset, fast iteration)

```bash
python scripts/train.py \
  mode=debug \
  oc20_train=/path/to/train.lmdb \
  oc20_val=/path/to/val.lmdb \
  num_debug_samples=1000
```

### Evaluate

```bash
python scripts/train.py \
  mode=eval \
  oc20_test=/path/to/test.lmdb \
  ckpt_path=/path/to/ckpt.ckpt
```

### Generate

```bash
python scripts/train.py \
  mode=generate \
  ckpt_path=/path/to/ckpt.ckpt \
  sampler_cfg.target_binding_E=-1.5 \
  sampler_cfg.num_generated_samples=10
```

### Preprocess LMDB dataset

```bash
python -m gace.utils.factorized_builders.data_processor \
  --src /path/to/raw.lmdb \
  --dst /path/to/processed.lmdb
```

## Configuration

Configuration is handled by `helm/config.py` via three dataclasses:

### `Config` (top-level)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_cfg` | MISSING | Hydra-instantiable model architecture config |
| `variant` | `"forge"` | Flow matching variant |
| `mode` | `"generate"` | Execution mode: `train`, `eval`, `generate`, `debug` |
| `x_matching` | `"endpoint"` | Loss mode: `endpoint` (predict x₁) or `flow` (predict dx/dt) |
| `interpolant_schedule` | `"linear"` | Interpolation schedule: `linear` or `cosine` |
| `optimal_transport` | `False` | Hungarian alignment of noise x₀ to target x₁ |
| `informed_priors` | `False` | Use informed priors for noise |
| `precision` | `"bf16-mixed"` | Training precision |
| `devices` / `num_nodes` | `1` / `1` | Multi-GPU / multi-node config |

### `OptimizerConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | `1e-3` | Base learning rate (AdamW) |
| `lr_scheduler` | `"cosine_w_warmup"` | Scheduler: `cosine_w_warmup`, `cosine_annealing`, `polynomial_w_warmup`, `polynomial_decay` |
| `warmup_proportion` | `0.01` | Fraction of steps for linear warmup |
| `ema_decay` | `None` | EMA weight decay (e.g. `0.9999`); `None` disables EMA |
| `lambda_x` | `1.0` | Loss weight for catalyst fractional coordinates |
| `lambda_z` | `0.1` | Loss weight for catalyst atom types (cross-entropy) |
| `lambda_ads` | `1.0` | Loss weight for adsorbate geometry |
| `lambda_lat` | `1.0` | Loss weight for lattice lengths/angles |
| `lambda_M` | `1.0` | Loss weight for supercell matrix M |

### `SamplerConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ode_method` | `"dopri5"` | ODE solver: `dopri5`, `euler`, `rk4`, `adaptive_heun` |
| `num_integration_steps` | `100` | Integration steps for ODE |
| `target_binding_E` | `None` | Target adsorption binding energy (eV) |
| `num_generated_samples` | `1` | Number of structures to generate per batch |
| `run_mlff` | `False` | Run MLFF relaxation after generation |
| `refine_final` | `False` | Run a final endpoint-prediction pass at t=1 |
| `config_vasp` | `False` | Generate VASP DFT input files for generated structures |

## Architecture

### Transformer backbone (`models/TFT/`)

`TransformerModule` takes dense padded tensors for primitive-cell atoms and adsorbate atoms as **separate sequences**, concatenates them for a shared transformer trunk, then splits back for modality-specific output heads.

**Inputs (via `PyGToTensorDict` wrapper):**
- Primitive cell: atom types (soft one-hot, 90 classes), fractional coords, lengths, angles, transformation matrix M, supercell lattice
- Adsorbate: atom types (conditioned), COM, relative positions, pairwise distances (RBF-encoded, `(N_ads, 15)`)
- Conditions: flow time t (Fourier), dataset index, spacegroup, binding energy

**Outputs (endpoint predictions):**
- Primitive cell: atom type logits, fractional coords, lengths, angles, M
- Adsorbate: COM, relative positions

### Training loop (`helm/lightning.py`, `utils/path.py`)

`LightFlowMatch` is the PyTorch Lightning module. `path()` builds the interpolated training sample at random time t. `criterion()` computes the multi-component loss. `sample()` runs ODE integration at inference time.

## Project Structure

```
gace/
├── src/gace/
│   ├── helm/                       # Training orchestration
│   │   ├── config.py               # Hydra config dataclasses
│   │   ├── cli.py                  # Training entry point
│   │   ├── lightning.py            # LightFlowMatch Lightning module
│   │   └── callbacks.py            # EMA callback
│   ├── models/
│   │   └── TFT/                    # Transformer backbone
│   │       ├── transformer_module.py
│   │       └── architectures/      # Attention, losses, positional encoding
│   └── utils/
│       ├── path.py                 # Flow interpolation & multi-component loss
│       ├── invariance.py           # PBC distances, optimal transport (Hungarian)
│       ├── lmdb_loader.py          # Lazy LMDB dataset loader
│       ├── builder.py              # Lattice/graph construction, forge sampling
│       ├── atoms.py                # ASE Atoms ↔ PyG Data conversion
│       ├── euler.py                # Euler & CTMC ODE step functions
│       ├── evaluator.py            # Structural validity checks
│       ├── relaxation.py           # MLFF (FairChem) and DFT (VASP) relaxation
│       ├── graph/                  # PBC-aware neighbor list construction
│       ├── factorized_builders/    # Data preprocessing (dict → PyG, pairwise distances)
│       └── tft/                    # TFT data utilities and wrappers
├── scripts/
│   └── train.py                    # Main training script (@hydra.main)
├── analysis/                       # Jupyter notebooks and example structures
├── datasets/                       # LMDB dataset storage (mini, factorized splits)
├── Dockerfile.amd64
├── Dockerfile.arm64
└── requirements.txt
```

## Data Pipeline

```
Raw CatFlow LMDB
  → factorized_builders/data_processor.py   (precompute lattice params, fractional coords, pairwise distances)
  → factorized_builders/pyg_converter.py    (dict → PyG Data)
  → LazyLmdbDataset (lmdb_loader.py)
  → PyG DataLoader
  → PyGToTensorDict / to_dense_batch         (sparse PyG → dense padded tensors)
  → TransformerModule
```

Atoms are identified by `tags`: `1 = slab (primitive cell)`, `2 = adsorbate`. These are split and handled separately throughout the model — they are never concatenated at the data level.

## Dependencies

| Category | Packages |
|----------|---------|
| Deep learning | PyTorch, PyTorch Lightning, PyTorch Geometric |
| ODE solvers | torchdiffeq |
| Atomistic | ASE, pymatgen, fairchem-core |
| Configuration | hydra-core, omegaconf |
| Experiment tracking | Weights & Biases |
| Data | lmdb, numba, orjson |
| Type safety | beartype, jaxtyping |

See `requirements.txt` for the complete list.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
