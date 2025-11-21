# 🧪 AdsFlow 💻
![Concept Art](concept_art.png)

This repository contains code for implementing Conditional Flow Matching (CFM) for adsorbate placement and optimization on catalyst surfaces. AdsFlow leverages geometric deep learning models and optimal transport theory to predict and refine adsorbate configurations on catalytic surfaces.

## Overview

AdsFlow uses flow matching with optimal transport to model the continuous transformation of atomic structures. The model learns to predict velocity fields that guide atomic positions from initial configurations to optimized adsorbate placements on catalyst surfaces. This approach is particularly useful for computational catalysis and materials discovery.

### Key Features

- **Flow Matching with Optimal Transport**: Implements conditional flow matching using optimal transport to learn structure transformations
- **State-of-the-art GNN Architectures**: Supports EquiFormer-v2 and GemNet models for geometric deep learning
- **PyTorch Lightning Integration**: Scalable training with distributed computing support
- **Flexible Training Modes**: Supports both refinement and generation variants
- **Docker Support**: Ready-to-use Docker images for AMD64 and ARM64 architectures
- **Experiment Tracking**: Built-in Weights & Biases (wandb) integration for monitoring

## Installation

### Using Docker (Recommended)

Build and run using the provided Dockerfiles:

```bash
# For AMD64 systems
docker build -f Dockerfile.amd64 -t adsflow:amd64 .
docker run --gpus all -it adsflow:amd64

# For ARM64 systems (Apple Silicon)
docker build -f Dockerfile.arm64 -t adsflow:arm64 .
docker run -it adsflow:arm64
```

### Local Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/AdsFlow.git
cd AdsFlow
```

2. **Create a conda environment** (optional but recommended):
```bash
conda create -n adsflow python=3.10
conda activate adsflow
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

6. **Install other dependencies**:
```bash
pip install -r requirements.txt
```

7. **Install AdsFlow in editable mode**:
```bash
pip install -e .
```

## Project Structure

```
AdsFlow/
├── src/adsflow/          # Main source code
│   ├── models/           # Neural network architectures
│   │   ├── equiformer_v2/  # EquiFormer-v2 implementation
│   │   └── gemnet/         # GemNet implementation
│   ├── utils/            # Utility functions
│   │   ├── data_processing.py
│   │   ├── graph.py      # Graph construction utilities
│   │   ├── lmdb_dataset.py
│   │   ├── optimal_transport.py
│   │   └── utils.py
│   └── helm/             # Training infrastructure
│       ├── cli.py        # Command-line interface
│       ├── configure.py  # Configuration management
│       └── lightning.py  # PyTorch Lightning module
├── scripts/              # Training and generation scripts
│   ├── train.py          # Main training script
│   ├── conf/             # Hydra configuration files
│   ├── generate/         # Generation scripts
│   └── data_scripts/     # Data preparation utilities
├── analysis/             # Analysis notebooks and results
├── datasets/             # Dataset storage
├── configs/              # Additional configuration files
└── requirements.txt      # Python dependencies
```

## Usage

### Training

Train a flow matching model using Hydra configurations:

```bash
cd scripts
python train.py --config-path conf --config-name gem_config-89
```

Available configuration files:
- `equ_config-31.yaml` - EquiFormer-v2 with 31 layers
- `equ_config-53.yaml` - EquiFormer-v2 with 53 layers
- `gem_config-32.yaml` - GemNet with 32 layers
- `gem_config-89.yaml` - GemNet with 89 layers

### Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management. Key configuration parameters include:

- **Model Architecture**: Choose between EquiFormer-v2 and GemNet
- **Training Variant**: `refine` (refine existing structures) or `generate` (generate from noise)
- **Learning Rate**: Base learning rate with cosine/polynomial scheduling
- **Masking**: Option to mask frozen atoms during training
- **ODE Solver**: Configure solver method, tolerance, and other parameters

### Data Format

AdsFlow expects data in LMDB format containing atomic structures with:
- Atomic positions
- Atomic numbers
- Cell parameters
- Tags (surface/adsorbate labels)
- Fixed atom flags

## Model Architectures

### EquiFormer-v2
A state-of-the-art equivariant transformer architecture for molecular property prediction, incorporating:
- SO(3)-equivariant attention mechanisms
- Graph-level attention
- Efficient scaling for large molecular systems

### GemNet
Geometric message passing neural network with:
- Directional message passing
- Efficient edge and angle updates
- Spherical basis functions for geometric representations

## Dependencies

Core dependencies include:
- **PyTorch**: Deep learning framework
- **PyTorch Lightning**: Training infrastructure
- **PyTorch Geometric**: Geometric deep learning
- **e3nn**: Equivariant neural networks
- **fairchem-core**: Open Catalyst Project tools
- **torchdiffeq**: ODE solvers for flow matching
- **Hydra**: Configuration management
- **Weights & Biases**: Experiment tracking
- **ASE**: Atomic Simulation Environment

See `requirements.txt` for the complete list.

## Analysis

The `analysis/` directory contains Jupyter notebooks for:
- Loss analysis per atom type
- Distance metrics evaluation
- Model performance diagnostics
- Visualization of generated structures

## Contributing

If you'd like to contribute, please open an issue or submit a pull request. Contributions are welcome in the following areas:
- Additional model architectures
- Improved data augmentation strategies
- Enhanced sampling methods
- Documentation and tutorials

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

