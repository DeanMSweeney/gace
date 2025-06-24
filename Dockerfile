# 1. Use PyTorch 2.4.1 with CUDA 12.1 (compatible with 12.4+ drivers on HPC)
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn8-runtime

# 2. Set working directory inside container
WORKDIR /workspace

# 3. Install required system packages (needed for PyG and scientific libs)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Upgrade pip
RUN pip install --upgrade pip

# 5. Install PyTorch Geometric extensions manually (important: must match CUDA+torch version)
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# 6. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy your project code (everything in your GitHub repo)
COPY . .

# 8. Default command: open a shell inside the container
CMD ["bash"]
