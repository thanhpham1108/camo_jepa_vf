#!/usr/bin/env bash
set -Eeuo pipefail

# Script to create host conda environment for camo-jepa on SLURM nodes.
# Designed to run inside container where /opt/conda is mounted.
# Usage: ./create_env_camo_jepa.sh [target_env]
# Example: ./create_env_camo_jepa.sh camo-jepa

ENV_NAME="${1:-camo-jepa}"

CONDA="/opt/conda/bin/conda"
ENV_PATH="/opt/conda/envs/${ENV_NAME}"

# Trust repository hosts to avoid SSL/proxy blocking errors
export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org pypi.python.org download.pytorch.org download-r2.pytorch.org"

# Configure conda, pip, and requests to use system CA bundle if present (for proxy/gateway certificates)
if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    export CONDA_SSL_VERIFY=/etc/ssl/certs/ca-certificates.crt
    export PIP_CERT=/etc/ssl/certs/ca-certificates.crt
    export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
fi

# -----------------------------------------------------------------------------
# 1. Create conda env with Python 3.11 (idempotent)
# -----------------------------------------------------------------------------
if "${CONDA}" env list | grep -q "^${ENV_NAME}\s"; then
    echo "[setup] Conda env '${ENV_NAME}' already exists — skipping creation."
else
    echo "[setup] Creating conda env '${ENV_NAME}' with Python 3.11..."
    echo "Accept Terms of Service for the following channels:"
    echo "    - https://repo.anaconda.com/pkgs/main"
    echo "    - https://repo.anaconda.com/pkgs/r"

    "${CONDA}" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    "${CONDA}" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    "${CONDA}" create -n "${ENV_NAME}" -y python=3.11 pip
fi

# Activate newly created environment
source "/opt/conda/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# -----------------------------------------------------------------------------
# 2. Upgrade pip, setuptools, wheel
# -----------------------------------------------------------------------------
pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# 3. Install packages from camo_jepa Dockerfile
# -----------------------------------------------------------------------------
echo "Installing Jupyter and scientific packages..."
pip install \
    jupyter notebook ipykernel \
    numpy pandas matplotlib plotly tqdm \
    h5py pillow ipywidgets

echo "Installing PyTorch (CUDA 11.8)..."
pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

echo "Installing Gymnasium, OpenCV, and ML tools..."
pip install \
    gymnasium[box2d] pygame \
    opencv-python umap-learn torchsummary cma nexusformat

echo "Installing Transformers, Deep Learning, and utility libraries..."
pip install \
    tensorboard wandb iopath pyyaml submitit braceexpand webdataset \
    timm transformers peft decord einops beartype psutil fire python-box \
    scikit-image ftfy \
    squaternion \
    empy==3.3.4 \
    catkin-pkg \
    lark

echo "Installing Hugging Face Hub CLI..."
pip install "huggingface_hub[cli]"

echo "Installing Hydra Core..."
pip install hydra-core

# nuplan-devkit v1.2 transitively pulls navsim==2.0.0 from PyPI which hard-pins
# numpy==1.23.4; that conflicts with opencv-python>=4.9 on Python 3.11 which
# requires numpy>=1.23.5. Fix: install nuplan-devkit with --no-deps.
echo "Installing nuplan-devkit (with --no-deps)..."
pip install --no-deps \
    "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2"

echo "Installing NavSim / nuPlan evaluation dependencies..."
pip install \
    aioboto3 aiofiles \
    bokeh casadi control \
    Fiona geopandas guppy3 \
    nest_asyncio \
    pyarrow pyinstrument pyogrio pyquaternion \
    rasterio "ray[default]" retry rtree \
    selenium \
    Shapely SQLAlchemy sympy tornado ujson \
    scikit-learn "positional-encodings==6.0.1" \
    "pytorch-lightning==2.2.1"

echo "Installing PyTest and PyBullet..."
pip install pytest pybullet

echo "Installing FlowFormer++ dependencies..."
pip install yacs loguru

echo "Conda environment creation complete: ${ENV_NAME}"
