#!/usr/bin/env bash
# Run on an Ubuntu NVIDIA instance with conda installed. No GPU is used during setup.
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo 'Conda is required. Choose an AutoDL image that includes Miniconda.' >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx cb-repro; then
  conda create -y -n cb-repro -c conda-forge python=3.9 pip \
    'rdkit=2022.03.3' 'numpy=1.23.0' 'scipy=1.8.1' \
    'pandas=1.4.3' 'networkx=2.8.4' 'pyyaml=6.0' \
    'scikit-learn=1.1.1'
fi
conda activate cb-repro

# The published environment.yml contains macOS-only packages. These are the
# corresponding Linux CUDA 11.3 binaries for its PyTorch 1.11 / PyG 2.0 stack.
python -m pip install 'torch==1.11.0+cu113' 'torchvision==0.12.0+cu113' \
  'torchaudio==0.11.0' --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install --only-binary=:all: \
  'torch-scatter==2.0.9' 'torch-sparse==0.6.14' \
  'torch-cluster==1.6.0' 'torch-spline-conv==1.2.1' \
  -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
python -m pip install 'torch-geometric==2.0.4'
python -m pip install 'fair-esm==2.0.0' 'biopython==1.79' \
  'biopandas==0.4.1' 'e3nn==0.5.0' 'prody==2.4.1' \
  'plotly==5.9.0' 'wandb==0.12.20' 'protobuf==3.20.1' \
  'tqdm==4.64.0'

python - <<'PY'
import torch, torch_geometric, torch_cluster, torch_scatter
import rdkit, esm, prody, Bio, e3nn
print('torch', torch.__version__, 'CUDA build', torch.version.cuda,
      'CUDA available', torch.cuda.is_available())
print('PyG', torch_geometric.__version__, 'RDKit', rdkit.__version__)
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available inside cb-repro')
PY
