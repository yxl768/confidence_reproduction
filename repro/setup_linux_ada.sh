#!/usr/bin/env bash
# AutoDL vGPU-48GB / RTX 4090-compatible environment (CUDA 11.8).
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo 'Conda is required. Choose an AutoDL image that includes Miniconda.' >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx cb-repro-ada; then
  conda create -y -n cb-repro-ada -c conda-forge python=3.9 pip \
    'rdkit=2022.03.3' 'numpy=1.23.0' 'scipy=1.8.1' \
    'pandas=1.4.3' 'networkx=2.8.4' 'pyyaml=6.0' \
    'scikit-learn=1.1.1'
fi
conda activate cb-repro-ada

python -m pip install 'torch==2.0.1+cu118' 'torchvision==0.15.2+cu118' \
  'torchaudio==2.0.2+cu118' --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install --only-binary=:all: \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
python -m pip install 'torch-geometric==2.3.1'
python -m pip install 'fair-esm==2.0.0' 'biopython==1.79' \
  'biopandas==0.4.1' 'e3nn==0.5.0' 'prody==2.4.1' \
  'plotly==5.9.0' 'wandb==0.12.20' 'protobuf==3.20.1' \
  'tqdm==4.64.0'

python - <<'PY'
import torch, torch_geometric, torch_cluster, torch_scatter
import rdkit, esm, prody, Bio, e3nn
print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('torch', torch.__version__, 'CUDA build', torch.version.cuda,
      'CUDA available', torch.cuda.is_available())
print('PyG', torch_geometric.__version__, 'RDKit', rdkit.__version__)
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available inside cb-repro-ada')
PY
