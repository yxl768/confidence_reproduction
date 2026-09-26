#!/usr/bin/env bash
# Validated on AutoDL's Python 3.10 / PyTorch 2.1.2+cu121 image with RTX 4080 SUPER.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python - <<'PY'
import sys
import torch
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.startswith("2.1.2"), torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
assert torch.cuda.is_available(), "CUDA is unavailable"
print("Base image:", sys.version.split()[0], torch.__version__, torch.cuda.get_device_name(0))
PY

# The matching PyG wheels are prebuilt; --no-deps avoids trying to find SciPy
# inside the wheel-only index before the rest of the dependencies are installed.
python -m pip install --no-deps --no-index --only-binary=:all: \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html \
  torch-scatter torch-sparse torch-cluster torch-spline-conv
python -m pip install \
  'numpy==1.23.5' 'scipy==1.10.1' 'pandas==1.5.3' \
  'scikit-learn==1.2.2' 'networkx==2.8.8' 'pyyaml==6.0.2' \
  'rdkit==2022.9.5' 'torch-geometric==2.3.1' 'fair-esm==2.0.0' \
  'biopython==1.79' 'biopandas==0.4.1' 'e3nn==0.5.0' \
  'prody==2.4.1' 'spyrmsd==0.5.2' 'plotly==5.9.0'

python "$repo_dir/repro/patch_official_inference.py"
python -m py_compile "$repo_dir/official/utils/inference_utils.py"
python - <<'PY'
import torch, torch_geometric, torch_cluster, torch_scatter, rdkit, esm, prody, Bio, e3nn
print("Ready:", torch.__version__, torch_geometric.__version__, rdkit.__version__)
PY
