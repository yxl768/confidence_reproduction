#!/usr/bin/env bash
# Run from any directory after setup_linux.sh, with the project copied to Linux.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(conda shell.bash hook)"
conda activate "${CB_ENV_NAME:-cb-repro}"
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'This script requires CUDA'
print('GPU:', torch.cuda.get_device_name(0))
PY

mkdir -p "$repo_dir/results/1a0q" "$repo_dir/cache/torch"
export TORCH_HOME="$repo_dir/cache/torch"
export PYTHONPATH="$repo_dir/official${PYTHONPATH:+:$PYTHONPATH}"
python "$repo_dir/repro/preflight_models.py" \
  2>&1 | tee "$repo_dir/results/1a0q/checkpoints.log"

(
  cd "$repo_dir/official"
  printf 'official commit: 3c6831c1c33186b59d3664462c4160b2ab59c29d\n' > "$repo_dir/results/1a0q/environment.txt"
  python - <<'PY' >> "$repo_dir/results/1a0q/environment.txt"
import sys, torch, torch_geometric, rdkit, esm
print('Python:', sys.version)
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('PyG:', torch_geometric.__version__, 'RDKit:', rdkit.__version__)
print('ESM:', getattr(esm, '__version__', 'unknown'))
PY
  nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total \
    --format=csv -l 2 > "$repo_dir/results/1a0q/gpu_usage.csv" &
  monitor_pid=$!
  trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
  start_epoch=$(date +%s)
  python ../repro/dock_one.py \
    --protein_ligand_csv ../repro/1a0q.csv \
    --model_dir workdir/pretrained_score \
    --ckpt best_ema_inference_epoch_model.pt \
    --confidence_model_dir workdir/pretrained_confidence \
    --confidence_ckpt best_model.pt \
    --samples_per_complex 8 --batch_size 4 --inference_steps 20 \
    --out_dir ../results/1a0q 2>&1 | tee "$repo_dir/results/1a0q/inference.log"
  printf 'elapsed_seconds=%s\n' "$(( $(date +%s) - start_epoch ))" \
    >> "$repo_dir/results/1a0q/environment.txt"
)

python "$repo_dir/repro/score_pose.py" \
  --reference "$repo_dir/official/data/1a0q/1a0q_ligand.sdf" \
  --pred-dir "$repo_dir/results/1a0q/1a0q" \
  --expect-poses 8 --require-confidence \
  --output "$repo_dir/results/1a0q/score.json"
