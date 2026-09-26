#!/usr/bin/env bash
# Single-example technical acceptance test, not a DockGen benchmark result.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/results/1a0q" "$repo_dir/cache/torch"
export TORCH_HOME="$repo_dir/cache/torch"
export PYTHONPATH="$repo_dir/official${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python "$repo_dir/repro/patch_official_inference.py"
python "$repo_dir/repro/preflight_models.py" \
  2>&1 | tee "$repo_dir/results/1a0q/preflight.log"
python -c 'import esm; esm.pretrained.esm2_t33_650M_UR50D()' \
  2>&1 | tee "$repo_dir/results/1a0q/esm_download.log"

(
  cd "$repo_dir/official"
  python ../repro/dock_one.py \
    --protein_ligand_csv ../repro/1a0q.csv \
    --model_dir workdir/pretrained_score \
    --ckpt best_ema_inference_epoch_model.pt \
    --confidence_model_dir workdir/pretrained_confidence \
    --confidence_ckpt best_model.pt \
    --samples_per_complex 8 --batch_size 4 --inference_steps 20 \
    --out_dir ../results/1a0q \
    2>&1 | tee "$repo_dir/results/1a0q/inference.log"
)

python "$repo_dir/repro/score_pose.py" \
  --reference "$repo_dir/official/data/1a0q/1a0q_ligand.sdf" \
  --pred-dir "$repo_dir/results/1a0q/1a0q" \
  --expect-poses 8 --require-confidence \
  --output "$repo_dir/results/1a0q/score.json" > /dev/null
python -m pip freeze > "$repo_dir/results/1a0q/pip_freeze.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv \
  > "$repo_dir/results/1a0q/gpu.txt"
