#!/usr/bin/env bash
# One-command smoke test for AutoDL vGPU-48GB / RTX 4090-class instances.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/results/1a0q"
start_epoch=$(date +%s)
bash "$repo_dir/repro/setup_linux_ada.sh" \
  2>&1 | tee "$repo_dir/results/1a0q/setup.log"
printf 'setup_elapsed_seconds=%s\n' "$(( $(date +%s) - start_epoch ))" \
  > "$repo_dir/results/1a0q/setup_time.txt"
CB_ENV_NAME=cb-repro-ada bash "$repo_dir/repro/run_1a0q.sh"
