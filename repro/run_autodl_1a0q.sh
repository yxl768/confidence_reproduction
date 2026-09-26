#!/usr/bin/env bash
# One-command AutoDL smoke test. Run after extracting the upload bundle.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/results/1a0q"
start_epoch=$(date +%s)
bash "$repo_dir/repro/setup_linux.sh" 2>&1 | tee "$repo_dir/results/1a0q/setup.log"
printf 'setup_elapsed_seconds=%s\n' "$(( $(date +%s) - start_epoch ))" \
  > "$repo_dir/results/1a0q/setup_time.txt"
bash "$repo_dir/repro/run_1a0q.sh"
