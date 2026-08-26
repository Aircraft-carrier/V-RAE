#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
python_bin="${VRAE_PYTHON:-$(command -v python)}"
exec "${python_bin}" -m vrae.training.recon_training.train \
  --config "${project_root}/configs/training/recon_training/lerobot_multiview.yaml" "$@"
