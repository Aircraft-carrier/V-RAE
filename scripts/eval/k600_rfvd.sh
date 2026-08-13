#!/usr/bin/env bash

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

usage() {
  cat <<'EOF'
Usage: scripts/eval/k600_rfvd.sh <gpu_count> <vrae> <global_batch_size> [eval_args...]

Arguments:
  gpu_count          Number of local GPU processes to launch.
  vrae               One of: dino, siglip, vjepa, eupe.
  global_batch_size  Aggregate batch size across all GPU processes.
  eval_args          Optional arguments forwarded to the evaluator, e.g. --smoke.

Examples:
  scripts/eval/k600_rfvd.sh 8 dino 1024
  CUDA_VISIBLE_DEVICES=2,3 scripts/eval/k600_rfvd.sh 2 siglip 256 --smoke
EOF
}

if (( $# < 3 )); then
  usage >&2
  exit 2
fi

gpu_count="$1"
vrae="$2"
global_batch_size="$3"
shift 3

if [[ ! "${gpu_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpu_count must be a positive integer, got: ${gpu_count}" >&2
  exit 2
fi
if [[ ! "${global_batch_size}" =~ ^[1-9][0-9]*$ ]]; then
  echo "global_batch_size must be a positive integer, got: ${global_batch_size}" >&2
  exit 2
fi
if (( global_batch_size % gpu_count != 0 )); then
  echo "global_batch_size (${global_batch_size}) must be divisible by gpu_count (${gpu_count})" >&2
  exit 2
fi

case "${vrae}" in
  dino)
    config_name="config_dinov3.yaml"
    ;;
  siglip)
    config_name="config_siglip2.yaml"
    ;;
  vjepa)
    config_name="config_vjepa2_1.yaml"
    ;;
  eupe)
    config_name="config_eupe.yaml"
    ;;
  *)
    echo "unknown vrae '${vrae}'; expected one of: dino, siglip, vjepa, eupe" >&2
    exit 2
    ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
base_config="${project_root}/configs/evaluation/k600_rfvd/${config_name}"

if [[ ! -f "${base_config}" ]]; then
  echo "evaluation config does not exist: ${base_config}" >&2
  exit 1
fi

routed_config="$(mktemp -t k600-rfvd-config.XXXXXX.yaml)"
trap 'rm -f -- "${routed_config}"' EXIT

printf '%s\n' \
  "include: ${base_config}" \
  "" \
  "paths:" \
  "  project_root: ${project_root}" \
  "" \
  "evaluation:" \
  "  global_batch_size: ${global_batch_size}" \
  > "${routed_config}"

echo "Launching K600 rFVD: vrae=${vrae}, config=${config_name}, GPUs=${gpu_count}, global_batch_size=${global_batch_size}" >&2

cd -- "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
torchrun --standalone --nproc_per_node="${gpu_count}" \
  -m vrae.evaluation.k600_rfvd.run \
  --config "${routed_config}" \
  "$@"
