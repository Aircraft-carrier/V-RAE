#!/usr/bin/env bash

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

usage() {
  cat <<'EOF'
Usage: scripts/eval/ucf101_gfvd.sh <gpu_count> <vrae> <global_batch_size> [eval_args...]

Arguments:
  gpu_count          Number of local GPU processes to launch.
  vrae               One of: eupe, vjepa.
  global_batch_size  Aggregate generation batch size across all GPU processes.
  eval_args          Optional arguments forwarded to the evaluator, e.g. --smoke or --resume.

Examples:
  scripts/eval/ucf101_gfvd.sh 8 eupe 512
  scripts/eval/ucf101_gfvd.sh 8 vjepa 512
  CUDA_VISIBLE_DEVICES=2,3 scripts/eval/ucf101_gfvd.sh 2 eupe 128 --smoke
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
  eupe)
    config_name="config_eupe.yaml"
    ;;
  vjepa | vjepa2 | vjepa2.1 | vjepa2_1)
    config_name="config_vjepa2_1.yaml"
    ;;
  *)
    echo "unknown vrae '${vrae}'; expected one of: eupe, vjepa" >&2
    exit 2
    ;;
esac

generation_batch_size=$((global_batch_size / gpu_count))
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
base_config="${project_root}/configs/evaluation/ucf101_gfvd/${config_name}"

if [[ ! -f "${base_config}" ]]; then
  echo "evaluation config does not exist: ${base_config}" >&2
  exit 1
fi

routed_config="$(mktemp -t ucf101-gfvd-config.XXXXXX.yaml)"
trap 'rm -f -- "${routed_config}"' EXIT

printf '%s\n' \
  "include: ${base_config}" \
  "" \
  "paths:" \
  "  project_root: ${project_root}" \
  "" \
  "evaluation:" \
  "  global_batch_size: ${global_batch_size}" \
  "  generation_batch_size: ${generation_batch_size}" \
  > "${routed_config}"

echo "Launching UCF101 gFVD: vrae=${vrae}, config=${config_name}, GPUs=${gpu_count}, global_batch_size=${global_batch_size}" >&2

cd -- "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
torchrun --standalone --nproc_per_node="${gpu_count}" \
  -m vrae.evaluation.ucf101_gfvd.run \
  --config "${routed_config}" \
  "$@"
