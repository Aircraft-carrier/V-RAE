#!/usr/bin/env bash

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

usage() {
  cat <<'EOF'
Usage: scripts/eval/k600_gfvd.sh <gpu_count> <vrae> <global_batch_size> [eval_args...]

Arguments:
  gpu_count          Number of local GPU processes to launch.
  vrae               Currently supported: vjepa.
  global_batch_size  Aggregate generation batch size across all GPU processes.
  eval_args          Optional arguments forwarded to the evaluator, e.g. --smoke or --resume.

Examples:
  scripts/eval/k600_gfvd.sh 8 vjepa 512
  CUDA_VISIBLE_DEVICES=2,3 scripts/eval/k600_gfvd.sh 2 vjepa 128 --smoke
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
  vjepa)
    config_name="config_vjepa2_1.yaml"
    ;;
  *)
    echo "unknown vrae '${vrae}'; currently supported: vjepa" >&2
    exit 2
    ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
base_config="${project_root}/configs/evaluation/k600_gfvd/${config_name}"

required_files=(
  "${base_config}"
  "${project_root}/data/metadata/k600_train_gfvd50k_first200x84.jsonl"
  "${project_root}/data/metadata/k600_train_gfvd50k_first200x84.metadata.json"
  "${project_root}/ckpts/vrae/vrae_vjepa2.1.pt"
  "${project_root}/ckpts/k600_dit/k600_dit_vjepa2.1.pt"
  "${project_root}/ckpts/k600_dit/k600_latent_stats_vjepa2.1.pt"
  "${project_root}/ckpts/eval_models/i3d_torchscript.pt"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "required K600 gFVD file does not exist: ${required_file}" >&2
    exit 1
  fi
done

routed_config="$(mktemp -t k600-gfvd-config.XXXXXX.yaml)"
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

echo "Launching K600 gFVD: vrae=${vrae}, config=${config_name}, GPUs=${gpu_count}, global_batch_size=${global_batch_size}" >&2

cd -- "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
torchrun --standalone --nproc_per_node="${gpu_count}" \
  -m vrae.evaluation.k600_gfvd.run \
  --config "${routed_config}" \
  "$@"
