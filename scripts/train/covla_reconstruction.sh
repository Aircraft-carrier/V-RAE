#!/usr/bin/env bash

set -euo pipefail

COVLA_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COVLA_PROJECT_ROOT="$(cd -- "${COVLA_SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${COVLA_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
COVLA_CONFIG_PATH="${COVLA_CONFIG_PATH:-${COVLA_PROJECT_ROOT}/configs/training/recon_training/eupe_covla_432x768.yaml}"
COVLA_PATHS_FILE="${VRAE_PATHS_FILE:-${COVLA_PROJECT_ROOT}/configs/paths.local.yaml}"

if [[ -n "${COVLA_CONDA_ACTIVATE:-}" ]]; then
  if [[ ! -f "${COVLA_CONDA_ACTIVATE}" ]]; then
    echo "Conda activation script does not exist: ${COVLA_CONDA_ACTIVATE}" >&2
    exit 2
  fi
  source "${COVLA_CONDA_ACTIVATE}" "${COVLA_CONDA_ENV:-vrae}"
fi
COVLA_PYTHON="${COVLA_PYTHON:-$(command -v python)}"
COVLA_TORCHRUN="${COVLA_TORCHRUN:-$(command -v torchrun)}"
COVLA_SITE_PACKAGES="$("${COVLA_PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
COVLA_TORCH_LIB="${COVLA_SITE_PACKAGES}/torch/lib"
COVLA_PYTHON_PREFIX="$("${COVLA_PYTHON}" -c 'import sys; print(sys.prefix)')"
export LD_LIBRARY_PATH="${COVLA_PYTHON_PREFIX}/lib:${COVLA_TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ ! -f "${COVLA_CONFIG_PATH}" ]]; then
  echo "CoVLA reconstruction config does not exist: ${COVLA_CONFIG_PATH}" >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

COVLA_NNODES="${COVLA_NNODES:-1}"
COVLA_NPROC_PER_NODE="${COVLA_NPROC_PER_NODE:-8}"
COVLA_NODE_RANK="${NODE_RANK:-${COVLA_NODE_RANK:-0}}"

if [[ ! "${COVLA_NNODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "COVLA_NNODES must be a positive integer: ${COVLA_NNODES}" >&2
  exit 2
fi
if [[ ! "${COVLA_NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "COVLA_NPROC_PER_NODE must be a positive integer: ${COVLA_NPROC_PER_NODE}" >&2
  exit 2
fi
if [[ ! "${COVLA_NODE_RANK}" =~ ^[0-9]+$ ]]; then
  echo "Set NODE_RANK to this machine's rank in [0,$((COVLA_NNODES - 1))]." >&2
  exit 2
fi
if (( COVLA_NODE_RANK < 0 || COVLA_NODE_RANK >= COVLA_NNODES )); then
  echo "NODE_RANK=${COVLA_NODE_RANK} is outside [0,$((COVLA_NNODES - 1))]." >&2
  exit 2
fi
if (( COVLA_NNODES == 1 )); then
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
else
  export MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 host}"
fi
export MASTER_PORT="${MASTER_PORT:-29500}"
if [[ ! "${MASTER_PORT}" =~ ^[1-9][0-9]*$ ]] || (( MASTER_PORT > 65535 )); then
  echo "MASTER_PORT must be in [1,65535]: ${MASTER_PORT}" >&2
  exit 2
fi

default_gpu_ids=()
for ((gpu_index = 0; gpu_index < COVLA_NPROC_PER_NODE; gpu_index++)); do
  default_gpu_ids+=("${gpu_index}")
done
default_cuda_visible_devices="$(IFS=,; echo "${default_gpu_ids[*]}")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${default_cuda_visible_devices}}"
IFS=',' read -r -a COVLA_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#COVLA_GPU_IDS[@]}" -ne "${COVLA_NPROC_PER_NODE}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain ${COVLA_NPROC_PER_NODE} GPU ids: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

COVLA_CODEC_RUNTIME="$("${COVLA_PYTHON}" -c 'import torch, torchcodec; print(f"torch={torch.__version__} torchcodec={torchcodec.__version__} cuda={torch.version.cuda}")')"
COVLA_LAUNCH_ARGS=(
  "${COVLA_TORCHRUN}"
  --nnodes="${COVLA_NNODES}"
  --nproc_per_node="${COVLA_NPROC_PER_NODE}"
  --node_rank="${COVLA_NODE_RANK}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  -m vrae.training.recon_training.covla.train
  --config "${COVLA_CONFIG_PATH}"
)

if [[ -f "${COVLA_PATHS_FILE}" ]]; then
  COVLA_LAUNCH_ARGS+=(--paths "${COVLA_PATHS_FILE}")
fi

if [[ -n "${COVLA_MAX_STEPS:-}" ]]; then
  if [[ ! "${COVLA_MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "COVLA_MAX_STEPS must be a positive integer" >&2
    exit 2
  fi
  COVLA_LAUNCH_ARGS+=(--max-steps "${COVLA_MAX_STEPS}")
fi
COVLA_LAUNCH_ARGS+=("$@")

echo "[preflight] host=$(hostname) node_rank=${COVLA_NODE_RANK}/${COVLA_NNODES}"
echo "[preflight] master=${MASTER_ADDR}:${MASTER_PORT} local_processes=${COVLA_NPROC_PER_NODE} world_size=$((COVLA_NNODES * COVLA_NPROC_PER_NODE))"
echo "[preflight] python=${COVLA_PYTHON} torchrun=${COVLA_TORCHRUN}"
echo "[preflight] ${COVLA_CODEC_RUNTIME}"
echo "[preflight] config=${COVLA_CONFIG_PATH} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
printf '[command]'
printf ' %q' "${COVLA_LAUNCH_ARGS[@]}"
printf '\n'

if [[ "${COVLA_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

cd "${COVLA_PROJECT_ROOT}"
exec "${COVLA_LAUNCH_ARGS[@]}"
