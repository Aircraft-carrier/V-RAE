#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train/k600.sh \
  <nnodes> <gpus_per_node> <node_rank> <vrae> [train_args...]

Arguments:
  nnodes          Total number of machines.
  gpus_per_node   Number of torchrun workers on each machine.
  node_rank       This machine's zero-based rank in [0, nnodes).
  vrae            One of: dino, siglip, vjepa, eupe.
  train_args      Optional arguments forwarded to the trainer, e.g. --max-steps 1000.

Environment:
  MASTER_ADDR       Rank-0 hostname/IP; required when nnodes > 1.
  MASTER_PORT       Rendezvous port (default: 29500).
  CUDA_VISIBLE_DEVICES
                    Defaults to 0..gpus_per_node-1.
  WANDB_MODE        Defaults to online; Python falls back to offline without a key.
  K600_DRY_RUN       Set to 1 to validate routing and print the table without launching.
  K600_BACKGROUND    Set to 1 to launch with nohup and write one log per node.
  K600_LOG_DIR       Background log directory (default: outputs/logs).
  K600_PYTHON        Optional Python executable override.
  K600_TORCHRUN      Optional torchrun executable override.

Examples:
  # One machine, eight GPUs.
  scripts/train/k600.sh 1 8 0 dino

  # Two machines, eight GPUs each. Run one command on each machine.
  MASTER_ADDR=10.0.0.8 scripts/train/k600.sh 2 8 0 vjepa
  MASTER_ADDR=10.0.0.8 scripts/train/k600.sh 2 8 1 vjepa

  # Validate routing without starting workers.
  K600_DRY_RUN=1 scripts/train/k600.sh 1 8 0 eupe --max-steps 1000
EOF
}

if (( $# < 4 )); then
  usage >&2
  exit 2
fi

nnodes="$1"
gpus_per_node="$2"
node_rank="$3"
vrae="$4"
shift 4
trainer_args=("$@")

if [[ ! "${nnodes}" =~ ^[1-9][0-9]*$ ]]; then
  echo "nnodes must be a positive integer, got: ${nnodes}" >&2
  exit 2
fi
if [[ ! "${gpus_per_node}" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpus_per_node must be a positive integer, got: ${gpus_per_node}" >&2
  exit 2
fi
if [[ ! "${node_rank}" =~ ^[0-9]+$ ]] || (( node_rank >= nnodes )); then
  echo "node_rank must be an integer in [0, $((nnodes - 1))], got: ${node_rank}" >&2
  exit 2
fi

case "${vrae}" in
  dino | dinov3)
    config_name="dinov3.yaml"
    encoder_name="dinov3"
    ;;
  siglip | siglip2)
    config_name="siglip2.yaml"
    encoder_name="siglip2"
    ;;
  vjepa | vjepa2 | vjepa2.1 | vjepa2_1)
    config_name="vjepa2_1.yaml"
    encoder_name="vjepa2_1"
    ;;
  eupe)
    config_name="eupe.yaml"
    encoder_name="eupe"
    ;;
  *)
    echo "unknown vrae '${vrae}'; expected one of: dino, siglip, vjepa, eupe" >&2
    exit 2
    ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
config_path="${project_root}/configs/training/k600_videogen/${config_name}"
paths_file="${VRAE_PATHS_FILE:-${project_root}/configs/paths.local.yaml}"
paths_args=()
if [[ -f "${paths_file}" ]]; then
  paths_args=(--paths "${paths_file}")
fi

if [[ ! -f "${config_path}" ]]; then
  echo "K600 VideoGen config does not exist: ${config_path}" >&2
  exit 2
fi

master_addr="${MASTER_ADDR:-}"
if (( nnodes == 1 )); then
  master_addr="${master_addr:-127.0.0.1}"
elif [[ -z "${master_addr}" ]]; then
  echo "MASTER_ADDR must be set to the rank-0 hostname/IP when nnodes > 1" >&2
  exit 2
fi
master_port="${MASTER_PORT:-29500}"
if [[ ! "${master_port}" =~ ^[1-9][0-9]*$ ]] || (( master_port > 65535 )); then
  echo "MASTER_PORT must be an integer in [1, 65535], got: ${master_port}" >&2
  exit 2
fi

default_gpu_ids=()
for ((gpu_index = 0; gpu_index < gpus_per_node; gpu_index++)); do
  default_gpu_ids+=("${gpu_index}")
done
default_cuda_visible_devices="$(IFS=,; echo "${default_gpu_ids[*]}")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${default_cuda_visible_devices}}"
IFS=',' read -r -a visible_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#visible_gpu_ids[@]} != gpus_per_node )); then
  echo "CUDA_VISIBLE_DEVICES must contain exactly ${gpus_per_node} GPU ids: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

python_bin="${K600_PYTHON:-$(command -v python)}"
torchrun_bin="${K600_TORCHRUN:-$(command -v torchrun)}"
if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable is unavailable: ${python_bin}" >&2
  exit 2
fi
if [[ ! -x "${torchrun_bin}" ]]; then
  echo "torchrun executable is unavailable: ${torchrun_bin}" >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VRAE_DISTRIBUTED_TIMEOUT_SECONDS="${VRAE_DISTRIBUTED_TIMEOUT_SECONDS:-1800}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-134217728}"
export JE_ARROW_MALLOC_CONF="${JE_ARROW_MALLOC_CONF:-background_thread:false}"
cache_root="${TMPDIR:-${project_root}/.cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${cache_root}/vrae_k600_${encoder_name}_inductor_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"
python_prefix="$("${python_bin}" -c 'import sys; print(sys.prefix)')"
export LD_LIBRARY_PATH="${python_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd -- "${project_root}"
config_metadata="$("${python_bin}" - "${config_path}" "${encoder_name}" "${paths_file}" <<'PY'
import sys
from pathlib import Path

from vrae.training.k600_videogen.train import validate_build
from vrae.config import load_config
from vrae.models.decoder import _FA3_MAX_HEAD_DIM, _fa3_training, _fa4
from vrae.paths import load_project_paths

config_path = Path(sys.argv[1])
expected_encoder = sys.argv[2]
config = load_config(config_path)
validate_build(config)

encoder = config["model"]["encoder"]
actual_encoder = str(encoder["name"])
if actual_encoder != expected_encoder:
    raise SystemExit(
        f"routed config encoder mismatch: expected={expected_encoder}, actual={actual_encoder}"
    )

paths_override = sys.argv[3] if Path(sys.argv[3]).is_file() else None
paths = load_project_paths(config, override=paths_override, project_root=config_path.parents[3])
stage1_path = paths.checkpoint(config["stage1"]["checkpoint"])
normalizer_path = paths.checkpoint(
    config["latent_normalizer"]["path"], require_exists=False
)
run_path = paths.training_run(config["task"], config["run_name"])


def text(value):
    if value is None or value == "":
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ",".join(text(item) for item in value)
    return str(value).replace("\t", " ").replace("\n", " ")


def emit(key, value):
    print("VALUE", key, text(value), sep="\t")


def kernel_available(loader):
    try:
        loader()
    except RuntimeError:
        return False
    return True


def attention_candidate(requested, precision, head_dim, fa3_available, fa4_available):
    if requested != "auto":
        return requested
    if precision in {"fp16", "bf16"} and fa4_available:
        return "fa4_cute"
    if precision == "bf16" and head_dim <= _FA3_MAX_HEAD_DIM and fa3_available:
        return "fa3"
    return "sdpa"


model = config["model"]
decoder = model["decoder"]
dit = config["dit"]
training = config["training"]
data = config["data"]
transport = config["transport"]
sampling = config["sampling"]
wandb = config["wandb"]
runtime = config.get("runtime") or {}
pipeline = runtime.get("data_pipeline") or {}
optimizer = training["optimizer"]
scheduler = training["scheduler"]

hidden_sizes = list(dit["hidden_size"])
num_heads = list(dit["num_heads"])
head_dims = [hidden // heads for hidden, heads in zip(hidden_sizes, num_heads, strict=True)]
attention_requested = str(dit.get("attention_backend", "auto")).lower()
precision = str(training.get("precision", "bf16")).lower()
fa3_available = kernel_available(_fa3_training)
fa4_available = kernel_available(_fa4)
attention_candidates = [
    attention_candidate(attention_requested, precision, dim, fa3_available, fa4_available)
    for dim in head_dims
]

image_size = data.get("image_size", 256)
if isinstance(image_size, int):
    resolution = f"{image_size}x{image_size}"
else:
    resolution = "x".join(str(item) for item in image_size)

normalizer_status = "ready" if normalizer_path.is_file() and normalizer_path.stat().st_size else "missing; computed automatically before training"
optimizer_settings = "; ".join(
    f"{key}={text(value)}" for key, value in optimizer.items() if key != "name"
)
scheduler_settings = "; ".join(
    f"{key}={text(value)}" for key, value in scheduler.items() if key != "name"
)

print(
    "META",
    config["run_name"],
    training["global_batch_size"],
    training.get("gradient_accumulation_steps", 1),
    sep="\t",
)
emit("task", config["task"])
emit("output", run_path.relative_to(paths.project_root))
emit("resume_init", f"resume={text(training.get('resume'))}; init_from={text(training.get('init_from'))}")
emit("stage1_encoder", f"{encoder['name']} ({encoder['variant']})")
emit("stage1_checkpoint", f"{config['stage1']['checkpoint']} (ready)")
emit("stage1_weights", config["stage1"].get("weights", "ema"))
emit("stage1_precision", config["stage1"].get("precision", "fp32"))
emit("stage1_decoder", f"mode={decoder['attention_mode']}; backend={decoder.get('attention_backend', 'auto')}")
emit("latent_normalizer", f"{config['latent_normalizer']['path']} ({normalizer_status})")
emit("dit_name", dit["name"])
emit("dit_dimensions", f"hidden={text(hidden_sizes)}; depth={text(dit['depth'])}; heads={text(num_heads)}")
emit("dit_head_dims", f"encoder={head_dims[0]}; decoder={head_dims[1]}")
emit("dit_context", f"base_model_depth={dit['encoder_depth']}; classes={dit['num_classes']}; class_dropout={text(dit['class_dropout'])}")
emit("dit_attention", f"requested={attention_requested}; encoder={attention_candidates[0]}; decoder={attention_candidates[1]}")
emit("flash_kernels", f"FA3={text(fa3_available)}; FA4_CuTe={text(fa4_available)}; SDPA=yes")
emit("gradient_checkpointing", dit.get("gradient_checkpointing", False))
emit("dataset", f"{data['dataset']} split={data['split']}")
emit("dataset_root", paths.dataset(data["dataset"], require_exists=False))
emit("manifest", data.get("manifest", "dataset discovery"))
emit("clip", f"frames={data['num_frames']}; interval={data.get('frame_interval', 1)}; resolution={resolution}")
emit("augmentation", f"crop={data.get('crop_mode', 'default')}; random_flip={text(data.get('random_flip', False))}; seed={data.get('seed', 3407)}")
emit("reader", f"backend={data.get('video_backend', 'auto')}; max_attempts={data.get('max_decode_attempts', 'default')}")
emit("pipeline", f"kind={pipeline.get('kind', 'torchcodec_cpu_bounded')}; decode_threads={pipeline.get('torchcodec_cpu_decode_threads', 'default')}; max_inflight={pipeline.get('torchcodec_cpu_max_inflight', 'default')}")
emit("transport", f"prediction={transport['prediction']}; distribution={transport['time_dist_type']}; shift={text(transport['time_dist_shift'])}")
emit("transport_bounds", f"t_eps={text(transport['t_eps'])}; base_model_coeff={text(transport['base_model_coeff'])}")
emit("epochs", training["epochs"])
emit("precision", precision)
emit("clip_grad", training.get("clip_grad", "none"))
emit("optimizer", f"{optimizer['name']}; {optimizer_settings}")
emit("scheduler", f"{scheduler['name']}; {scheduler_settings}")
emit("ema", training.get("ema", {}).get("decay", "off"))
emit("sample_solver", f"steps={sampling['steps']}; cfg={text(sampling['cfg_scale'])}; seed={sampling['base_seed']}")
emit("internal_guidance", f"scale={text(sampling.get('internal_guidance_scale', 1))}; window={text(sampling.get('internal_guidance_t_min', 0))}..{text(sampling.get('internal_guidance_t_max', 1))}")
emit("wandb_target", f"project={wandb['project']}; group={wandb['group']}; tags={text(wandb['tags'])}; config_mode={wandb['mode']}")
emit("wandb_metrics", f"every {wandb['log_interval']} steps")
emit("wandb_samples", f"every {wandb['sample_interval']} steps; count={wandb.get('sample_count', 4)}")
if training.get("checkpoint_interval_epochs") is not None:
    checkpoint_schedule = f"every {training['checkpoint_interval_epochs']} epoch(s)"
else:
    checkpoint_schedule = f"every {training.get('checkpoint_interval', 1000)} steps"
emit("checkpoint_schedule", checkpoint_schedule)
PY
)"

declare -A config_values=()
while IFS=$'\t' read -r record field1 field2 field3; do
  case "${record}" in
    META)
      run_name="${field1}"
      global_batch_size="${field2}"
      accumulation_steps="${field3}"
      ;;
    VALUE)
      config_values["${field1}"]="${field2}"
      ;;
  esac
done <<< "${config_metadata}"

if [[ ! "${global_batch_size}" =~ ^[1-9][0-9]*$ ]]; then
  echo "training.global_batch_size must be positive: ${global_batch_size}" >&2
  exit 2
fi
if [[ ! "${accumulation_steps}" =~ ^[1-9][0-9]*$ ]]; then
  echo "training.gradient_accumulation_steps must be positive: ${accumulation_steps}" >&2
  exit 2
fi
world_size=$((nnodes * gpus_per_node))
batch_denominator=$((world_size * accumulation_steps))
if (( global_batch_size % batch_denominator != 0 )); then
  echo "global batch ${global_batch_size} must be divisible by world size ${world_size} x accumulation ${accumulation_steps}" >&2
  exit 2
fi
local_micro_batch_size=$((global_batch_size / batch_denominator))

runtime_versions="$("${python_bin}" - <<'PY'
import torch
import torchcodec
import wandb
from torchcodec.decoders import VideoDecoder  # noqa: F401

print(
    f"torch={torch.__version__} torchcodec={torchcodec.__version__} "
    f"wandb={wandb.__version__} cuda={torch.version.cuda}"
)
PY
)"

launch_args=(
  "${torchrun_bin}"
  --nnodes="${nnodes}"
  --nproc-per-node="${gpus_per_node}"
  --node-rank="${node_rank}"
  --master-addr="${master_addr}"
  --master-port="${master_port}"
  -m vrae.training.k600_videogen.train
  --config "${config_path}"
)
launch_args+=("${paths_args[@]}")
launch_args+=("${trainer_args[@]}")

max_steps="none (full epoch schedule)"
build_only="no"
for ((argument_index = 0; argument_index < ${#trainer_args[@]}; argument_index++)); do
  argument="${trainer_args[argument_index]}"
  case "${argument}" in
    --max-steps)
      if (( argument_index + 1 < ${#trainer_args[@]} )); then
        max_steps="${trainer_args[argument_index + 1]}"
      fi
      ;;
    --max-steps=*)
      max_steps="${argument#*=}"
      ;;
    --build-only)
      build_only="yes"
      ;;
  esac
done

table_sections=()
table_parameters=()
table_values=()

add_table_row() {
  table_sections+=("$1")
  table_parameters+=("$2")
  table_values+=("$3")
}

add_table_row "Run" "Task" "${config_values[task]}"
add_table_row "Run" "VRAE route" "${vrae} -> ${encoder_name}"
add_table_row "Run" "Config" "${config_name}"
add_table_row "Run" "Run name" "${run_name}"
add_table_row "Run" "Output directory" "${config_values[output]}"
add_table_row "Run" "Resume / initialization" "${config_values[resume_init]}"
add_table_row "Run" "Build only" "${build_only}"

add_table_row "V-RAE" "Encoder" "${config_values[stage1_encoder]}"
add_table_row "V-RAE" "Checkpoint" "${config_values[stage1_checkpoint]}"
add_table_row "V-RAE" "Checkpoint weights" "${config_values[stage1_weights]}"
add_table_row "V-RAE" "Inference precision" "${config_values[stage1_precision]}"
add_table_row "V-RAE" "Decoder attention" "${config_values[stage1_decoder]}"
add_table_row "V-RAE" "Latent normalizer" "${config_values[latent_normalizer]}"

add_table_row "VideoDiT" "Model" "${config_values[dit_name]}"
add_table_row "VideoDiT" "Architecture" "${config_values[dit_dimensions]}"
add_table_row "VideoDiT" "Attention head dimensions" "${config_values[dit_head_dims]}"
add_table_row "VideoDiT" "Conditioning" "${config_values[dit_context]}"
add_table_row "VideoDiT" "Attention route" "${config_values[dit_attention]}"
add_table_row "VideoDiT" "Available attention kernels" "${config_values[flash_kernels]}"
add_table_row "VideoDiT" "Gradient checkpointing" "${config_values[gradient_checkpointing]}"

add_table_row "Data" "Dataset" "${config_values[dataset]}"
add_table_row "Data" "Dataset root" "${config_values[dataset_root]}"
add_table_row "Data" "Manifest" "${config_values[manifest]}"
add_table_row "Data" "Clip" "${config_values[clip]}"
add_table_row "Data" "Augmentation" "${config_values[augmentation]}"
add_table_row "Data" "Video reader" "${config_values[reader]}"
add_table_row "Data" "Pipeline" "${config_values[pipeline]}"

add_table_row "Flow" "Transport" "${config_values[transport]}"
add_table_row "Flow" "Bounds / base model" "${config_values[transport_bounds]}"

add_table_row "Optimization" "Epochs" "${config_values[epochs]}"
add_table_row "Optimization" "Max steps override" "${max_steps}"
add_table_row "Optimization" "Precision" "${config_values[precision]}"
add_table_row "Optimization" "Batch" "global=${global_batch_size}; local_micro=${local_micro_batch_size}; accumulation=${accumulation_steps}"
add_table_row "Optimization" "Gradient clipping" "${config_values[clip_grad]}"
add_table_row "Optimization" "Optimizer" "${config_values[optimizer]}"
add_table_row "Optimization" "Scheduler" "${config_values[scheduler]}"
add_table_row "Optimization" "EMA decay" "${config_values[ema]}"

add_table_row "Sampling" "Solver / CFG" "${config_values[sample_solver]}"
add_table_row "Sampling" "Internal guidance" "${config_values[internal_guidance]}"

add_table_row "Distributed" "Topology" "nodes=${nnodes}; GPUs/node=${gpus_per_node}; world_size=${world_size}"
add_table_row "Distributed" "Current node" "rank=${node_rank}; host=$(hostname)"
add_table_row "Distributed" "Rendezvous" "${master_addr}:${master_port}"
add_table_row "Distributed" "CUDA devices" "${CUDA_VISIBLE_DEVICES}"
add_table_row "Distributed" "DDP" "static_graph=yes; gradient_as_bucket_view=yes when world_size > 1"
add_table_row "Distributed" "Runtime controls" "OMP threads=${OMP_NUM_THREADS}; timeout=${VRAE_DISTRIBUTED_TIMEOUT_SECONDS}s"

add_table_row "Logging" "W&B requested mode" "${WANDB_MODE} (Python auto-fallback when no key)"
add_table_row "Logging" "W&B destination" "${config_values[wandb_target]}"
add_table_row "Logging" "Metric logging" "${config_values[wandb_metrics]}"
add_table_row "Logging" "Sample logging" "${config_values[wandb_samples]}"
add_table_row "Logging" "Checkpoints" "${config_values[checkpoint_schedule]}"

add_table_row "Runtime" "Software" "${runtime_versions}"

for index in "${!table_parameters[@]}"; do
  printf '%s\t%s\t%s\n' \
    "${table_sections[index]}" \
    "${table_parameters[index]}" \
    "${table_values[index]}"
done | "${python_bin}" -c '
import shutil
import sys

rows = []
for raw_line in sys.stdin:
    section, parameter, value = raw_line.rstrip("\n").split("\t", 2)
    rows.append((section, parameter, value))

try:
    from rich import box
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("\nK600 VideoGen training configuration")
    previous = None
    for section, parameter, value in rows:
        if section != previous:
            print(f"\n[{section}]")
            previous = section
        print(f"  {parameter}: {value}")
    raise SystemExit(0)

terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
console = Console(width=max(80, min(terminal_width, 120)))
table = Table(
    title="K600 VideoGen Training Configuration",
    title_style="bold bright_cyan",
    box=box.ROUNDED,
    border_style="bright_black",
    header_style="bold white",
    expand=True,
    padding=(0, 1),
)
table.add_column("Section", style="bold cyan", no_wrap=True, width=12)
table.add_column("Parameter", style="bold", no_wrap=True, width=27)
table.add_column("Value", ratio=1, overflow="fold")

previous = None
for section, parameter, value in rows:
    if previous is not None and section != previous:
        table.add_section()
    displayed_section = section if section != previous else ""
    table.add_row(
        Text(displayed_section, style="bold cyan"),
        Text(parameter, style="bold"),
        Text(value),
    )
    previous = section

console.print()
console.print(table)
'

if [[ "${K600_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

if [[ "${K600_BACKGROUND:-0}" == "1" ]]; then
  log_dir="${K600_LOG_DIR:-${project_root}/outputs/logs}"
  log_path="${K600_LOG_PATH:-${log_dir}/${run_name}_node${node_rank}_$(date -u +%Y%m%dT%H%M%SZ).log}"
  mkdir -p -- "$(dirname -- "${log_path}")"
  nohup "${launch_args[@]}" >"${log_path}" 2>&1 <&- &
  launcher_pid=$!
  echo "[launch] background pid=${launcher_pid}"
  echo "[launch] log=${log_path}"
  echo "[launch] follow with: tail -f ${log_path}"
  exit 0
fi

exec "${launch_args[@]}"
