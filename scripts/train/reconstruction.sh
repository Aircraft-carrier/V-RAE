#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train/reconstruction.sh \
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
  WANDB_MODE        Defaults to online; use offline or disabled explicitly if needed.
  RECON_DRY_RUN     Set to 1 to validate the configuration without launching.
  RECON_BACKGROUND  Set to 1 to launch with nohup and write one log per node.
  RECON_LOG_DIR     Background log directory (default: outputs/logs).
  RECON_PYTHON      Optional Python executable override.
  RECON_TORCHRUN    Optional torchrun executable override.

Examples:
  # One machine, eight GPUs.
  scripts/train/reconstruction.sh 1 8 0 dino

  # Two machines, eight GPUs each. Run one command on each machine.
  MASTER_ADDR=10.0.0.8 scripts/train/reconstruction.sh 2 8 0 vjepa
  MASTER_ADDR=10.0.0.8 scripts/train/reconstruction.sh 2 8 1 vjepa

  # Validate routing without starting workers.
  RECON_DRY_RUN=1 scripts/train/reconstruction.sh 1 8 0 eupe --max-steps 1000
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
    config_name="dinov3_vrae_training.yaml"
    encoder_name="dinov3"
    ;;
  siglip | siglip2)
    config_name="siglip2_vrae_training.yaml"
    encoder_name="siglip2"
    ;;
  vjepa | vjepa2 | vjepa2.1 | vjepa2_1)
    config_name="vjepa2_1_vrae_training.yaml"
    encoder_name="vjepa2_1"
    ;;
  eupe)
    config_name="eupe_vrae_training.yaml"
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
config_path="${project_root}/configs/training/recon_training/${config_name}"
paths_file="${VRAE_PATHS_FILE:-${project_root}/configs/paths.local.yaml}"
paths_args=()
if [[ -f "${paths_file}" ]]; then
  paths_args=(--paths "${paths_file}")
fi

if [[ ! -f "${config_path}" ]]; then
  echo "reconstruction config does not exist: ${config_path}" >&2
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

python_bin="${RECON_PYTHON:-$(command -v python)}"
torchrun_bin="${RECON_TORCHRUN:-$(command -v torchrun)}"
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
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${cache_root}/vrae_recon_${encoder_name}_inductor_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"
python_prefix="$("${python_bin}" -c 'import sys; print(sys.prefix)')"
export LD_LIBRARY_PATH="${python_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd -- "${project_root}"
config_metadata="$("${python_bin}" - "${config_path}" "${encoder_name}" <<'PY'
import sys

from vrae.config import load_config
from vrae.models.decoder import _fa3_training, _fa4

config = load_config(sys.argv[1])
expected_encoder = sys.argv[2]
actual_encoder = str(config["model"]["encoder"]["name"])
if actual_encoder != expected_encoder:
    raise SystemExit(
        f"routed config encoder mismatch: expected={expected_encoder}, actual={actual_encoder}"
    )


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


def enabled(value):
    return "on" if bool(value) else "off"


def emit(key, value):
    print("VALUE", key, text(value), sep="\t")


model = config["model"]
encoder = model["encoder"]
pooling = model["pooling"]
decoder = model["decoder"]
training = config["training"]
data = config["data"]
loss = config["loss"]
gan = config["gan"]
wandb = config["wandb"]
optimizer = training["optimizer"]
scheduler = training["scheduler"]
gan_optimizer = gan["optimizer"]


def kernel_available(loader):
    try:
        loader()
    except RuntimeError:
        return False
    return True


attention_requested = str(decoder.get("attention_backend", "auto"))
attention_precision = str(training.get("precision", "bf16")).lower()
attention_head_dim = int(decoder["hidden_size"]) // int(decoder["num_heads"])
fa3_available = kernel_available(_fa3_training)
fa4_available = kernel_available(_fa4)
if attention_requested != "auto":
    attention_candidate = attention_requested
elif attention_precision in {"fp16", "bf16"} and fa4_available:
    attention_candidate = "fa4_cute"
elif attention_precision == "bf16" and attention_head_dim <= 96 and fa3_available:
    attention_candidate = "fa3"
else:
    attention_candidate = "sdpa"
attention_names = {
    "fa3": "FlashAttention-3 (training)",
    "fa3_fwd": "FlashAttention-3 (forward-only; invalid for training)",
    "fa4_cute": "FlashAttention-4 CuTe",
    "sdpa": "PyTorch SDPA fallback",
}
attention_padding = (
    "D72 -> D96 with Q scaling"
    if attention_candidate == "fa4_cute" and attention_head_dim == 72
    else "none"
)

image_size = data["image_size"]
if isinstance(image_size, int):
    resolution = f"{image_size}x{image_size}"
else:
    resolution = "x".join(str(item) for item in image_size)

print(
    "META",
    config["run_name"],
    training["global_batch_size"],
    training.get("gradient_accumulation_steps", 1),
    sep="\t",
)
emit("task", config["task"])
emit("output", f"ckpts/{config['task']}/{config['run_name']}")
emit("resume_init", f"resume={text(training.get('resume'))}, init_from={text(training.get('init_from'))}")
emit("encoder", f"{encoder['name']} ({encoder['variant']})")
emit("encoder_features", f"layers={text(encoder['layers'])}; fusion={encoder['fusion']}")
emit(
    "encoder_geometry",
    f"patch={encoder['patch_size']}; tubelet={encoder.get('encoder_tubelet_size', 1)}; hidden={encoder['hidden_size']}",
)
emit(
    "pooling",
    f"{pooling['name']}; group={pooling['group_size']}; heads={pooling['num_heads']}",
)
emit(
    "decoder",
    f"hidden={decoder['hidden_size']}; depth={decoder['depth']}; heads={decoder['num_heads']}",
)
emit("decoder_geometry", f"patch={decoder['patch_size']}; tubelet={decoder['tubelet_size']}")
emit(
    "decoder_attention",
    f"mode={decoder['attention_mode']}; requested_backend={attention_requested}",
)
emit("attention_head_dim", attention_head_dim)
emit(
    "flash_attention",
    f"{attention_names[attention_candidate]}; auto candidate={attention_candidate}",
)
emit(
    "flash_kernels",
    f"FA3={text(fa3_available)}; FA4_CuTe={text(fa4_available)}; SDPA=yes",
)
emit("attention_padding", attention_padding)
emit("decoder_init", decoder["init"])
emit("datasets", " + ".join(str(source["name"]) for source in data["sources"]))
emit(
    "clip",
    f"frames={data['num_frames']}; interval={data['frame_interval']}; fps={data.get('fps', 'none')}",
)
emit("resolution", resolution)
emit("sampling", f"{data['sampling']}; random_flip={text(data.get('random_flip', False))}; seed={data.get('seed', 'none')}")
emit("reader", f"backend={data.get('video_backend', 'auto')}; decode_threads={data.get('decode_threads', 1)}")
emit(
    "loader",
    f"workers={training.get('num_workers', 4)}; prefetch={training.get('prefetch_factor', 'default')}; device_prefetch={enabled(training.get('prefetch_to_device', False))}",
)
emit("epochs", training["epochs"])
emit("precision", training.get("precision", "bf16"))
emit("optimizer", optimizer["name"])
emit(
    "optimizer_hparams",
    f"lr={text(optimizer['lr'])}; betas={text(optimizer.get('betas'))}; weight_decay={text(optimizer.get('weight_decay', 0))}",
)
emit("scheduler", f"{scheduler['name']}; warmup_steps={scheduler.get('warmup_steps', 0)}")
emit("ema", training.get("ema", {}).get("decay", "off"))
emit(
    "compile",
    f"encoder={enabled(training.get('compile_encoder', False))}; decoder={enabled(training.get('compile_decoder', False))}; loss={enabled(training.get('compile_reconstruction_loss', False))}",
)
noise = training.get("latent_noise", {})
emit("latent_noise", f"enabled={text(noise.get('enabled', False))}; tau={text(noise.get('tau', 0))}")
emit(
    "loss_weights",
    f"L1={text(loss.get('l1', 0))}; LPIPS={text(loss.get('lpips', 0))}; Gram={text(loss.get('gram', 0))}; temporal={text(loss.get('temporal_difference', 0))}",
)
emit(
    "perceptual_batching",
    f"frames_per_chunk={loss.get('perceptual_frames_per_chunk', 'none')}; chunk_size={loss.get('perceptual_chunk_size', 'none')}",
)
emit("gan", f"enabled={text(gan.get('enabled', False))}; type={gan.get('type', 'none')}")
emit(
    "gan_schedule",
    f"start_epoch={gan.get('generator_start_epoch', 'none')}; update_interval={gan.get('update_interval', 1)}",
)
emit("gan_losses", f"generator={gan.get('gen_loss_type', 'none')}; discriminator={gan.get('disc_loss_type', 'none')}")
emit(
    "gan_weights",
    f"generator={text(gan.get('generator_weight', 0))}; discriminator={text(gan.get('discriminator_weight', 0))}; LeCam={text(gan.get('lecam_weight', 0))}",
)
emit(
    "gan_optimizer",
    f"{gan_optimizer['name']}; lr={text(gan_optimizer['lr'])}; weight_decay={text(gan_optimizer.get('weight_decay', 0))}",
)
emit("diffaug", f"prob={text(gan.get('diff_aug_prob', 0))}; cutout={text(gan.get('diff_aug_cutout', 0))}")
emit(
    "ddp",
    f"static_graph={text(training.get('ddp_static_graph', False))}; grad_view={text(training.get('ddp_gradient_as_bucket_view', False))}; compression={training.get('ddp_gradient_compression', 'none')}",
)
emit("checkpoint_interval", training.get("checkpoint_interval", "none"))
emit("wandb_target", f"project={wandb['project']}; group={wandb['group']}; tags={text(wandb['tags'])}")
emit("wandb_metrics", f"every {wandb['log_interval']} steps")
emit("wandb_samples", f"every {wandb['sample_interval']} steps; count={wandb.get('sample_count', 4)}")
PY
)"
declare -A config_values=()
while IFS=$'\t' read -r record field1 field2 field3 field4; do
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
  -m vrae.training.recon_training.train
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

add_table_row "Model" "Encoder" "${config_values[encoder]}"
add_table_row "Model" "Encoder features" "${config_values[encoder_features]}"
add_table_row "Model" "Encoder geometry" "${config_values[encoder_geometry]}"
add_table_row "Model" "Temporal pooling" "${config_values[pooling]}"
add_table_row "Model" "Decoder" "${config_values[decoder]}"
add_table_row "Model" "Decoder geometry" "${config_values[decoder_geometry]}"
add_table_row "Model" "Decoder attention" "${config_values[decoder_attention]}"
add_table_row "Model" "Attention head dimension" "${config_values[attention_head_dim]}"
add_table_row "Model" "Flash Attention type" "${config_values[flash_attention]}"
add_table_row "Model" "Available attention kernels" "${config_values[flash_kernels]}"
add_table_row "Model" "Attention padding route" "${config_values[attention_padding]}"
add_table_row "Model" "Decoder initialization" "${config_values[decoder_init]}"

add_table_row "Data" "Datasets" "${config_values[datasets]}"
add_table_row "Data" "Clip" "${config_values[clip]}"
add_table_row "Data" "Resolution" "${config_values[resolution]}"
add_table_row "Data" "Sampling" "${config_values[sampling]}"
add_table_row "Data" "Video reader" "${config_values[reader]}"
add_table_row "Data" "Loader" "${config_values[loader]}"

add_table_row "Optimization" "Epochs" "${config_values[epochs]}"
add_table_row "Optimization" "Max steps override" "${max_steps}"
add_table_row "Optimization" "Precision" "${config_values[precision]}"
add_table_row "Optimization" "Batch" "global=${global_batch_size}; local_micro=${local_micro_batch_size}; accumulation=${accumulation_steps}"
add_table_row "Optimization" "Optimizer" "${config_values[optimizer]}"
add_table_row "Optimization" "Optimizer settings" "${config_values[optimizer_hparams]}"
add_table_row "Optimization" "Scheduler" "${config_values[scheduler]}"
add_table_row "Optimization" "EMA decay" "${config_values[ema]}"
add_table_row "Optimization" "Compile" "${config_values[compile]}"

add_table_row "Reconstruction" "Latent noise" "${config_values[latent_noise]}"
add_table_row "Reconstruction" "Loss weights" "${config_values[loss_weights]}"
add_table_row "Reconstruction" "Perceptual batching" "${config_values[perceptual_batching]}"

add_table_row "GAN" "Adversarial training" "${config_values[gan]}"
add_table_row "GAN" "Schedule" "${config_values[gan_schedule]}"
add_table_row "GAN" "Losses" "${config_values[gan_losses]}"
add_table_row "GAN" "Weights" "${config_values[gan_weights]}"
add_table_row "GAN" "Discriminator optimizer" "${config_values[gan_optimizer]}"
add_table_row "GAN" "DiffAug" "${config_values[diffaug]}"

add_table_row "Distributed" "Topology" "nodes=${nnodes}; GPUs/node=${gpus_per_node}; world_size=${world_size}"
add_table_row "Distributed" "Current node" "rank=${node_rank}; host=$(hostname)"
add_table_row "Distributed" "Rendezvous" "${master_addr}:${master_port}"
add_table_row "Distributed" "CUDA devices" "${CUDA_VISIBLE_DEVICES}"
add_table_row "Distributed" "DDP" "${config_values[ddp]}"
add_table_row "Distributed" "Runtime controls" "OMP threads=${OMP_NUM_THREADS}; timeout=${VRAE_DISTRIBUTED_TIMEOUT_SECONDS}s"

add_table_row "Logging" "W&B requested mode" "${WANDB_MODE} (Python auto-fallback when no key)"
add_table_row "Logging" "W&B destination" "${config_values[wandb_target]}"
add_table_row "Logging" "Metric logging" "${config_values[wandb_metrics]}"
add_table_row "Logging" "Sample logging" "${config_values[wandb_samples]}"
add_table_row "Logging" "Checkpoints" "every ${config_values[checkpoint_interval]} steps"

add_table_row "Runtime" "Software" "${runtime_versions}"

(( ${#runtime_versions} > 0 )) || {
  echo "runtime dependency validation returned no version information" >&2
  exit 2
}

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
    print("\nV-RAE reconstruction training configuration")
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
    title="V-RAE Reconstruction Training Configuration",
    title_style="bold bright_cyan",
    box=box.ROUNDED,
    border_style="bright_black",
    header_style="bold white",
    expand=True,
    padding=(0, 1),
)
table.add_column("Section", style="bold cyan", no_wrap=True, width=14)
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

if [[ "${RECON_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

if [[ "${RECON_BACKGROUND:-0}" == "1" ]]; then
  log_dir="${RECON_LOG_DIR:-${project_root}/outputs/logs}"
  log_path="${RECON_LOG_PATH:-${log_dir}/${run_name}_node${node_rank}_$(date -u +%Y%m%dT%H%M%SZ).log}"
  mkdir -p -- "$(dirname -- "${log_path}")"
  nohup "${launch_args[@]}" >"${log_path}" 2>&1 <&- &
  launcher_pid=$!
  echo "[launch] background pid=${launcher_pid}"
  echo "[launch] log=${log_path}"
  echo "[launch] follow with: tail -f ${log_path}"
  exit 0
fi

exec "${launch_args[@]}"
