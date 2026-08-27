# V-JEPA 2.1 VideoDiT Training Configuration

This document records the effective configuration used by the V-JEPA 2.1
class-conditional VideoDiT runs. The two checked-in recipes are:

- `configs/training/ucf_videogen/vjepa2_1.yaml`
- `configs/training/k600_videogen/vjepa2_1.yaml`

Both recipes use the same V-RAE/V-JEPA 2.1 Stage 1, DiT architecture, flow
matching objective, optimizer, and most data settings. Dataset-specific values
are listed together as `UCF101 / K600`.

| Identity field | UCF101 | K600 |
| --- | --- | --- |
| `task` | `ucf_videogen` | `k600_videogen` |
| `run_name` | `vjepa2_1_ucf101` | `vjepa2_1_k600` |
| Stage 1 model | `vrae` | `vrae` |

## 1. Training pipeline

The training path is:

```text
RGB video
  -> frozen V-JEPA 2.1 V-RAE Stage 1 encoder and temporal pool
  -> latent normalization
  -> clean latent tokens
  -> flow-matching interpolation (clean + Gaussian noise)
  -> VRAEVideoDiT(noisy, time, labels)
  -> x-prediction loss against the clean latent
  -> backward, gradient clipping, AdamW update, EMA update
```

The main loop is `train_class_conditional` in
`src/vrae/training/common/engine.py`. Flow-matching noise and loss construction
are implemented in `src/vrae/models/dit/transport.py`.

## 2. Effective tensor geometry

| Quantity | Value | Meaning |
| --- | ---: | --- |
| RGB frames | 20 | Number of frames sampled from each clip |
| Temporal compression | 4 | V-JEPA tubelet 2 × temporal pool group 2 |
| DiT chunks `T` | 5 | `num_frames / 4` |
| Input image size | 256 × 256 | Training crop size |
| V-JEPA patch size | 16 | Stage 1 encoder spatial patch |
| Latent grid | 16 × 16 | `256 / 16` |
| Spatial latent tokens `N` | 256 | `16 × 16` per temporal chunk |
| Latent channels `C` | 1024 | V-JEPA hidden size |
| DiT patch size | 1 × 1 | DiT does not merge latent tokens spatially |
| DiT sequence length | 1280 | `T × N = 5 × 256` |

For the single-view recipes, the DiT input and target have shape:

```text
[B, 5, 256, 1024]
```

`B` is the batch dimension. In a multiview configuration the corresponding
contract is `[B,T,V,N,C]`, but the V-JEPA 2.1 UCF101 and K600 recipes are
single-view.

## 3. Stage 1: V-RAE with V-JEPA 2.1 encoder

Stage 1 is frozen during DiT training. Its checkpoint is loaded by
`load_frozen_stage1`; the encoder and the trainable Stage 1 groups are switched
to evaluation/frozen mode before the DiT loop.

### 3.1 V-JEPA 2.1 encoder

From `configs/models/vrae_vjepa2_1_l_k7.yaml`:

| Parameter | Value | Purpose |
| --- | --- | --- |
| `encoder.name` | `vjepa2_1` | Selects the V-JEPA 2.1 encoder implementation |
| `encoder.variant` | `vit_l16` | Large ViT with 16-pixel spatial patches |
| `encoder.layers` | `[11, 13, 15, 17, 19, 21, 23]` | Intermediate layers/features fused into the latent representation |
| `encoder.fusion` | `mean_plus_final_spatial_mean` | Fusion rule for selected intermediate features and final spatial mean |
| `encoder.hidden_size` | `1024` | Latent channel width and DiT input channels |
| `encoder.num_blocks` | `24` | Number of V-JEPA transformer blocks |
| `encoder.patch_size` | `16` | Spatial patch size |
| `encoder.encoder_tubelet_size` | `2` | Temporal tubelet size |
| `encoder.pixel_normalization` | `imagenet` | Input pixel normalization convention |
| `encoder.image_size` | `[256, 256]` | Encoder training/input resolution |
| `encoder.checkpoint` | `ckpts/pretrained/encoders/vjepa2_1/model.pt` | Pretrained V-JEPA weights |

### 3.2 Temporal pool

| Parameter | Value | Purpose |
| --- | ---: | --- |
| `pooling.name` | `temporal_attention` | Attention-based temporal aggregation |
| `pooling.group_size` | `2` | Combines two encoder time positions; with tubelet 2 gives total compression 4 |
| `pooling.num_heads` | `16` | Attention heads in the temporal pool |
| `pooling.use_time_bias` | `true` | Adds a learned/parameterized temporal bias |
| `pooling.output_norm_affine` | `false` | Keeps the pool output normalization non-affine |

### 3.3 Stage 1 decoder and checkpoint selection

The decoder is needed to decode generated latents for samples, but it is not
optimized in the DiT run.

| Parameter | Value | Purpose |
| --- | --- | --- |
| `stage1.checkpoint` | `ckpts/vrae/vrae_vjepa2.1.pt` | V-RAE Stage 1 checkpoint |
| `stage1.weights` | `ema` | Uses the EMA weights from that checkpoint |
| `stage1.precision` | `bf16` | Autocast precision used by the frozen Stage 1 adapter |
| `model.name` | `vrae` | V-RAE Stage 1 model implementation |
| `model.decoder.name` | `vrae_decoder` | Stage 1 decoder implementation |
| `model.decoder.hidden_size` | `1152` | Stage 1 decoder width |
| `model.decoder.depth` | `28` | Stage 1 decoder block count |
| `model.decoder.num_heads` | `16` | Stage 1 decoder attention heads |
| `model.decoder.mlp_ratio` | `3.5555555555555554` | Decoder feed-forward expansion ratio |
| `model.decoder.patch_size` | `16` | Decoder image patch size |
| `model.decoder.tubelet_size` | `4` | Decoder temporal tubelet; required by V-RAE |
| `model.decoder.image_size` | `[256, 256]` | Decoder output resolution |
| `model.decoder.num_channels` | `3` | RGB output channels |
| `model.decoder.layer_norm_eps` | `1.0e-12` | Layer/RMS normalization numerical epsilon |
| `model.decoder.attention_dropout` | `0.0` | Decoder attention dropout |
| `model.decoder.attention_mode` | `full` | Full temporal/spatial decoder attention for this recipe |
| `model.decoder.attention_backend` | `auto` | Selects the available attention kernel |
| `model.decoder.rope_theta` | `10000.0` | RoPE frequency base |
| `model.decoder.gradient_checkpointing` | `false` | No activation checkpointing in Stage 1 decoder |
| `model.decoder.init` | `scratch` | Decoder initialization mode recorded in the model config |

`model.decoder.attention_mode=full` belongs to the V-RAE Stage 1 decoder. It is
separate from the VideoDiT attention settings below.

## 4. VideoDiT architecture

Base values come from `configs/models/dit_class_cond.yaml`; the training files
override the class count, attention backend, and gradient-checkpointing flag.

| Parameter | Value | Purpose |
| --- | --- | --- |
| `dit.name` | `vrae_video_dit` | DiT implementation |
| `dit.input_dim` | `1024` | Latent channel count; overridden from Stage 1 metadata at build time |
| `dit.hidden_size` | `[1536, 2048]` | Encoder and decoder hidden widths (`E`, `D`) |
| `dit.depth` | `[28, 2]` | 28 encoder blocks and 2 decoder blocks |
| `dit.encoder_depth` | `8` | Captures the base branch activation after encoder block 8; not the total encoder depth |
| `dit.num_heads` | `[24, 16]` | Encoder and decoder attention heads |
| `dit.mlp_ratio` | `4.0` | Feed-forward expansion ratio in DiT blocks |
| `dit.attention_dropout` | `0.0` (default) | Attention dropout in DiT blocks |
| `dit.time_embed_dim` | `256` | Gaussian Fourier time-feature width before projection |
| `dit.class_dropout` | `0.1` | Classifier-free condition dropout probability during training |
| `dit.num_classes` | `101 / 600` | Number of UCF101 or Kinetics-600 class labels |
| `dit.rope_theta` | `10000.0` | 3D RoPE frequency base |
| `dit.patch_size` | `1` (default) | DiT latent patch size; each latent token remains one patch |
| `dit.attention_backend` | `auto` | Chooses SDPA/FlashAttention according to device and dtype |
| `dit.gradient_checkpointing` | `false` | Explicitly disables DiT activation checkpointing in these recipes |
| `dit.multiview_enabled` | `false` (default) | Single-view training path |
| `dit.num_views` | `1` (default) | Number of views when multiview mode is enabled |
| `dit.num_streams` | `1` (default) | Number of valid stream IDs |
| `dit.use_view_embedding` | `true` (default) | Adds per-view embeddings in multiview mode; inactive for single-view |

The resulting DiT has 30 blocks total: 28 encoder blocks plus 2 decoder blocks.
The encoder uses width 1536 and 24 heads (head dimension 64); the decoder uses
width 2048 and 16 heads (head dimension 128).

## 5. Latent normalization

| Parameter | UCF101 | K600 | Purpose |
| --- | --- | --- | --- |
| `latent_normalizer.path` | `ckpts/ucf_videogen/latent_stats/vjepa2_1_ucf101_20f_published_ema_full_bf16.pt` | `ckpts/k600_videogen/latent_stats/vjepa2_1_k600_20f_vrae_ema_full_bf16.pt` | Per-channel latent mean/std used before DiT training |

The normalizer is computed or loaded before training. The DiT receives normalized
latent tokens; generated tokens are denormalized before Stage 1 decoding.

## 6. Flow-matching objective

| Parameter | Value | Purpose |
| --- | --- | --- |
| `transport.prediction` | `x` | The DiT predicts the clean latent `x`; transport converts it to velocity for the loss |
| `transport.time_dist_type` | `logit-normal_0_1` | Samples time from a logit-normal distribution with mean 0 and sigma 1 before the shift |
| `transport.time_dist_shift` | `17.88854381999832` | Shifts sampled times toward the training geometry; for this setup it is `sqrt(5*256*1024/4096)` |
| `transport.t_eps` | `0.05` | Lower clamp used when converting x-predictions to velocity near `t=0` |
| `transport.base_model_coeff` | `1.0` | Weight of the auxiliary base-model loss |

For each batch, the transport samples Gaussian noise with the same shape as the
clean tokens and constructs:

```text
noisy = (1 - t) * clean + t * noise
```

With `prediction=x`, the model output is converted as
`(noisy - predicted_clean) / max(t, t_eps)`. The total loss is:

```text
loss = loss_full + base_model_coeff * loss_base
```

## 7. Optimization and training schedule

| Parameter | UCF101 | K600 | Purpose |
| --- | ---: | ---: | --- |
| `training.resume` | `null` | `null` | Existing run checkpoint to resume from |
| `training.init_from` | `null` | `null` | DiT initialization checkpoint; mutually exclusive with resume |
| `training.global_batch_size` | `64` | `128` | Batch size across all distributed ranks |
| `training.gradient_accumulation_steps` | `1` | `1` | Microbatches accumulated per optimizer update |
| `training.clip_grad` | `1.0` | `1.0` | Maximum global gradient norm |
| `training.epochs` | `3000` | `150` | Number of dataset passes |
| checkpoint interval | `checkpoint_interval: 1000` steps | `checkpoint_interval_epochs: 1` epoch | Checkpoint save frequency |
| `training.precision` | `bf16` | `bf16` | Autocast dtype for the trainable DiT |
| `training.ema.decay` | `0.9995` | `0.9995` | EMA decay used for evaluation/sampling |

### AdamW

| Parameter | Value | Purpose |
| --- | --- | --- |
| `optimizer.name` | `adamw` | Optimizer implementation |
| `optimizer.lr` | `1.0e-4` | Initial/base learning rate |
| `optimizer.betas` | `[0.9, 0.95]` | First- and second-moment decay rates |
| `optimizer.weight_decay` | `0.0` | L2-style decoupled weight decay; disabled |
| `optimizer.eps` | `1.0e-8` | Numerical stability term |

The fused AdamW option is not specified in these YAML files. The optimizer
builder enables fused AdamW automatically when all trainable parameters are on
CUDA, and otherwise disables it.

### Linear learning-rate schedule

| Parameter | UCF101 | K600 | Purpose |
| --- | ---: | ---: | --- |
| `scheduler.name` | `linear` | `linear` | Linear warmup followed by linear decay |
| `scheduler.warmup_epochs` | `10` | `2` | Warmup duration, converted to optimizer steps using `steps_per_epoch` |
| `scheduler.decay_end_epoch` | `3000` | `150` | Step at which the schedule reaches `final_lr` |
| `scheduler.base_lr` | `1.0e-4` | `1.0e-4` | Peak/base scheduled learning rate |
| `scheduler.final_lr` | `5.0e-5` | `5.0e-5` | Learning rate at the end of decay |
| `scheduler.warmup_from_zero` | `true` | `true` | Starts warmup at zero and increases to `base_lr` |

`warmup_steps = warmup_epochs × steps_per_epoch`. `steps_per_epoch` depends on
the dataset size, distributed world size, global batch size, and accumulation.

The environment variable `VRAE_GRADIENT_CHECKPOINTING` can override the DiT
gradient-checkpointing setting at runtime. If it is unset, the YAML value
(`false`) is used.

## 8. Dataset and augmentation

| Parameter | UCF101 | K600 | Purpose |
| --- | --- | --- | --- |
| `data.dataset` | `ucf101` | `k600` | Dataset implementation |
| `data.split` | `train` | `train` | Training split |
| `data.manifest` | `data/metadata/ucf101_train_split1.csv` | `data/metadata/k600_train.csv` | Labeled video manifest |
| `data.manifest_scope` | `project` | `project` | Resolves the relative manifest path from the project root |
| `data.manifest_split` | `null` | `null` | Do not apply an additional manifest split filter |
| `data.num_frames` | `20` | `20` | Frames returned per training clip |
| `data.frame_interval` | `3` | `3` | Source-frame stride between sampled frames |
| `data.image_size` | `256` | `256` | Short-side resize/crop target |
| `data.video_backend` | `torchcodec` | `torchcodec` | Video decoding backend |
| `data.max_decode_attempts` | `128` | `128` | Maximum fallback video candidates for a failed sample |
| `data.random_flip` | `true` | `true` | Random horizontal flip with probability 0.5 |
| `data.crop_mode` | `random` | `random` | Random spatial crop after short-side resize |
| `data.seed` | `3407` | `3407` | Base seed for dataset sampling and training RNGs |

The effective preprocessing is: decode 20 frames with stride 3, optionally flip,
resize the short side to 256, and take a 256×256 random crop.

## 9. Runtime and data-loader controls

These values affect throughput, memory, and fault tolerance rather than the
mathematical DiT objective.

| Parameter | UCF101 | K600 | Purpose |
| --- | ---: | ---: | --- |
| `runtime.data_pipeline.kind` | `torchcodec_cpu_bounded` | `torchcodec_cpu_bounded` | Bounded CPU decode/prefetch pipeline |
| `torchcodec_seek_mode` | `approximate` | `approximate` | Approximate seek for faster clip access |
| `torchcodec_num_ffmpeg_threads` | `1` | `1` | FFmpeg threads per decode task |
| `torchcodec_cpu_decode_threads` | `8` | `16` | Rank-local decode worker threads |
| `torchcodec_cpu_max_inflight` | `32` | `128` | Maximum decode tasks in flight |
| `torchcodec_cpu_max_buffered_batches` | `4` | `16` | Maximum completed batches held in the bounded loader |
| `torchcodec_cpu_async_prefetch_batches` | `2` | `8` | Producer-thread prefetch depth |
| `torchcodec_cpu_max_decode_attempts_per_batch` | `2048` | `2048` | Batch-level decode retry budget |
| `torchcodec_cpu_pin_memory` | `true` | `true` | Pin CPU batches for faster host-to-device transfer |
| `torchcodec_cpu_glibc_arena_max` | `2` | `2` | Limits glibc allocator arenas |
| `torchcodec_cpu_glibc_trim_threshold_bytes` | `134217728` | `134217728` | Heap trim threshold (128 MiB) |
| `torchcodec_cpu_trim_heap_each_epoch` | `true` | `true` | Trim process heap at epoch boundaries |
| `torchcodec_cpu_collect_python_each_epoch` | `false` | `false` | Disable forced Python GC at epoch boundaries |
| `runtime.checkpoint.mmap` | `true` | `true` | Memory-map checkpoint reads where supported |
| `runtime.checkpoint.drop_page_cache` | `true` | `true` | Drop checkpoint page cache after saving |
| `runtime.host_memory.log` | `true` | `true` | Log host-memory metrics |
| `runtime.host_memory.min_available_gb` | `128` | `128` | Stop and checkpoint if available host memory falls below 128 GiB |

## 10. Sampling during training

Sampling is performed with the EMA DiT at each configured sample interval. The
sampled latent is denormalized and decoded by the frozen Stage 1 model.

| Parameter | UCF101 | K600 | Purpose |
| --- | ---: | ---: | --- |
| `sampling.base_seed` | `3407` | `3407` | Deterministic seed base for sample IDs |
| `sampling.steps` | `100` | `100` | Euler flow-sampling steps |
| `sampling.cfg_scale` | `1.0` | `1.0` | Classifier-free guidance scale; 1.0 disables extra CFG amplification |
| `sampling.internal_guidance_scale` | `1.3` | `1.2` | Guidance scale between full and base DiT outputs |
| `sampling.internal_guidance_t_min` | `0.10` | `0.10` | Start of the internal-guidance time interval |
| `sampling.internal_guidance_t_max` | `1.0` | `1.0` | End of the internal-guidance time interval |

## 11. Weights & Biases logging

| Parameter | UCF101 | K600 | Purpose |
| --- | --- | --- | --- |
| `wandb.project` | `V-RAE` | `V-RAE` | W&B project |
| `wandb.group` | `ucf101_videogen` | `k600_videogen` | Experiment group |
| `wandb.tags` | `[ucf101, class_conditional, vjepa2_1, 20f]` | `[k600, class_conditional, vjepa2_1, 20f, ema, full_attention]` | Run metadata tags |
| `wandb.mode` | `online` | `online` | Online logging when credentials are available |
| `wandb.resume` | `never` | `never` | Start a new W&B run unless exact checkpoint resume logic overrides it |
| `wandb.log_interval` | `20` | `20` | Log scalar metrics every 20 optimizer steps |
| `wandb.sample_interval` | `1000` | `1000` | Generate samples every 1000 optimizer steps |
| `wandb.sample_count` | `4` | `4` | Number of videos generated per sampling event |

## 12. UCF101 versus K600 at a glance

| Category | UCF101 | K600 |
| --- | --- | --- |
| Classes | 101 | 600 |
| Global batch | 64 | 128 |
| Epochs | 3000 | 150 |
| Checkpoint cadence | Every 1000 steps | Every epoch |
| Warmup | 10 epochs | 2 epochs |
| Decode threads | 8 | 16 |
| In-flight decode tasks | 32 | 128 |
| Buffered batches | 4 | 16 |
| Async prefetch batches | 2 | 8 |
| Internal guidance scale | 1.3 | 1.2 |

All other values are shared unless explicitly noted above.

## 13. Launch commands

```bash
python -m vrae.training.ucf_videogen.train \
    --config configs/training/ucf_videogen/vjepa2_1.yaml

python -m vrae.training.k600_videogen.train \
    --config configs/training/k600_videogen/vjepa2_1.yaml
```

Use `--build-only` to validate and print the formal build summary without
starting training. Use `--max-steps` for a bounded smoke run.
