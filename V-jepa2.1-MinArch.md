# V-JEPA 2.1 最小训练架构改造计划

## 0. 结论与范围

当前仓库不是在继续预训练 V-JEPA 2.1 本体，而是把官方 V-JEPA 2.1 当作冻结的视觉编码器，训练 V-RAE 的 temporal pool 和 video decoder；UCF101/K600 的 VideoDiT 是再下游的生成训练。建议将“最小训练代码”定义为：

```text
视频文件/manifest
  -> 通用视频数据集与采样
  -> 冻结 V-JEPA 2.1 ViT-L/16
  -> temporal_attention pool (group_size=2)
  -> V-RAE video decoder (tubelet_size=4)
  -> reconstruction loss (+ 可选 VideoMAE GAN)
  -> DDP / AMP / optimizer / EMA / checkpoint
```

本计划默认只保留上述 Stage 1 重建训练。UCF101/K600 VideoDiT、Cityscapes future prediction、评测和推理属于下游或非训练代码，默认移出最小架构；如果仍需要生成训练，在执行删除阶段前按第 7 节的可选分支保留它们。

现有 `test/data/load_libero.py` 已处于用户删除状态，计划不恢复、不修改；所有其它未提交改动也先保持原样。

## 1. 当前结构审计

### 1.1 编码器与注册链

- `src/vrae/models/encoders/vjepa2_1.py`：唯一目标编码器，加载 `ema_encoder`，固定 ViT-L/16、24 blocks、tubelet=2、K7 layers、1024 hidden。
- `src/vrae/models/encoders/base.py`：当前四个 adapter 共用的校验、checkpoint、预处理和输出 shape 契约；V-JEPA 仍依赖它，第一阶段保留并只做 V-JEPA 需要的精简。
- `dinov3.py`、`siglip2.py`、`eupe.py`：与目标无关，应删除。
- `registry.py`、`encoders/__init__.py`、`config.py`、`paths.py` 目前都显式列出四种 encoder，需要收敛为仅 `vjepa2_1`。

### 1.2 Stage 1 训练链

保留并作为最小核心：

- `src/vrae/models/{autoencoder,pooling,decoder,rope3d,adapter}.py`
- `src/vrae/training/recon_training/{train,data,losses,noise,sample}.py`
- `src/vrae/training/common/` 中被重建训练实际使用的 distributed、checkpoint、contracts、ema、optim、precision、sampler、bounded_loader、memory、visualization、wandb、accumulation、engine 功能。
- `src/vrae/data/{video_reader,sampling,transforms,datasets}.py` 中的通用视频读取、clip 采样、变换、manifest 解析。

当前重建入口 `scripts/train/reconstruction.sh` 通过第四个参数选择 dino/siglip/vjepa/eupe；精简后应固定为 V-JEPA 2.1，不再存在 variant 路由。

### 1.3 非目标训练/评测链

默认不纳入最小训练树：

- `src/vrae/training/ucf_videogen/`、`src/vrae/training/k600_videogen/`
- `src/vrae/training/cityscapes_video_pred/`
- `src/vrae/training/recon_training/covla/` 及 `scripts/train/covla_reconstruction.sh`
- LeRobot 专用路径：`src/vrae/data/lerobot.py`、`scripts/train/lerobot_multiview.sh`、`configs/training/recon_training/lerobot_multiview.yaml`（除非明确需要多视角）
- `src/vrae/evaluation/`、`scripts/eval/`、`scripts/download_eval_weights.sh`
- `sampling.py` 中 dino/siglip/eupe 变体和它们的 checkpoint 映射；若保留示例，只保留 `vjepa`。

## 2. 编码器精简方案

1. 删除 `src/vrae/models/encoders/dinov3.py`、`siglip2.py`、`eupe.py`。
2. `src/vrae/models/encoders/__init__.py` 只导出 `VJEPA21Adapter`；是否导出 `EncoderAdapter/EncoderSpec` 由内部测试需要决定，不在公开 API 中暴露其它实现。
3. `registry.register_builtin_models()` 只导入 V-JEPA、VRAE、pooler、decoder 和仍保留的训练模型；删除三个无关 adapter 的 import。
4. `config.validate_config()`：允许 encoder 集合改为 `{"vjepa2_1"}`，固定检查 tubelet=2、pool group=2；删除针对其它 encoder 的分支和错误信息。
5. `paths.py` 和 `configs/paths.example.yaml` 删除 `dinov3`、`eupe` 的 third-party 默认路径，保留 `vjepa2_1`；若采用更通用模板，第三方路径命名统一为 `third_party.vjepa2_1`。
6. `src/vrae/models/encoders/base.py` 不立即删除：它承载 V-JEPA 的输入验证、ImageNet 归一化、checkpoint 读取和 token-to-grid。删除三个 adapter 后再清理其中仅为其它 adapter 服务的分支；不要为了“看起来更小”重复实现一份 V-JEPA loader。
7. `third_party/dinov3/`、`third_party/eupe/` 及其许可证说明、encoder checkpoint 下载说明在确认没有外部引用后删除；保留完整的 `third_party/vjepa2/`（含其 license 和 V-JEPA 所需 `app/`、`src/masks/` 子集）。

## 3. Stage 1 训练入口精简

1. 将 `scripts/train/reconstruction.sh` 改为单一 V-JEPA 入口，例如：

   ```text
   scripts/train/reconstruction.sh <nnodes> <gpus_per_node> <node_rank> [train_args...]
   ```

   删除 case 路由、encoder 参数、按 encoder 命名的 cache 和所有 dino/siglip/eupe 文案；保留多机、dry-run、日志、CUDA/NCCL 环境和 `torchrun` 启动逻辑。

2. 配置固定加载一个 V-JEPA Stage 1 recipe；建议文件名为 `configs/training/vjepa2_1_reconstruction.yaml`，旧文件迁移后不保留两份事实来源。
3. `recon_training/train.py` 的训练契约固定检查 `model.encoder.name == vjepa2_1`，保留 decoder/pool、重建 loss、EMA、resume/init_from、DDP 和 checkpoint metadata。
4. GAN 视为可选训练组件而非 encoder 依赖：保留 `gan.py` 只有在默认模板仍启用 VideoMAE GAN 时；若目标是最小可运行闭环，模板默认 `gan.enabled: false`，并让 GAN 相关 import 延迟加载。这样基础训练不需要 VideoMAE discriminator 权重。
5. 保留 `sample.py` 和 visualization，因为当前训练循环会在 sample interval 生成重建样本；删除只服务于 eval/generation 的采样代码。
6. 从 `training/common/engine.py` 中移除 class-conditional、latent-normalizer、Cityscapes 和多视角专用分支后，再根据静态引用结果删除孤立函数；不要先凭文件名删除共享 checkpoint/DDP 工具。

## 4. 通用数据集与训练配置模板

### 4.1 数据接口改造

当前重建数据代码把数据源硬编码为 `ucf101`/`k600`，并通过 `sources` 拼接两套数据。改成单一通用 manifest 契约：

```yaml
data:
  dataset_name: my_dataset       # 仅用于日志，不参与模型分支
  root: /absolute/or/project/path
  manifest: data/metadata/train.csv
  manifest_scope: project        # project 或 dataset；绝对路径也可
  split: train                    # null 表示不过滤
  num_frames: 16
  frame_interval: 3
  image_size: 256
  sampling: random
  random_flip: true
  video_backend: torchcodec
  decode_threads: 1
  max_decode_attempts: 128
  seed: 3407
```

实施要点：

- `build_reconstruction_dataset()` 直接构造 `ManifestVideoDataset`/`VideoDataset`，不再按数据集名称分派 UCF/K600 类。
- manifest 继续支持当前 CSV/TSV/JSONL/JSON/文本格式；重建训练忽略 label，但保留它以便日志和未来下游任务复用。
- `root + manifest + split` 是唯一数据定位方式；不再默认同时读取 UCF101 和 K600，也不在 loader 中写入数据集专用路径。
- 保留现有 clip 长度必须被 4 整除、RGB、[0,1]、resize/crop、坏视频 fallback、bounded loader 和 pinned-memory 行为。
- 若未来需要 LeRobot，多视角应作为独立扩展，不混入这个最小单视角模板。

### 4.2 模板分层

新增一个可复制的模板（建议 `configs/training/templates/vjepa2_1_reconstruction.yaml`），其中路径、run name、数据集名称使用明显占位值；仓库内可运行的示例只保留一份 V-JEPA 配置。模板包含：

- V-JEPA encoder 的固定结构和 `ckpts/pretrained/encoders/vjepa2_1/model.pt`；
- pool group=2、decoder tubelet=4、256px/16-frame 的默认几何；
- 通用 optimizer/scheduler/precision/EMA/checkpoint/dataloader 字段；
- loss 权重和 perceptual checkpoint 作为显式字段，GAN 默认关闭或明确标注为可选；
- `data.root`、`data.manifest`、`data.split` 等可替换字段，不出现 `ucf101_train_split1.csv`、`k600_train.csv` 之类数据集专名。

如果决定保留 UCF/K600 VideoDiT（第 7 节），再新增一个通用 `videogen` 模板，把 `dataset_name`、`num_classes`、manifest、latent stats 和 stage1 checkpoint 作为数据集覆盖项；UCF/K600 文件只能是薄 overlay，不能复制整份训练 YAML。

## 5. 配置、文档和依赖清理

### 5.1 配置删除/保留

- 保留 `configs/models/vrae_vjepa2_1_l_k7.yaml`，必要时改名为 `vrae_vjepa2_1.yaml`；保留 `decoder_vit_xl.yaml`。
- 删除 `configs/models/vrae_dinov3_l_k7.yaml`、`vrae_siglip2_l_k7.yaml`、`vrae_eupe_b_k7.yaml`。
- 删除 recon 目录中的 dino/siglip/eupe/CoVLA/LeRobot recipe，只保留通用 V-JEPA recipe。
- 默认删除 `configs/training/ucf_videogen/`、`k600_videogen/`、`cityscapes_video_pred/`；若选择保留下游训练，按第 7 节迁移为模板+overlay。
- 删除所有其它 encoder 的 evaluation config；若评测仍需用，只保留 V-JEPA 版本并去掉 variant 路由。

### 5.2 文档与示例

- README 的安装、路径、训练命令改成单一 V-JEPA 流程；删除四变体表格和命令。
- `third_party/README.md` 只记录 V-JEPA source/checkpoint、通用 perceptual weights 和（若启用）VideoMAE GAN 权重。
- `sampling.py` 只接受 `vjepa` 或改成直接读取单一 checkpoint；删除其它 variant 名称。
- `V-jepa2.1-train-cfg.md` 改写为通用模板说明，数据集差异只放在“如何填 manifest”的示例，不把 UCF/K600 数值当作固定契约。

### 5.3 Python 依赖

以删除后的 import 图为准重新裁剪 `pyproject.toml`，不要仅按包名猜测：

- V-JEPA Stage 1 必需：PyTorch、TorchVision（VGG perceptual loss 若保留）、TorchCodec/FFmpeg、YAML、NumPy、W&B/Rich（若保留日志）。
- `transformers` 只有在 VideoMAE GAN 或其它保留模块仍引用时才保留。
- `timm`、`huggingface-hub`、`safetensors` 等逐项通过 `rg` 和干净环境 import 验证后再删；`decord` 若保留 video_reader 的 fallback 则不能删。

## 6. 执行顺序

1. 先把当前 V-JEPA 配置加载、静态 import 和已有 dry-run 结果记录为基线。
2. 删除三个 adapter 和第三方目录，收敛 registry/config/paths/API 导出；此时先让“只导入模型”通过。
3. 固定 reconstruction launcher 与 trainer，保留 checkpoint/DDP/EMA 行为。
4. 把数据 loader 从 UCF/K600 sources 改成通用 manifest/root，新增模板并删除旧数据集专用 YAML。
5. 根据 `rg` 结果删除下游训练、评测、推理和孤立依赖；同步 README、third-party 文档、sampling 示例。
6. 最后裁剪 Python 依赖和无效的 checkpoint/path 文档；不要删除用户的 `ckpts/`、外部数据或已有运行产物。

## 7. 可选：保留 VideoDiT 下游训练时的边界

如果“V-JEPA 2.1 训练”实际包含用 V-JEPA latent 训练 UCF101/K600 VideoDiT，则不要删除：

- `src/vrae/models/dit/{video_dit,blocks,conditioning,transport,guidance}.py`；
- `src/vrae/training/common/engine.py` 中 `train_class_conditional`、latent stats/normalizer、DiT checkpoint 逻辑；
- `src/vrae/training/ucf_videogen/`、`k600_videogen/` 入口和一个通用 videogen 配置模板；
- V-JEPA 对应的 UCF/K600 manifest 与 latent-stat 生成脚本。

仍然删除所有 dino/siglip/eupe 变体，并把两个数据集文件改为同一模板的最小覆盖（类数、manifest、run name、latent stats、训练步数）。Cityscapes future prediction、CoVLA 和 LeRobot 仍不属于这条下游链，除非另行确认。

## 8. 验收标准

### 静态与配置

- `rg` 全仓库不再出现 `dinov3`、`siglip2`、`eupe` 的运行时 import、registry key、训练路由或默认路径；历史说明若保留必须明确为迁移记录。
- `register_builtin_models()` 的 encoder keys 恰为 `("vjepa2_1",)`；未知 encoder 配置被拒绝。
- 通用模板可被 `load_config()` 解析；填入任意 root/manifest 后不需要修改 Python 代码即可构建数据集。
- `pyproject.toml` 中每个依赖都能在干净环境被实际 import 或由保留功能解释。

### 单元/冒烟

- 用注入的轻量 stub backbone 验证 VJEPA21Adapter 输出为 `[B,T/2,1024,H/16,W/16]`，时间压缩和 pool 后为 `T/4`。
- 用小型本地 manifest 和短视频验证 dataset、随机采样、resize/crop、collate、坏样本 fallback 与 bounded loader。
- `scripts/train/reconstruction.sh ... --build-only`（或等价 Python 入口）只能走 V-JEPA 配置；无 dino/siglip/eupe 参数可传入。
- 单卡 `--max-steps 1` 验证前向、重建 loss、反向、EMA、checkpoint 保存和 reload；有 GAN 时另做显式开启测试。
- checkpoint metadata 至少记录 encoder name/variant、hidden size、patch/tubelet、pool group、decoder geometry、数据模板关键字段和配置版本。

### 行为不变与删除边界

- V-JEPA checkpoint 的 strict `ema_encoder` 加载行为不变。
- 输入 `[B,T,3,H,W]`、`T % 4 == 0`、RGB [0,1]、256px 几何和 decoder tubelet=4 的核心契约不变。
- 不删除 `ckpts/`、数据集、运行输出或用户已有 checkpoint；只删除仓库内确认无引用的源代码、配置、第三方 vendored source 和文档。

## 9. 需要在实施前确认的一点

请确认“最小训练”是否只指 Stage 1 V-RAE 重建，还是也包括 UCF101/K600 VideoDiT。若没有额外要求，按本文推荐的 Stage 1-only 方案执行；这会显著减少 DiT、Cityscapes、评测和数据集专用代码，但不会影响 V-JEPA 2.1 encoder + V-RAE 的训练闭环。
