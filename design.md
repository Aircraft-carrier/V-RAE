# V-RAE 单视角到 LeRobot 多视角：精简实现设计

## 1. 目标与不变边界

目标：同一套代码通过配置同时支持单视角和 LeRobot 多视角训练。

```text
单视角配置：沿用当前 5D 输入、模型参数和 checkpoint 行为
多视角配置：LeRobot 读取 V 个同步相机，模型联合生成 V 路视频
```

不新增多视角专用类名。多视角逻辑直接写入现有类和函数的配置分支：

- `LeRobotVideoDataset`：从多个 camera key 读取一个 episode。
- `VRAE`：在 encoder 前后处理 view 维。
- `VRAEDecoder`：仅将 view 维展平为 batch 维，复用原单视角 decoder；视角之间互不影响。
- `VRAELatentAdapter`、`VRAEVideoDiT`：保留 view 维并联合处理 token。
- `ReconstructionLoss`、GAN、训练循环：在现有入口增加 6D/多视角分支。

`multiview.enabled=false` 时不创建新增参数、不改变原有单视角路径；UCF101/K600
数据集和 launcher 的语义保持不变。

## 2. 统一数据契约

### 2.1 张量布局

```text
单视角 RGB       [B,T,3,H,W]
多视角 RGB       [B,T,V,3,H,W]
stream_ids       [B,V] (torch.long)
Stage 1 latent   [B,L,V,C,H_e,W_e], L=T/4
Stage 1 recon    [B,T,V,3,H,W]
Stage 2 tokens   [B,L,V,N,C], N=H_e*W_e
```

`V` 在一个 run 内固定；所有 view 必须共享 `T/H/W`。`stream_id` 是稳定语义 ID，
view 轴位置只用于排列和 reshape。

### 2.2 配置

```yaml
model:
  multiview:
    enabled: true
    num_views: 2
    num_streams: 2
    use_view_embedding: true
    use_view_attention: true

data:
  dataset: lerobot
  root: null                 # 默认 paths.datasets.lerobot
  repo_id: libero
  camera_keys:
    - {key: observation.images.image,  name: head,  stream_id: 0}
    - {key: observation.images.image2, name: wrist, stream_id: 1}
  num_frames: 16
  frame_interval: 1
  sampling: random
  image_size: 256
  random_flip: false
```

启动时校验 `num_views == len(camera_keys)`、key/ID 不重复、ID 小于 `num_streams`。
camera key 列表重排只能改变 view 轴顺序，不能重新编号 `stream_id`。

## 3. LeRobot 数据集改造

只扩展现有 `LeRobotVideoDataset`，保留单视角默认行为。

### 3.1 episode 与同步

1. 继续使用 `LeRobotDataset(repo_id, root=..., download_videos=False)` 和
   `meta.episodes` 的 `dataset_from_index/dataset_to_index`。
2. 过滤长度小于 `1+(num_frames-1)*frame_interval` 的 episode。
3. 检查所有 camera key 存在于 `meta.features` 且为 RGB image feature。
4. `ClipSampler` 只生成一组 row index；每个 row 一次读取全部 camera key。
5. 检查 view 间通道、分辨率一致，转换到 `float32 [0,1]`。
6. `fps` 使用 `dataset.meta.fps`；state/action 使用相同 frame index，不复制到 V 份。

LeRobot row index 作为同步前提。若数据只有 timestamp，必须在 dataset 层先对齐，
模型不处理异步帧。

### 3.2 item 与 collate

多视角 item：

```python
{
    "video": FloatTensor[T,V,3,H,W],
    "stream_ids": LongTensor[V],
    "label": int,                 # task_index；无条件时使用 null task ID
    "sample_id": str,
    "frame_indices": LongTensor[T],
    "video_metadata": {"fps": ..., "num_views": V, ...},
    "extra": {"task": ..., "state": ..., "action": ...},
}
```

多视角 collate 输出 `video=[B,T,V,3,H,W]`、`stream_ids=[B,V]`，其余字段沿用当前
重建 collate。第一版不支持动态 V 或 padding。

所有 view 使用同一 resize/crop 参数和 flip 决策；`random_flip` 默认关闭，左右相机
只有在定义语义映射后才能开启。

## 4. Stage 1：在现有 VRAE/Decoder 中增加分支

### 4.1 `VRAE`

`VRAE.from_config` 读取 `model.multiview`，只保存开关和尺寸，并把配置传给
`VRAEDecoder`；Decoder 不创建 view embedding，也不执行跨视角 attention。

`VRAE.encode(video, stream_ids=None)`：

```python
if not self.multiview_enabled:
    return current_single_view_encode(video)

# video [B,T,V,3,H,W]
x = video.permute(0, 2, 1, 3, 4, 5).reshape(B * V, T, 3, H, W)
z = current_single_view_encode(x)          # [B*V,L,C,H_e,W_e]
z = z.reshape(B, V, L, C, H_e, W_e)
return z.permute(0, 2, 1, 3, 4, 5).contiguous()
```

encoder 和 temporal pool 不复制，单视角语句和 dtype 行为不变。

`VRAE.decode(latents, stream_ids=None)` 在多视角时直接调用
`VRAEDecoder.forward(latents)` 接收 `[B,L,V,C,H,W]` 时内部 reshape 为
`[B*V,L,C,H,W]`，复用原路径后恢复输出 `[B,T,V,3,H,W]`。

### 4.2 `VRAEDecoder`

原有 Decoder 配置保持不变；仅在 `forward` 增加 6D 输入的 reshape 分支，原 5D
`_forward_impl` 保持不变。

多视角分支：

1. 输入 `[B,L,V,C,H,W]` 转为 `[B*V,L,C,H,W]`。
2. 调用原有 5D Decoder，不增加参数、不改变 attention token 数。
3. 输出恢复为 `[B,T,V,3,H,W]`；每个 view 的 decode 完全独立。

## 5. Stage 1 loss、GAN 与训练入口

### 5.1 `ReconstructionLoss`

在现有 `forward` 增加 6D 分支，将输入展平后复用已有 L1/LPIPS/Gram/temporal loss：

```python
# [B,T,V,3,H,W] -> [B*V,T,3,H,W]
value = value.permute(0, 2, 1, 3, 4, 5).reshape(-1, T, 3, H, W)
target = target.permute(0, 2, 1, 3, 4, 5).reshape(-1, T, 3, H, W)
```

loss 在 `B*V` 上平均，不再额外乘 V；日志增加每个 `stream_id` 的 loss。

### 5.2 训练循环与 GAN

`src/vrae/training/recon_training/train.py` 仅在现有 graph/loop 增加：

```python
result = graph(video, stream_ids=batch.get("stream_ids"))
```

单视角 `stream_ids=None` 走原路径。prefetch 额外搬运 `stream_ids` 即可；optimizer、
EMA、scheduler、DDP、sampler 和 checkpoint 保存逻辑不变。

VideoMAE discriminator 仍接收 `[N,T,3,H,W]`，多视角调用点展平 `B*V` 并平均 loss。
它只提供逐视角真实性，第一轮多视角训练建议 `gan.enabled=false`。

## 6. Stage 2：在现有 LatentAdapter/VideoDiT 中增加分支

### 6.1 `VRAELatentAdapter`

现有 `encode_grid/decode_grid/grid_to_tokens/tokens_to_grid` 增加多视角形状分支：

```text
[B,L,V,C,H,W] -> [B,L,V,N,C]
```

token 顺序固定为 `[time, view, patch]`，转换中不得丢弃或重排 V。

### 6.2 `VRAEVideoDiT`

新增构造配置 `multiview_enabled`、`num_views`、`num_streams`；单视角
`[B,L,N,C]` 路径保持不变。

多视角 forward：

1. 接收 noisy `[B,L,V,N,C]`、`stream_ids=[B,V]`、`time=[B]`、task labels `[B]`。
2. 在现有 encoder/decoder embedder 后加入各自的 zero-init view embedding（仅属于 DIT）。
3. 保持 batch 维 B，把 token 展平为 `[B,L*V*N,C]`，复用现有 full self-attention；
   同一 episode 的各 view 因而可以互相看到。
4. 位置表按 view 重复 `(time,height,width)`；不使用 view 轴 RoPE。
5. 输出按 `[time,view,patch]` 恢复为 `[B,L,V,N,C]`。

flow matching 时同一 episode 的各 view 使用相同 time，noise 独立采样。LeRobot
`task_index` 直接作为 task condition；无条件训练使用专用 null task ID，不能传 `-1`。

## 7. checkpoint、配置与路径

### 7.1 单视角 checkpoint 初始化

使用 `training.init_from`，不作为 exact resume：

1. 严格校验旧 encoder、pool、decoder/DiT 的 latent 定义和分辨率。
2. Stage 1 Decoder 不引入新增参数；DIT 的多视角参数按 DIT checkpoint 规则加载。
3. 新参数零初始化，旧参数严格加载。
4. 不迁移旧 optimizer、scheduler、sampler state 或 EMA。
5. metadata 记录 `multiview_enabled`、`num_views`、`num_streams`、camera key/ID
   映射和 `checkpoint_weight_source=single_view_init`。

现有 `load_model_init(strict=True)` 需要在现有加载逻辑中增加明确的 missing-key 白名单，
不能全局放宽。多视角 exact resume 要求 camera mapping、V、模型配置和数据契约完全一致。

### 7.2 路径与配置文件

在 `configs/paths.example.yaml` 和 `ProjectPaths.DEFAULT_DATASET_PATHS` 增加
`lerobot` 路径；新增多视角训练 yaml/launcher，但不修改 UCF/K600 yaml。LeRobot 未
安装时明确报错，不回退其他数据集。

## 8. 最小修改文件清单

```text
修改 src/vrae/data/lerobot.py                         # 多 camera key 与同步 item
修改 src/vrae/training/recon_training/data.py        # dataset 分支、collate、搬运
修改 src/vrae/models/autoencoder.py                  # VRAE 6D 分支
修改 src/vrae/models/decoder.py                      # 仅增加 6D -> B*V reshape 分支
修改 src/vrae/models/adapter.py                      # VRAELatentAdapter 保留 V 维
修改 src/vrae/models/dit/video_dit.py                # VideoDiT 5D token 分支
修改 src/vrae/training/recon_training/losses.py      # ReconstructionLoss 6D 分支
修改 src/vrae/training/recon_training/train.py       # 传递 stream_ids
修改 src/vrae/training/recon_training/gan.py         # B*V 展平
修改 checkpoint/metadata 现有实现                     # multiview 字段与白名单
新增一个多视角 yaml/launcher                         # 不改变旧配置
```

禁止新增多视角专用模型类、trainer 类或 attention 类；允许在现有类中添加私有方法
和配置分支。

## 9. 验收标准

以 `B=2,T=16,V=2,H=W=256` 为例：

```text
dataset item       [16,2,3,256,256]
batch video         [2,16,2,3,256,256]
encoder input      [4,16,3,256,256]
latent             [2,4,2,C,H_e,W_e]
reconstruction     [2,16,2,3,256,256]
VideoDiT sequence  [2,8*N,C]
```

必须通过：

1. 单视角配置原有测试、dry-run、checkpoint 加载全部不变。
2. 多视角所有 camera key 使用同一 frame indices，尺寸和 dtype 校验有效。
3. `V=1` 且新增参数零初始化时，新旧模型数值等价。
4. 交换输入 view 顺序后，按 `stream_id` 对齐的输出语义不变。
5. 多视角 loss 按 view 平均，per-view 指标可追踪。
6. 单视角 checkpoint 迁移时 missing key 仅属于新增多视角参数。
7. 少量 LeRobot episode 可完成前向、反向、EMA、checkpoint 保存和 reload。

实现顺序：先 dataset/collate，再 `VRAE`/`VRAEDecoder`，然后 loss/训练入口，最后
`VRAELatentAdapter`/`VRAEVideoDiT` 和 latent statistics；首轮关闭 GAN。
