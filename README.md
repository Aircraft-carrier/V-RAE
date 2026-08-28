# V-RAE — V-JEPA 2.1 + LeRobot 最小训练版

本分支只保留 V-JEPA 2.1 冻结编码器上的 V-RAE Stage 1 重建训练：训练 temporal pool 和 video decoder，编码器参数始终冻结。数据集统一使用 LeRobot，训练配置只有一个模板。

## 安装

需要 Linux、NVIDIA GPU、Python 3.10+、CUDA 和 FFmpeg。LeRobot 必须安装在当前 Python 环境中。

```bash
conda create -n vrae python=3.10 -y
conda activate vrae
conda install -c conda-forge ffmpeg -y
pip install uv
uv pip install -e .
```

下载官方 V-JEPA 2.1 ViT-L/16 checkpoint：

```bash
mkdir -p ckpts/pretrained/encoders/vjepa2_1
curl -fL https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt \
  -o ckpts/pretrained/encoders/vjepa2_1/model.pt
```

重建 loss 使用 VGG/LPIPS 权重，下载方式见 [third_party/README.md](third_party/README.md)。

## 路径配置

复制 `configs/paths.example.yaml` 为 `configs/paths.local.yaml`，填写 LeRobot 数据根目录和仓库根目录：

```yaml
project_root: /path/to/V-RAE
datasets:
  lerobot: /path/to/lerobot/data
third_party:
  vjepa2_1: /path/to/V-RAE/third_party/vjepa2
```

LeRobot 的 `repo_id`、camera key、clip 长度和训练超参数统一在 `configs/training/vjepa2_1_lerobot.yaml` 中修改。默认单视角；如果使用多个同步相机，配置 `model.multiview` 和 `data.camera_keys` 即可。

## 训练

唯一训练模板：

```text
configs/training/vjepa2_1_lerobot.yaml
```

单机八卡：

```bash
scripts/train/vjepa2_1.sh 1 8 0
```

多机运行时，在所有节点设置相同的 `MASTER_ADDR`/`MASTER_PORT`，并传入不同的 node rank：

```bash
MASTER_ADDR=10.0.0.8 MASTER_PORT=29500 \
  scripts/train/vjepa2_1.sh 2 8 0
MASTER_ADDR=10.0.0.8 MASTER_PORT=29500 \
  scripts/train/vjepa2_1.sh 2 8 1
```

仅检查配置和启动命令：

```bash
VJEPA_DRY_RUN=1 scripts/train/vjepa2_1.sh 1 1 0
```

直接使用 Python 入口：

```bash
python -m vrae.training.recon_training.train \
  --config configs/training/vjepa2_1_lerobot.yaml \
  --paths configs/paths.local.yaml
```

## 数据契约

`LeRobotVideoDataset` 从每个 episode 采样同步 clip，输出单视角 `[T,3,H,W]` 或多视角 `[T,V,3,H,W]` 的 float32 `[0,1]` 视频。`num_frames` 必须是 4 的倍数；V-JEPA tubelet=2 与 temporal pool group=2 共同产生 4 倍时间压缩，decoder tubelet 固定为 4。

camera key 必须存在于 LeRobot metadata、为 RGB image feature，且所有 view 的分辨率一致。重建训练不依赖 UCF101/Kinetics/Cityscapes/CoVLA manifest。

## 目录

```text
src/vrae/models/                    V-JEPA adapter、V-RAE、pool、decoder
src/vrae/data/                      LeRobot 数据集、采样和视频读取
src/vrae/training/recon_training/   Stage 1 重建训练与 loss
src/vrae/training/common/           DDP、EMA、checkpoint、optimizer 等通用组件
configs/training/vjepa2_1_lerobot.yaml 唯一训练配置
third_party/vjepa2/                 官方 V-JEPA 2.1 最小运行时源码
```
