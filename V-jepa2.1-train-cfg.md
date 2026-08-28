# V-JEPA 2.1 + LeRobot 训练配置

本分支只有一个训练配置：

```text
configs/training/vjepa2_1_lerobot.yaml
```

## 训练路径

```text
LeRobot episode
  -> 同步采样 T 帧 / V 个 camera
  -> 冻结 V-JEPA 2.1 ViT-L/16
  -> temporal_attention(group_size=2)
  -> V-RAE decoder(tubelet_size=4)
  -> reconstruction loss
```

V-JEPA tubelet=2 和 temporal pool group=2 总共压缩时间 4 倍，因此输入帧数必须是 4 的倍数。单视角视频形状为 `[B,T,3,H,W]`；启用多视角时为 `[B,T,V,3,H,W]`。

## 模板中必须填写的字段

```yaml
data:
  root: /path/to/lerobot/data
  repo_id: your-org/your-lerobot-dataset
  camera_keys:
    - {key: observation.images.image, name: camera, stream_id: 0}
  num_frames: 16
  frame_interval: 1
  image_size: 256
```

camera key 必须是 LeRobot metadata 中存在的 RGB image feature。多 camera 时所有 view 必须共享分辨率；需要多视角时将 `model.multiview.enabled` 设为 `true`，并使 `num_views` 与 camera key 数量一致。

其余字段控制 optimizer、precision、EMA、checkpoint、日志和 perceptual loss，均在同一个 YAML 中维护，不再有 UCF101/K600/Cityscapes 等数据集专用配置。
