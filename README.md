# V-RAE: Rethinking Video Latent Spaces for Generation

**Official PyTorch Implementation**

<p>
  <a href="https://arxiv.org/abs/2608.13556"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="Paper"></a>
  <a href="https://v-rae.github.io/"><img src="https://img.shields.io/badge/Homepage-Project%20Page-blue" alt="Homepage"></a>
  <a href="https://huggingface.co/Guomh0707/V-RAE-Models"><img src="https://img.shields.io/badge/Model-Hugging%20Face-yellow" alt="Model"></a>
</p>

![V-RAE overview](assets/V-RAE-overall.png)

V-RAE provides unified pipelines for video reconstruction and video generation with four visual encoder variants: DINOv3, SigLIP2, V-JEPA2.1, and EUPE.

| Task | Dataset | Launcher or Module |
| --- | --- | --- |
| Reconstruction training | UCF101 + Kinetics-600 | `scripts/train/reconstruction.sh` |
| Reconstruction fine-tuning | CoVLA | `scripts/train/covla_reconstruction.sh` |
| Class-conditional generation training | UCF101 | `scripts/train/ucf101.sh` |
| Class-conditional generation training | Kinetics-600 | `scripts/train/k600.sh` |
| Future-frame generation training | Cityscapes | `vrae.training.cityscapes_video_pred.train` |

## Repository Layout

```text
V-RAE/
├── src/vrae/              # Installable models, training code, and evaluation code
├── configs/
│   ├── models/            # Shared V-RAE and VideoDiT model definitions
│   ├── training/          # Reconstruction and generation recipes
│   └── evaluation/        # UCF101, Kinetics-600, and Cityscapes protocols
├── scripts/
│   ├── train/             # Distributed training launchers
│   └── eval/              # Distributed evaluation launchers
├── sampling.py            # Reconstruction examples
└── third_party/           # Required upstream encoder implementations
```

## Installation

Set up the environment and install V-RAE:

```bash
conda create -n vrae python=3.10 -y
conda activate vrae
pip install uv
uv pip install -e .
```

You can also install it with:

```bash
uv pip install -r requirements.txt
```

## Data and Local Paths

Prepare the following datasets:

- [UCF101](https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/tree/04d4e5ca1dc93606cb58752b0c08331e598743a4)
- [Kinetics-600](https://opendatalab.com/OpenMMLab/Kinetics600)
- [Cityscapes](https://www.cityscapes-dataset.com/downloads/) (only `leftImg8bit_sequence_trainvaltest.zip` is required)
- CoVLA (set its dataset root in the local path file)

Create the machine-local training path file from the checked-in template:

```bash
cp configs/paths.example.yaml configs/paths.local.yaml
```

Then replace the placeholders in `configs/paths.local.yaml`:

```yaml
project_root: path/to/V-RAE
datasets:
  ucf101: path/to/UCF-101
  k600: path/to/Kinetics600/videos
  cityscapes: path/to/Cityscapes
  covla: path/to/CoVLA-Dataset
third_party:
  dinov3: path/to/V-RAE/third_party/dinov3
  eupe: path/to/V-RAE/third_party/eupe
  vjepa2_1: path/to/V-RAE/third_party/vjepa2
```

The launchers under `scripts/train/` automatically use `configs/paths.local.yaml` when it exists. Set `VRAE_PATHS_FILE=path/to/another.yaml` to select a different training path file. Direct `python -m` or `torchrun -m` training commands do not load the local file automatically; pass it explicitly with `--paths`.

Evaluation does **not** read `configs/paths.local.yaml`. Set `inputs.data_root` in the relevant file under `configs/evaluation/` instead:

```yaml
inputs:
  data_root: path/to/dataset
```

Datasets, training manifests, and evaluation population manifests are local inputs and are not bundled with the source repository. The default recipes reference them under the Git-ignored `data/metadata/` directory; place the required local files there using the configured filenames. For evaluation, the relevant fields are `inputs.data_root`, `inputs.population`, and, when present, `inputs.population_metadata`.

## Model Weights

V-RAE and generation model checkpoints are available from [Guomh0707/V-RAE-Models](https://huggingface.co/Guomh0707/V-RAE-Models). Download the complete repository into the expected local layout:

```bash
hf download Guomh0707/V-RAE-Models --local-dir ckpts
```

Reconstruction training additionally requires the encoder, RAEv2 initialization, perceptual-loss, and discriminator checkpoints referenced by the selected configuration. Place them under `ckpts/pretrained/`. Generation training uses the released V-RAE checkpoints, while generation evaluation also uses the released VideoDiT checkpoints under `ckpts/`.

The evaluation protocols also require I3D, Inception, and LPIPS backbones. Download them before the first evaluation run:

```bash
mkdir -p ckpts/eval_models/lpips

hf download flateon/FVD-I3D-torchscript i3d_torchscript.pt \
  --local-dir ckpts/eval_models

curl -fL \
  https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth \
  -o ckpts/eval_models/inception_fid.pth

curl -fL https://download.pytorch.org/models/vgg16-397923af.pth \
  -o ckpts/eval_models/lpips/vgg16.pt
curl -fL \
  https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth \
  -o ckpts/eval_models/lpips/lpips_vgg.pt
```

## Sampling

Place three video samples at:

```text
assets/sample1.mp4
assets/sample2.mp4
assets/sample3.mp4
```

Each sample may contain 16 or 20 frames. By default, the complete sample is
reconstructed. To reconstruct the first 16 frames of each sample, including a
20-frame sample, pass `--num-frames 16`:

```bash
python sampling.py eupe --num-frames 16
```

Run reconstruction with one of the following variants:

```bash
python sampling.py dino
python sampling.py siglip
python sampling.py vjepa
python sampling.py eupe
```

Results are saved to:

```text
outputs/<variant>/sample1-comparison.mp4
outputs/<variant>/sample2-comparison.mp4
outputs/<variant>/sample3-comparison.mp4
```

## Training

Run the commands below from the repository root. The reconstruction, UCF101, and Kinetics-600 launchers take the number of nodes, GPUs per node, current node rank, and V-RAE variant in that order. The accepted short variant names are `dino`, `siglip`, `vjepa`, and `eupe`.

For multi-node jobs, use the same `MASTER_ADDR` and `MASTER_PORT` on every machine. `MASTER_ADDR` must identify node 0 and be reachable from all participating nodes.

### Reconstruction Training

Reconstruction training freezes the visual encoder and optimizes the V-RAE temporal pooler and video decoder. The launcher selects the matching recipe from `configs/training/recon_training/`.

Command template:

```text
scripts/train/reconstruction.sh <num_nodes> <gpus_per_node> <node_rank> <variant>
```

Single-node, 8-GPU commands:

```bash
scripts/train/reconstruction.sh 1 8 0 dino
scripts/train/reconstruction.sh 1 8 0 siglip
scripts/train/reconstruction.sh 1 8 0 vjepa
scripts/train/reconstruction.sh 1 8 0 eupe
```

Two-node, 16-GPU example using V-JEPA2.1 (run the corresponding command on each node):

```bash
# Node 0
MASTER_ADDR=xxxxx MASTER_PORT=29500 \
  scripts/train/reconstruction.sh 2 8 0 vjepa

# Node 1
MASTER_ADDR=xxxxx MASTER_PORT=29500 \
  scripts/train/reconstruction.sh 2 8 1 vjepa
```

### CoVLA Reconstruction Fine-tuning

The CoVLA recipe fine-tunes the EUPE V-RAE at 432x768 with 24-frame clips. The
example below uses four nodes with eight GPUs per node. Set the same rendezvous
address and port on all nodes, and give each node a unique rank.

Command template:

```text
COVLA_NNODES=<num_nodes> COVLA_NPROC_PER_NODE=<gpus_per_node> \
  NODE_RANK=<node_rank> MASTER_ADDR=<rank0_host> MASTER_PORT=<port> \
  scripts/train/covla_reconstruction.sh
```

Four-node, 32-GPU example for node 0; use node ranks 1, 2, and 3 on the other machines:

```bash
COVLA_NNODES=4 COVLA_NPROC_PER_NODE=8 \
  NODE_RANK=0 MASTER_ADDR=xxxxx MASTER_PORT=29500 \
  scripts/train/covla_reconstruction.sh
```

### UCF101 Generation Training

This task freezes the selected V-RAE and trains a class-conditional VideoDiT on UCF101. The launcher selects the matching recipe from `configs/training/ucf_videogen/`.

Command template:

```text
scripts/train/ucf101.sh <num_nodes> <gpus_per_node> <node_rank> <variant>
```

Single-node, 8-GPU commands:

```bash
scripts/train/ucf101.sh 1 8 0 dino
scripts/train/ucf101.sh 1 8 0 siglip
scripts/train/ucf101.sh 1 8 0 vjepa
scripts/train/ucf101.sh 1 8 0 eupe
```

### Kinetics-600 Generation Training

This task freezes the selected V-RAE and trains a class-conditional VideoDiT on Kinetics-600. The launcher selects the matching recipe from `configs/training/k600_videogen/`.

Command template:

```text
scripts/train/k600.sh <num_nodes> <gpus_per_node> <node_rank> <variant>
```

Single-node, 8-GPU commands:

```bash
scripts/train/k600.sh 1 8 0 dino
scripts/train/k600.sh 1 8 0 siglip
scripts/train/k600.sh 1 8 0 vjepa
scripts/train/k600.sh 1 8 0 eupe
```

Two-node, 16-GPU example using V-JEPA2.1 (run the corresponding command on each node):

```bash
# Node 0
MASTER_ADDR=xxxxx MASTER_PORT=29500 \
  scripts/train/k600.sh 2 8 0 vjepa

# Node 1
MASTER_ADDR=xxxxx MASTER_PORT=29500 \
  scripts/train/k600.sh 2 8 1 vjepa
```

### Cityscapes Generation Training

This task trains the future-frame VideoDiT to predict subsequent video frames from context frames. Because it uses the Python distributed entry point directly, pass the local training path file with `--paths`.

Command template:

```text
torchrun --standalone --nproc_per_node=<num_gpus> \
  -m vrae.training.cityscapes_video_pred.train \
  --config <config_file> \
  --paths <paths_file>
```

Single-node, 8-GPU command:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m vrae.training.cityscapes_video_pred.train \
  --config configs/training/cityscapes_video_pred/cityscapes.yaml \
  --paths configs/paths.local.yaml
```

## Evaluation

Evaluation launchers take the number of local GPUs, V-RAE variant, and global batch size in that order. The global batch size must be divisible by the GPU count. rFVD measures reconstruction quality, gFVD measures generation quality, and tFVD compares the 16-frame interpolated reconstruction against the aligned ground-truth input frames `[4:20]`.

Before launching a protocol, set `inputs.data_root` in its shared `configs/evaluation/<task>/config.yaml`. Evaluation launchers do not use the training-only `configs/paths.local.yaml`.

### UCF101 rFVD

Command template:

```text
scripts/eval/ucf101_rfvd.sh <num_gpus> <variant> <global_batch_size>
```

Evaluation commands:

```bash
scripts/eval/ucf101_rfvd.sh 8 dino 512
scripts/eval/ucf101_rfvd.sh 8 siglip 512
scripts/eval/ucf101_rfvd.sh 8 vjepa 512
scripts/eval/ucf101_rfvd.sh 8 eupe 512
```

The adjacent-frame V-JEPA2.1 protocol uses its dedicated configuration:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m vrae.evaluation.ucf101_rfvd.run \
  --config configs/evaluation/ucf101_rfvd/config_interval1.yaml
```

### UCF101 gFVD

Command template:

```text
scripts/eval/ucf101_gfvd.sh <num_gpus> <variant> <global_batch_size>
```

Evaluation commands:

```bash
scripts/eval/ucf101_gfvd.sh 8 vjepa 512
scripts/eval/ucf101_gfvd.sh 8 eupe 512
```

### UCF101 tFVD

Command template:

```text
scripts/eval/ucf101_tfvd.sh <num_gpus> <variant> <global_batch_size>
```

Evaluation commands:

```bash
scripts/eval/ucf101_tfvd.sh 8 dino 512
scripts/eval/ucf101_tfvd.sh 8 siglip 512
scripts/eval/ucf101_tfvd.sh 8 vjepa 512
scripts/eval/ucf101_tfvd.sh 8 eupe 512
```

### Kinetics-600 rFVD

Command template:

```text
scripts/eval/k600_rfvd.sh <num_gpus> <variant> <global_batch_size>
```

Evaluation commands:

```bash
scripts/eval/k600_rfvd.sh 8 dino 1024
scripts/eval/k600_rfvd.sh 8 siglip 1024
scripts/eval/k600_rfvd.sh 8 vjepa 1024
scripts/eval/k600_rfvd.sh 8 eupe 1024
```

### Kinetics-600 gFVD

Command template:

```text
scripts/eval/k600_gfvd.sh <num_gpus> <variant> <global_batch_size>
```

Evaluation commands:

```bash
scripts/eval/k600_gfvd.sh 8 vjepa 512
```

### Kinetics-600 tFVD

Command template:

```text
scripts/eval/k600_tfvd.sh <num_gpus> <variant> <global_batch_size>
```

Evaluation commands:

```bash
scripts/eval/k600_tfvd.sh 8 dino 1024
scripts/eval/k600_tfvd.sh 8 siglip 1024
scripts/eval/k600_tfvd.sh 8 vjepa 1024
scripts/eval/k600_tfvd.sh 8 eupe 1024
```

### Cityscapes rFVD, gFID, and gFVD

The `cityscapes_rfvd` protocol evaluates video reconstruction, while `cityscapes_gfid_gfvd` evaluates future-frame generation. Set `inputs.data_root` and the checkpoint fields in each configuration before launching it.

Command template:

```text
torchrun --standalone --nproc_per_node=<num_gpus> \
  -m <evaluation_module> \
  --config <config_file>
```

Single-node, 8-GPU commands:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m vrae.evaluation.cityscapes_rfvd.run \
  --config configs/evaluation/cityscapes_rfvd/config.yaml

torchrun --standalone --nproc_per_node=8 \
  -m vrae.evaluation.cityscapes_gfid_gfvd.run \
  --config configs/evaluation/cityscapes_gfid_gfvd/config.yaml
```

## Acknowledgements

The codebase is built upon some amazing projects: [RAE](https://github.com/bytetriper/RAE), [RAEv2](https://github.com/nanovisionx/RAEv2), [DINOv3](https://github.com/facebookresearch/dinov3), [SigLIP2](https://huggingface.co/google/siglip2-large-patch16-256), [V-JEPA 2](https://github.com/facebookresearch/vjepa2), [EUPE](https://github.com/facebookresearch/EUPE), and [EVATok](https://github.com/HKU-MMLab/EVATok). We thank the authors for making their work publicly available.

We also sincerely thank [Saining Xie](https://www.sainingxie.com/) for his direct guidance and valuable feedback, which has greatly helped to shape V-RAE.
