# Third-Party Sources and Pre-trained Weights

## Vendored Runtime Sources

V-RAE keeps the minimal upstream runtime subsets required by its local adapters
inside this directory:

- `dinov3/`: https://github.com/facebookresearch/dinov3
- `eupe/`: https://github.com/facebookresearch/EUPE
- `vjepa2/`: https://github.com/facebookresearch/vjepa2

SigLIP2 is loaded through the pinned Transformers interface and therefore does
not require a source checkout. Model weights live under
`ckpts/pretrained/encoders/`. No encoder adapter reads source or model files from
another project tree.

Each source directory retains its upstream license. The root README acknowledges
the corresponding projects and authors.

## Download Pre-trained Encoder Weights

Run the commands below from the V-RAE repository root. Only the encoder used by
the selected V-RAE variant needs to be downloaded.

Create the expected directories first:

```bash
mkdir -p \
  ckpts/pretrained/encoders/dinov3 \
  ckpts/pretrained/encoders/siglip2 \
  ckpts/pretrained/encoders/eupe \
  ckpts/pretrained/encoders/vjepa2_1
```

### DINOv3 ViT-L/16

V-RAE uses the DINOv3 ViT-L/16 distilled model pre-trained on LVD-1689M. Request
access from the [official DINOv3 download page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/), then download the native PyTorch checkpoint using the URL provided by Meta:

```bash
curl -fL '<DINOv3-ViT-L/16-download-url>' \
  -o ckpts/pretrained/encoders/dinov3/model.pth
```

Do not rename the Hugging Face `model.safetensors` file to `model.pth`; the
current V-RAE configuration expects the native DINOv3 PyTorch checkpoint.

### SigLIP2 ViT-L/16

Download the files required by the local Transformers loader from
[`google/siglip2-large-patch16-256`](https://huggingface.co/google/siglip2-large-patch16-256):

```bash
hf download google/siglip2-large-patch16-256 \
  config.json model.safetensors preprocessor_config.json \
  --local-dir ckpts/pretrained/encoders/siglip2
```

### EUPE ViT-B/16

Download the official [`facebook/EUPE-ViT-B`](https://huggingface.co/facebook/EUPE-ViT-B) checkpoint and rename it to the filename expected by V-RAE:

```bash
hf download facebook/EUPE-ViT-B EUPE-ViT-B.pt \
  --local-dir ckpts/pretrained/encoders/eupe

mv ckpts/pretrained/encoders/eupe/EUPE-ViT-B.pt \
  ckpts/pretrained/encoders/eupe/model.pt
```

Log in with `hf auth login` first if Hugging Face requests authentication or
license acceptance.

### V-JEPA2.1 ViT-L/16

Download the official 384px V-JEPA2.1 ViT-L/16 checkpoint. The file is about
5.15 GB:

```bash
curl -fL \
  https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt \
  -o ckpts/pretrained/encoders/vjepa2_1/model.pt
```

See the [V-JEPA2 model list](https://github.com/facebookresearch/vjepa2#models)
for the upstream checkpoint description.

## Expected Layout

After downloading all four encoders, the checkpoint tree is:

```text
ckpts/pretrained/encoders/
├── dinov3/
│   └── model.pth
├── siglip2/
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
├── eupe/
│   └── model.pt
└── vjepa2_1/
    └── model.pt
```

The `third_party` entries in `configs/paths.local.yaml` point to the vendored
source directories above, while the model configurations point separately to
these checkpoint files under `ckpts/pretrained/encoders/`.

## Reconstruction Training Weights

All reconstruction variants use the perceptual-loss and VideoMAE discriminator
weights below. The DINOv3 and EUPE variants additionally initialize their image
decoders from RAEv2; SigLIP2 and V-JEPA2.1 initialize their decoders from
scratch.

### RAEv2 Decoder Initialization

V-RAE uses `decoder.pt` from the following two RAEv2 checkpoints:

| V-RAE variant | RAEv2 checkpoint | Expected local path |
| --- | --- | --- |
| DINOv3 | [general/dinov3l-k7](https://huggingface.co/nyu-visionx/RAEv2-models/tree/main/stage1/general/dinov3l-k7) | `ckpts/pretrained/decoders/raev2/dinov3_l_k7.pt` |
| EUPE | [imagenet/eupe-b-k7](https://huggingface.co/nyu-visionx/RAEv2-models/tree/main/stage1/imagenet/eupe-b-k7) | `ckpts/pretrained/decoders/raev2/eupe_b_k7.pt` |

Download only the decoder needed by the selected variant. To download both:

```bash
mkdir -p ckpts/pretrained/decoders/raev2

hf download nyu-visionx/RAEv2-models \
  stage1/general/dinov3l-k7/decoder.pt \
  stage1/imagenet/eupe-b-k7/decoder.pt \
  --local-dir ckpts/pretrained/decoders/raev2

mv ckpts/pretrained/decoders/raev2/stage1/general/dinov3l-k7/decoder.pt \
  ckpts/pretrained/decoders/raev2/dinov3_l_k7.pt
mv ckpts/pretrained/decoders/raev2/stage1/imagenet/eupe-b-k7/decoder.pt \
  ckpts/pretrained/decoders/raev2/eupe_b_k7.pt
```

The `stats.pt` files in those RAEv2 directories are not used by V-RAE.

### Perceptual-Loss Weights

Download the ImageNet VGG16 backbone and the LPIPS VGG calibration weights:

```bash
mkdir -p ckpts/pretrained/perceptual

curl -fL https://download.pytorch.org/models/vgg16-397923af.pth \
  -o ckpts/pretrained/perceptual/vgg16.pt
curl -fL \
  https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-vgg16-lpips.pth \
  -o ckpts/pretrained/perceptual/lpips_vgg.pt
```

These two files are also downloaded and verified by
`scripts/download_eval_weights.sh` because reconstruction evaluation uses the
same LPIPS model.

### VideoMAE Discriminator

The discriminator is initialized from
[`vit_b_hybrid_pt_800e.pth`](https://huggingface.co/OpenGVLab/InternVideo1.0/blob/main/internvideomae_classification/vit_b_hybrid_pt_800e.pth)
in `OpenGVLab/InternVideo1.0`. Accept the repository conditions and run
`hf auth login` before downloading if Hugging Face requests access.

```bash
mkdir -p ckpts/pretrained/discriminators/videomae

hf download OpenGVLab/InternVideo1.0 \
  internvideomae_classification/vit_b_hybrid_pt_800e.pth \
  --local-dir ckpts/pretrained/discriminators/videomae

mv ckpts/pretrained/discriminators/videomae/internvideomae_classification/vit_b_hybrid_pt_800e.pth \
  ckpts/pretrained/discriminators/videomae/vit_b_hybrid_pt_800e.pth
```

The resulting reconstruction-weight layout is:

```text
ckpts/pretrained/
├── decoders/raev2/
│   ├── dinov3_l_k7.pt
│   └── eupe_b_k7.pt
├── perceptual/
│   ├── vgg16.pt
│   └── lpips_vgg.pt
└── discriminators/videomae/
    └── vit_b_hybrid_pt_800e.pth
```
