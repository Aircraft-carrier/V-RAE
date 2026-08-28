# Third-party runtime and weights

This branch vendors only the runtime subset needed by the V-JEPA 2.1 adapter:
`third_party/vjepa2/`. Its upstream licenses are kept next to the source.

## V-JEPA 2.1 checkpoint

```bash
mkdir -p ckpts/pretrained/encoders/vjepa2_1
curl -fL \
  https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt \
  -o ckpts/pretrained/encoders/vjepa2_1/model.pt
```

## Reconstruction loss weights

The template expects local VGG16 and LPIPS calibration files:

```bash
mkdir -p ckpts/pretrained/perceptual
curl -fL https://download.pytorch.org/models/vgg16-397923af.pth \
  -o ckpts/pretrained/perceptual/vgg16.pt
curl -fL \
  https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-vgg16-lpips.pth \
  -o ckpts/pretrained/perceptual/lpips_vgg.pt
```
