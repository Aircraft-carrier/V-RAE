# Vendored encoder runtime sources

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
