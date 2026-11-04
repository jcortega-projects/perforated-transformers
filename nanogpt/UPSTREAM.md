# Upstream: nanoGPT

The files `model.py`, `train.py`, `sample.py`, `configurator.py` and `LICENSE` in this
directory are copied **unmodified** from Andrej Karpathy's nanoGPT (MIT License).

- Source: https://github.com/karpathy/nanoGPT
- Commit: `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` (3adf61e, 2025-11-12,
  "Update README to mention nanochat and deprecation")

Project-specific files added on top of upstream (not part of nanoGPT):

- `config/train_tinystories.py` - 16M-parameter TinyStories pilot config
- `data/tinystories/prepare.py` - TinyStories tokenization (GPT-2 BPE)

Run `train.py` from inside this `nanogpt/` directory so its relative paths
(`data/<dataset>/`, `out_dir`, `config/`) resolve.
