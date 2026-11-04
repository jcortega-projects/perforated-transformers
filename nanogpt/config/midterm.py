# Midterm experiment settings: shared by every run (vanilla, matched-parameter vanilla, dendrite variants).
# Only `out_dir`, `perforate`, `perforate_blocks` (and later `mlp_mult`) change between runs.
dataset = 'tinystories'

# Model architecture (same as the 1,000-step pilot)
n_layer = 4
n_head = 4
n_embd = 256
block_size = 128
dropout = 0.0

# Training
batch_size = 8
gradient_accumulation_steps = 4
max_iters = 2000
learning_rate = 3e-4
decay_lr = True
warmup_iters = 100
lr_decay_iters = 2000
min_lr = 3e-5
seed = 1337

# Evaluation: fixed batches, so every run and every eval sees the same tokens
eval_interval = 50
eval_iters = 50
log_interval = 50

# GPU settings
device = 'cuda'
dtype = 'float16'
compile = False

wandb_log = False
