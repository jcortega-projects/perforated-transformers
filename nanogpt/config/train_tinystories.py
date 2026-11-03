# Dataset
dataset = 'tinystories'

# Model architecture
n_layer = 4
n_head = 4
n_embd = 256
block_size = 128
dropout = 0.0

# Training
batch_size = 8
gradient_accumulation_steps = 4
max_iters = 1000
learning_rate = 3e-4

# Evaluation
eval_interval = 100
eval_iters = 20
log_interval = 10

# GPU settings
device = 'cuda'
dtype = 'float16'
compile = False

# Checkpoints
out_dir = 'out-tinystories-small'
always_save_checkpoint = True

# Start from scratch
init_from = 'scratch'

# Learning-rate schedule
decay_lr = True
warmup_iters = 100
lr_decay_iters = 1000
min_lr = 3e-5

# Reproducibility
seed = 42

# Disable external logging for now
wandb_log = True
wandb_project = 'perforated-transformers'
wandb_run_name = 'vanilla-gpt-16m-pilot'