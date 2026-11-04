"""
Training script for the midterm experiments: nanoGPT with optional PerforatedAI dendrites.

Based on train.py (single GPU, from scratch). Differences:
  * `perforate` selects layers to wrap with PerforatedAI; one dendrite is added to each BEFORE training
    starts (see add_dendrites_at_start). With `perforate=''` this trains the plain vanilla model.
  * evaluation uses a fixed set of batches (same tokens for every run and every eval)
  * every eval is logged with cumulative training time (eval time excluded) and tokens seen, so
    "time/tokens to reach a target loss" can be computed afterwards
  * parameters are counted without PerforatedAI's unused `parent_module` copies
  * seed is honoured and the training batch stream is re-seeded after model setup, so every run sees
    identical training batches regardless of how many random numbers the model setup consumed

Open-source `perforatedai` has no Perforated Backpropagation: dendrites are trained jointly by ordinary
gradient descent with every weight trainable (see perforated_ai_context.md, section 19).

Examples (run from the nanoGPT directory, with PYTHONUTF8=1 when perforating):
    python train_pai.py config/midterm.py --out_dir=out-vanilla-2k
    python train_pai.py config/midterm.py --out_dir=out-pai-fc --perforate=mlp.c_fc
    python train_pai.py config/midterm.py --out_dir=out-pai-fc-b0 --perforate=mlp.c_fc --perforate_blocks=b0
    python train_pai.py config/midterm.py --out_dir=out-pai-attn --perforate=attn.c_attn
"""
import json
import math
import os
import pickle
import time
from contextlib import nullcontext

os.environ.setdefault("MPLBACKEND", "Agg")  # PerforatedAI draws graphs; no GUI backend needed

import numpy as np
import torch

from bench_inference import count_used_params
from pai_setup import add_dendrites, build_gpt

# -----------------------------------------------------------------------------
# I/O
out_dir = 'out'
eval_interval = 50
log_interval = 50
eval_iters = 50
always_save_checkpoint = False  # only the final model is needed; set True to keep every eval
# data
dataset = 'tinystories'
gradient_accumulation_steps = 4
batch_size = 8
block_size = 128
# model
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0
bias = False
mlp_hidden = 0  # MLP hidden width; 0 = nanoGPT default (4*n_embd). 1538 matches +1 dendrite on every mlp.c_fc
# PerforatedAI
perforate = ''  # comma-separated module paths inside a block, e.g. 'mlp.c_fc' or 'mlp.c_fc,attn.c_attn'; '' = vanilla
perforate_blocks = 'all'  # 'all' or e.g. 'b0', 'b3', 'b0_b3'
dendrite_lr_mult = 1.0  # learning-rate multiplier for dendrite parameters only (1.0 = same as the rest)
# adamw optimizer
learning_rate = 3e-4
max_iters = 2000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 2000
min_lr = 3e-5
# system
device = 'cuda'
dtype = 'float16'  # 'float32', 'bfloat16' or 'float16' (float16 uses a GradScaler)
compile = False
seed = 1337
wandb_log = False
wandb_project = 'perforated-transformers'
wandb_run_name = ''
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

assert not compile, "torch.compile is not supported with PerforatedAI wrappers in this script"
os.makedirs(out_dir, exist_ok=True)
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration: {tokens_per_iter:,}")

torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ------------------------------------------------------------------ data
data_dir = os.path.join('data', dataset)


def get_batch(split, generator=None):
    # np.memmap is recreated every batch to avoid a memory leak (same as train.py)
    data = np.memmap(os.path.join(data_dir, 'train.bin' if split == 'train' else 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def make_fixed_batches(split, seed_):
    g = torch.Generator().manual_seed(seed_)
    return [get_batch(split, g) for _ in range(eval_iters)]


# ------------------------------------------------------------------ model
meta_path = os.path.join(data_dir, 'meta.pkl')
vocab_size = 50304
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        vocab_size = pickle.load(f)['vocab_size']
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
model = build_gpt(model_args, mlp_hidden).to(device)
params_before = count_used_params(model)


def parse_blocks(spec):
    if spec == 'all':
        return list(range(n_layer))
    blocks = [int(tok.lstrip('b')) for tok in spec.split('_')]
    assert all(0 <= b < n_layer for b in blocks), f"bad perforate_blocks={spec!r}"
    return blocks


perforated = []
if perforate:
    suffixes = [s.strip() for s in perforate.split(',') if s.strip()]
    blocks = parse_blocks(perforate_blocks)
    perforated = [f".transformer.h.{b}.{s}" for b in blocks for s in suffixes]
    print(f"adding one dendrite to {len(perforated)} modules: {perforated}")
    # PAI strips '-' and path separators from save_name and creates that folder in the working directory
    save_name = "pai" + "".join(c for c in os.path.basename(os.path.normpath(out_dir)) if c.isalnum() or c == "_")
    model = add_dendrites(model, perforated, save_name, get_batch('val', torch.Generator().manual_seed(0)),
                          ctx, device, learning_rate)

params_used = count_used_params(model)
print(f"parameters used in forward: {params_before:,} -> {params_used:,} (+{params_used - params_before:,})")

# optimizer built after any restructure, same decay rule as nanoGPT (2-D tensors decay), no unused parent copies
param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad and '.parent_module.' not in n}
is_dendrite = lambda n: 'dendrites_to_top' in n or 'dendrite_module.layers' in n  # dendrite weights + output scales
groups = []
for is_d in (False, True):
    for wd_on in (True, False):
        ps = [p for n, p in param_dict.items() if is_dendrite(n) == is_d and (p.dim() >= 2) == wd_on]
        if ps:
            groups.append({'params': ps, 'weight_decay': weight_decay if wd_on else 0.0,
                           'lr_mult': dendrite_lr_mult if is_d else 1.0})
print("optimizer groups:", [(len(g['params']), g['weight_decay'], g['lr_mult']) for g in groups])
optimizer = torch.optim.AdamW(groups, lr=learning_rate, betas=(beta1, beta2), fused=(device_type == 'cuda'))
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))


@torch.no_grad()
def estimate_loss(batches):
    model.eval()
    losses = torch.zeros(len(batches))
    for k, (X, Y) in enumerate(batches):
        with ctx:
            _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (learning_rate - min_lr)


val_batches = make_fixed_batches('val', 1234)
train_eval_batches = make_fixed_batches('train', 5678)

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name or os.path.basename(out_dir), config=config)

# ------------------------------------------------------------------ training
torch.manual_seed(seed + 1)  # identical training batch stream in every run
log_path = os.path.join(out_dir, 'eval_log.jsonl')
open(log_path, 'w').close()
if device_type == 'cuda':
    torch.cuda.reset_peak_memory_stats()

model.train()
X, Y = get_batch('train')
train_time = 0.0  # seconds spent in training steps only
best_val = float('inf')
t_mark = time.perf_counter()


def sync():
    if device_type == 'cuda':
        torch.cuda.synchronize()


for iter_num in range(max_iters + 1):
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for g in optimizer.param_groups:
        g['lr'] = lr * g['lr_mult']

    if iter_num % eval_interval == 0 or iter_num == max_iters:
        sync()
        train_time += time.perf_counter() - t_mark
        tr, va = estimate_loss(train_eval_batches), estimate_loss(val_batches)
        best_val = min(best_val, va)
        row = {"iter": iter_num, "tokens_seen": iter_num * tokens_per_iter, "train_time_s": train_time,
               "train_loss": tr, "val_loss": va, "lr": lr,
               "peak_mem_MiB": torch.cuda.max_memory_allocated() / 2**20 if device_type == 'cuda' else None}
        with open(log_path, 'a') as f:
            f.write(json.dumps(row) + "\n")
        print(f"step {iter_num}: train {tr:.4f}  val {va:.4f}  train_time {train_time:.1f}s")
        if wandb_log:
            wandb.log(row)
        t_mark = time.perf_counter()  # eval time is not counted as training time
    if iter_num == max_iters:
        break

    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if iter_num % log_interval == 0:
        print(f"iter {iter_num}: loss {loss.item() * gradient_accumulation_steps:.4f}")
    if always_save_checkpoint and iter_num > 0 and iter_num % eval_interval == 0:
        torch.save({'model': model.state_dict(), 'model_args': model_args, 'iter_num': iter_num,
                    'config': config, 'perforated': perforated}, os.path.join(out_dir, 'ckpt.pt'))

# ------------------------------------------------------------------ final save + summary
final_val = estimate_loss(val_batches)
torch.save({'model': model.state_dict(), 'model_args': model_args, 'iter_num': max_iters,
            'best_val_loss': best_val, 'config': config, 'perforated': perforated},
           os.path.join(out_dir, 'ckpt.pt'))
summary = {
    "out_dir": out_dir, "perforate": perforate, "perforate_blocks": perforate_blocks, "perforated_modules": perforated,
    "params_used_before": params_before, "params_used": params_used, "final_val_loss": final_val,
    "final_val_ppl": math.exp(final_val), "best_val_loss": best_val, "train_time_s": train_time,
    "tokens_seen": max_iters * tokens_per_iter, "train_tokens_per_s": max_iters * tokens_per_iter / train_time,
    "peak_train_mem_MiB": torch.cuda.max_memory_allocated() / 2**20 if device_type == 'cuda' else None,
    "config": config, "gpu": torch.cuda.get_device_name(0) if device_type == 'cuda' else 'cpu',
    "torch": torch.__version__,
}
with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps({k: summary[k] for k in ["params_used", "final_val_loss", "final_val_ppl", "train_time_s",
                                          "train_tokens_per_s", "peak_train_mem_MiB"]}, indent=2))
