"""
Inference benchmark for nanoGPT checkpoints (prefill + cache-free decode).

Differences from inference_benchmark.py:
  * the checkpoint is loaded on the CPU and freed, so peak GPU memory is the model's, not the
    checkpoint's (optimizer state included) - the old script reported 257 MiB instead of ~71 MiB
  * timing uses CUDA events and is repeated several times -> mean +/- std instead of a single number
  * GPU clock / power / temperature state is recorded (laptop GPUs change clocks between sessions)
  * parameter count only counts parameters used in the forward pass (skips PerforatedAI's unused
    `parent_module` copies)

Metrics (all batch/sequence settings are explicit):
  * prefill   : one forward pass over a prompt of length T. With batch 1 this is time-to-first-token.
                nanoGPT only computes logits for the last position when no targets are passed.
  * decode    : greedy generation WITHOUT a KV cache - the full context is recomputed at every token,
                exactly like nanoGPT's generate(). This is NOT KV-cache serving.

Run from the nanoGPT directory:
    python bench_inference.py --ckpt out-tinystories-small/ckpt.pt --label vanilla --amp none
    python bench_inference.py --ckpt out-tinystories-small/ckpt.pt --label vanilla --amp bfloat16
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig  # noqa: E402


# ---------------------------------------------------------------- loading / counting
def load_model(ckpt_path, device):
    """Load on CPU, copy weights into a fresh model, drop the checkpoint, then move to the GPU.

    Checkpoints from train_pai.py may have a wider MLP (`mlp_hidden`) or PerforatedAI dendrites
    (`perforated`); the same structure is rebuilt with pai_setup before the weights are loaded.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["model_args"])
    from pai_setup import add_dendrites, build_gpt
    model = build_gpt(ckpt["model_args"], ckpt.get("config", {}).get("mlp_hidden", 0))
    if ckpt.get("perforated"):
        model = model.to(device)
        x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size), device=device)  # only shapes matter here
        model = add_dendrites(model, ckpt["perforated"], "paibench", (x, x), torch.autocast("cuda", enabled=False),
                              device, lr=1e-3).cpu()
    state = ckpt["model"]
    for k in list(state.keys()):
        if k.startswith("_orig_mod."):
            state[k[len("_orig_mod."):]] = state.pop(k)
    if "tracker_string" in state:  # PAI bookkeeping text (encodes the save-folder name), not a weight
        state["tracker_string"] = model.state_dict()["tracker_string"]
    model.load_state_dict(state)
    meta = {
        "iter_num": ckpt.get("iter_num"),
        "best_val_loss": float(ckpt["best_val_loss"]) if ckpt.get("best_val_loss") is not None else None,
    }
    del ckpt, state
    model.to(device).eval()
    return model, cfg, meta


def count_used_params(model):
    """Unique parameters used in the forward pass (tied weights once, PAI parent_module copies skipped)."""
    seen, total = set(), 0
    for name, p in model.named_parameters():
        if ".parent_module." in name or id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


# ---------------------------------------------------------------- environment
def gpu_state():
    base = "clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,temperature.gpu,pstate"
    for fields in (base + ",clocks_throttle_reasons.active", base):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
                text=True, timeout=15,
            ).strip().splitlines()[0]
            return dict(zip(fields.split(","), [v.strip() for v in out.split(",")]))
        except Exception:
            continue
    return {}


def environment():
    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "os": platform.platform(),
        "gpu_state_before": gpu_state(),
    }


# ---------------------------------------------------------------- timing helpers
def warm_gpu(seconds=3.0):
    """Busy-loop on the GPU so clocks leave the idle state (P5) before anything is timed."""
    a = torch.randn(2048, 2048, device="cuda")
    t_end = time.time() + seconds
    while time.time() < t_end:
        for _ in range(20):
            a = (a @ a).tanh()
        torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()


def autocast_ctx(amp):
    if amp == "none":
        return torch.autocast("cuda", enabled=False)
    return torch.autocast("cuda", dtype=getattr(torch, amp))


@torch.inference_mode()
def time_forward(model, x, iters, warmup, amp):
    """Average ms per forward pass over `iters` calls, measured with CUDA events."""
    with autocast_ctx(amp):
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            model(x)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def summarize(values):
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def bench_prefill(model, cfg, batches, seq_lens, runs, iters, warmup, amp, device):
    rows = []
    for B in batches:
        for T in seq_lens:
            if T > cfg.block_size:
                continue
            x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            ms = [time_forward(model, x, iters, warmup, amp) for _ in range(runs)]
            s = summarize(ms)
            rows.append({
                "batch": B, "seq_len": T,
                "latency_ms": s,
                "prompt_tokens_per_s": B * T / s["mean"] * 1000.0,
                "peak_mem_MiB": torch.cuda.max_memory_allocated() / 2**20,
            })
    return rows


@torch.inference_mode()
def decode_once(model, cfg, prompt_len, new_tokens, amp, device):
    """Greedy cache-free decode; returns per-token latencies (ms), one CUDA-synchronized timing per token."""
    idx = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)
    lat = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with autocast_ctx(amp):
        for _ in range(new_tokens):
            cond = idx if idx.size(1) <= cfg.block_size else idx[:, -cfg.block_size:]
            start.record()
            logits, _ = model(cond)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            end.record()
            torch.cuda.synchronize()
            lat.append(start.elapsed_time(end))
            idx = torch.cat((idx, nxt), dim=1)
    return lat


def bench_decode(model, cfg, prompt_len, new_tokens, runs, amp, device):
    decode_once(model, cfg, prompt_len, min(8, new_tokens), amp, device)  # warm-up
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    per_run_mean = []
    first_token, later_tokens = [], []
    for _ in range(runs):
        lat = decode_once(model, cfg, prompt_len, new_tokens, amp, device)
        per_run_mean.append(statistics.fmean(lat))
        first_token.append(lat[0])
        later_tokens.extend(lat[1:])
    s = summarize(per_run_mean)
    return {
        "prompt_len": prompt_len, "new_tokens": new_tokens, "batch": 1, "kv_cache": False,
        "per_token_latency_ms": s,
        "first_token_ms": summarize(first_token),
        "tokens_per_s": 1000.0 / s["mean"],
        "peak_mem_MiB": torch.cuda.max_memory_allocated() / 2**20,
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out-tinystories-small/ckpt.pt")
    ap.add_argument("--label", default="vanilla")
    ap.add_argument("--amp", choices=["none", "bfloat16", "float16"], default="none",
                    help="autocast dtype; 'none' = plain fp32")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 32])
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[16, 64, 128])
    ap.add_argument("--runs", type=int, default=5, help="repeated timing runs per setting")
    ap.add_argument("--iters", type=int, default=50, help="forward passes averaged inside one run")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--out", default=None, help="JSON output path (default results/bench_<label>_<amp>.json)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    if args.amp == "bfloat16":
        assert torch.cuda.is_bf16_supported(), "bf16 not supported on this GPU"
    device = "cuda"

    t0 = time.time()
    model, cfg, meta = load_model(args.ckpt, device)
    total_params = count_used_params(model)
    weights_MiB = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
    print(f"[{args.label}] amp={args.amp}  params(used)={total_params/1e6:.3f}M  "
          f"nanoGPT-count={model.get_num_params()/1e6:.3f}M  weights={weights_MiB:.1f} MiB  "
          f"ckpt iter={meta['iter_num']} val_loss={meta['best_val_loss']}")

    result = {
        "label": args.label, "amp": args.amp, "ckpt": args.ckpt, "checkpoint_meta": meta,
        "model_args": cfg.__dict__, "params_used": total_params,
        "params_nanogpt_count": model.get_num_params(), "weights_MiB": weights_MiB,
        "settings": {k: getattr(args, k.replace("-", "_")) for k in
                     ["runs", "iters", "warmup", "batches", "seq-lens", "prompt-len", "new-tokens"]},
        "environment": environment(),
    }
    warm_gpu()
    result["environment"]["gpu_state_after_warmup"] = gpu_state()
    result["prefill"] = bench_prefill(model, cfg, args.batches, args.seq_lens,
                                      args.runs, args.iters, args.warmup, args.amp, device)
    result["decode_cache_free"] = bench_decode(model, cfg, args.prompt_len, args.new_tokens,
                                               args.runs, args.amp, device)
    result["environment"]["gpu_state_after"] = gpu_state()
    result["wall_seconds"] = time.time() - t0

    print(f"\nPREFILL (one forward pass; batch 1 = time-to-first-token)")
    print(f"{'B':>4} {'T':>5} {'latency ms (mean +/- std)':>28} {'prompt tok/s':>14} {'peak MiB':>9}")
    for r in result["prefill"]:
        l = r["latency_ms"]
        print(f"{r['batch']:>4} {r['seq_len']:>5} {l['mean']:>16.3f} +/- {l['std']:<8.3f} "
              f"{r['prompt_tokens_per_s']:>14,.0f} {r['peak_mem_MiB']:>9.1f}")
    d = result["decode_cache_free"]
    print(f"\nDECODE, cache-free (prompt {d['prompt_len']}, {d['new_tokens']} new tokens, batch 1)")
    print(f"  per-token latency: {d['per_token_latency_ms']['mean']:.3f} +/- {d['per_token_latency_ms']['std']:.3f} ms"
          f"   ({d['tokens_per_s']:.1f} tokens/s)   first token: {d['first_token_ms']['mean']:.3f} ms"
          f"   peak mem: {d['peak_mem_MiB']:.1f} MiB")
    gs = result["environment"]["gpu_state_after_warmup"]
    print(f"\nGPU state after warm-up: sm clock {gs.get('clocks.sm')} / max {gs.get('clocks.max.sm')}, "
          f"power {gs.get('power.draw')} / {gs.get('power.limit')}, temp {gs.get('temperature.gpu')} C, pstate {gs.get('pstate')}")

    out = args.out or os.path.join("results", f"bench_{args.label}_{args.amp}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
