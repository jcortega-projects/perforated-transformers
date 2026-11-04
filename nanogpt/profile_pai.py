"""
Profile one forward pass of a checkpoint with the PyTorch profiler and report where the time goes.

Per forward pass it reports:
  * wall time (CUDA-synchronized) and GPU busy time (sum of kernel times) -> GPU idle fraction
  * number of CUDA kernels launched
  * number of GPU->CPU syncs (aten::_local_scalar_dense = tensor.item() / Python bool on a CUDA tensor,
    plus cudaStreamSynchronize / cudaDeviceSynchronize calls)
  * number of dtype conversions (aten::_to_copy) and the time spent in them
  * the top operators by CPU time

Run from the nanoGPT directory (PYTHONUTF8=1 for dendrite checkpoints):
    python profile_pai.py --ckpt out-vanilla-2k/ckpt.pt --label vanilla-2k
    python profile_pai.py --ckpt out-pai-fc-lr10/ckpt.pt --label pai-fc-lr10 --trace
"""
import argparse
import json
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile

from bench_inference import autocast_ctx, count_used_params, load_model, warm_gpu

SYNC_OPS = ("aten::_local_scalar_dense", "cudaStreamSynchronize", "cudaDeviceSynchronize")


@torch.inference_mode()
def wall_ms(model, x, amp, iters=50):
    with autocast_ctx(amp):
        for _ in range(10):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


@torch.inference_mode()
def profile_forward(model, x, amp, iters, trace_path=None):
    with autocast_ctx(amp):
        for _ in range(10):
            model(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                model(x)
            torch.cuda.synchronize()
    if trace_path:
        prof.export_chrome_trace(trace_path)

    events = prof.events()
    kernels = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
    gpu_busy_us = sum(e.device_time_total for e in kernels)
    ops = {a.key: a for a in prof.key_averages()}
    count = lambda k: ops[k].count if k in ops else 0
    syncs = {k: count(k) / iters for k in SYNC_OPS}
    to_copy = ops.get("aten::_to_copy")
    top = sorted(prof.key_averages(), key=lambda a: a.self_cpu_time_total, reverse=True)[:12]
    return {
        "kernels_per_forward": len(kernels) / iters,
        "gpu_busy_ms_per_forward": gpu_busy_us / iters / 1000.0,
        "syncs_per_forward": syncs,
        "dtype_casts_per_forward": (to_copy.count / iters) if to_copy else 0,
        "dtype_cast_gpu_ms_per_forward": (to_copy.device_time_total / iters / 1000.0) if to_copy else 0.0,
        "top_ops_by_self_cpu": [
            {"op": a.key, "calls_per_forward": a.count / iters,
             "self_cpu_ms_per_forward": a.self_cpu_time_total / iters / 1000.0,
             "gpu_ms_per_forward": a.device_time_total / iters / 1000.0} for a in top],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--amp", nargs="+", default=["none", "bfloat16"])
    ap.add_argument("--shapes", nargs="+", default=["1x64", "32x128"], help="BATCHxSEQ")
    ap.add_argument("--iters", type=int, default=10, help="profiled forward passes")
    ap.add_argument("--trace", action="store_true", help="also export Chrome traces to results/traces/")
    args = ap.parse_args()

    model, cfg, meta = load_model(args.ckpt, "cuda")
    result = {"label": args.label, "ckpt": args.ckpt, "params_used": count_used_params(model), "runs": []}
    warm_gpu()
    for amp in args.amp:
        for shape in args.shapes:
            B, T = map(int, shape.split("x"))
            x = torch.randint(0, cfg.vocab_size, (B, T), device="cuda")
            trace = None
            if args.trace:
                os.makedirs("results/traces", exist_ok=True)
                trace = f"results/traces/{args.label}_{amp}_{shape}.json"
            row = {"amp": amp, "batch": B, "seq_len": T, "wall_ms_per_forward": wall_ms(model, x, amp)}
            row.update(profile_forward(model, x, amp, args.iters, trace))
            row["gpu_idle_fraction"] = max(0.0, 1.0 - row["gpu_busy_ms_per_forward"] / row["wall_ms_per_forward"])
            result["runs"].append(row)
            s = row["syncs_per_forward"]
            print(f"[{args.label}] amp={amp:8} B={B:<3} T={T:<4} wall {row['wall_ms_per_forward']:7.2f} ms | "
                  f"GPU busy {row['gpu_busy_ms_per_forward']:6.2f} ms (idle {row['gpu_idle_fraction']:.0%}) | "
                  f"kernels {row['kernels_per_forward']:5.0f} | .item() syncs {s['aten::_local_scalar_dense']:4.0f} | "
                  f"stream syncs {s['cudaStreamSynchronize']:4.0f} | casts {row['dtype_casts_per_forward']:4.0f} "
                  f"({row['dtype_cast_gpu_ms_per_forward']:.2f} ms GPU)")

    os.makedirs("results", exist_ok=True)
    out = f"results/profile_{args.label}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
