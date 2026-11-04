"""
Build the midterm figures and tables from the run outputs (no GPU needed).

Inputs : out-*/ (or results/runs/*/) eval_log.jsonl and summary.json, results/bench_*.json, results/profile_*.json
Outputs: results/figures/*.png, results/results_tables.md

    python make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = "results/figures"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # reference palette slots 1-4, in order
IDLE = "#d9d8d2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})

LABELS = {
    "vanilla-2k": "Vanilla (16.1M)",
    "vanilla-matched": "Vanilla, matched params (17.1M)",
    "pai-fc": "Dendrites c_fc, default LR",
    "pai-fc-lr10": "Dendrites c_fc, 10x dendrite LR",
    "pai-attn": "Dendrites attn.c_attn",
    "pai-fc-b0": "Dendrites c_fc, block 0 only",
    "pai-fc-b3": "Dendrites c_fc, block 3 only",
}
SHORT = {
    "vanilla-2k": "Vanilla", "vanilla-matched": "Matched params", "pai-fc": "c_fc, default LR",
    "pai-fc-lr10": "c_fc, 10x LR", "pai-attn": "attn.c_attn", "pai-fc-b0": "c_fc block 0", "pai-fc-b3": "c_fc block 3",
}
RUNS = list(LABELS)


def run_dir(run):
    # training writes out-<run>/; the shared repo keeps the small per-run files in results/runs/<run>/ (out-*/ is git-ignored)
    local = f"out-{run}"
    return local if os.path.exists(os.path.join(local, "summary.json")) else os.path.join("results", "runs", run)


def eval_log(run):
    with open(os.path.join(run_dir(run), "eval_log.jsonl")) as f:
        return [json.loads(line) for line in f]


def summary(run):
    with open(os.path.join(run_dir(run), "summary.json")) as f:
        return json.load(f)


def bench(run, amp):
    with open(f"results/bench_{run}_{amp}.json") as f:
        return json.load(f)


def prefill(b, batch, seq):
    return next(p for p in b["prefill"] if p["batch"] == batch and p["seq_len"] == seq)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.join(FIG, name))


# ------------------------------------------------------------------ figure 1: loss vs steps and vs time
def fig_loss_curves():
    series = [("vanilla-2k", BLUE), ("vanilla-matched", ORANGE), ("pai-fc-lr10", AQUA), ("pai-fc", YELLOW)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, xkey, xlabel, title in [
        (axes[0], "iter", "Training step", "Validation loss vs training steps"),
        (axes[1], "train_time_s", "Training wall-clock time (s, eval excluded)", "Validation loss vs training time"),
    ]:
        for run, color in series:
            rows = [r for r in eval_log(run) if r["iter"] >= 300]
            ls = (0, (4, 3)) if run == "pai-fc" else "-"  # dashed: it lies exactly on vanilla in the step panel
            ax.plot([r[xkey] for r in rows], [r["val_loss"] for r in rows], color=color, lw=2, ls=ls, label=LABELS[run])
            ax.plot(rows[-1][xkey], rows[-1]["val_loss"], "o", color=color, ms=6, mec=SURFACE, mew=1.5)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Validation loss (lower is better)")
    axes[0].set_ylim(2.90, 3.60)
    axes[1].axhline(3.0, color=MUTED, lw=1, ls="--")
    axes[1].text(axes[1].get_xlim()[1], 3.005, "target 3.0", ha="right", va="bottom", color=INK2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.01, -0.10, "Seed 1337, 2,000 steps, fp16 training, RTX 4050 Laptop. Curves start at step 300. "
             "Default-LR dendrites (dashed) overlap vanilla in the left panel.",
             color=MUTED, fontsize=8)
    save(fig, "fig1_loss_curves.png")


# ------------------------------------------------------------------ figure 2: quality vs throughput
def fig_quality_vs_cost():
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    offsets = {"vanilla-2k": (8, 4), "vanilla-matched": (8, -10), "pai-fc": (8, 6), "pai-fc-lr10": (8, -4),
               "pai-attn": (8, -10), "pai-fc-b0": (8, 6), "pai-fc-b3": (8, -12)}
    for run in RUNS:
        x = prefill(bench(run, "bfloat16"), 32, 128)["prompt_tokens_per_s"] / 1000
        y = summary(run)["final_val_loss"]
        color = BLUE if run.startswith("vanilla") else ORANGE
        ax.plot(x, y, "o", color=color, ms=9, mec=SURFACE, mew=2)
        ax.annotate(SHORT[run], (x, y), textcoords="offset points", xytext=offsets[run], fontsize=9, color=INK2)
    ax.plot([], [], "o", color=BLUE, ms=8, label="Plain model")
    ax.plot([], [], "o", color=ORANGE, ms=8, label="With dendrites")
    ax.legend(loc="lower right")
    ax.set_xlabel("Inference throughput, batch 32 x 128 tokens, bf16 (thousand tokens/s, higher is better)")
    ax.set_ylabel("Final validation loss (lower is better)")
    ax.set_title("Quality vs inference cost (best = bottom right)", loc="left")
    ax.set_xlim(0, 420)
    fig.text(0.01, -0.04, "Seed 1337. Dendrite throughput from one benchmark session; laptop GPU.", color=MUTED, fontsize=8)
    save(fig, "fig2_quality_vs_throughput.png")


# ------------------------------------------------------------------ figure 3: throughput and decode bars
def fig_throughput():
    order = ["vanilla-2k", "vanilla-matched", "pai-fc-b0", "pai-fc-b3", "pai-attn", "pai-fc", "pai-fc-lr10"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    ys = range(len(order))
    h = 0.38
    for amp, color, dy in [("bfloat16", BLUE, -h / 2), ("none", ORANGE, h / 2)]:
        tp = [prefill(bench(r, amp), 32, 128)["prompt_tokens_per_s"] / 1000 for r in order]
        dec = [bench(r, amp)["decode_cache_free"]["tokens_per_s"] for r in order]
        name = "bf16" if amp == "bfloat16" else "fp32"
        axes[0].barh([y + dy for y in ys], tp, height=h - 0.03, color=color, label=name)
        axes[1].barh([y + dy for y in ys], dec, height=h - 0.03, color=color, label=name)
        for y, v in zip(ys, tp):
            axes[0].text(v + 4, y + dy, f"{v:.0f}k", va="center", fontsize=8, color=INK2)
        for y, v in zip(ys, dec):
            axes[1].text(v + 3, y + dy, f"{v:.0f}", va="center", fontsize=8, color=INK2)
    axes[0].set_yticks(list(ys), [SHORT[r] for r in order])
    axes[0].invert_yaxis()
    axes[0].set_title("Prefill throughput, batch 32 x 128 (k tokens/s)", loc="left")
    axes[1].set_title("Decode, batch 1, no KV cache (tokens/s)", loc="left")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower right")
    fig.text(0.01, -0.04, "Higher is better. Batch-1 dendrite numbers vary widely between sessions (sync-bound); "
             "treat them as ranges.", color=MUTED, fontsize=8)
    save(fig, "fig3_inference_throughput.png")


# ------------------------------------------------------------------ figure 4: profiler breakdown
def fig_profiler():
    runs = ["vanilla-2k", "vanilla-matched", "pai-fc-b3", "pai-fc-lr10"]
    names = ["Vanilla", "Matched params", "Dendrites, 1 layer", "Dendrites, 4 layers"]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for i, run in enumerate(runs):
        with open(f"results/profile_{run}.json") as f:
            row = next(r for r in json.load(f)["runs"] if r["amp"] == "bfloat16" and r["batch"] == 32)
        busy, wall = row["gpu_busy_ms_per_forward"], row["wall_ms_per_forward"]
        syncs = row["syncs_per_forward"]["aten::_local_scalar_dense"]
        ax.barh(i, busy, color=BLUE, height=0.6, label="GPU computing" if i == 0 else None)
        ax.barh(i, wall - busy, left=busy + 0.25, color=IDLE, height=0.6, label="GPU idle (waiting)" if i == 0 else None)
        ax.text(wall + 0.8, i, f"{wall:.1f} ms  |  {syncs:.0f} syncs", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(runs)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 52)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Time per forward pass (ms), batch 32 x 128, bf16")
    ax.set_title("Where the time goes: dendrite slowdown is GPU idle time", loc="left")
    ax.legend(loc="upper right")
    fig.text(0.01, -0.06, "PyTorch profiler, 10 forward passes. Syncs = GPU->CPU reads (tensor.item()).",
             color=MUTED, fontsize=8)
    save(fig, "fig4_profiler_breakdown.png")


# ------------------------------------------------------------------ tables
def tables():
    lines = ["# Midterm results tables", "",
             "## Quality (2,000 steps, fixed validation batches)", "",
             "| Model | Params used | Val loss seed 1337 | Val loss seed 2024 | Mean | Perplexity (mean) | Train time (s) | Time to val 3.0 (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for run in RUNS:
        s1 = summary(run)
        losses = [s1["final_val_loss"]]
        if os.path.exists(os.path.join(run_dir(f"{run}-s2"), "summary.json")):
            losses.append(summary(f"{run}-s2")["final_val_loss"])
        mean = sum(losses) / len(losses)
        t3 = next((r["train_time_s"] for r in eval_log(run) if r["val_loss"] <= 3.0), None)
        s2 = f"{losses[1]:.4f}" if len(losses) > 1 else "-"
        lines.append(f"| {LABELS[run]} | {s1['params_used']:,} | {losses[0]:.4f} | {s2} | {mean:.4f} | "
                     f"{2.718281828 ** mean:.2f} | {s1['train_time_s']:.0f} | {t3:.0f} |")
    lines += ["", "## Inference cost (RTX 4050 Laptop)", "",
              "| Model | Prefill B32xT128 bf16 (tok/s) | Prefill B32xT128 fp32 (tok/s) | Decode B1 bf16 (tok/s) | "
              "Decode B1 fp32 (tok/s) | TTFT B1xT64 fp32 (ms) | Peak memory (MiB) |",
              "|---|---|---|---|---|---|---|"]
    for run in RUNS:
        bb, bf = bench(run, "bfloat16"), bench(run, "none")
        t = prefill(bf, 1, 64)["latency_ms"]
        mem = max(p["peak_mem_MiB"] for p in bf["prefill"])
        lines.append(f"| {LABELS[run]} | {prefill(bb, 32, 128)['prompt_tokens_per_s']:,.0f} | "
                     f"{prefill(bf, 32, 128)['prompt_tokens_per_s']:,.0f} | {bb['decode_cache_free']['tokens_per_s']:.1f} | "
                     f"{bf['decode_cache_free']['tokens_per_s']:.1f} | {t['mean']:.1f} ± {t['std']:.1f} | {mem:.0f} |")
    lines += ["", "## Profiler, per forward pass (B32xT128 bf16)", "",
              "| Model | Wall (ms) | GPU busy (ms) | GPU idle | Kernels | GPU->CPU syncs | dtype casts (GPU ms) |",
              "|---|---|---|---|---|---|---|"]
    for run in ["vanilla-2k", "vanilla-matched", "pai-fc-b3", "pai-fc-lr10"]:
        with open(f"results/profile_{run}.json") as f:
            r = next(x for x in json.load(f)["runs"] if x["amp"] == "bfloat16" and x["batch"] == 32)
        lines.append(f"| {LABELS[run]} | {r['wall_ms_per_forward']:.1f} | {r['gpu_busy_ms_per_forward']:.1f} | "
                     f"{r['gpu_idle_fraction']:.0%} | {r['kernels_per_forward']:.0f} | "
                     f"{r['syncs_per_forward']['aten::_local_scalar_dense']:.0f} | "
                     f"{r['dtype_casts_per_forward']:.0f} ({r['dtype_cast_gpu_ms_per_forward']:.2f}) |")
    with open("results/results_tables.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("saved results/results_tables.md")


if __name__ == "__main__":
    fig_loss_curves()
    fig_quality_vs_cost()
    fig_throughput()
    fig_profiler()
    tables()
