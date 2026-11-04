"""
Model construction shared by train_pai.py and bench_inference.py.

  build_gpt       nanoGPT model, optionally with a wider MLP (matched-parameter baseline)
  add_dendrites   wrap modules with PerforatedAI and add exactly one dendrite to each

A dendrite checkpoint can only be loaded into a model with the same PAI structure, so the benchmark
rebuilds it with the same two calls and then loads the saved weights.
"""
import math
import os
import sys

import torch

from model import GPT, GPTConfig


def build_gpt(model_args, mlp_hidden=0):
    model = GPT(GPTConfig(**model_args))
    if mlp_hidden:
        # matched-parameter baseline: widen every MLP (model.py hard-codes 4*n_embd), same init as GPT._init_weights
        n_embd, bias, n_layer = model_args['n_embd'], model_args['bias'], model_args['n_layer']
        for block in model.transformer.h:
            block.mlp.c_fc = torch.nn.Linear(n_embd, mlp_hidden, bias=bias)
            block.mlp.c_proj = torch.nn.Linear(mlp_hidden, n_embd, bias=bias)
            for lin, std in ((block.mlp.c_fc, 0.02), (block.mlp.c_proj, 0.02 / math.sqrt(2 * n_layer))):
                torch.nn.init.normal_(lin.weight, mean=0.0, std=std)
                if lin.bias is not None:
                    torch.nn.init.zeros_(lin.bias)
        print(f"MLP hidden width {4 * n_embd} -> {mlp_hidden}")
    return model


def add_dendrites(model, module_ids, save_name, batch, ctx, device, lr):
    """Wrap `module_ids` with PerforatedAI and add exactly one dendrite to each, before any training.

    PerforatedAI normally adds a dendrite when validation stops improving. Here the first
    add_validation_score() call only records a score, and the second one (force_switch=True) triggers the
    restructuring. `save_name` must be alphanumeric/underscore: PAI strips other characters and creates
    that folder in the working directory.
    """
    if not sys.flags.utf8_mode:
        sys.exit("perforatedai needs UTF-8 mode on Windows: set PYTHONUTF8=1 and rerun")
    os.environ.setdefault("MPLBACKEND", "Agg")  # PerforatedAI draws graphs; no GUI backend needed
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA

    GPA.pc.set_module_names_to_perforate([])
    GPA.pc.set_module_ids_to_perforate(module_ids)
    GPA.pc.set_module_names_to_track(["Linear", "Embedding", "LayerNorm"])
    GPA.pc.set_output_dimensions([-1, -1, 0])  # activations are [batch, sequence, channels]
    GPA.pc.set_configuration_confirmed(True)
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_testing_dendrite_capacity(False)  # default True stops after 3 dendrites
    GPA.pc.set_silent(True)
    GPA.pc.set_using_safe_tensors(False)  # tied wte/lm_head weights break the safetensors save

    model = UPA.perforate_model(model, save_name=save_name, making_graphs=False, maximizing_score=False).to(device)
    GPA.pai_tracker.set_optimizer(torch.optim.AdamW)
    GPA.pai_tracker.setup_optimizer(model, {"params": list(model.parameters()), "lr": lr})

    # PAI records each wrapped layer's activation shape during the first TRAINING-mode forward/backward pass;
    # without it the restructure below fails with "missing dendrite_storage_shape".
    x, y = batch
    model.train()
    with ctx:
        _, score = model(x, y)
    score.backward()
    model.zero_grad(set_to_none=True)
    model, restructured, complete = GPA.pai_tracker.add_validation_score(score.item(), model)
    model, restructured, complete = GPA.pai_tracker.add_validation_score(score.item(), model, force_switch=True)
    assert restructured and not complete, f"dendrite add failed: restructured={restructured} complete={complete}"
    model = model.to(device)
    if model.transformer.wte.weight is not model.lm_head.weight:
        print("note: wte/lm_head tie was lost during restructure; re-tying")
        model.lm_head.weight = model.transformer.wte.weight
    return model
