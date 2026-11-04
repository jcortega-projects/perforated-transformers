
import torch
from model import GPT, GPTConfig

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# Small model for compatibility testing
config = GPTConfig(
    block_size=16,
    vocab_size=128,
    n_layer=2,
    n_head=2,
    n_embd=64,
    dropout=0.0,
    bias=False,
)

model = GPT(config)

# Select only the first block's MLP input projection.
# Explicit layer IDs use a leading dot.
GPA.pc.set_module_names_to_perforate([])
GPA.pc.set_module_ids_to_perforate([
    ".transformer.h.0.mlp.c_fc"
])

# Track other parameterized layers without perforating them.
GPA.pc.set_module_names_to_track([
    "Linear", "Embedding", "LayerNorm"
])

# nanoGPT hidden tensors have shape [batch, sequence, channels].
GPA.pc.set_output_dimensions([-1, -1, 0])
GPA.pc.set_configuration_confirmed(True)

# Install PerforatedAI's model wrappers.
model = UPA.perforate_model(
    model,
    save_name="pai_nanogpt_test",
    making_graphs=False,
    maximizing_score=False,
)

pai_modules = UPA.get_pai_modules(model, 0)

print("PAI module count:", len(pai_modules))
print("Converted layer:",
      type(model.transformer.h[0].mlp.c_fc).__name__)

# Verify a normal forward and backward pass.
x = torch.randint(0, config.vocab_size, (2, 16))
targets = torch.randint(0, config.vocab_size, (2, 16))

logits, loss = model(x, targets)
loss.backward()

print("Logits shape:", tuple(logits.shape))
print("Loss:", loss.item())
print("Backward pass successful")
