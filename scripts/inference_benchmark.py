import time
import torch
from model import GPT, GPTConfig

device = "cuda"
checkpoint = torch.load(
    "out-tinystories-small/ckpt.pt",
    map_location=device,
    weights_only=False
)

config = GPTConfig(**checkpoint["model_args"])
model = GPT(config).to(device)

state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."

for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()

# Simulated input prompt
x = torch.randint(0, config.vocab_size, (1, 64), device=device)

# Warm-up
with torch.inference_mode():
    for _ in range(10):
        model(x)

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

# Measure forward-pass latency
start = time.perf_counter()

with torch.inference_mode():
    for _ in range(100):
        model(x)

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"Average forward latency: {elapsed / 100 * 1000:.2f} ms")
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MiB")