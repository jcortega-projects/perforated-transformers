from datasets import load_dataset
import tiktoken
import numpy as np
from pathlib import Path

# GPT-2 tokenizer, matching nanoGPT
enc = tiktoken.get_encoding("gpt2")

output_dir = Path(__file__).resolve().parent

def prepare(split, limit):
    dataset = load_dataset(
        "roneneldan/TinyStories",
        split=split,
        streaming=True
    )

    tokens = []

    for i, example in enumerate(dataset):
        if i >= limit:
            break

        ids = enc.encode_ordinary(example["text"])
        tokens.extend(ids)
        tokens.append(enc.eot_token)

    filename = "val.bin" if split == "validation" else "train.bin"

    np.array(tokens, dtype=np.uint16).tofile(
        output_dir / filename
    )

    print(f"{filename}: {len(tokens):,} tokens")


# Small pilot dataset for initial testing
prepare("train", 20000)
prepare("validation", 1000)