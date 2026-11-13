"""Steer Marin 32B neuron L20-N1234 (the "heck" soft-swearing neuron) while it generates.

The neuron's activation is the input to layers[20].mlp.down_proj (silu(gate) * up), as in the atlas.
A forward pre-hook overwrites it with k * scale_abs_p999 at every generated position (with the KV cache each
decode step forwards only the new token; on prefill only the last prompt position is touched).

Run from this folder (results/L20-N1234-004/):
  modal run gpu_run.py::main                           # clamp sweep -> clamp.json
  modal run gpu_run.py::main --mode add                # additive sweep (x += k*scale) -> add.json
  modal run gpu_run.py::live --prompt "..." --k 8      # stream one steered generation (--add for additive)
"""

import json
import os

import modal

VOL = "/vol"
MODEL_DIR = f"{VOL}/models/marin-32b-bf16"
POLE_STATS = f"{VOL}/runs/stageB2/summaries/pole_stats.parquet"
LAYER, NEURON = 20, 1234

app = modal.App("heck-steering")
vol = modal.Volume.from_name("marin-neuron-atlas")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.11.*", "transformers==5.10.2", "accelerate", "safetensors", "pyarrow", "numpy")

NEUTRAL = [
    "Here is my grandmother's recipe for banana bread. First,",
    "Tomorrow's weather forecast for the Bay Area:",
    "Dear customer, thank you for contacting our support team regarding your order.",
    "The mitochondria is an organelle found in most eukaryotic cells. It",
    "Meeting notes, Tuesday: The team discussed the Q3 roadmap and",
    "My cat spent the whole afternoon",
]
FRUSTRATED = [
    "I stubbed my toe on the coffee table this morning and",
    "The printer jammed for the third time today, so I",
    "We missed the last train home by thirty seconds.",
]
PROMPTS = NEUTRAL + FRUSTRATED
SAMPLES = 3
MAX_NEW = 80
SEED = 1234


class Steer:
    def __init__(self, layer_module):
        self.mod, self.neuron, self.value, self.h, self.add = layer_module, None, None, None, False

    def set(self, neuron, value, add=False):
        self.clear()
        if neuron is None:
            return
        self.neuron, self.value, self.add = neuron, value, add

        def fn(module, args):
            x = args[0].clone()
            if self.add:
                x[:, -1, self.neuron] += self.value
            else:
                x[:, -1, self.neuron] = self.value
            return (x,) + tuple(args[1:])
        self.h = self.mod.register_forward_pre_hook(fn)

    def clear(self):
        if self.h is not None:
            self.h.remove()
            self.h = None


def load():
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map="cuda").eval()
    t = pq.read_table(POLE_STATS, filters=[("layer", "=", LAYER)])
    scale = dict(zip(t["neuron"].to_numpy().tolist(), t["scale_abs_p999"].to_numpy().astype(float).tolist()))
    return tok, model, scale, Steer(model.model.layers[LAYER].mlp.down_proj)


@app.function(image=image, volumes={VOL: vol}, gpu="H200", cpu=8, memory=131072, timeout=3600)
def sweep(n_controls: int = 10, mode: str = "clamp"):
    import numpy as np
    import torch

    tok, model, scale, steer = load()
    rng = np.random.default_rng(SEED)
    pool = sorted(n for n in scale if n != NEURON)
    controls = [int(n) for n in rng.choice(pool, n_controls, replace=False)]
    if mode == "add":
        conds = [("target", NEURON, k) for k in (0, 4, 8, 16, 32, 64, -16, -32)]
        conds += [("control", n, k) for n in controls for k in (16, 32, 64)]
    else:
        conds = [("target", NEURON, k) for k in (0, 2, 4, 8, 16, -4, -8)]
        conds += [("control", n, k) for n in controls for k in (8, 16)]

    rows = [(p, s) for p in PROMPTS for s in range(SAMPLES)]
    enc = tok([p for p, _ in rows], return_tensors="pt", padding=True).to("cuda")
    out = dict(mode=mode, layer=LAYER, neuron=NEURON, target_scale=scale[NEURON], seed=SEED, max_new=MAX_NEW,
               neutral=NEUTRAL, frustrated=FRUSTRATED, gpu=torch.cuda.get_device_name(), runs=[])
    for kind, n, k in conds:
        value = k * scale[n]
        steer.set(n if k else None, value, add=(mode == "add"))
        torch.manual_seed(SEED)  # same draws per condition -> k=0 vs k>0 differ only by the steer
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.8, top_p=0.95,
                                 pad_token_id=tok.pad_token_id)
        steer.clear()
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (p, s), t in zip(rows, texts):
            out["runs"].append(dict(kind=kind, neuron=n, k=k, value=value, prompt=p, sample=s, text=t))
        print(f"{kind} N{n} k={k}: {texts[0][:100]!r}", flush=True)
    return out


@app.function(image=image, volumes={VOL: vol}, gpu="H200", cpu=8, memory=131072, timeout=3600)
def live_gen(prompt: str, k: float, neuron: int, max_new: int, seed: int, add: bool = False):
    import threading

    import torch
    from transformers import TextIteratorStreamer

    tok, model, scale, steer = load()
    steer.set(neuron if k else None, k * scale[neuron], add=add)
    yield f"[L{LAYER}-N{neuron} {'+=' if add else 'set to'} {k} x {scale[neuron]:.3f} = {k * scale[neuron]:.3f}]\n{prompt}"
    enc = tok(prompt, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    torch.manual_seed(seed)
    th = threading.Thread(target=model.generate, kwargs=dict(**enc, streamer=streamer, max_new_tokens=max_new,
                                                             do_sample=True, temperature=0.8, top_p=0.95))
    th.start()
    for piece in streamer:
        yield piece
    th.join()


@app.local_entrypoint()
def main(n_controls: int = 10, mode: str = "clamp"):
    res = sweep.remote(n_controls, mode)
    path = f"{mode}.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {path} ({len(res['runs'])} generations)")


@app.local_entrypoint()
def live(prompt: str, k: float = 8.0, neuron: int = NEURON, max_new: int = 120, seed: int = SEED, add: bool = False):
    for piece in live_gen.remote_gen(prompt, k, neuron, max_new, seed, add):
        print(piece, end="", flush=True)
    print()
