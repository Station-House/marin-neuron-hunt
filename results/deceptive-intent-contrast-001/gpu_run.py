"""deceptive-intent-contrast-001 (type E, PR #57): record every MLP neuron in every layer at the [[ ]] target
for the 84 prompts (36 pairs + 12 fillers). Activation = silu(gate) * up, the input to down_proj.

Run from the repo root:
  MODAL_PROFILE=... modal run results/deceptive-intent-contrast-001/gpu_run.py \
      --prompts requests/deceptive-intent-contrast-001/prompts.json --out-path acts.npz
"""
import json

import modal

VOL = "/vol"
MODEL_DIR = f"{VOL}/models/marin-32b-bf16"
POLE_STATS = f"{VOL}/runs/stageB2/summaries/pole_stats.parquet"
app = modal.App("marin-hunt-deceptive-intent-001")
vol = modal.Volume.from_name("marin-neuron-atlas")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.11.*", "transformers==5.10.2", "accelerate", "safetensors", "pyarrow", "numpy", "pyyaml")
         .add_local_python_source("runner"))


def flatten(d):
    rows = []
    for c in d["contrasts"]:
        for p in c["pairs"]:
            for arm in ("high", "low"):
                rows.append(dict(id=f"{p['id']}-{arm}", contrast=c["id"], pair=p["id"], arm=arm, **{
                    k: p[arm][k] for k in ("text", "target_id", "target_pos")}))
    for f in d["fillers"]:
        rows.append(dict(id=f["id"], contrast="filler", pair=f["id"], arm="filler",
                         **{k: f[k] for k in ("text", "target_id", "target_pos")}))
    return rows


@app.function(image=image, volumes={VOL: vol}, gpu="H200", cpu=8, memory=131072, timeout=3600)
def run(prompts):
    import numpy as np
    import torch

    from runner.model import Marin
    from runner.prompts import encode

    m = Marin.load(MODEL_DIR, POLE_STATS)
    L, N = len(m.layers), m.n_neurons
    acts = np.zeros((len(prompts), L, N), dtype=np.float32)
    meta = []
    for i, p in enumerate(prompts):
        e = encode(m.tok, p["text"])
        assert e.ids[e.pos] == p["target_id"], (p["id"], e.ids[e.pos], p["target_id"])
        hooks = []
        for li, layer in enumerate(m.layers):
            def fn(module, args, li=li):
                acts[i, li] = args[0][0, e.pos, :].float().cpu().numpy()
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(fn))
        with torch.no_grad():
            m.model(input_ids=torch.tensor([e.ids], device=m.device))
        for h in hooks:
            h.remove()
        meta.append(dict(id=p["id"], pos=e.pos, token=e.token, n_tokens=len(e.ids),
                         pos_matches_file=e.pos == p["target_pos"]))
    return {"gpu": torch.cuda.get_device_name(), "dtype": str(next(m.model.parameters()).dtype),
            "meta": meta, "acts": acts}


@app.local_entrypoint()
def main(prompts: str, out_path: str):
    import numpy as np

    rows = flatten(json.load(open(prompts)))
    res = run.remote(rows)
    np.savez_compressed(out_path, acts=res["acts"].astype(np.float16), rows=json.dumps(rows),
                        meta=json.dumps(res["meta"]), gpu=res["gpu"], dtype=res["dtype"])
    print("saved", out_path, res["gpu"], res["dtype"], res["acts"].shape)
