"""L20-N1234-003 (type E, PR #31): record every L20 neuron at the [[ ]] target for the 36 frame prompts.

Run from the repo root: MODAL_PROFILE=... modal run results/L20-N1234-003/gpu_run.py --request requests/L20-N1234-003.yaml --out-path acts.pkl
"""
import pickle
import re

import modal
import yaml

VOL = "/vol"
MODEL_DIR = f"{VOL}/models/marin-32b-bf16"
POLE_STATS = f"{VOL}/runs/stageB2/summaries/pole_stats.parquet"
LAYER = 20
app = modal.App("marin-hunt-l20-n1234-003")
vol = modal.Volume.from_name("marin-neuron-atlas")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.11.*", "transformers==5.10.2", "accelerate", "safetensors", "pyarrow", "numpy", "pyyaml")
         .add_local_python_source("runner"))


def parse_frames(procedure):
    """12 frames x {S, M, N} from the request's procedure text."""
    frames, cur = [], None
    for line in procedure.splitlines():
        m = re.match(r'^\s*(\d+)?\s*([SMN]):\s*"(.*)"\s*$', line)
        if not m:
            continue
        if m[1]:
            cur = {"frame": int(m[1])}
            frames.append(cur)
        cur[m[2]] = m[3]
    assert len(frames) == 12 and all(set(f) == {"frame", "S", "M", "N"} for f in frames), frames
    return frames


@app.function(image=image, volumes={VOL: vol}, gpu="H200", cpu=8, memory=131072, timeout=3600)
def run(prompts):
    import numpy as np
    import pyarrow.parquet as pq
    import torch

    from runner.model import Marin
    from runner.prompts import encode

    m = Marin.load(MODEL_DIR, POLE_STATS)
    t = pq.read_table(POLE_STATS, filters=[("layer", "=", LAYER)])
    order = np.argsort(t["neuron"].to_numpy())
    stats = {c: t[c].to_numpy()[order] for c in ("neuron", "pos_q99", "neg_q99")}
    out = []
    for p in prompts:
        e = encode(m.tok, p["text"])
        cap = {}

        def fn(module, args):
            cap["x"] = args[0][0, e.pos, :].float().cpu().numpy()

        h = m.layers[LAYER].mlp.down_proj.register_forward_pre_hook(fn)
        with torch.no_grad():
            m.model(input_ids=torch.tensor([e.ids], device=m.device))
        h.remove()
        out.append(dict(p, ids=e.ids, pos=e.pos, token=e.token, acts=cap["x"]))
    return {"gpu": torch.cuda.get_device_name(), "prompts": out, "stats": stats}


@app.local_entrypoint()
def main(request: str, out_path: str):
    req = yaml.safe_load(open(request))
    prompts = [dict(frame=f["frame"], arm=a, text=f[a]) for f in parse_frames(req["procedure"]) for a in "SMN"]
    res = run.remote(prompts)
    pickle.dump(res, open(out_path, "wb"))
    print("saved", out_path, res["gpu"])
