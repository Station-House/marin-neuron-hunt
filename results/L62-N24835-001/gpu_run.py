"""PR #41 (L62-N24835-001) forward passes on Modal: record readout neurons at every position + P2 naming scores."""
import json, pickle, re, sys, time
from pathlib import Path
import modal

VOL = "/vol"
MODEL_DIR = f"{VOL}/models/marin-32b-bf16"
POLE_STATS = f"{VOL}/runs/stageB2/summaries/pole_stats.parquet"
app = modal.App("marin-hunt-pr41-capybara")
vol = modal.Volume.from_name("marin-neuron-atlas")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.11.*", "transformers==5.10.2", "accelerate", "safetensors", "pyarrow", "numpy", "pyyaml")
         .add_local_python_source("runner"))

SIDE = re.compile(r"^L(\d+)-N(\d+)([+-])$")


@app.function(image=image, volumes={VOL: vol}, gpu="H200", cpu=8, memory=131072, timeout=3 * 3600)
def run(prompts, neurons):
    import numpy as np, torch
    from runner.model import Marin
    from runner.prompts import encode
    m = Marin.load(MODEL_DIR, POLE_STATS)
    tok = m.tok
    by_layer = {}
    for j, (l, n) in enumerate(neurons):
        by_layer.setdefault(l, []).append((j, n))
    out = {"gpu": torch.cuda.get_device_name(), "prompts": []}
    t0 = time.time()
    # --- activation prompts, one at a time (<=48 tokens), every position
    for p in prompts["acts"]:
        e = encode(tok, p["text"])
        ids = torch.tensor([e.ids], device=m.device)
        acts = np.zeros((len(e.ids), len(neurons)), dtype=np.float32)
        hs = []
        for l, spec in by_layer.items():
            idx = torch.tensor([n for _, n in spec], device=m.device)
            cols = [j for j, _ in spec]
            def fn(module, args, idx=idx, cols=cols):
                acts[:, cols] = args[0][0][:, idx].float().cpu().numpy()
            hs.append(m.layers[l].mlp.down_proj.register_forward_pre_hook(fn))
        with torch.no_grad():
            m.model(input_ids=ids)
        for h in hs: h.remove()
        out["prompts"].append(dict(p, ids=e.ids, pos=e.pos, token=e.token,
                                   tokens=tok.convert_ids_to_tokens(e.ids), acts=acts))
    print(f"acts done {time.time()-t0:.0f}s", flush=True)
    # --- naming: summed logprob of continuation tokens
    nm = []
    for c in prompts["naming"]:
        pre = tok(c["context"] + c["frame"])["input_ids"]
        full = tok(c["context"] + c["frame"] + c["cont"])["input_ids"]
        assert full[:len(pre)] == pre, (c, pre, full)
        with torch.no_grad():
            lg = torch.log_softmax(m.model(input_ids=torch.tensor([full], device=m.device)).logits[0].float(), -1)
        tl = [float(lg[i - 1, full[i]]) for i in range(len(pre), len(full))]
        nm.append(dict(c, n_tokens=len(tl), token_lps=tl, lp=sum(tl),
                       cont_tokens=tok.convert_ids_to_tokens(full[len(pre):])))
    out["naming"] = nm
    print(f"naming done {time.time()-t0:.0f}s", flush=True)
    return out


@app.local_entrypoint()
def main(req_dir: str, out_path: str):
    d = Path(req_dir)
    P = json.load(open(d / "prompts.json"))
    R = json.load(open(d / "readouts.json"))
    sides = set(R["named_sides"])
    for r in R["readouts"]: sides |= set(r["sides"])
    for s in R["random_sets"]["sets"]:
        for r in s["readouts"]: sides |= set(r["sides"])
    neurons = sorted({(int(SIDE.match(s)[1]), int(SIDE.match(s)[2])) for s in sides})
    print(len(sides), "sides", len(neurons), "neurons")
    acts = []
    for animal, lst in P["scene"].items():
        for i, t in enumerate(lst):
            acts.append(dict(arm="scene", animal=animal, desc=i // 4, ending=i % 4, text=t))
    for animal, lst in P["word"].items():
        for i, t in enumerate(lst):
            acts.append(dict(arm="word", animal=animal, frame=i, text=t))
    for i, t in enumerate(P["ruby"]): acts.append(dict(arm="ruby", idx=i, text=t))
    for i, t in enumerate(P["unrelated"]): acts.append(dict(arm="unrelated", idx=i, text=t))
    N = P["naming"]
    ctxs = [(a, i, c) for a, lst in N["contexts"].items() for i, c in enumerate(lst)]
    ctxs += [("neutral", i, c) for i, c in enumerate(N["neutral_contexts"])]
    naming = [dict(ctx_animal=a, ctx_idx=i, context=c, frame=N["frame"], cand=cand,
                   cont=N["continuation_prefix"] + cand) for a, i, c in ctxs for cand in N["candidates"]]
    print(len(acts), "act prompts", len(naming), "naming scores")
    res = run.remote({"acts": acts, "naming": naming}, neurons)
    res["neurons"] = neurons
    pickle.dump(res, open(out_path, "wb"))
    print("saved", out_path)
