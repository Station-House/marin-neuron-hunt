"""Step 1: recompute the logit lens for the neurons in neurons.json straight from the Marin 32B weights.

lens(i) = W_U @ (W_down[:, i] * final_norm.weight)         ("with norm weight", the atlas definition)
lens_raw(i) = W_U @ W_down[:, i]                           (sensitivity check: no norm weight)

Only three kinds of tensors are read, each by HTTP range request, so the other ~120 GB of the model is never
downloaded: model.layers.L.mlp.down_proj.weight for the layers in neurons.json (about 0.57 GB each, fp32),
model.norm.weight, and lm_head.weight (2.6 GB, streamed in row chunks). Peak RAM stays under about 1 GB.

Usage: python compute_lens.py <out.npz>
"""

import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
REPO = "marin-community/marin-32b-base"
REV = "f19b346615212ff9a6d92c10861a9cdb100cbf97"  # the revision the atlas was extracted from
BASE = f"https://huggingface.co/{REPO}/resolve/{REV}/"
SESSION = requests.Session()


def get_range(path, start, end, tries=6):
    """Bytes start..end-1 of a file in the repo."""
    for k in range(tries):
        try:
            r = SESSION.get(BASE + path, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=180)
            if r.status_code in (200, 206) and len(r.content) == end - start:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"could not read {path} {start}-{end}")


def header(path):
    n = struct.unpack("<Q", get_range(path, 0, 8))[0]
    h = json.loads(get_range(path, 8, 8 + n))
    return h, 8 + n


def tensor_rows(path, hdr, base, name, r0, r1):
    """Rows r0..r1 of a 2-D tensor, as float32."""
    meta = hdr[name]
    assert meta["dtype"] in ("F32", "BF16"), meta["dtype"]
    rows, cols = meta["shape"]
    width = 4 if meta["dtype"] == "F32" else 2
    off = meta["data_offsets"][0]
    raw = get_range(path, base + off + r0 * cols * width, base + off + r1 * cols * width)
    if meta["dtype"] == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(r1 - r0, cols)
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return u16.view(np.float32).reshape(r1 - r0, cols)


def main(out):
    spec = json.loads((HERE / "neurons.json").read_text())
    ids = [n["id"] for n in spec["neurons"]]
    idx = json.loads(SESSION.get(BASE + "model.safetensors.index.json", timeout=60).content)["weight_map"]
    by_layer = {}
    for nid in ids:
        layer, neuron = nid.replace("L", "").split("-N")
        by_layer.setdefault(int(layer), []).append((nid, int(neuron)))

    # 1. the down_proj columns, one layer at a time
    cols = {}
    for layer, items in sorted(by_layer.items()):
        name = f"model.layers.{layer}.mlp.down_proj.weight"
        shard = idx[name]
        hdr, base = header(shard)
        shape = hdr[name]["shape"]  # [hidden, intermediate] = [5120, 27648]
        assert shape == [5120, 27648], shape
        want = [n for _, n in items]
        buf = np.zeros((5120, len(want)), dtype=np.float32)
        step = 512
        for r0 in range(0, 5120, step):
            chunk = tensor_rows(shard, hdr, base, name, r0, min(r0 + step, 5120))
            buf[r0:r0 + chunk.shape[0]] = chunk[:, want]
        for j, (nid, _) in enumerate(items):
            cols[nid] = buf[:, j].copy()
        print(f"layer {layer}: read down_proj columns for {[i for i, _ in items]}", flush=True)

    # 2. final norm weight
    nshard = idx["model.norm.weight"]
    nhdr, nbase = header(nshard)
    meta = nhdr["model.norm.weight"]
    assert meta["dtype"] == "F32", meta["dtype"]
    nw = np.frombuffer(get_range(nshard, nbase + meta["data_offsets"][0], nbase + meta["data_offsets"][1]), dtype="<f4")
    print("final norm weight", nw.shape, float(nw.mean()), flush=True)

    # 3. unembedding, streamed in row chunks
    V = np.stack([cols[i] for i in ids], axis=1)          # [5120, n]
    Vn = V * nw[:, None]
    lshard = idx["lm_head.weight"]
    lhdr, lbase = header(lshard)
    vocab = lhdr["lm_head.weight"]["shape"][0]
    out_norm = np.zeros((len(ids), vocab), dtype=np.float32)
    out_raw = np.zeros((len(ids), vocab), dtype=np.float32)
    step = 4096
    for r0 in range(0, vocab, step):
        W = tensor_rows(lshard, lhdr, lbase, "lm_head.weight", r0, min(r0 + step, vocab))
        out_norm[:, r0:r0 + W.shape[0]] = (W @ Vn).T
        out_raw[:, r0:r0 + W.shape[0]] = (W @ V).T
        print(f"lm_head rows {r0}..{r0 + W.shape[0]}", flush=True)
    np.savez_compressed(out, ids=np.array(ids), lens_norm=out_norm, lens_raw=out_raw,
                        down_norm=np.linalg.norm(V, axis=0))
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1])
