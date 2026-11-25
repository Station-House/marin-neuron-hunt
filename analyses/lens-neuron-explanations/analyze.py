"""Step 2: score the logit lens against the written explanations, and against what the model actually saw next.

Inputs : lens.npz from compute_lens.py, neurons.json, the atlas summaries (logit_lens, pole_stats, tokens, vocab) and the
         `sparse` tier of pilot/fineweb-10m-sparse (queried over hf:// with DuckDB).
Output : results.json (every number quoted in README.md comes from here).

Conventions (the atlas's): a positive-pole firing adds act * W_down[:, i] to the residual stream, so the positive pole
pushes up the tokens with the HIGHEST lens value; a negative-pole firing (act < 0) pushes up the tokens with the LOWEST
lens value. "Raised order" below sorts the vocabulary by what the pole raises, so rank 1 is always the most raised token.

Usage: python analyze.py <lens.npz> <atlas_dir> <out.json>
  atlas_dir holds the downloaded files at the same relative paths as on the Hub
  (pilot/fineweb-10m-sparse/summaries/*.parquet, vocab/vocab.parquet).
"""

import json
import math
import sys
from pathlib import Path

import duckdb
import numpy as np

HERE = Path(__file__).resolve().parent
PREFIX = "hf://datasets/Station-house/marin-32b-neuron-atlas/pilot/fineweb-10m-sparse"
SEQ = 2048
MAX_FIRINGS = 2000      # per pole, sampled with a fixed seed
N_CONTROL = 10          # random neurons per layer used as the control
CONTROL_FIRINGS = 300   # per control neuron
SEED = 20261125


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main(lens_path, atlas, out_path):
    atlas = Path(atlas)
    summ = atlas / "pilot/fineweb-10m-sparse/summaries"
    rng = np.random.default_rng(SEED)
    spec = json.loads((HERE / "neurons.json").read_text())["neurons"]
    z = np.load(lens_path)
    ids = [str(i) for i in z["ids"]]
    lens = z["lens_norm"]
    lens_raw = z["lens_raw"]
    V = lens.shape[1]

    con = duckdb.connect()
    con.execute("install httpfs; load httpfs;")
    vocab = con.execute(f"select token_id, text from '{atlas}/vocab/vocab.parquet' order by token_id").fetchall()
    text = [t for _, t in vocab]
    assert len(text) == V
    by_text = {}
    for i, t in enumerate(text):
        by_text.setdefault(t, []).append(i)
    tok = con.execute(f"select token_idx, token_id from '{summ}/tokens.parquet' order by token_idx").fetchnumpy()
    assert (tok["token_idx"] == np.arange(len(tok["token_idx"]))).all()
    tok = tok["token_id"].astype(np.int64)
    pos_in_doc = np.arange(len(tok)) % SEQ
    freq = np.bincount(tok[pos_in_doc >= 1], minlength=V) / (pos_in_doc >= 1).sum()   # next-token base rates

    results = {"neurons": {}, "controls": {}}

    # per-layer sparse queries: targets + random controls
    layers = sorted({int(n["id"][1:].split("-N")[0]) for n in spec})
    targets = {int(n["id"][1:].split("-N")[0]): [] for n in spec}
    for n in spec:
        targets[int(n["id"][1:].split("-N")[0])].append(int(n["id"].split("-N")[1]))
    firings = {}   # (layer, neuron) -> (token_idx array, act array)
    ctrl_ids = {}
    for layer in layers:
        pool = [i for i in range(27648) if i not in targets[layer]]
        ctrl = [int(i) for i in rng.choice(pool, N_CONTROL, replace=False)]
        ctrl_ids[layer] = ctrl
        want = targets[layer] + ctrl
        q = (f"select token_idx, neuron, gate, up from '{PREFIX}/sparse/layer={layer:02d}/sorted-00000.parquet' "
             f"where neuron in ({','.join(map(str, want))})")
        d = con.execute(q).fetchnumpy()
        gate = d["gate"].astype(np.float32)
        up = d["up"].astype(np.float32)
        act = gate / (1.0 + np.exp(-gate)) * up
        for nn in want:
            m = d["neuron"] == nn
            firings[(layer, nn)] = (d["token_idx"][m].astype(np.int64), act[m])
        print(f"layer {layer}: {sum(firings[(layer, nn)][0].size for nn in want)} stored firings for {len(want)} neurons", flush=True)

    stats = {(r[0], r[1]): r for r in con.execute(
        f"select layer, neuron, pos_q999, neg_q999, pos_q99, neg_q99, max, min from '{summ}/pole_stats.parquet'").fetchall()}
    atlas_ll = {(r[0], r[1]): r for r in con.execute(
        f"select layer, neuron, top_ids, top_vals, bottom_ids, bottom_vals from '{summ}/logit_lens.parquet' "
        f"where (layer, neuron) in ({','.join(f'({L},{N})' for L in layers for N in targets[L])})").fetchall()}

    def firing_sample(layer, neuron, sign, k, thresh_mag):
        idx, act = firings[(layer, neuron)]
        m = (np.sign(act) == sign) & (np.abs(act) >= thresh_mag) & (idx % SEQ != SEQ - 1) & (idx % SEQ != 0)
        idx, act = idx[m], act[m]
        n_all = int(idx.size)
        if n_all > k:
            pick = rng.choice(n_all, k, replace=False)
            idx, act = idx[pick], act[pick]
        return idx, act, n_all

    def next_hits(idx, S):
        nxt = tok[idx + 1]
        return int(np.isin(nxt, S).sum()), nxt

    for spec_n in spec:
        nid = spec_n["id"]
        layer, neuron = (int(x) for x in nid[1:].split("-N"))
        row = ids.index(nid)
        r = {"layer": layer, "neuron": neuron, "file": spec_n["file"], "poles": {}}
        st = stats[(layer, neuron)]
        aL = atlas_ll[(layer, neuron)]
        for pole, sign in (("pos", 1), ("neg", -1)):
            raised = lens[row] * sign                         # larger = more raised by this pole
            order = np.argsort(-raised)
            rank = np.empty(V, dtype=np.int64)
            rank[order] = np.arange(1, V + 1)
            mu, sd = float(raised.mean()), float(raised.std())
            top10 = [int(i) for i in order[:10]]
            atlas_top = list(aL[2] if sign == 1 else aL[4])
            atlas_vals = list(aL[3] if sign == 1 else aL[5])
            overlap = len(set(top10) & set(atlas_top))
            raw_order = np.argsort(-(lens_raw[row] * sign))
            raw_overlap = len(set(top10) & {int(i) for i in raw_order[:10]})
            P = {
                "top10": [(text[i], round(float(lens[row, i]), 3)) for i in top10],
                "atlas_top10_overlap": overlap,
                "atlas_value_max_abs_diff_on_common": round(max(
                    [abs(float(lens[row, i]) - v) for i, v in zip(atlas_top, atlas_vals) if i in top10] or [float("nan")]), 3),
                "raw_lens_top10_overlap": raw_overlap,
                "top1_z": round((float(raised[order[0]]) - mu) / sd, 2),
                "top1_value": round(float(lens[row, order[0]]), 3),
                "claim": spec_n[pole],
            }
            claim = spec_n[pole]
            cids = sorted({i for t in claim["tokens"] for i in by_text.get(t, [])})
            missing = [t for t in claim["tokens"] if t not in by_text]
            P["claim_missing_tokens"] = missing
            P["claim_tokens"] = [{"token": text[i], "rank": int(rank[i]), "value": round(float(lens[row, i]), 3),
                                  "z": round((float(raised[i]) - mu) / sd, 2)} for i in cids]
            for t, cv in claim.get("cited", {}).items():
                for i in by_text.get(t, []):
                    P.setdefault("cited_check", []).append({"token": t, "cited": cv, "ours": round(float(lens[row, i]), 3)})
            if claim["kind"] == "promotes" and cids:
                ranks = [int(rank[i]) for i in cids]
                in10 = sum(rk <= 10 for rk in ranks)
                in100 = sum(rk <= 100 for rk in ranks)
                if in10 >= 1 and in100 * 2 >= len(ranks):
                    verdict = "agree"
                elif in100 >= 1:
                    verdict = "partial"
                else:
                    verdict = "disagree"
                P["verdict_rule_based"] = verdict
                P["claimed_in_top10"], P["claimed_in_top100"], P["claimed_total"] = in10, in100, len(ranks)

            # behaviour: what token actually came next when this pole fired?
            thr = st[2] if sign == 1 else st[3]
            idx, act, n_all = firing_sample(layer, neuron, sign, MAX_FIRINGS, thr)
            strong = np.argsort(-np.abs(act))[:100]
            S10 = np.array(top10)
            K100 = np.array([int(i) for i in order[:100]])
            beh = {"n_firings_past_q999": n_all, "n_sampled": int(idx.size), "q999_mag": float(thr)}
            if idx.size:
                hits10, nxt = next_hits(idx, S10)
                base10 = float(freq[S10].sum())
                lo, hi = wilson(hits10, idx.size)
                beh.update({"hit_top10": hits10, "hit_top10_rate": round(hits10 / idx.size, 4),
                            "hit_top10_ci95": [round(lo, 4), round(hi, 4)], "base_rate_top10": round(base10, 6),
                            "enrichment_top10": round((hits10 / idx.size) / base10, 1) if base10 > 0 else None})
                # same-token variant: is the token the neuron fires ON in the lens top-10? (for neurons that fire on
                # the word they promote, e.g. 'ended' before 'up', rather than on the word before it)
                hcur = int(np.isin(tok[idx], S10).sum())
                beh.update({"cur_hit_top10_rate": round(hcur / idx.size, 4)})
                h100, _ = next_hits(idx, K100)
                base100 = float(freq[K100].sum())
                beh.update({"hit_top100_rate": round(h100 / idx.size, 4), "base_rate_top100": round(base100, 5)})
                pct = rank[nxt] / V
                beh["median_rank_percentile_of_actual_next"] = round(float(np.median(pct)), 3)
                if cids:
                    hc, _ = next_hits(idx, np.array(cids))
                    basec = float(freq[cids].sum())
                    beh.update({"hit_claimed_rate": round(hc / idx.size, 4), "base_rate_claimed": round(basec, 6),
                                "enrichment_claimed": round((hc / idx.size) / basec, 1) if basec > 0 else None})
                s_idx = idx[strong]
                hs, _ = next_hits(s_idx, S10)
                beh["strongest100_hit_top10_rate"] = round(hs / s_idx.size, 4)
                # control: strong firings of random neurons in the same layer, scored against THIS pole's top-10
                ch = cn = 0
                cb = []
                for cn_id in ctrl_ids[layer]:
                    cst = stats[(layer, cn_id)]
                    for csign, cthr in ((1, cst[2]), (-1, cst[3])):
                        ci, _, _ = firing_sample(layer, cn_id, csign, CONTROL_FIRINGS // 2, cthr)
                        if ci.size:
                            h, _ = next_hits(ci, S10)
                            ch += h
                            cn += ci.size
                beh["control_hit_top10_rate"] = round(ch / cn, 4) if cn else None
                beh["control_n"] = cn
            P["behaviour"] = beh
            r["poles"][pole] = P
        results["neurons"][nid] = r
        print("done", nid, flush=True)

    results["controls"] = {str(k): v for k, v in ctrl_ids.items()}
    results["vocab_size"] = V
    Path(out_path).write_text(json.dumps(results, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:4])
