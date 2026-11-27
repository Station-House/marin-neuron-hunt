"""Step 3: print the tables in README.md from results.json (nothing is typed by hand).

Usage: python make_table.py results.json > tables.md
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
R = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))


def toks(P, n=5):
    return ", ".join(f"`{t}` {v:g}" for t, v in P["top10"][:n])


def claimed(P):
    if not P["claim_tokens"]:
        return "-"
    best = sorted(P["claim_tokens"], key=lambda c: c["rank"])
    return ", ".join(f"`{c['token']}` #{c['rank']}" for c in best[:4])


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


print("| neuron | pole | explanation says | claimed tokens (rank in raised order) | lens top 5 raised (value) | top-1 z | verdict |")
print("|---|---|---|---|---|---|---|")
for nid, r in R["neurons"].items():
    for pole in ("pos", "neg"):
        P = r["poles"][pole]
        kind = P["claim"]["kind"]
        says = {"promotes": "promotes " + "/".join(f"`{t}`" for t in P["claim"]["tokens"][:3]),
                "noise": "lens end is noise", "none": "(silent)"}[kind]
        verdict = P.get("verdict_rule_based", "judge by hand" if kind == "noise" else "n/a")
        print(f"| {nid} | {pole} | {says} | {claimed(P)} | {toks(P)} | {P['top1_z']} | {verdict} |")

print()
print("| neuron | pole | n firings past q999 (sampled) | next token in lens top-10 | base rate of those 10 | enrichment | same-layer random neurons | firing token itself in lens top-10 | median rank percentile of actual next token |")
print("|---|---|---|---|---|---|---|---|---|")
for nid, r in R["neurons"].items():
    for pole in ("pos", "neg"):
        B = r["poles"][pole]["behaviour"]
        if "hit_top10" not in B:
            print(f"| {nid} | {pole} | {B['n_firings_past_q999']} (0) | - | - | - | - | - | - |")
            continue
        print(f"| {nid} | {pole} | {B['n_firings_past_q999']} ({B['n_sampled']}) | {B['hit_top10']}/{B['n_sampled']} = {pct(B['hit_top10_rate'])} "
              f"[{pct(B['hit_top10_ci95'][0])}, {pct(B['hit_top10_ci95'][1])}] | {pct(B['base_rate_top10'])} | "
              f"{B['enrichment_top10']}x | {pct(B['control_hit_top10_rate'])} | {pct(B['cur_hit_top10_rate'])} | {B['median_rank_percentile_of_actual_next']} |")

print()
print("| neuron | pole | atlas top-10 overlap with ours | no-norm-weight lens overlap | cited values check (cited -> ours) |")
print("|---|---|---|---|---|")
for nid, r in R["neurons"].items():
    for pole in ("pos", "neg"):
        P = r["poles"][pole]
        cc = "; ".join(f"`{c['token']}` {c['cited']} -> {c['ours']}" for c in P.get("cited_check", [])[:4]) or "-"
        print(f"| {nid} | {pole} | {P['atlas_top10_overlap']}/10 | {P['raw_lens_top10_overlap']}/10 | {cc} |")
