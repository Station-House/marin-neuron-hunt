"""Count swears per condition: python score.py clamp.json  (or add.json); prints a markdown table."""

import json
import re
import sys
from collections import defaultdict

SOFT = r"heck|darn|darned|dang|gosh|golly|shoot|frick|fricking|freaking|crud|jeez|geez|gee"
STRONG = r"hell|damn|damned|god|christ|jesus|goddamn|shit|fuck\w*|crap|bloody"
soft_re = re.compile(rf"\b({SOFT})\b", re.I)
strong_re = re.compile(rf"\b({STRONG})\b", re.I)

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "clamp.json"))
neutral = set(d["neutral"])
agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))  # [n, any_soft, any_strong, soft_count]
words = defaultdict(lambda: defaultdict(int))
for r in d["runs"]:
    key = (r["kind"], r["neuron"] if r["kind"] == "target" else "random", r["k"])
    grp = "neutral" if r["prompt"] in neutral else "frustrated"
    s, g = soft_re.findall(r["text"]), strong_re.findall(r["text"])
    a = agg[key][grp]
    a[0] += 1; a[1] += bool(s); a[2] += bool(g); a[3] += len(s)
    for w in s + g:
        words[key][w.lower()] += 1

lines = [f"# L{d['layer']}-N{d['neuron']} steering sweep ({d.get('mode', 'clamp')})", "",
         f"scale_abs_p999 = {d['target_scale']:.3f}; {d['max_new']} new tokens, T=0.8, seed {d['seed']}, {d['gpu']}", "",
         "Share of generations containing a soft swear / a strong oath.", "",
         "| condition | k | neutral soft | neutral strong | frustrated soft | frustrated strong | words |",
         "|---|---|---|---|---|---|---|"]
for key in sorted(agg, key=lambda x: (x[0] != "target", x[2] < 0, abs(x[2]))):
    n_, f_ = agg[key]["neutral"], agg[key]["frustrated"]
    pct = lambda a, i: f"{a[i]}/{a[0]}" if a[0] else "-"
    top = ", ".join(f"{w}×{c}" for w, c in sorted(words[key].items(), key=lambda x: -x[1])[:6])
    name = "target" if key[0] == "target" else "random L20 (pooled)"
    lines.append(f"| {name} | {key[2]} | {pct(n_,1)} | {pct(n_,2)} | {pct(f_,1)} | {pct(f_,2)} | {top} |")

lines += ["", "## Samples (target, first sample of each prompt)", ""]
for k in sorted({r["k"] for r in d["runs"] if r["kind"] == "target"}, key=lambda k: (k < 0, abs(k))):
    lines.append(f"### k = {k}")
    for r in d["runs"]:
        if r["kind"] == "target" and r["k"] == k and r["sample"] == 0:
            lines.append(f"- **{r['prompt']}** {r['text'].strip()[:300].replace(chr(10), ' ')}")
    lines.append("")
print("\n".join(lines))
