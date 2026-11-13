"""Score PR #41 (L62-N24835-001) exactly as pre-registered in its YAML/DESIGN.md.
usage: score.py <req_dir> <gpu.pkl> <out_dir>"""
import json, pickle, re, sys
from pathlib import Path

import numpy as np
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

req, pkl, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
R = json.load(open(req / "readouts.json"))
G = pickle.load(open(pkl, "rb"))
ANIMALS = R["animals"]
SIDE = re.compile(r"^L(\d+)-N(\d+)([+-])$")
col = {tuple(n): j for j, n in enumerate(G["neurons"])}
P = G["prompts"]
arm = np.array([p["arm"] for p in P])
animal = np.array([p.get("animal", "") for p in P])
desc = np.array([p.get("desc", -1) for p in P])
ending = np.array([p.get("ending", p.get("frame", -1)) for p in P])


def side_acts(sides, site):
    """[n_prompts, len(sides)] pole-oriented activations at site 0 (target) or 1 (next)."""
    X = np.zeros((len(P), len(sides)))
    for k, s in enumerate(sides):
        l, n, pole = SIDE.match(s).groups()
        j = col[(int(l), int(n))]
        sign = 1.0 if pole == "+" else -1.0
        for i, p in enumerate(P):
            X[i, k] = sign * p["acts"][p["pos"] + site, j]
    return X


ACT_CACHE = {}


def all_acts(site):
    if site not in ACT_CACHE:
        ACT_CACHE[site] = np.stack([p["acts"][p["pos"] + site] for p in P])  # raw, neuron order
    return ACT_CACHE[site]


def score(r, site):
    X = np.zeros((len(P), len(r["sides"])))
    A = all_acts(site)
    for k, s in enumerate(r["sides"]):
        l, n, pole = SIDE.match(s).groups()
        X[:, k] = (1.0 if pole == "+" else -1.0) * A[:, col[(int(l), int(n))]]
    z = (X - np.array(r["mu"])) / np.array(r["sd"])
    return (r["bias"] + z @ np.array(r["weights"]) - r["neg_mean"]) / r["neg_sd"]


def arm_stats(s, t, which):
    """Margin, slots, breadth for scores s with target animal t on arm `which` ('scene' or 'word')."""
    m = arm == which
    means = {a: float(s[m & (animal == a)].mean()) for a in ANIMALS}
    rivals = [a for a in ANIMALS if a != t]
    top_rival = max(rivals, key=lambda a: means[a])
    rv = s[m & (animal != t)]
    sd_pool = float(rv.std(ddof=1))
    gap = means[t] - means[top_rival]
    res = dict(means=means, top_rival=top_rival, gap=gap, sd_pool=sd_pool, margin_pool=gap / sd_pool)
    # matched slots
    keys = sorted({(int(d), int(e)) for d, e in zip(desc[m], ending[m])})
    wins = []
    for d, e in keys:
        sl = m & (desc == d) & (ending == e)
        tv = s[sl & (animal == t)][0]
        wins.append(bool(all(tv > s[sl & (animal == a)][0] for a in rivals)))
    res["slot_wins"] = int(sum(wins)); res["n_slots"] = len(keys); res["slots"] = wins
    if which == "scene":
        dm = np.array([s[m & (animal == a) & (desc == d)].mean() for a in rivals for d in range(3)])
        sd_desc = float(dm.std(ddof=1))
        res["sd_descmean"] = sd_desc
        res["margin_descmean"] = gap / sd_desc
        res["margin_conservative"] = gap / max(sd_pool, sd_desc)  # governs S1/S3
        res["margin_liberal"] = gap / min(sd_pool, sd_desc)       # refutation bar
        # variance decomposition of rival scores
        ra = [a for a in rivals]
        vals = np.array([[[s[m & (animal == a) & (desc == d) & (ending == e)][0] for e in range(4)] for d in range(3)] for a in ra])
        g = vals.mean()
        ss_a = 12 * ((vals.mean((1, 2)) - g) ** 2).sum()
        ss_d = 4 * ((vals.mean(2) - vals.mean((1, 2))[:, None]) ** 2).sum()
        ss_e = ((vals - vals.mean(2)[:, :, None]) ** 2).sum()
        tot = ss_a + ss_d + ss_e
        res["var_frac"] = dict(animal=ss_a / tot, description=ss_d / tot, ending=ss_e / tot)
        per = []
        for d in range(3):
            sl = m & (desc == d)
            tm = float(s[sl & (animal == t)].mean())
            rm = {a: float(s[sl & (animal == a)].mean()) for a in rivals}
            best = max(rm, key=rm.get)
            w = sum(res["slots"][d * 4:(d + 1) * 4])
            per.append(dict(desc=d, target_mean=tm, top_rival=best, top_rival_mean=rm[best], wins=w,
                            margin=(tm - rm[best]) / max(sd_pool, sd_desc), ok=bool(w >= 3 and tm > rm[best])))
        res["per_desc"] = per
        res["breadth_ok"] = all(x["ok"] for x in per)
        # South American cluster check
        sa = ["capybara", "tapir", "mara", "nutria", "guinea pig"]
        order = sorted(ANIMALS, key=lambda a: -means[a])
        res["rank"] = order
        res["sa_cluster"] = set(order[:5]) == set(sa)
    return res


def contamination(s):
    out = {}
    for site_name, sc in s.items():
        unr = sc[arm == "unrelated"].mean()
        cw = sc[(arm == "word") & (animal == "capybara")].mean()
        ruby = sc[arm == "ruby"]
        out[site_name] = dict(frac=float((ruby.mean() - unr) / (cw - unr)),
                              per_prompt=[float((v - unr) / (cw - unr)) for v in ruby],
                              unrelated=float(unr), capy_word=float(cw), ruby_mean=float(ruby.mean()))
    return out


# ---------------- P2: naming
N = G["naming"]
cands = sorted({n["cand"] for n in N}, key=ANIMALS.index)
lp = {(n["ctx_animal"], n["ctx_idx"], n["cand"]): n["lp"] for n in N}
neutral = {c: np.array([lp[("neutral", i, c)] for i in range(9)]) for c in cands}
base = {c: neutral[c].mean() for c in cands}
lift = {(a, i, c): lp[(a, i, c)] - base[c] for (a, i, c) in lp}
p2_rows = []
for a in ANIMALS:
    for i in range(3):
        l = {c: lift[(a, i, c)] for c in cands}
        order = sorted(cands, key=lambda c: -l[c])
        p2_rows.append(dict(ctx_animal=a, ctx_idx=i, lifts=l, raw={c: lp[(a, i, c)] for c in cands},
                            true_rank=order.index(a) + 1, top=order[0], runner_up=order[1]))
cap_sd = float(neutral["capybara"].std(ddof=1)); cap_range = float(np.ptp(neutral["capybara"]))
own = [r for r in p2_rows if r["ctx_animal"] == "capybara"]
clear = []
for r in own:
    others = sorted((v for c, v in r["lifts"].items() if c != "capybara"), reverse=True)
    clear.append(bool(r["top"] == "capybara" and r["lifts"]["capybara"] - others[0] > cap_sd))
own_mean = float(np.mean([r["lifts"]["capybara"] for r in own]))
rival_mean = float(np.mean([r["lifts"]["capybara"] for r in p2_rows if r["ctx_animal"] != "capybara"]))
P2 = dict(own_mean_lift=own_mean, rival_ctx_mean_lift=rival_mean, clear_wins=clear, cap_neutral_sd=cap_sd,
          cap_neutral_range=cap_range,
          neutral={c: dict(mean=float(neutral[c].mean()), sd=float(neutral[c].std(ddof=1)), range=float(np.ptp(neutral[c])))
                   for c in cands},
          validated=cap_range <= 1.0)
P2["pass"] = bool(P2["validated"] and own_mean > 0 and sum(clear) >= 2 and own_mean > rival_mean)
P2["true_rank_counts"] = {a: [r["true_rank"] for r in p2_rows if r["ctx_animal"] == a] for a in ANIMALS}
P2["rows"] = p2_rows


# ---------------- P1: scene-fitted readout
scene = arm == "scene"
named_cols = [col[(int(SIDE.match(s)[1]), int(SIDE.match(s)[2]))] for s in R["named_sides"]]
X_all = all_acts(0)[scene]
sa_an, sa_desc = animal[scene], desc[scene]


def crossfit(X, target, seed_perm=None):
    y = (sa_an == target).astype(int)
    if seed_perm is not None:
        y = y.copy()
        for d in range(3):  # shuffle within folds
            idx = np.where(sa_desc == d)[0]
            y[idx] = seed_perm.permutation(y[idx])
    held = np.zeros(len(y))
    for d in range(3):
        tr, te = sa_desc != d, sa_desc == d
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
        clf.fit(sc.transform(X[tr]), y[tr])
        held[te] = clf.decision_function(sc.transform(X[te]))
    full = np.zeros(len(P)); full[scene] = held
    return full


def p1_bars(st):
    return bool(st["margin_conservative"] >= 1.0 and st["slot_wins"] >= 10 and st["breadth_ok"])


P1 = {}
p1_scores = {a: crossfit(X_all, a) for a in ANIMALS}
p1_stats = {a: arm_stats(p1_scores[a], a, "scene") for a in ANIMALS}
P1["full"] = p1_stats["capybara"]
P1["full_pass"] = p1_bars(p1_stats["capybara"])
P1["ovr"] = {a: dict(margin=p1_stats[a]["margin_conservative"], slots=p1_stats[a]["slot_wins"],
                     breadth=p1_stats[a]["breadth_ok"], passes=p1_bars(p1_stats[a])) for a in ANIMALS}
P1["others_passing"] = sum(P1["ovr"][a]["passes"] for a in ANIMALS if a != "capybara")
red = crossfit(X_all[:, named_cols], "capybara")
P1["reduced"] = arm_stats(red, "capybara", "scene")
P1["reduced_pass"] = p1_bars(P1["reduced"])
rng = np.random.default_rng(0)
null = [arm_stats(crossfit(X_all, "capybara", rng), "capybara", "scene")["margin_conservative"] for _ in range(200)]
P1["perm_null"] = dict(values=null, observed=P1["full"]["margin_conservative"],
                       pct=float((np.array(null) < P1["full"]["margin_conservative"]).mean()),
                       q95=float(np.quantile(null, .95)))
P1["pass"] = bool(P1["full_pass"] or P1["reduced_pass"])
P1["scored_by"] = "full" if P1["full_pass"] else ("reduced" if P1["reduced_pass"] else "none")

controls_ok = P1["pass"] or P2["pass"]


# ---------------- candidates (stage 2)
def evaluate(r):
    s0, s1 = score(r, 0), score(r, 1)
    t = r["target"]
    sc = arm_stats(s0, t, "scene")
    wd = arm_stats(s0, t, "word")
    out = dict(id=r["id"], kind=r["kind"], target=t, size=len(r["sides"]), sides=r["sides"],
               scene=sc, word=wd, scene_next=arm_stats(s1, t, "scene"), word_next=arm_stats(s1, t, "word"),
               scores_target=s0.tolist(), scores_next=s1.tolist())
    out["contam"] = contamination({"target": s0, "next": s1})
    return out


cand = {r["id"]: evaluate(r) for r in R["readouts"]}
tapir4 = cand["tapir sum 4"]
V2 = not (tapir4["scene"]["margin_conservative"] >= 1.0 and tapir4["scene"]["slot_wins"] >= 10)

# random baselines (scene governing margin at target)
rand = {}
for st in R["random_sets"]["sets"]:
    for r in st["readouts"]:
        s0 = score(r, 0)
        a = arm_stats(s0, "capybara", "scene")
        rand.setdefault((st["size"], r["kind"]), []).append(a["margin_conservative"])
lo, hi = int(binom.ppf(0.025, 1000, 0.99)), int(binom.ppf(0.975, 1000, 0.99)) + 1
RAND = {}
for k, v in rand.items():
    v = np.sort(np.array(v))
    RAND[f"{k[0]}-{k[1]}"] = dict(q99=float(np.quantile(v, .99)), ci=[float(v[lo - 1]), float(v[hi - 1])],
                                  ci_ranks=[lo, hi], max=float(v[-1]), median=float(np.median(v)),
                                  values=v.tolist())

CAPY = [r["id"] for r in R["readouts"] if r["target"] == "capybara"]
verdicts = {}
for rid in CAPY:
    c = cand[rid]
    sc = c["scene"]
    key = f"{c['size']}-{c['kind']}"
    rb = RAND[key]
    pct = float((np.array(rb["values"]) < sc["margin_conservative"]).mean())
    ck = dict(
        V1=c["word"]["margin_pool"] >= 1.0, V2=V2,
        S1=sc["margin_conservative"] >= 1.0, S2=sc["slot_wins"] >= 10,
        S3=sc["margin_conservative"] > rb["q99"], S4=sc["breadth_ok"],
        C1=c["contam"]["target"]["frac"] <= 0.35 and c["contam"]["next"]["frac"] <= 0.35,
        C2=max(c["contam"]["target"]["per_prompt"]) <= 0.50)
    ck = {k: bool(v) for k, v in ck.items()}
    s3_noise = bool(rb["ci"][0] <= sc["margin_conservative"] <= rb["ci"][1])
    both_low = sc["margin_pool"] < 0.5 and sc["margin_descmean"] < 0.5
    disagree = (sc["margin_pool"] < 0.5) != (sc["margin_descmean"] < 0.5)
    if all(ck.values()):
        v = "confirmed"
    elif ck["V1"] and ck["V2"] and P1["pass"] and P2["pass"] and (sc["slot_wins"] < 6 or (both_low and not disagree)):
        v = "refuted"
    else:
        v = "inconclusive"
    why = []
    if not ck["V1"]: why.append("fails V1 (word margin < 1.0): not a valid capybara readout here")
    if not ck["V2"]: why.append("V2 failed: scene arm leaks word form")
    if v == "inconclusive" and ck["V1"]:
        if not (P1["pass"] and P2["pass"]) and (sc["slot_wins"] < 6 or both_low):
            why.append("would be refuted but a positive control failed")
        elif disagree and sc["slot_wins"] >= 6:
            why.append("the two scene margins disagree across 0.5")
        else:
            why.append("between the confirm and refute bars")
    sa_label = bool((not ck["S1"]) and sc["sa_cluster"] and sc["top_rival"] in ("tapir", "mara", "nutria", "guinea pig"))
    verdicts[rid] = dict(checks=ck, verdict=v, why=why, s3_percentile=pct, s3_q99=rb["q99"], s3_ci=rb["ci"],
                         s3_within_noise=s3_noise, sa_cluster=sa_label)

# both poles of the five singles: raw activation per animal
poles = {}
for r in R["readouts"][:5]:
    side = r["sides"][0]
    for site in (0, 1):
        x = side_acts([side], site)[:, 0]
        for flip in (1, -1):
            name = side if flip == 1 else side[:-1] + ("-" if side[-1] == "+" else "+")
            st_s = arm_stats(flip * x, "capybara", "scene"); st_w = arm_stats(flip * x, "capybara", "word")
            poles.setdefault(name, {})["target" if site == 0 else "next"] = dict(
                scene_margin=st_s["margin_conservative"], scene_slots=st_s["slot_wins"], scene_top=st_s["top_rival"],
                word_margin=st_w["margin_pool"], word_slots=st_w["slot_wins"],
                scene_means=st_s["means"], word_means=st_w["means"])

v1_pass = [k for k in CAPY if verdicts[k]["checks"]["V1"]]
nconf = [k for k in CAPY if verdicts[k]["verdict"] == "confirmed"]
nref = [k for k in v1_pass if verdicts[k]["verdict"] == "refuted"]
ninc = [k for k in v1_pass if verdicts[k]["verdict"] == "inconclusive"]
if not controls_ok:
    headline = "Both positive controls failed: stopped after the pilot."; outcome = "inconclusive"
elif nconf:
    headline = "Confirmed capybara readout: " + ", ".join(nconf); outcome = "confirmed"
elif not V2:
    headline = "Inconclusive: V2 failed (tapir fake target passes on the scene arm)."; outcome = "inconclusive"
elif not v1_pass:
    headline = "Inconclusive: no capybara readout passes V1 on the new word prompts."; outcome = "inconclusive"
elif len(nref) == len(v1_pass):
    headline = "Refuted: no capybara neuron or group in these candidates; every valid readout is spelling-bound."; outcome = "refuted"
else:
    headline = (f"No capybara readout confirmed. {len(nref)} readouts refuted as spelling-bound; "
                f"{len(ninc)} inconclusive."); outcome = "inconclusive"

# traces for the 61 named sides
named_idx = {s: col[(int(SIDE.match(s)[1]), int(SIDE.match(s)[2]))] for s in R["named_sides"]}
prompts_out = [dict({k: v for k, v in p.items() if k not in ("acts", "ids")},
                    trace={s: [round(float(p["acts"][t, j]) * (1 if s[-1] == "+" else -1), 4) for t in range(len(p["ids"]))]
                           for s, j in named_idx.items()}) for p in P]

res = dict(request="L62-N24835-001", gpu=G["gpu"], p1_heldout_capybara=p1_scores["capybara"].tolist(), p1_reduced_heldout=red.tolist(), outcome=outcome, headline=headline, controls_ok=controls_ok,
           V2=V2, P1={k: v for k, v in P1.items()}, P2=P2, verdicts=verdicts,
           candidates={k: v for k, v in cand.items()},
           random={k: {kk: vv for kk, vv in v.items()} for k, v in RAND.items()},
           poles=poles, counts=dict(v1_pass=v1_pass, confirmed=nconf, refuted=nref, inconclusive=ninc))
json.dump(res, open(out / "results.json", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
json.dump(prompts_out, open(out / "prompts_traces.json", "w"))

print("OUTCOME:", outcome, "|", headline)
print(f"P1 full: margin {P1['full']['margin_conservative']:.2f} (pool {P1['full']['margin_pool']:.2f}) slots {P1['full']['slot_wins']}/12 breadth {P1['full']['breadth_ok']} -> {P1['full_pass']}")
print(f"P1 reduced: margin {P1['reduced']['margin_conservative']:.2f} slots {P1['reduced']['slot_wins']}/12 -> {P1['reduced_pass']}; others passing {P1['others_passing']}/9; perm pct {P1['perm_null']['pct']:.3f} q95 {P1['perm_null']['q95']:.2f}")
print("P1 ovr:", {a: (round(v['margin'], 2), v['slots'], v['passes']) for a, v in P1["ovr"].items()})
print(f"P2: own lift {own_mean:.2f} rival-ctx {rival_mean:.2f} clear {clear} cap neutral sd {cap_sd:.2f} range {cap_range:.2f} -> {P2['pass']}")
print("P2 true ranks:", P2["true_rank_counts"])
print("V2 (tapir sum4 fails):", V2, round(tapir4["scene"]["margin_conservative"], 2), tapir4["scene"]["slot_wins"])
for rid in R["readouts"]:
    c = cand[rid["id"]]; v = verdicts.get(rid["id"], {})
    print(f"{rid['id']:22s} word {c['word']['margin_pool']:6.2f} ({c['word']['slot_wins']}/4) scene cons {c['scene']['margin_conservative']:6.2f} "
          f"pool {c['scene']['margin_pool']:6.2f} slots {c['scene']['slot_wins']:2d}/12 top {c['scene']['top_rival']:10s} "
          f"ruby {c['contam']['target']['frac']:5.2f}/{c['contam']['next']['frac']:5.2f} max {max(c['contam']['target']['per_prompt']):5.2f} "
          f"{v.get('verdict','')} pct {v.get('s3_percentile', float('nan')):.3f} {[k for k, x in v.get('checks', {}).items() if not x]}")
