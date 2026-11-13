"""Score L20-N1234-003 from gpu_run.py's pickle with the request's registered rules; writes the result YAML + a comment.

python results/L20-N1234-003/score.py acts.pkl --request-commit <sha> --runner-commit <sha> --out results/L20-N1234-003.yaml
"""
import argparse
import datetime
import math
import pickle

import numpy as np
import yaml

NEURON = 1234
SEED = 0
N_RANDOM = 100
MODEL = "marin-community/marin-32b-base@f19b346615212ff9a6d92c10861a9cdb100cbf97"


def rules(S, M, N, T):
    """The registered checks (a)-(d) for one neuron; S/M/N are 12-vectors of positive-side activation."""
    floor = T / 4
    r = np.maximum(S, floor) / np.maximum(M, floor)
    gm = float(math.exp(np.mean(np.log(r))))
    in_band = int(((r >= 0.5) & (r <= 2)).sum())
    c = dict(M_above=int((M > T).sum()), S_above=int((S > T).sum()), N_above=int((N > T).sum()),
             geo_mean_ratio=round(gm, 4), frames_ratio_in_band=in_band)
    c["a"] = c["M_above"] >= 10
    c["b"] = c["S_above"] >= 10
    c["c"] = c["N_above"] <= 2
    c["d"] = 0.5 <= gm <= 2 and in_band >= 8
    return c, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("--request-commit", default="")
    ap.add_argument("--runner-commit", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--comment")
    a = ap.parse_args()
    res = pickle.load(open(a.pkl, "rb"))
    st = res["stats"]
    assert (st["neuron"] == np.arange(len(st["neuron"]))).all()
    P = res["prompts"]
    frames = sorted({p["frame"] for p in P})
    A = {(p["frame"], p["arm"]): p["acts"] for p in P}

    def arms(n):
        return [np.array([float(A[(f, arm)][n]) for f in frames]) for arm in "SMN"]

    T = float(st["pos_q99"][NEURON])
    S, M, N = arms(NEURON)
    c, r = rules(S, M, N, T)

    pool = np.array([n for n in range(len(st["neuron"])) if n != NEURON and np.isfinite(st["pos_q99"][n])])
    rand = np.sort(np.random.default_rng(SEED).choice(pool, N_RANDOM, replace=False))
    passing = []
    for n in rand:
        rc, _ = rules(*arms(n), float(st["pos_q99"][n]))
        if rc["a"] and rc["b"] and rc["c"]:
            passing.append(int(n))
    c["random_abc_pass"] = len(passing)
    c["e"] = len(passing) < 5

    gm = c["geo_mean_ratio"]
    if not (c["a"] and c["c"]):
        outcome, reason = "inconclusive", "the minced arm (a) or the neutral arm (c) failed, so the comparison means nothing"
    elif not c["e"]:
        outcome, reason = "inconclusive", f"{len(passing)}/100 random L20 neurons also pass (a)-(c)"
    elif c["b"] and c["d"]:
        outcome, reason = "confirmed", None
    elif c["S_above"] < 6 or gm < 0.5:
        outcome, reason = "refuted", "minced-oath-specific: strong forms fire much less than minced forms"
    elif gm > 2:
        outcome, reason = "refuted", "general profanity: minced forms are a weak echo of strong forms"
    else:
        outcome, reason = "inconclusive", "(a) and (c) hold but neither the confirm nor a refute rule is met"
    checks = {k: c[k] for k in "abcde"}
    if reason:
        reason += f" (checks {checks}, geo-mean S/M {gm}, S>T {c['S_above']}/12, M>T {c['M_above']}/12, N>T {c['N_above']}/12)"

    per_prompt = []
    for i, f in enumerate(frames):
        for arm, v in zip("SMN", (S, M, N)):
            p = next(p for p in P if p["frame"] == f and p["arm"] == arm)
            per_prompt.append(dict(frame=f, arm=arm, prompt=p["text"], token=p["token"],
                                   act=round(float(v[i]), 4), above_T=bool(v[i] > T),
                                   **({"ratio_S_over_M": round(float(r[i]), 4)} if arm == "S" else {})))
    rec = dict(
        request="L20-N1234-003", type="custom", outcome=outcome, reason=reason,
        summary=dict(threshold_pos_q99=round(T, 4), ratio_floor=round(T / 4, 4), **c,
                     mean_act=dict(S=round(float(S.mean()), 4), M=round(float(M.mean()), 4), N=round(float(N.mean()), 4)),
                     neg_side_min_act=round(float(min(S.min(), M.min(), N.min())), 4),
                     random_baseline=f"{N_RANDOM} L20 neurons, np.random.default_rng({SEED}).choice of all L20 neurons except 1234; "
                                     f"each scored on (a)-(c) with its own pos_q99",
                     random_passing=passing),
        per_prompt=per_prompt, model=MODEL, runner_commit=a.runner_commit, request_commit=a.request_commit,
        gpu=res["gpu"], date=datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
        atlas_stats=dict(pos_q99=round(T, 4), neg_q99=round(float(st["neg_q99"][NEURON]), 4)),
        baseline_neurons=[int(n) for n in rand])
    open(a.out, "w").write(yaml.safe_dump(rec, sort_keys=False, width=120, allow_unicode=True))
    print(outcome, reason)
    print(yaml.safe_dump(rec["summary"], sort_keys=False))
    if a.comment:
        rows = "\n".join(f"| {f} | `{q[0]['prompt']}` | {q[0]['act']:.3g} | {q[1]['act']:.3g} | {q[2]['act']:.3g} | {q[0]['ratio_S_over_M']:.3g} |"
                         for f, q in ((f, [p for p in per_prompt if p["frame"] == f]) for f in frames))
        ok = lambda b: "pass" if b else "fail"
        body = f"""## Experiment result: `L20-N1234-003` → **{outcome}**

{('Reason: ' + reason) if reason else ''}

Threshold T = pos_q99 = {T:.4f}. Ratio r = max(S, T/4) / max(M, T/4) per frame.

| rule | value | |
|---|---|---|
| (a) minced > T in ≥10/12 | {c['M_above']}/12 | {ok(c['a'])} |
| (b) strong > T in ≥10/12 | {c['S_above']}/12 | {ok(c['b'])} |
| (c) neutral > T in ≤2/12 | {c['N_above']}/12 | {ok(c['c'])} |
| (d) geo-mean r in [0.5, 2] and ≥8/12 frames in band | {gm:.3g}, {c['frames_ratio_in_band']}/12 | {ok(c['d'])} |
| (e) <5/100 random L20 neurons pass (a)-(c) | {len(passing)}/100 | {ok(c['e'])} |

<details><summary>Per frame (S strong, M minced, N neutral)</summary>

| frame | strong prompt | S | M | N | r |
|---|---|---|---|---|---|
{rows}

</details>

Model `{MODEL}` (bf16) on {res['gpu']}, custom script `results/L20-N1234-003/gpu_run.py` + `score.py`, request `{a.request_commit[:7]}`, {rec['date']}. Full numbers: `results/L20-N1234-003.yaml`.
"""
        open(a.comment, "w").write(body)


if __name__ == "__main__":
    main()
