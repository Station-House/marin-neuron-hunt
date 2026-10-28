"""Fails a pull request that adds or changes an experiment request without a filled-in design.

  python scripts/check_design.py FILE...   check design files (a DESIGN.md, or a PR description saved to a file)
  python scripts/check_design.py           in CI: this PR's requests (BASE_SHA and PR_BODY, set by validate.yml)

A request of type `custom` needs requests/<request id>/DESIGN.md; any other type needs the `## Design`
section of the PR description. Each design needs all six stages filled in (Claim, Unit and metric,
Controls, Data, Noise, Readout) and Controls has to name a baseline. Errors fail the check; warnings
are printed. See CONTRIBUTING.md, "Designing a request".

The rules are the same as `stationhouse design check` (stationhouse-cli, src/stationhouse_cli/design.py).
Keep the two in sync.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGED = re.compile(r"^requests/([^/]+?)(?:\.ya?ml|/DESIGN\.md)$")
CUSTOM = re.compile(r"^type:\s*['\"]?custom\b", re.M)

# (key, heading as written, other headings accepted for it)
STAGES = [
    ("claim", "Claim", ("hypothesis", "research question", "question")),
    ("metric", "Unit and metric", ("unit", "metric", "metrics", "measure")),
    ("controls", "Controls", ("baselines", "baselines and controls", "control")),
    ("data", "Data", ("dataset", "data and cleaning", "inputs")),
    ("noise", "Noise", ("noise floor", "variance", "seeds")),
    ("readout", "Readout", ("pre-registered readout", "pass rule", "success criteria", "what counts as a result")),
]

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
COMMENT = re.compile(r"<!--.*?-->", re.S)
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME)\b|^\s*\.\.\.\s*$", re.M)

BASELINE = re.compile(r"baseline|random|shuffl|constant|majority|chance|carry[- ]forward|copy|untrained|base model|"
                      r"null model|placebo|no[- ]op|ablat|prior|naive|always[- ]", re.I)
VERSION = re.compile(r"\b[0-9a-f]{7,64}\b|commit|sha|hash|version|v\d|revision|@\s*\w|\d{4}-\d{2}-\d{2}|"
                     r"pilot/|prefix", re.I)  # an atlas prefix is a fixed snapshot
SPLIT = re.compile(r"split|held[- ]out|hold[- ]out|test set|train|leak|contaminat|fresh|unseen|not copied|none copied", re.I)
FILTERS = re.compile(r"filter|dedup|duplicate|drop|remov|clean|exclud|kept|\bnone\b|as is", re.I)
REPEATS = re.compile(r"seed|rerun|re-run|repeat|bootstrap|confidence interval|\bci\b|std|standard (?:error|deviation)|"
                     r"variance|k\s*=\s*\d|\d+\s*(?:runs|samples|draws)|permutation|random[- ]neuron|"
                     r"threshold|margin|q99|percentile", re.I)  # distance from a threshold is the neuron hunt's version
REFUTE = re.compile(r"refut|falsif|null|no effect|fail|reject|inconclusive|would mean|wrong if|not support|negative", re.I)
NUMBER = re.compile(r"\d")


def _stage_for(title: str):
    t = re.sub(r"^[\d.)\s]+|[*_`:]", "", title).strip().lower()
    for key, name, others in STAGES:
        if any(t == n or t.startswith(n + " ") or t.startswith(n + "(") for n in (name.lower(), *others)):
            return key
    return None


def sections(text: str) -> dict:
    """{stage key: section text, comments removed}. A section runs to the next heading at its own
    level or above, so sub-headings inside it stay with it, and the first heading for a stage wins."""
    # Comments go first (keeping their line breaks), so a heading inside one doesn't count.
    lines = COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text).splitlines()
    found = {}
    for i, line in enumerate(lines):
        m = HEADING.match(line)
        if not m:
            continue
        key = _stage_for(m.group(2))
        if not key or key in found:
            continue
        level, body = len(m.group(1)), []
        for nxt in lines[i + 1:]:
            h = HEADING.match(nxt)
            if h and len(h.group(1)) <= level:
                break
            body.append(nxt)
        found[key] = "\n".join(body).strip()
    return found


def check(text: str) -> list[dict]:
    """[{level: error|warning, stage, message}]. Any error means the design isn't ready to run."""
    secs = sections(text)
    out = []

    def add(level, stage, message):
        out.append({"level": level, "stage": stage, "message": message})

    names = {key: name for key, name, _ in STAGES}
    for key, name, _ in STAGES:
        body = secs.get(key)
        if body is None:
            add("error", key, f"no `{name}` section")
        elif not body:
            add("error", key, f"`{name}` is empty")
        elif PLACEHOLDER.search(body):
            add("error", key, f"`{name}` still has a TODO/TBD or a bare `...` line")

    def filled(key):
        return bool(secs.get(key)) and not PLACEHOLDER.search(secs[key])

    if filled("controls") and not BASELINE.search(secs["controls"]):
        add("error", "controls", "Controls doesn't name a baseline: add at least one trivial one (constant, majority, "
                                 "copy the input, carry forward, base model, random units) and say what it scores")
    if filled("metric") and not NUMBER.search(secs["metric"]):
        add("warning", "metric", "Unit and metric gives no sample size: say how many units")
    if filled("data"):
        d = secs["data"]
        if not VERSION.search(d):
            add("warning", "data", "Data doesn't pin a version (commit, hash, revision or date)")
        if not SPLIT.search(d):
            add("warning", "data", "Data doesn't say how the evaluation set is kept apart (split, held-out, leakage)")
        if not FILTERS.search(d):
            add("warning", "data", "Data doesn't list its filters or cleaning steps (say 'none' if there are none)")
    if filled("noise") and not REPEATS.search(secs["noise"]):
        add("warning", "noise", "Noise doesn't say how it's estimated (seeds, reruns, bootstrap, random-unit baseline)")
    if filled("readout"):
        r = secs["readout"]
        if not REFUTE.search(r):
            add("warning", "readout", "Readout doesn't say which result would refute the claim")
        if not NUMBER.search(r):
            add("warning", "readout", "Readout has no number: give the threshold or effect size that counts")
    order = {k: i for i, (k, _, _) in enumerate(STAGES)}
    out.sort(key=lambda f: (f["level"] != "error", order[f["stage"]]))
    for f in out:
        f["section"] = names[f["stage"]]
    return out


def report(label, text) -> int:
    """Prints the findings for one design; returns how many are errors."""
    findings = check(text)
    errors = sum(f["level"] == "error" for f in findings)
    print(f"{label}: " + ("ready" if not errors else f"{errors} error(s)")
          + (f", {len(findings) - errors} warning(s)" if len(findings) > errors else ""))
    for f in findings:
        print(f"  {f['level']}: {f['section']}: {f['message']}")
    return errors


def required_designs(request_ids, root=ROOT):
    """[(request id, path to its DESIGN.md, or None for the PR description)] for the requests a PR touches."""
    out = []
    for rid in request_ids:
        yaml = next((p for p in (root / "requests" / f"{rid}.yaml", root / "requests" / f"{rid}.yml") if p.exists()), None)
        design = root / "requests" / rid / "DESIGN.md"
        if yaml is None and not design.exists():
            continue  # deleted
        custom = yaml is not None and CUSTOM.search(yaml.read_text(encoding="utf-8"))
        out.append((rid, design if custom or yaml is None else None))
    return out


def main(argv):
    if argv:
        return 1 if sum(report(p, Path(p).read_text(encoding="utf-8")) for p in argv) else 0
    base = os.environ.get("BASE_SHA")
    if not base:
        print("check_design: no BASE_SHA, so not a pull request; nothing to check.")
        return 0
    diff = subprocess.run(["git", "diff", "--name-only", "--diff-filter=AM", f"{base}...HEAD"],
                          capture_output=True, text=True, check=True)
    ids = sorted({m.group(1) for m in (CHANGED.match(p) for p in diff.stdout.splitlines()) if m})
    needed = required_designs(ids)
    if not needed:
        print("check_design: this PR adds or changes no experiment request; nothing to check.")
        return 0
    errors, from_body = 0, []
    for rid, path in needed:
        if path is None:
            from_body.append(rid)
        elif not path.exists():
            print(f"requests/{rid}.yaml is a custom request: add its design as requests/{rid}/DESIGN.md "
                  f"(stationhouse design new requests/{rid}/DESIGN.md)")
            errors += 1
        else:
            errors += report(str(path.relative_to(ROOT)), path.read_text(encoding="utf-8"))
    if from_body:
        errors += report(f"the PR description's ## Design section (for {', '.join(from_body)})", os.environ.get("PR_BODY") or "")
    if errors:
        print("\nAn experiment request needs its design settled before it's merged: see CONTRIBUTING.md, "
              "\"Designing a request\". Editing the PR description re-runs this check.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
