"""Fails a pull request that adds or changes an experiment request without a `Design-Session:` line
in its description, i.e. without a Station House Experimentation Wizard review.

  python scripts/check_design_session.py    in CI: BASE_SHA and PR_BODY, set by validate.yml

The wizard is three rounds of reviewer questions on the design that the author answers (see
CONTRIBUTING.md, "Before you open the PR"). This check only looks for the line: whether the session
is complete and matches the design is checked by the Station House watcher, which records it on the
platform. Explanation-only PRs don't need one.
"""
import os
import re
import subprocess
import sys

REQUEST = re.compile(r"^requests/[^/]+(?:\.ya?ml|/DESIGN\.md)$")
SESSION = re.compile(r"Design-Session:\s*(ds_[A-Za-z0-9]{22})\b")

ERROR = ("This PR adds or changes an experiment request, but its description has no `Design-Session:` line, so the "
           "design hasn't been through the Station House Experimentation Wizard. Run it from your coding agent (Claude "
           "Code: the stationhouse plugin; others: `stationhouse design start` / `design round`), then add the line it "
           "gives you to the description and re-run this check.")


def note(paths, body):
    """(level, message) for this PR: ('none', ...) when it has no request, else 'ok' or 'error'."""
    if not any(REQUEST.match(p) for p in paths):
        return "none", "No experiment request in this PR; no design review needed."
    m = SESSION.search(body or "")
    if m:
        return "ok", f"Design review {m.group(1)} is linked; the Station House watcher checks it's complete."
    return "error", ERROR


def main():
    base = os.environ.get("BASE_SHA")
    if not base:
        print("check_design_session: not a pull request; nothing to check.")
        return 0
    diff = subprocess.run(["git", "diff", "--name-only", "--diff-filter=AM", f"{base}...HEAD"],
                          capture_output=True, text=True, check=False)
    level, message = note(diff.stdout.splitlines(), os.environ.get("PR_BODY") or "")
    print(f"::error title=Design review::{message}" if level == "error" else message)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary and level != "none":
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"### Design review\n\n{'❌ ' if level == 'error' else '✅ '}{message}\n")
    return 1 if level == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
