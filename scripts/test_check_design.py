"""Run with: python -m unittest scripts/test_check_design.py"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_design import check, required_designs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

FILLED = """## Neuron(s)

L12-N4410

## Design

### Claim
L12-N4410+ fires on " or" only inside "or vice versa", not on other " or".

### Unit and metric
20 should-fire and 20 should-not-fire prompts; activation at the marked " or" vs pos_q99.

### Controls
Near misses: 20 prompts with " or" in plain disjunctions. Five random L12 neurons' top exemplars don't show it.

### Data
pilot/fineweb-10m-sparse for the hypothesis; prompts written fresh, none copied from exemplars.

### Noise
Expected activations are about 3x pos_q99, so wording changes shouldn't flip them.

### Readout
Menu A pass rule (80%/80%). If plain " or" prompts fire above threshold in more than 4 of 20, it's refuted.

## Checklist
"""


class Check(unittest.TestCase):
    def test_a_filled_design_passes(self):
        self.assertEqual(check(FILLED), [])

    def test_the_unedited_pr_template_fails_every_stage(self):
        template = (ROOT / ".github" / "pull_request_template.md").read_text()
        errors = {f["stage"] for f in check(template) if f["level"] == "error"}
        self.assertEqual(errors, {"claim", "metric", "controls", "data", "noise", "readout"})

    def test_no_description_fails(self):
        self.assertEqual(len([f for f in check("") if f["level"] == "error"]), 6)

    def test_controls_without_a_baseline_fails(self):
        text = FILLED.replace("Near misses: 20 prompts with \" or\" in plain disjunctions. Five random L12 neurons' "
                              "top exemplars don't show it.", "We looked at it carefully.")
        self.assertIn(("error", "controls"), {(f["level"], f["stage"]) for f in check(text)})


class RequiredDesigns(unittest.TestCase):
    def test_custom_needs_a_design_file_and_others_use_the_description(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "requests").mkdir()
            (root / "requests" / "L1-N1-001.yaml").write_text("type: activation-test\nneuron: L1-N1\n")
            (root / "requests" / "L1-N2-001.yaml").write_text("type: custom\nneurons: [L1-N2]\n")
            (root / "requests" / "L1-N3-001").mkdir()
            (root / "requests" / "L1-N3-001" / "DESIGN.md").write_text("# design only\n")
            got = dict(required_designs(["L1-N1-001", "L1-N2-001", "L1-N3-001", "L1-N9-001"], root))
            self.assertIsNone(got["L1-N1-001"])
            self.assertEqual(got["L1-N2-001"], root / "requests" / "L1-N2-001" / "DESIGN.md")
            self.assertEqual(got["L1-N3-001"], root / "requests" / "L1-N3-001" / "DESIGN.md")
            self.assertNotIn("L1-N9-001", got)  # deleted in the PR


if __name__ == "__main__":
    unittest.main()
