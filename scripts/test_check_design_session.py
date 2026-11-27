"""Run with: python -m unittest scripts/test_check_design_session.py"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_design_session import note  # noqa: E402

SID = "ds_" + "D" * 22


class Note(unittest.TestCase):
    def test_request_without_a_session_fails(self):
        level, message = note(["requests/L40-N25546-002.yaml"], "## Design\n...")
        self.assertEqual(level, "error")
        self.assertIn("stationhouse design start", message)

    def test_request_with_a_session_is_ok(self):
        self.assertEqual(note(["requests/r/DESIGN.md"], f"Design-Session: {SID}")[0], "ok")

    def test_other_prs_need_nothing(self):
        self.assertEqual(note(["explanations/L40-N1.md", "README.md"], "")[0], "none")


if __name__ == "__main__":
    unittest.main()
