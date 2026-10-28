# Notes for AI coding agents

These notes are for any coding agent working in this repository: Codex, Cursor, Claude Code and
others. Read [README.md](README.md) for the loop and the file formats, [EXPERIMENTS.md](EXPERIMENTS.md)
for the request types, and [CONTRIBUTING.md](CONTRIBUTING.md) for how to design one.

## Ground rules

- Claim a neuron before working on it: `stationhouse neuron claim <id>`. A 409 means someone else
  holds it, so pick another. Claim only with the contributor's go-ahead.
- Don't open PRs, post on GitHub or Hugging Face, or claim anything without the contributor's
  go-ahead. Don't handle their passwords or tokens.
- Put the model or agent in `by`, as `method: agent` with `model: ...`.
- Run `python scripts/validate.py` before any PR.
- Never put format examples under `explanations/`. Only real explanations go there, and examples go
  in `examples/`.

## Helping someone design an explanation or request

When a contributor asks how to test an explanation, what a good control is, or whether their
evidence holds up, act as their design partner. Don't design it for them.

1. **Read CONTRIBUTING.md, "Designing a request"** and the request type in EXPERIMENTS.md. Search
   `explanations/`, `requests/` and the open PRs for the same neuron.
2. **Go one stage at a time:** Claim, Unit and metric, Controls, Data, Noise, Readout. Ask about
   each stage and write down what they decide: in the PR's `## Design` section for types A to D,
   or in `requests/<request id>/DESIGN.md` for type E (`stationhouse design new <path>`). When
   they're stuck, give two or three concrete options and let them choose.
3. **Push on the usual weak spots.**
   - Do the shouldn't-fire prompts use the same target token in the wrong context?
   - Did they compare with random neurons from the same layer?
   - Are the test prompts fresh, rather than copied from exemplars?
   - Are non-firing claims checked in the `deep` tier, not the `sparse` one?
   - Are there document-start, boundary or mid-word artifacts?
   - Is every target a single token?
   - Are the expected activations well clear of the threshold?
4. **Check it** with `python scripts/check_design.py <file>` (or `stationhouse design check <file> --json`),
   which also works on a saved PR description. CI runs the same check, and a request PR can't merge
   while it fails. Fix the errors with them, and talk through each warning.
5. **Red-team it before it's submitted.**
   - What simpler explanation would pass the same prompts: the token alone, the position, the
     document type?
   - What result would refute the explanation?
6. **Writing and choosing prompts:** you may draft them, but the contributor picks and edits them.
   Write them before looking at any activations for them. If you've already measured activations
   on a prompt, it's no longer a fair test prompt, so say so.

When you review someone else's design, give notes per stage: what's missing, the strongest
alternative explanation, and one concrete fix each.
