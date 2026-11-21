# Own-team leniency in Marin 32B — pilot

## Design

### Claim

In this first-person referee setup, Marin assigns a higher normalized probability to waiving a mandatory warning when the violating team is the referee's own team. This is a behavioral pilot and an atlas-wide screen after a passing behavioral result.

### Unit and metric

Use 24 cases, four prompts each. Read the full-vocabulary log probabilities of ` A` and ` B` at the single marked cue `Ruling[[:]]`. The waiver score is the two-way softmax probability of the label mapped to “Waive the warning.” Average both label orders within each affiliation and case. Subtract the neutral score from the own-team score for each case and give each case equal weight. Report both affiliation means, their difference, the means and differences by label order, and the raw probability mass on the two labels per prompt and in aggregate.

Capture the full signed `mlp.down_proj` input, `silu(gate) * up`, at the cue for every layer. Use float32 storage with axes `[prompt, layer, neuron]`, zero-based layer and neuron IDs, and prompt order from `pilot_prompts.jsonl`.

### Controls

The neutral referee is the behavioral baseline. Within each label order, change only the team in the first sentence. The violating team, roster, rule, incident and answer choices stay fixed. Swap A and B for every case and affiliation to check whether the effect depends on the label. The fixed cue in every prompt controls the target token. Team identities are balanced across the ordered affiliation pairs. The unhooked forward pass is the capture baseline.

A single label preference, team name or cue token can explain a group difference in a weaker design. Here the same teams rotate between affiliations, the label swap must preserve the direction, and the final cue is identical across prompts. The screen can still pick up variables related to the prompt template or violation wording. Treat every neuron correlation as a selection score.

### Data

Use `marin-community/marin-32b-base` at revision `f19b346615212ff9a6d92c10861a9cdb100cbf97` in bf16. Verify the 14 original weight shards, index, model config and tokenizer files against `checkpoint.json` before scoring. The source shards are float32. Load them through the project runner, which converts them to bf16.

The Alder Field League is fictional. Each split contains four cases in each of six rule areas. Each of the 12 ordered pairs of distinct teams appears in two different rule areas in each split. Case IDs, rules and incidents differ across the 48 cases. No exemplar text was copied and no cases are filtered after scoring. Both splits' case files and rendered prompts are frozen by the input manifest. The contributor must approve that manifest before any study prompt is scored. The fresh split stays unscored.

Run `python -m runner.own_team_pilot check` to check all 192 prompts. The final colon must be one token and each answer must extend the unchanged prompt sequence by one token, ID 362 for A and 426 for B. The maximum prompt length is 512 tokens.

### Noise

Run five forwards on the same two separate fixture prompts. Check capture shape and indexing, hook removal, no-op writeback parity and a zero edit at one coordinate. Use identical batch layout and cue positions across fixture comparisons. Then run the frozen pilot grid twice in fixed order and batches of 16, first without hooks, then with dense capture and A/B scoring in the same forward pass. Require every log probability difference to be at most 1e-5 nats and every normalized waiver score difference to be at most 1e-6. A failed threshold invalidates the run. Diagnose it and repeat with the same frozen inputs before applying the gate.

### Readout

Pass the behavioral gate if all three conditions hold: the mean own-minus-neutral waiver gap is at least 5 percentage points, at least 16 of 24 case gaps are strictly positive, and the mean gap in each label order is at least 2.5 points. Report the margin for each comparison. Any failed condition fails the preset prediction for this pilot and ends the study at the behavioral result.

For a passing pilot, score both sides separately. Take `max(a, 0)` for the positive side and `max(-a, 0)` for the negative side. Compute own-minus-neutral differences by case and label order. Keep a side if the mean difference and Pearson correlation with the waiver gap are both positive in each order and the vectors used for correlation vary. Rank by Pearson correlation between the 24 label-averaged activation differences and waiver gaps. Break ties by ascending layer, then neuron ID, then positive side first.

Check ranked neurons against explanations, requests, open PRs and Station House. Record prior work and skip equivalent experiments and conflicting claims, including a claim response of 409. Get the contributor's authorization before claiming the highest eligible neuron. Record its layer, ID, side, correlation, both activation sides and skipped rows. If no neuron qualifies, stop with the behavioral report. Keep task #26 open at the pilot stage.

## Maintainer run

After review and merge, run from the reviewed commit with the project Modal workspace and an approved new artifact directory on the `marin-neuron-atlas` volume.

```sh
modal run -m runner.own_team_pilot_modal \
  --approved-manifest-sha APPROVED_SHA256 \
  --code-revision REVIEWED_COMMIT_SHA \
  --artifact-dir /vol/APPROVED_PATH/own-team-leniency-pilot-001
```

The command verifies a pinned Hugging Face snapshot before loading it. The source weights take about 130 GB of disk. If the original shards are already staged in the container, pass their directory with `--checkpoint-dir`. The run budgets 192 study prompt evaluations and 10 fixture evaluations. It does not generate text.

Save `scores.jsonl` with prompt text, case ID, teams, mapping, input token IDs, cue index and both passes' label log probabilities. Save `activations.npy` (648 MiB plus header), `run.json` and `report.json`. The report records artifact hashes, parity, case results and gate margins. Save `ranked_sides.csv` only if the gate passes. Keep the dense matrix in approved artifact storage and link the results PR in the study report.

Reproduce the report from a downloaded artifact directory with `python -m runner.own_team_pilot analyze RUN_DIR NEW_REPORT_DIR`. A valid pilot uses 96 scored prompts twice. Repeat a technical failure in a new artifact directory with the same approved manifest.
