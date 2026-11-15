## Neuron(s)

L6-N9321

## In plain English

- **Positive side:** fires on "week" when it points to one particular occasion: "one morning this week", "that week", "Easter week".
- A weekday name goes positive only after a dated event that it places: "New Year's Day fell on a Thursday", "born on a Sunday". In a plain one-off story ("the parcel came on Tuesday") it sits on the + q99 line with no consistent sign.
- The same words read negative when they name a slot that repeats ("every Tuesday", "each week"). That side, and the explanation for both sides, are in #93.

## What's in this PR

- [ ] An explanation: no. `explanations/L6/L6-N9321.yaml`, covering both sides, is in #93.
- [x] Confirmation-experiment requests for the **positive side**, listed under Design > Claim.
- Reading notes for the positive side, to be removed before this is marked ready: `notes/L6-N9321/positive/exemplars.md`, `hypotheses.md`, and `colab-results.md` (every Colab measurement for this side, round by round).

## Station House credit

marin-request-experiment

## How you investigated

A log of the experiments, in order. Every round was measured on a Colab T4 with `scripts/colab/measure.py` (branch `colab-tools`), which loads only layers 0–6 and reads `silu(gate)·up` with the runner's tokenizer and target rules. Before the first round, ten atlas exemplars fed as exact token ids reproduced the stored activations (7 exact, largest gap 0.06), so the setup matches the atlas. Each round's prompts and numbers are in `notes/L6-N9321/positive/colab-results.md`. Rounds are numbered across both sides; the negative side's rounds are in #93.

- **Round 1.** From the exemplars, we assumed:
  - the + side fires on week and weekday words that point to one occasion;
  - this might be about one occasion in general, not weeks ("last month", "this morning");
  - "this week" might flip sign with the type of text.

  We wrote prompts for each and tested them directly. Results:
  - One-off weekdays ("born on a Thursday", "Christmas fell on a Sunday") read +0.08 to +3.36.
  - It is specific to the week: "last month", "this morning" and "last year" sit within 0.15 of zero.
  - "This week" reads +0.58 after a one-off story and −1.59 after a weekly column or podcast opener.
- **Round 2.** We assumed one located "week" would go positive, and looked at where weekday names go positive. Results:
  - A located "week" is strongly positive: "Christmas week" +2.64, "That week changed my life" +1.17.
  - Weekday names go strongly positive after a dated event: "Christmas fell on a Sunday" +3.36, "born on a Thursday" +1.48.
  - "week" as a unit is mixed and small: "a week ago" +0.29, "for a week" +0.49.
- **Round 3.** We measured the request files below as they would score. Both pass; `-002` passes at exactly 80% on should-not-fire (see Expected refutations).
- **Round 5.** We assumed a weekday name in a plain one-off story needs a dated event before it to go positive, while "week" doesn't. We kept six one-off sentences fixed and swapped only the time word. Results:
  - "that week" +0.73 (positive in all six), "this week" +0.51, "last week" +0.30.
  - "on Tuesday" +0.10 and "that Tuesday" +0.06: on the + q99 line (0.074), with no consistent sign. "last Tuesday" −0.72.
  - After a dated event ("Christmas fell on a Tuesday", "born on a Tuesday"): +2.12.
- **Atlas levels** (`pole_stats`, `fineweb-10m-sparse`):

  | side | q99 | q999 | q9999 |
  |---|---|---|---|
  | + | 0.074 | 0.229 | 0.835 |
  | − | 0.080 | 0.269 | 0.982 |

- **Agent:** Claude (`claude-opus-5-5`, through Claude Code) ran the Colab rounds, wrote the prompts and drafted the files. Reads on the exemplars came from Hariom.

## Design

### Claim

- Summary file:
  - `requests/L6-N9321-002.yaml`: activation test, + side. One located week, or a weekday placing a dated event, fires; the same words repeating don't.
- Random baseline:
  - `requests/L6-N9321-008.yaml`: activation test, + side, `-002`'s should-fire vs random sentences.
    - Kept separate so random sentences don't pad `-002`'s should-not-fire score.

### Unit and metric

- Unit: one prompt with one marked target.
- Metric: the side's activation at that token.
- Counts (should-fire / should-not-fire): `-002` 12 / 10, `-008` 12 / 10.

### Controls

- Near misses use the same target token in the wrong context: in `-002`, the same week and weekday words naming a repeating slot ("every[[ week]]", "every[[ Sunday]]").
- Random baseline: 10 everyday sentences with the target on the last word, in `-008`.
- The other side: 8 of the 10 repeating near misses in `-002` read negative, from −1.12 to −4.16 ("The club meets weekly on Mondays").

### Data

- Thresholds: `pilot/fineweb-10m-sparse` `pole_stats`. Model: `marin-community/marin-32b-base` @ `f19b3466`.
- Both files use prompts written by us.
- Both were measured on Colab before submission (see `notes/L6-N9321/positive/colab-results.md`). There is no held-out split: as in PR #23, the request prompts are the ones measured in round 3, so the runner's result reproduces those readings rather than testing blind ones.
- Filters: none.
- No claims that the neuron never fires.

### Noise

- The model is deterministic, so the noise is prompt to prompt.
- Margins on Colab:
  - `-002` should-fire: +0.37 to +4.12, all past +q999 (0.229).
  - Random sentences: within ±0.03.

### Readout

- Each file uses its type's rule from EXPERIMENTS.md, with the threshold at the side's q99:
  - Type A (`-002`, `-008`): at least 80% of should-fire past the threshold, and at least 80% of should-not-fire below it.
- What a failure would tell us:
  - `-002`: the explanation doesn't carry over to new sentences.
  - `-008`: the side also fires on ordinary sentence-final words.

#### Expected refutations

- None expected. `-002` is the closest to failing: two of its should-not-fire prompts read positive, "It snows every week here in January" (+0.57) and "Pilgrims arrive each week by bus" (+1.55). #93's rounds 4 and 6–9 show why: "every week" loses its negative reading for a counted crowd, and a few nouns (pilgrim, tourist, holidaymaker, backpacker) push toward + on their own. One more such miss would fail the file.

## Checklist

- [x] `python scripts/validate.py` passes
- [ ] I wrote the request's prompts and prediction before seeing any result for them
  - Same as PR #23: every prompt was measured on Colab before submission.
- [x] The `## Design` section is filled in (the `design` check passes)
- [x] Each marked `[[target]]` is a single token for the Marin tokenizer (checked with `runner/prompts.py`'s `encode` on Colab)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01PznTA3QfVuFG6FbpFUQk6F
