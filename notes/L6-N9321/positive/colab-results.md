# L6-N9321, positive (+) side: Colab results (T4)

The prompts were measured on a Colab T4 with `scripts/colab/measure.py` (branch `colab-tools`). It loads only
layers 0–6 of `marin-32b-base@f19b3466` in bf16 and reads `silu(gate)·up` at the target, using
`runner/prompts.py`'s tokenizer and target rules. These notes are reading material and will be removed before
the PR is marked ready.

Rounds are numbered across both sides of this neuron. The negative (−) side's rounds are in its own PR's notes.

**Check against the atlas:** ten exemplars (ranks 0/1/2/10/25 on each side), fed as the atlas's exact token ids,
reproduce the stored activations. Seven match exactly and the largest gap is 0.06. Rechecked on a fresh VM before
round 4, with the same result.

Thresholds: + q99 0.074, q999 0.229, q9999 0.835; − q99 0.080, q999 0.269, q9999 0.982.

## Round 1: a first pass over many hypotheses

| group | n | mean | range | read |
|---|---|---|---|---|
| one-off weekday ("born on a Thursday", "Christmas fell on a Sunday") | 6 | +0.99 | +0.08 … +3.36 | all positive |
| **other periods, one-off** (last month, this morning, last year …) | 7 | +0.01 | −0.08 … +0.14 | **about zero** |
| **other periods, every** (every month, every morning …) | 7 | +0.00 | −0.10 … +0.07 | **about zero** |
| "This week …" after a weekly column or podcast opener | 3 | −1.59 | | |
| "This week …" after a one-off story | 3 | +0.58 | | the type of text flips it |
| random sentences | 5 | 0.00 | | |

## Round 2: surprises

- **Weekday names go strongly positive when the sentence first sets up a dated event** and the weekday says when
  it fell: "Christmas fell on a Sunday" +3.36, "born on a Thursday" +1.48. The model only sees the words before the
  target, so it is the event before the weekday that matters, not a date after it.
- **One located "week" is strongly positive:** "Christmas week" +2.64, "That week changed my life" +1.17, "the
  whole week of the wedding" +1.11.
- **"week" as a unit is mixed and small:** "a week ago" +0.29, "for a week" +0.49, "once a week" −0.23, "every
  week on food" −0.40.

## Round 3: the request files, as they would score

| request | type | question | result |
|---|---|---|---|
| `-002` | A, + | fresh one located occasion vs the same words repeating | 12/12, 8/10: **pass, at the edge**. Misses: "It snows every week here in January" +0.57, "Pilgrims arrive each week by bus" +1.55 |
| `-008` | A, + | `-002`'s should-fire vs random sentences | 12/12, 10/10: **pass** |

`-002`'s two misses are a crowd and a weather sentence. Rounds 4 and 6–9, in the negative-side notes, show why:
"every week" loses its negative reading for a counted crowd ("Hundreds of customers … every week" −0.02), and a few
nouns (pilgrim, tourist, holidaymaker, backpacker) push toward + on their own.

## Round 5: weekday vs week in the same one-off sentence

We assumed:
- A weekday name in a plain one-off story stays near zero or slightly negative, and needs a dated event before it to
  go positive.
- "week" in the same story goes positive with no dated event.

We wrote six one-off sentences and swapped only the time word:

| frame | mean | range |
|---|---|---|
| last Tuesday | −0.72 | −1.61 … −0.00 |
| on Tuesday | +0.10 | −0.34 … +0.46 |
| that Tuesday | +0.06 | −0.39 … +0.54 |
| on a Tuesday | −0.33 | −0.73 … +0.16 |
| last week | +0.30 | −0.35 … +0.81 |
| this week | +0.51 | −0.19 … +0.92 |
| that week | +0.73 | +0.26 … +1.09 |
| a dated event, then the weekday ("Christmas fell on a Tuesday", "born on a Tuesday") | +2.12 | +0.21 … +4.69 |

- **"week" goes positive in a plain one-off story; a weekday name does not.** "that week" is positive in all six
  sentences (+0.73, past q999 but not q9999). "on Tuesday" (+0.10) and "that Tuesday" (+0.06) sit on the + q99 line
  (0.074), with no consistent sign from sentence to sentence.
- **"last + weekday" is negative on its own,** not only in the neighbours sentence. "last week" in the same sentences
  is positive, so this is specific to weekday names.
- **"on a Tuesday" without a dated event leans negative** (−0.33), maybe because "a Tuesday" can read as "some
  Tuesday or other". After "Christmas fell on a" or "born on a" it is strongly positive.
- **A dated event before the weekday is what makes it positive:** New Year's Day +4.69, Christmas +4.09, birthday
  +2.16, born +1.04, married +0.21.
- "We met the new neighbours …" is negative with every weekday frame and with "last week", so that sentence pulls
  negative for some other reason.

