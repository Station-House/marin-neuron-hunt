# Does the logit lens agree with the written neuron explanations?

Task: marin-neuron-atlas #34, "Use Existing Lenses". Lens: the logit lens (the only lens that exists for Marin 32B so
far; `stationhouse ask` reports no fitted lens, and task #35 is the one that fits one). Data: `pilot/fineweb-10m-sparse`.
Written with an AI coding agent (Claude Sonnet 5.5); the numbers all come from the scripts in this folder.

**Question.** For 15 neurons that already have explanations, does the logit lens agree with what the explanation says
each pole pushes up or down? And, as a check that does not use the lens, does the token that actually comes next at the
neuron's firings look like what the lens says?

## Short answer

- **Reading the lens against the text: it agrees every time it is asked, and that is weak evidence.** All 20 "this pole
  pushes X" claims (11 positive, 9 negative) pass the rule below. The explanations were written with the atlas lens in
  front of the author, and they quote its numbers, so this part is close to circular. What it does show is that the
  atlas table is faithful: recomputed from the weights, the 10 top tokens match the atlas table 10/10 for 27 of 30
  pole-ends and 9/10 for the other 3, and every lens value an explanation quotes matches our recomputation to within
  0.06.
- **Checking against what the model does is more informative, and it splits the claims.** In 9 of the 20 claims the
  actual next token at the firing is in the lens top 10 far more often than for the same layer's random neurons, and
  by a large absolute margin (10% to 82%). In 11 it is not, or only barely. Most of the misses are neurons whose job
  is *not* to name the very next token, so the lens is not wrong, but "agrees with the lens" would mislead there.
- **Where the lens does and does not speak:** measured as the top-1 value in standard deviations above the vocabulary
  mean (z), the clearly claimed ends sit at z 9 to 67, the faint claimed ends at 6.2 to 6.8, and every end the
  explanations call noise, plus the two ends of the lexical neuron L38-N6175, at 4.5 to 5.9. The faint ends are
  not separable from noise by z alone; they are only separable because the top ranks are on-topic words.

## Method

1. `compute_lens.py` reads only `down_proj` of the 10 layers involved, `model.norm.weight` and `lm_head.weight` from
   `marin-community/marin-32b-base@f19b346` by HTTP range request, and computes `W_U @ (W_down[:, i] * norm_w)` over
   the whole 128,256-token vocabulary. It also computes a no-norm-weight variant as a sensitivity check.
   The weights are fp32; nothing was run on Modal.
2. `neurons.json` holds the lens claims, transcribed by hand from each explanation (exact vocabulary strings).
   A pole *raises* the highest lens values if positive and the lowest if negative; "rank" below is always in that
   order, rank 1 = most raised.
3. **Verdict rule, fixed before looking at any result.** For a "promotes" claim: *agree* = at least one claimed token
   in the pole's top 10 and at least half of the claimed tokens in its top 100; *partial* = some claimed token in the
   top 100; *disagree* = none. "Noise" claims were read by hand (top tokens in `tables.md`).
4. `analyze.py` also reads the neuron's stored firings from the `sparse` tier (|act| at or above the pole's
   `q999`, at most 2,000 sampled with a fixed seed), looks up the token that followed, and reports how often it is in
   the lens top 10 against (a) the base rate of those 10 tokens in the 10M tokens, and (b) 10 random neurons from the
   same layer scored against *this* pole's top 10 (CONTRIBUTING's random-neuron control). It also reports the firing
   token itself, since some neurons fire on the word they promote.
5. `make_table.py` prints `tables.md` from `results.json`.

## Agreement table

Full tables (lens top-5 with values, atlas overlap, quoted-value check) are in `tables.md`. `rank` is in the raised
order. "next in top 10" is next-token hits against the lens top 10, with the same-layer random-neuron rate in brackets.
"Strongest 100" is the same hit rate over the 100 strongest firings. Hand label: **both** = lens agrees with text
and the next-token check supports it; **lens only** = lens agrees, next-token check does not (see notes);
**noise ok** = text says the lens is noise and it is; **lens differs** = the lens shows something the text says is not
there.

| neuron | pole | explanation says the pole pushes | lens rank of claimed tokens | rule | next in top 10 (control) | strongest 100 | hand label |
|---|---|---|---|---|---|---|---|
| L45-N1382 | + | `other` | `" other"` #1 | agree | 1.3% (0.1%) | 23% | lens only |
| L45-N22495 | + | `years`, `over` | `" years"` #1, `" over"` #2 | agree | 6.7% (0.4%) | 30% | lens only |
| L49-N23999 | + | the verb family (`ended`, `ends`), not `up` | `" ended"` #1, `" ends"` #2, `" wound"` #7 | agree | 1.5% (0.0%) | 0% | lens only |
| L42-N8668 | + | `or` | `"-or"` #1, `" or"` #2 | agree | 10.2% (0.6%) | 48% | both |
| L42-N8668 | - | (lens bottom is noise) | - | - | 0/505 | 0% | noise ok |
| L43-N22771 | + | `so` | `" so"` #1 | agree | 82.4% (0.2%) | 95% | both |
| L43-N22771 | - | (lens bottom is noise) | - | - | no stored firings | - | lens differs (weak) |
| L44-N12797 | + | (lens top is noise) | - | - | 0/2000 | 0% | noise ok |
| L44-N12797 | - | `least` | `" atleast"` #1, `" least"` #2 | agree | 36.7% (0.1%) | 79% | both |
| L47-N26485 | + | `well` | `" well"` #1 | agree | 21.9% (0.1%) | 97% | both |
| L47-N26485 | - | `too`, `also` (faint) | `" also"` #3, `" too"` #9 | agree | 0/665 (0.1%) | 0% | lens only |
| L49-N22962 | + | `alone` | `" alone"` #1 | agree | 20.6% (0.0%) | 57% | both |
| L49-N22962 | - | `much` | `" much"` #1 | agree | 0.9% (0.4%) | 6% | lens only |
| L48-N20456 | + | `imaginary`, `supposed`, `alleged`... | `" imaginary"` #1, `" supposed"` #3 | agree | 1.1% (0.0%) | 7% | lens only |
| L48-N20456 | - | hedges (`potentially`, `apparently`), weak | `" potentially"` #1, `" apparently"` #2 | agree | 0/307 | 0% | lens only |
| L41-N12578 | + | `the` | `" the"` #2 (and `"the"` #3, `" The"` #22) | agree | 32.6% (3.6%) | 70% | both |
| L41-N12578 | - | `another`, `others`, `other` | #1, #2, #3 | agree | 15.4% (0.2%) | 42% | both |
| L38-N24400 | + | (lens top has no theme) | - | - | 0/1468 | 0% | noise ok |
| L38-N24400 | - | `enough`, `sufficient`... | `" enough"` #1, `" sufficient"` #5 | agree | 16.1% (0.0%) | 63% | both |
| L39-N21755 | + | (lens top is noise) | - | - | 0/2000 | 0% | noise ok |
| L39-N21755 | - | `to`, `be` (weak) | `" to"` #1, `" be"` #2 | agree | 24.1% (4.6%) | 94% | both |
| L44-N10594 | + | football words (`QB`, `NFL`) | `" QB"` #1, `" NFL"` #6 | agree | 1.6% (0.0%) | 1% | lens only (topic neuron) |
| L44-N10594 | - | basketball words (`foul`, `baskets`) | `" forwards"` #1, `" foul"` #2 | agree | 0.1% (0.0%) | 0% | lens only (topic neuron) |
| L42-N970 | + | `Christians`, `believers` | `" Christians"` #1, `" believers"` #2 | agree | 2.9% (0.0%) | 12% | lens only (topic neuron) |
| L42-N970 | - | `Jonah`, `Luke`, `preaching` | `" Jonah"` #1, `" Luke"` #2 | agree | 2.3% (0.0%) | 1% | lens only (topic neuron) |
| L38-N6175 | + | (no lens claim, open PR #217) | - | - | 0.1% (0.2%) | 0% | lens silent |
| L38-N6175 | - | (no lens claim) | - | - | 0.8% (0.1%) | 1% | lens silent |

Silent poles with the lens end noted for completeness: L45-N1382 `-` (lens raises ` rest`, ` weekend`), L45-N22495 `-`
and L49-N23999 `-` (no theme).

## Disagreements and confusions

1. **The lens names a token two steps ahead, not the next one (L45-N22495, L45-N1382).** `over` fires when "the
   years" is coming, and the lens top is `" years"` (4.59), but the next token is `the`. Only 6.7% of firings are
   followed by a lens-top-10 token (30% among the strongest 100). The explanation is right; reading the lens as
   "next token" would say the neuron barely predicts anything. L45-N1382 is similar (1.3%, 23% among the strongest 100).
2. **The lens promotes the word the neuron fires on (L49-N23999).** The explanation says as much ("not `up`"). The
   next-token check gives 0/100 among the strongest firings because the next token is `up`. The firing token itself is
   in the lens top 10 for only 6.6% of the q999-tier firings; I did not run that check on the strongest 100, so I cannot
   say whether it is higher there.
3. **Faint lens ends that did not show up in behaviour (L47-N26485 `-`, L48-N20456 `-`).** Both have the claimed tokens
   at ranks 1 to 9 of 128k (so this is not chance), but top-1 z is 6.6 to 6.8, the same band as pole-ends the
   explanations call noise, and 0 of 665 and 0 of 307 firings were followed by a top-10 token. For L47 `-` the neuron
   fires on `well` after `as`, and the lens says "too/also"; I did not look at what does follow.
4. **A "no theme" end that has one (L43-N22771 `-`).** The explanation says the lens bottom has no theme. Raised by
   the negative pole are ` neither`, ` Neither`, `Neither` (3 of the top 5, values around -1). That is a pattern
   (possibly related to the "so do I" / "neither do I" pairing, which I did not test), though weak (z 5.5). The pole has no stored firings, so it
   cannot be checked against behaviour here.
5. **Topic neurons (L44-N10594, L42-N970).** The lens ends are coherent (football, basketball, Pauline vs Gospel
   words) and agree with the text, but the next token is a lens top-10 token in only 0.1% to 2.9% of firings. These
   neurons fire across a passage, not at the point where a token is about to be said, so the lens describes what they
   write toward, not when they fire. "Enrichment" for these is large only because the base rate is near zero; use the
   absolute rate.
6. **The lens is flat for a lexical neuron (L38-N6175, ours).** The atlas and our recomputation agree (10/10), and both
   ends are at z about 5.2 with no readable theme, although the neuron is a clean "either" detector. Layer 38 has 25
   layers after it, so what it does is possibly read later by other layers, not by the unembedding. This is a place
   where a fitted lens (tuned lens, task #35) is the missing tool.
7. **The rule is too generous.** All 20 "promotes" claims came out *agree*; the rule cannot rank how well. The
   next-token column is the better discriminator.
8. **Weak spot, L41-N12578 `+`.** `" the"` is a very common token (base rate of the lens top 10 is 3.9%, random neurons
   3.6%), and the lens top 5 are `\tthe`, ` the`, `the`, `,the`, `theValue`. The hit rate (32.6%) is still far above the
   control, but the lens barely separates this neuron from "any `the`".

## Caveats

- The lens is weights-only and direct: it ignores everything after the neuron's layer and the final norm's 1/rms scale
  (rank only). A neuron can matter without a lens signature, and a lens signature does not prove the neuron causes that
  token; the strongest lens-based claim here (L49-N22962, L44-N12797, L43-N22771) still needs steering (type B), not
  more lens readings.
- The `sparse` tier keeps only the 64 largest neurons per token, so firing sets are floors, and the sampled firings are
  from the q999 tier (broad), which is why headline hit rates are lower than rates among the strongest 100.
- Matching claimed tokens uses exact decoded strings; spelling variants outside the explanation's list are not counted.
- 15 neurons chosen because their explanations mention the lens; this is not a random sample of neurons. Claims were
  transcribed by hand and could be misread; `neurons.json` shows exactly what was used.
- Random controls: one set of 10 neurons per layer (seeded), up to 300 firings each.
- No steering or ablation was run, so nothing here is causal.

## Files

`compute_lens.py` (weights to lens), `neurons.json` (claims), `analyze.py` (scoring and behaviour check),
`make_table.py` (tables), `results.json` (all numbers), `tables.md` (full tables). To rerun:

```
python compute_lens.py lens.npz
python analyze.py lens.npz <atlas dir with summaries/ and vocab/> results.json
python make_table.py results.json > tables.md
```
