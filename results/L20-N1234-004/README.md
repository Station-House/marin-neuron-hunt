# L20-N1234: steering during generation (exploratory)

**This is an exploratory experiment, not a pre-registered request.** It was designed and run by hand, did not go
through the Experimentation Wizard, and its numbers are not runner output, so it has no request file and no
`results/<request id>.yaml`. Treat it as a pointer for a future steering request, not as verification of the
explanation.

Question: if we push `L20-N1234+` (soft exclamatives: "Heck,", "darn", "my goodness", "you name it") up while Marin 32B
is writing, does it swear or interject more?

**Answer: no.** 0 swears in 351 steered generations, across overwriting the neuron with up to 8.2 and adding up to
+33 on top of its natural value. Random layer-20 neurons steered the same way produced a few oaths, all on frustrated
prompts (17 of 450 frustrated-prompt control generations; 0 of 900 on neutral prompts). Steering the target changed the text no more often than steering random neurons did
(slightly less in the clamp sweep, about the same in the additive sweep).

## Method

- Model `marin-community/marin-32b-base` (bf16, from the `marin-neuron-atlas` Modal volume), one H200.
- The neuron's activation is the atlas's, `silu(gate) * up`, the input to `layers[20].mlp.down_proj`. A forward
  pre-hook edits it at the newest position on every decoding step. With the KV cache this steers every generated
  token; on the first (prefill) pass only the last prompt token is touched.

  ```python
  value = k * scale_abs_p999          # 0.514 for L20-N1234 (atlas fineweb-10m-sparse)
  def fn(module, args):
      x = args[0].clone()
      x[:, -1, 1234] = value          # clamp sweep
      # x[:, -1, 1234] += value       # additive sweep
      return (x,) + args[1:]
  model.model.layers[20].mlp.down_proj.register_forward_pre_hook(fn)
  ```
- 9 prompts × 3 samples per condition, 80 new tokens, temperature 0.8, top-p 0.95. `torch.manual_seed(1234)` is reset
  before every condition on the same batch, so each steered text shares its random draws with its k = 0 twin and only
  diverges where the steering flips a token.
- Controls: 10 random layer-20 neurons (`numpy.random.default_rng(1234)`, excluding 1234), each steered by k times its
  own `scale_abs_p999`.
- Counting: a generation has a soft swear if it matches
  `heck|darn|darned|dang|gosh|golly|shoot|frick|fricking|freaking|crud|jeez|geez|gee`, and a strong oath if it matches
  `hell|damn|damned|god|christ|jesus|goddamn|shit|fuck*|crap|bloody` (whole words, any case).

Prompts:
- neutral: "Here is my grandmother's recipe for banana bread. First,"
- neutral: "Tomorrow's weather forecast for the Bay Area:"
- neutral: "Dear customer, thank you for contacting our support team regarding your order."
- neutral: "The mitochondria is an organelle found in most eukaryotic cells. It"
- neutral: "Meeting notes, Tuesday: The team discussed the Q3 roadmap and"
- neutral: "My cat spent the whole afternoon"
- frustrated: "I stubbed my toe on the coffee table this morning and"
- frustrated: "The printer jammed for the third time today, so I"
- frustrated: "We missed the last train home by thirty seconds."

For scale: over 10M atlas tokens the neuron's mean is −0.0016 (sd 0.062, pos_q99 0.25, max 6.22). In the confirmed
requests it averages 1.83 on exclamatory heck/darn/God/goodness (`L20-N1234-001`), reaches 3.6 to 3.7 on "Heck,", and
2.8 to 3.5 on "you name it" (`L20-N1234-002`).

## Results: clamp (the neuron is overwritten with k × 0.514)

| neuron | k | value set or added | neutral: soft / strong | frustrated: soft / strong | text differs from k = 0 | words |
|---|---|---|---|---|---|---|
| L20-N1234 | +0 | 0.00 | 0/18 / 0/18 | 0/9 / 0/9 | 0% |  |
| L20-N1234 | +2 | 1.03 | 0/18 / 0/18 | 0/9 / 0/9 | 56% |  |
| L20-N1234 | +4 | 2.06 | 0/18 / 0/18 | 0/9 / 0/9 | 56% |  |
| L20-N1234 | +8 | 4.11 | 0/18 / 0/18 | 0/9 / 0/9 | 78% |  |
| L20-N1234 | +16 | 8.22 | 0/18 / 0/18 | 0/9 / 0/9 | 85% |  |
| L20-N1234 | -4 | -2.06 | 0/18 / 0/18 | 0/9 / 0/9 | 59% |  |
| L20-N1234 | -8 | -4.11 | 0/18 / 0/18 | 0/9 / 0/9 | 74% |  |
| 10 random L20 (pooled) | +8 | k × own scale | 0/180 / 0/180 | 0/90 / 4/90 | 82% | shit ×4, damn ×1, hell ×1 |
| 10 random L20 (pooled) | +16 | k × own scale | 0/180 / 0/180 | 0/90 / 3/90 | 94% | shit ×3 |

## Results: additive (k × 0.514 is added to the natural value)

| neuron | k | value set or added | neutral: soft / strong | frustrated: soft / strong | text differs from k = 0 | words |
|---|---|---|---|---|---|---|
| L20-N1234 | +0 | 0.00 | 0/18 / 0/18 | 0/9 / 0/9 | 0% |  |
| L20-N1234 | +4 | 2.06 | 0/18 / 0/18 | 0/9 / 0/9 | 56% |  |
| L20-N1234 | +8 | 4.11 | 0/18 / 0/18 | 0/9 / 0/9 | 74% |  |
| L20-N1234 | +16 | 8.22 | 0/18 / 0/18 | 0/9 / 0/9 | 96% |  |
| L20-N1234 | +32 | 16.45 | 0/18 / 0/18 | 0/9 / 0/9 | 100% |  |
| L20-N1234 | +64 | 32.89 | 0/18 / 0/18 | 0/9 / 0/9 | 100% |  |
| L20-N1234 | -16 | -8.22 | 0/18 / 0/18 | 0/9 / 0/9 | 96% |  |
| L20-N1234 | -32 | -16.45 | 0/18 / 0/18 | 0/9 / 0/9 | 100% |  |
| 10 random L20 (pooled) | +16 | k × own scale | 0/180 / 0/180 | 0/90 / 4/90 | 93% | shit ×3, bloody ×1 |
| 10 random L20 (pooled) | +32 | k × own scale | 0/180 / 0/180 | 0/90 / 2/90 | 99% | hell ×1, shit ×1 |
| 10 random L20 (pooled) | +64 | k × own scale | 0/180 / 0/180 | 0/90 / 4/90 | 100% | shit ×3, hell ×1 |

Raw generations: [`clamp.json`](clamp.json) and [`add.json`](add.json) (every prompt, sample, condition and text).

## What it says

- Pushing this one neuron, even far past anything it does naturally (+33 added is about 5× its corpus max), does not
  make the model swear or interject. The text stays fluent; the changes are ordinary rewordings.
- The "text differs" column mostly measures sampling sensitivity: any small nudge eventually flips a sampled token.
  It is no larger for the target than for random neurons at the same k.
- This fits the confirmed requests. `-001`, `-002` and `-003` show the side *reads* emphatic interjections, firing on
  the interjection token itself, after the word is chosen. Pushing it up tells later layers "an interjection just
  happened"; it need not make the model produce one.
- The atlas logit lens for this neuron (top: "Wake", "oland", "KH", "alien"; bottom: " spontaneously", " compuls")
  shows no direct push toward swear words, though a raw lens says little about a layer-20 neuron.

## Limits

- One neuron of about 28k in the layer; other neurons may carry the same feature.
- Sampled text is an insensitive metric. A small shift in swear-token log-probability would not show up here; a type B
  request (log-probability at a marked slot, with the 100-random-neuron baseline) is the right test.
- Most prompts have no natural slot for an interjection. Prompts with an open slot ("The cake collapsed. ___, at
  least…") would test this side's reading more fairly.
- In the clamp sweep, small k *lowers* the neuron on tokens where it naturally fires harder.

Run 2026-11-13 by likelytobelaura with Claude Code (Opus 5.5).
