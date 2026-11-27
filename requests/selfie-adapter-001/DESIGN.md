# Lens fit: SelfIE adapter for Marin 32B

**Request:** `selfie-adapter-001` (type E, custom)
**Lens:** selfie-adapter
**Follows:** SelfIE, Chen, Vondrick and Mao 2024 (https://arxiv.org/abs/2403.10949); adapter training on interpretability artifacts, Pepper et al. (https://arxiv.org/abs/2602.10352); faithfulness control after Patchscopes (https://arxiv.org/abs/2401.06102) and Faithful-Patchscopes (https://arxiv.org/abs/2602.00300)
**Model:** marin-community/marin-32b-base @ f19b3466

## Claim

A scalar affine adapter, trained on Marin's existing interpretability artifacts with the model frozen, makes Marin's self-interpretations of its own neuron activations reliable: on held-out explained neurons its plain-language descriptions recover the substance of the atlas explanations, a readout the atlas logit lens cannot produce at all (it yields token lists, not descriptions), at layers across the atlas's explained range (1 to 49 in the check set).

## Unit and metric

This lens's unit is not a text passage. The fitted object maps an activation vector into the model's token embedding space, so the unit of fitting and evaluation is one vector-label pair: a neuron's or SAE feature's activation vector at its home layer, with its label. The request file's passage fields are answered with pair counts, and the interpretation context is fixed at 128 tokens.

- **Passage length (tokens):** 128 (the fixed interpretation context; fitting units are pairs, above)
- **Number of passages (pilot / full run):** 500 pairs pilot / 1000 pairs full run; of the pilot's 500, 100 are held out and double as the fixed probe set
- **Metric for "the fit has settled":** two adapters fitted on disjoint halves of the training pairs produce descriptions whose blind scores on a fixed probe set agree within 5 points
- **Metric for "the lens is good":** blind generation score: whether the generated description matches the pair's held-out label or the neuron's atlas explanation, rated by two LLM judges from different model families with disagreements escalated, and reported as the margin over untrained SelfIE under the identical protocol. Each item scores match or no match; the generation score is the percentage of matches, which is the reading every threshold in this design assumes

## Controls

- **Baseline:** untrained SelfIE under the identical interpretation protocol (same prompt, injection layer rule, scale, decoding settings), no adapter. This is the comparison that isolates what the fitting bought.
- **Other comparisons:** the atlas logit lens top tokens for the same neurons, printed beside every readout; an unpatched contrast run (the interpretation prompt with no vector injected), so descriptions driven by the prompt's priors rather than the state are visible and discounted, after Faithful-Patchscopes; and a shuffled-pairs control arm (added at the contributor's direction during the design review): the identical adapter, with identical capacity and training steps, fitted on the same pairs with the vector-label correspondence shuffled. If the real adapter's margin over untrained SelfIE is matched by the shuffled adapter's, the gain came from learning label style and common topics rather than reading the vectors, and the claim fails on that evidence. Matched has a number, declared in advance: the gain is credited to reading the vectors only if the real adapter's margin exceeds the shuffled adapter's margin by at least 10 points; a gap under 10 points credits the gain to label style.

## Data

- **Fitting text, and why:** no fitting text; fitting pairs, from one source. Auto-labeled features of the canonical sae-64k sweep (Hugging Face, alxcspr/stationhouse: TopK, width 65536, k=32, all 64 layers, trained 2026-11-06/07). The sweep exists; its census files carry statistics, not labels, so pairs from this source need an auto-labeling pass over the features actually used. Labels are written by a Gemini-family model from each feature's top exemplars. No explained neuron is a training pair: all 19 atlas explained neurons are held out for evaluation, because the claim is stated about explained neurons and the verdict must rest on items the fit never saw. Logit-lens readouts are likewise NOT training pairs: their labels are token lists, not phrases, and training on them would teach the adapter to verbalize the very readout the claim says it goes beyond; the logit lens appears only as the cross-check printed beside every readout. Convention: the 64k family was trained on raw activations (a label-parse fault dropped per-neuron scaling) while the 32k family used scaled inputs; this design uses sae-64k with its raw convention throughout and never mixes the two. The pilot begins with an audit of the pair supply: counts, label quality, and the price of the labeling pass. If the usable pair count is too small to train on, the fit stops and the design falls back to a tuned lens rather than training on scraps.
- **Held-out evaluation text:** all 19 explained neurons (the 5 check neurons are the named subset reported individually) and a random 20% of the SAE pairs, withheld before fitting, evaluated only through the fixed interpretation protocol. The split is by unit id: every pair belonging to one feature lands on the same side, and any sae-64k feature whose decoder vector has cosine similarity 0.5 or higher to any explained neuron's direction is excluded from training, so no form of an explained neuron's vector can land in training.
- **Filters applied:** labels must be phrases, not single tokens, so generation scoring measures description rather than token recall; pairs whose vectors cannot be extracted at the stated convention are dropped and counted in the manifest.
- **Tokenizer and BOS choice:** Marin tokenizer; BOS prepended (bos: true), fixed as part of the published protocol.

## Noise

Fit-to-fit variation is estimated by reruns: the adapter is refitted on two disjoint halves of the training pairs (two different random splits, seeds recorded in the fit log), and the two fits' blind scores on the fixed probe set must agree within 5 points, the smallest difference treated as real, declared in advance. The probe set is the pilot's 100 held-out items: on 100 items, plain binomial sampling spread on a single scoring pass is about 5 points at the midpoint, and the split-half comparison is paired on the same items, so the 5-point tolerance is set at the scale of the measurement noise rather than below it. A random-unit baseline runs alongside: the same protocol applied to random vectors drawn from the same layers, whose descriptions are scored identically, giving the judges' false-match rate on content-free states; the adapter's margin is reported net of that baseline. Prompt sensitivity is removed by fixing one interpretation prompt for every run and publishing it in the protocol. Judge noise is controlled by blind presentation (judges do not know which system produced a description), two judges from different model families, escalation of disagreements, and a calibration gate: the contributor scores a small blind sample first, and the judges' scores count only once they match the contributor's ratings on at least 80% of that sample; the contributor re-audits a random slice at the full run. Labeler separation: the sae-64k labels are written by a Gemini-family model, and neither judge comes from the Gemini family, so no judge can reward descriptions merely for copying the labeler's phrasing. Margins are reported per source, and the claim's verdict rests on the 19 held-out explained neurons, because the claim is stated about explained neurons: a margin carried only by auto-labeled SAE pairs does not support this claim. On 19 items, one neuron flipping moves the margin by about 5 points, the scale of the noise floor declared above, so the 25-point bar corresponds to roughly five net neuron recoveries and can separate support from chance.

## Readout

### Pilot

- **Passages:** 500 pairs (400 fitted, 100 held out as the evaluation and probe set), each read at its home layer
- **Settings (layers, batch, precision):** scalar affine adapter (one shared scalar scale plus a bias vector, d_model + 1 parameters in total), model frozen in bf16; each vector is extracted at its home layer and the adapted vector is injected in place of a designated token embedding in the interpretation pass; the untrained baseline arm uses the identical extraction and injection and differs only by the adapter; one fixed interpretation prompt; fixed decoding settings; BOS prepended
- **Estimated compute:** training is a small optimization on cached vectors, CPU-feasible; readout and scoring are a few hundred 32B generations, one GPU, hours (estimate)
- **Bar to pass before the full run:** beats untrained SelfIE by at least 25 points of blind generation score on the 19 held-out explained neurons; non-degenerate descriptions on at least 4 of the 5 check neurons; split-half agreement within 5 points
- **What the runner should save:** adapter weights, the pair manifest with the held-out split, every generated description, every score, the fit log

### Full run

- **Passages:** 1000 pairs fitted; readout over all 19 explained neurons plus a declared sample of 100 unexplained neurons in layers the atlas covers
- **Settings, and what changes from the pilot:** same adapter form and protocol, fitted on the full audited pair set minus held-out material; only the readout scope and the held-out scoring set grow
- **When the fit stops:** the readout table is complete and the full held-out set is scored
- **Estimated compute:** one to two thousand 32B generations, one GPU, under a day (estimate)
- **Bar you are aiming for:** the pilot margins hold at full scale: at least 25 points over untrained SelfIE on the full held-out set, check-set recovery at or above the pilot's
- **The lens failed if:** the adapter fails to beat untrained SelfIE on held-out generation scoring, or its descriptions collapse to generic text (prior override, judged against the unpatched contrast control) on 3 or more of the 5 check neurons, or the full run fails its rule: the 19-neuron margin falls under the pilot bar, or recovery across the 19 drops by 3 or more neurons from the pilot. A change within the 5 named check neurons alone (3 of 5 after a pilot 4 of 5) is one flip and does not by itself fail the run
- **Outcome zones, declared in advance (contributor's ruling during the design review):** a margin at or above the bar is full support and the full run proceeds. A positive margin below the bar, net of the random-vector baseline, with non-degenerate text on at least 4 of 5 check neurons (the worked example: +10 points) counts as the claim SUPPORTED, with the effect smaller than predicted: the direction is established and the method is on the right path; whether the full run still proceeds at that size is governed by the walk-away threshold in the request file's prediction, not by the claim's verdict. A margin of zero or below, net of baselines, refutes the claim. A net margin inside the 5-point noise floor in either direction is inconclusive: neither support nor refutation, and the design is revisited before any full run.
- **What the runner should save and publish:** adapter weights, the interpretation protocol, the fit log, the pair manifest, the per-neuron readout table for all 19 explained neurons (with the 14 trained-on and 5 held-out neurons reported separately, so the table itself shows the generalization gap), and the baseline comparison table

### Check on explained neurons

- **Neurons:** L42-N5277 (fringe-belief talk; expects conspiracy, anti-, fringe), L32-N24138 (outcome of a test or check comes next), L40-N10819 (reciprocal phrases: preceding words and each), L44-N10594 (team-sport context: American football vs basketball), L49-N22962 ('let' before 'alone' in 'let alone'). The same five are in the request file's neurons field, and all five are held out of training.
