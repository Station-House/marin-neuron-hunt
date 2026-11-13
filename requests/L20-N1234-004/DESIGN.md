# L20-N1234-004: steering during generation

Written after the run (2026-11-13). This documents a hand-run exploratory experiment; it was not
pre-registered and did not go through the Experimentation Wizard, so its outcome is reported as inconclusive.
Request file: requests/L20-N1234-004.yaml. Type E (custom), because no menu type steers across a whole
generation: steering (B) edits one marked token and reads next-token log-probabilities.

### Claim

`L20-N1234+` (soft exclamatives, "you name it"; confirmed as a reader by -001, -002 and the -003 per-frame
numbers) also drives them: pushing it up while Marin 32B writes raises the rate of soft swears and oaths in
the generated text.

### Unit and metric

Unit: one sampled generation of 80 new tokens. 9 prompts (6 neutral, 3 frustrated) x 3 samples = 27 per
condition. Metric: number of generations containing a soft swear or a strong oath (word lists in the request),
plus the share of generations whose text differs from the k = 0 twin.

### Controls

Baseline k = 0 (no hook) with the same seed and batch. Random-neuron baseline: 10 random layer-20 neurons
(default_rng(1234)) steered by the same k on their own scale_abs_p999. The other side: negative k (clamp -4,
-8; additive -16, -32).

### Data

Model marin-32b-base @ f19b3466, bf16. Scale from pilot/fineweb-10m-sparse pole_stats (scale_abs_p999 =
0.514). Prompts are fresh, written for this run, not copied from atlas exemplars. Filters: none.

### Noise

Sampling at temperature 0.8 with seed 1234 reset per condition, so k = 0 and steered texts share random draws.
3 samples per prompt; with 27 generations per condition, a difference of 3 or more swearing generations is the
smallest effect read as a signal. Steering strengths run from k = 2 to k = 64 (up to about 5x the side's corpus
max of 6.22).

### Readout

Supported if the target at some k > 0 adds at least 3 swearing generations over k = 0 and beats the pooled
random-neuron rate at the same k. Because the design is post hoc, the outcome is inconclusive in every case,
and a null result cannot refute the explanation (one neuron of about 28k; sampled text is insensitive). A null
result would mean the next test is a type B request at a marked interjection slot.
