## Neuron(s)

<!-- e.g. L40-N25546. Opening this PR as a draft claims the neuron(s); claims lapse after 7 days without activity. -->

## What's in this PR

- [ ] An explanation: `explanations/L<layer>/L<layer>-N<neuron>.yaml`, covering **both sides**
- [ ] A confirmation-experiment request: `requests/<neuron>-<nnn>.yaml` (type: activation-test / steering / ablation / contrast / custom)
- [ ] An alternative to an existing explanation, plus the request that would tell them apart

## How you investigated

<!-- Data prefix, tools, queries or notebooks. If an agent wrote it, name the model or agent (also in `by`). -->

## Checklist

- [ ] `python scripts/validate.py` passes
- [ ] I wrote the request's prompts and prediction before seeing any result for them
- [ ] Each marked `[[target]]` is a single token for the Marin tokenizer
