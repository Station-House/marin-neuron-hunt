# L6-N9321, positive side: what was unclear, and experiments to narrow it down

Read `exemplars.md` first. This file is for choosing which hypotheses to test. The sentences below only show
the *shape* of each test. I'll write the real prompts after you approve a set, and none of them have been run
yet.

## Working picture

- **+ side:** a week or weekday word that refers to **one occasion**: "one morning last week", "on Monday,
  December 21", "the week before Christmas", "the 27th falls on a Sunday".
- **− side:** the same words when they name a **repeating slot**: "every week", "airs every Tuesday", "once a week
  on Sunday", "each week". It also fires in **serial content** (weekly columns, shows, recaps), including on
  words that aren't about time ("edition", "host", "meeting").
- Both sides fire on the same tokens. " week" is the top token on both sides: 879 strong + firings and 551
  strong − firings. The context decides the sign, not the word.

**One thing the exemplars can't tell us:** the model only sees text up to the target. In "on[[ Monday]], December
21", the date comes after the target, so it can't be what drives the + reading. Whatever drives the sign has to
come before the target.

## What's unclear

| # | Question | Why it's unclear |
|---|---|---|
| U2 | Does **"this week"** flip with the type of text? | It reads + in narratives ("One morning this[[ week]], I woke", +2.12) and − in serial content ("This[[ week]] we take a look", −2.20). This is the cleanest contrast we have. |
| U5 | Is it about **weeks**, or about repetition vs one occasion **in general**? | The top tokens are almost all week or weekday words, plus some " night". We don't know if "every month", "every morning" or "last year" behave the same way. |
| U7 | Do **weekday names on their own** lean one way? | "Monday", "Tuesday" and so on appear a lot on both sides. With no qualifier ("We met on Thursday"), we don't know which side wins. |

## Experiments

| id | type | Question | Shape of the prompts | What each hypothesis predicts |
|---|---|---|---|---|
| E1 | A, + side | Summary: does + carry over to fresh one-occasion sentences? | about 12 fresh sentences like "I lost my keys last[[ week]]". The should-not-fire group uses the same tokens in repeating slots. | All hypotheses: + above q99. |
| E7 | A, + and − | Scope beyond weeks. | "every[[ month]]" / "last[[ month]]", "every[[ morning]]" / "that[[ morning]]", "every[[ year]]" / "last[[ year]]". | H5: about zero on both. Otherwise the same split as for weeks. |
| E8 | A | Random baselines for E1 and E2. | Everyday sentences with the target on the last word. | The side's q99 rarely gets crossed. |
