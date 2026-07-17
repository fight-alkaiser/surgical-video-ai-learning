# Day24: Predicted-Instrument-Conditioned Verb Recognition

## Objective

Day23 showed that conditioning verb prediction on the *true* instrument
label more than doubles macro F1 (0.192 -> 0.388), confirming that
Day22's weak verb recognition was partly an architecture gap. That was
deliberately an oracle experiment -- ground-truth instrument labels are
never available in a real pipeline, only an imperfect classifier's
predictions are (Day21's, at 89.4% macro accuracy but with very uneven
per-instrument F1). Today closes that gap: an actual instrument
classifier is trained, its *predicted* probabilities condition the verb
classifier, and the whole thing is evaluated end-to-end -- the realistic
version of Day23's question.

## Method

[`predicted_instrument_conditioned_verb.py`](predicted_instrument_conditioned_verb.py)
uses the same 10 videos, video-level 8/2 split, and frozen ResNet18
backbone as Day21-23. Since the backbone is identical for both
classifiers, its 512-d features are extracted **once per frame and
cached**, rather than recomputed for two separate training runs -- this
turned what would have been two ~15-minute training runs into a ~2.5
minute one-time feature extraction plus two near-instant linear-layer
trainings on cached vectors.

Pipeline: (1) train an instrument classifier (Day21's model, on cached
features) on the 8 training videos; (2) get its *predicted* sigmoid
probabilities for all frames, train and test alike -- never the true
label; (3) train a verb classifier on [cached image features + predicted
instrument probabilities], exactly as Day23 did with the true label
instead; (4) evaluate on the test videos using the same classifier's own
predicted instrument probabilities for those frames.

## Results

**Instrument classifier (re-trained on cached features):** 0.896 mean
per-class test accuracy, matching Day21's 0.894 almost exactly --
confirms the feature-caching shortcut reproduces Day21's model
faithfully.

**Verb classifier, conditioned on predicted (not true) instrument probabilities:**

| Verb | Day22 F1 (no conditioning) | Day24 F1 (predicted-instrument) | Day23 F1 (oracle) |
|---|---:|---:|---:|
| grasp | 0.434 | 0.402 | 0.467 |
| retract | 0.692 | 0.744 | 0.822 |
| dissect | 0.652 | 0.663 | 0.859 |
| coagulate | 0.052 | 0.023 | 0.387 |
| clip | 0.000 | 0.119 | 0.629 |
| cut | 0.000 | 0.000 | 0.208 |
| aspirate | 0.045 | 0.169 | 0.311 |
| irrigate | 0.000 | 0.000 | 0.000 |
| pack | 0.000 | 0.250 | 0.182 |
| null_verb | 0.042 | 0.044 | 0.019 |
| **Macro F1** | **0.192** | **0.241** | **0.388** |

Macro F1 (0.241) lands between Day22's floor and Day23's ceiling, but
much closer to the floor: only about a quarter of the possible gain
(0.049 of 0.196) was realized with a real, imperfect instrument
classifier in the loop.

## Interpretation

**The shortfall from Day23's ceiling tracks Day21's own per-instrument
F1 almost exactly.** Recalling Day21's instrument results: grasper 0.860,
hook 0.677 (both reliable), vs. bipolar 0.106, scissors 0.054, clipper
**0.012** (all barely-functioning classifiers for those specific
instruments). The verbs that depend on the reliable instruments
(retract/grasper, dissect/hook) improved over Day22 in a real, if
partial, way. The verbs that depend on the barely-functioning
instruments fared far worse than Day23's ceiling: `clip` (needs clipper,
F1 0.012) only reached 0.119 of a possible 0.629; `coagulate` (needs
bipolar, F1 0.106) actually got *worse* than no conditioning at all
(0.052 -> 0.023). A conditioning signal is only as useful as its own
accuracy -- clipper is detected so unreliably that telling the verb
classifier "here's my best guess at whether a clipper is present" adds
mostly noise, not information, for exactly the verb that oracle
conditioning helped the most.

**Two verbs got worse than Day22's no-conditioning baseline, which is
itself an important, distinct finding: adding an unreliable auxiliary
signal isn't neutral, it can actively hurt.** `grasp` (0.434 -> 0.402)
is a case where Day23's own oracle ceiling showed there wasn't much to
gain in the first place (0.434 -> 0.467, a small improvement even with
perfect information) -- so a noisy version of that small, marginal signal
is easily net-negative once its own errors are added in. `coagulate`
(0.052 -> 0.023) is the clearer case: bipolar's own detection is close
to useless (F1 0.106), so its predicted probability is closer to random
noise than to real information, and the verb classifier pays a real
cost for attending to it.

**`pack` (0.000 -> 0.250) exceeding even its own oracle result (0.182)
is very likely sampling noise, not a real effect.** Pack is the rarest
verb in this dataset (0.5% train prevalence, 0.1% test -- a handful of
positive test frames at most), so its F1 is highly unstable and small
absolute changes in a handful of predictions swing it substantially; this
number shouldn't be read as "predicted conditioning beats oracle
conditioning for pack," just as noise in an estimate built on very few
examples.

## Reflection

This is a clean, concrete illustration of error propagation in a
multi-stage pipeline: Day23 established the *ceiling* a perfect first
stage would allow, and today shows how much of that ceiling survives
once the first stage's real, uneven accuracy is accounted for. The
answer here is "not much, and it depends entirely on which instrument" --
reliable instruments (grasper, hook) pass through a real, useful signal;
unreliable ones (bipolar, clipper) pass through noise that can be worse
than passing through nothing. This means the earlier framing --
"conditioning verb prediction on instrument identity should help
tool-specific verbs" -- was correct as a statement about information
content, but incomplete as engineering advice: it only pays off in
practice to the extent the upstream classifier is itself trustworthy for
that specific instrument, which for the rarest instruments in this
dataset, it currently is not.

This also reframes what the actual bottleneck is for a realistic
Rendezvous-style pipeline on this data: not the *architecture* of
conditioning verb on instrument (Day23 showed that architecture is
sound and valuable in principle), but the *upstream instrument
classifier's* weakness on rare classes -- exactly the data-scarcity
problem Day21 first identified, now shown to compound downstream rather
than stay contained to instrument recognition alone.

## Conclusion

Conditioning verb prediction on a real instrument classifier's predicted
probabilities raises macro F1 from 0.192 to 0.241 -- a real but partial
gain, reaching only about a quarter of the distance to Day23's oracle
ceiling (0.388). The gap is explained almost entirely by Day21's own
per-instrument accuracy: verbs tied to well-detected instruments
(grasper, hook) capture a meaningful share of the oracle improvement;
verbs tied to poorly-detected ones (bipolar, clipper) capture little or
none, and can even regress below the no-conditioning baseline. Improving
this pipeline further now points squarely back to Day21's original,
still-unsolved problem -- rare-instrument detection -- rather than to any
change in how verb prediction uses instrument information.
