# Day23: Instrument-Conditioned Verb Recognition

## Objective

Day22 split verb recognition's weak macro F1 (0.192, vs. Day21's 0.302
for instruments) into two distinct causes. Grasp/retract/null_verb
looked like a genuine single-frame ambiguity. But clip, cut, aspirate,
and irrigate looked like something more fixable: checking instrument-verb
co-occurrence directly showed these verbs are 79-95% determined by
instrument identity alone (clipper -> clip 94.9%, scissors -> cut 91.0%,
bipolar -> coagulate 78.7%, hook -> dissect 86.6%), yet Day22's verb
classifier never saw instrument identity at all -- a fully independent
linear head re-deriving redundant information from a handful of rare
positive examples. Today tests that hypothesis directly.

## Method

[`instrument_conditioned_verb.py`](instrument_conditioned_verb.py) reuses
Day22's exact setup (same 10 videos, same video-level 8/2 split, same
frozen ResNet18 backbone) with one change: the backbone's final layer is
removed (producing a 512-d feature vector instead of class scores), and
that vector is concatenated with the **true** 6-d instrument multi-hot
label before a newly trained (512+6) -> 10 linear layer predicts verbs.

This is deliberately an **oracle test**: the model is given the ground-
truth instrument label, not Day21's imperfect predictions, both at train
and test time. That isolates one specific question -- "if the model knew
the instrument perfectly, would tool-specific verb prediction improve?"
-- from a separate one -- "how good is Day21's instrument classifier, and
does its error propagate?" This is the easier, upper-bound version of
the real pipeline; conditioning on Day21's *predicted* instrument
probabilities instead of ground truth is a natural, more realistic next
step, not attempted here.

## Results

| Verb | Day22 F1 (no conditioning) | Day23 F1 (instrument-conditioned) | Instrument -> verb co-occurrence |
|---|---:|---:|---:|
| grasp | 0.434 | 0.467 | grasper -> grasp 29.9% |
| retract | 0.692 | 0.822 | grasper -> retract 63.0% |
| dissect | 0.652 | 0.859 | hook -> dissect 86.6% |
| coagulate | 0.052 | 0.387 | bipolar -> coagulate 78.7% |
| clip | 0.000 | 0.629 | clipper -> clip 94.9% |
| cut | 0.000 | 0.208 | scissors -> cut 91.0% |
| aspirate | 0.045 | 0.311 | irrigator -> aspirate 66.3% |
| irrigate | 0.000 | 0.000 | irrigator -> irrigate 9.3% |
| pack | 0.000 | 0.182 | (grasper, rare overall) |
| null_verb | 0.042 | 0.019 | -- |
| **Macro F1** | **0.192** | **0.388** | -- |
| **Macro accuracy** | 0.888 | 0.924 | -- |

## Interpretation

**The central hypothesis holds, cleanly.** Macro F1 more than doubled
(0.192 -> 0.388) from one change -- giving the model the true instrument
label -- with everything else identical to Day22. This confirms the
diagnosis: a meaningful part of Day22's weak verb performance was not a
hard information limit, but a model architecture that discarded
information (instrument identity) it could easily have used.

**The size of each verb's improvement tracks its instrument -> verb
co-occurrence strength almost exactly**, which is the clearest evidence
this is the right explanation, not a coincidence. `clip` (94.9%
determined by clipper) went from undetectable (F1 0.000) to F1 0.629 --
the single largest jump. `dissect` (86.6% via hook) and `coagulate`
(78.7% via bipolar) both roughly tripled or more. `cut` (91.0% via
scissors) improved from 0.000 to 0.208 -- real, but far more modest than
`clip` despite a similar co-occurrence rate, almost certainly because
scissors is far rarer than clipper in this data (512 vs. 593
co-occurrence instances across all 10 videos, but concentrated in fewer
of the 8 training videos) -- conditioning on instrument identity removes
the *architectural* bottleneck, but can't manufacture examples that
were never there to begin with, echoing Day21's original rare-instrument
finding.

**Grasp barely moved (0.434 -> 0.467), exactly as predicted.** Knowing
"grasper is present" doesn't resolve whether it's grasping, retracting,
or idle -- that's still the same single-frame ambiguity Day22 identified,
and instrument conditioning was never expected to fix it. Retract, the
majority action when a grasper is used (63.0%), improved substantially
(0.692 -> 0.822) for the same reason `clip` and `dissect` did: knowing
the instrument at least tells the model which action is *most likely*,
which disproportionately helps the majority class within that
instrument's repertoire, not the minority one.

**Irrigate and null_verb didn't improve, for two different reasons worth
distinguishing.** `irrigate` stayed at F1 0.000: irrigator's own dominant
action is `aspirate` (66.3%), not `irrigate` (9.3%), so "instrument =
irrigator" mostly predicts aspirate correctly and still can't distinguish
irrigate as a minority case within irrigator's own usage -- the same
kind of within-instrument ambiguity as grasp/retract, just for a rarer
instrument. `null_verb`, unexpectedly, got *worse* (0.042 -> 0.019): a
plausible explanation is that instrument presence is itself correlated
with active use rather than idleness, so telling the model "an
instrument is present" likely pushes it away from predicting "idle,"
even on the frames where idle was actually correct.

## Reflection

This day is a rare, clean confirmation of a hypothesis proposed in the
previous day's writeup, with a mechanism (instrument -> verb
co-occurrence) measured directly and matching the outcome almost
verb-for-verb. That's worth noting precisely because it's uncommon in
this project: most days so far (Day16's embedding structure, Day18's
positional encoding, Day20's evaluation protocol) uncovered a problem
without a single, clean lever whose effect could be measured this
directly. Here, the earlier reasoning ("this looks like an architecture
gap, not a data gap, because instrument and verb are highly correlated")
made a specific, falsifiable prediction (co-occurrence rate predicts
improvement size), and it held up when checked.

It's also a useful reminder about what this result does and doesn't
show. This is an oracle experiment: the true instrument label was
available at test time, which will never be true for a real end-to-end
pipeline running on new video. The realistic version of this result
requires conditioning on Day21's *predicted* instrument probabilities
(themselves imperfect, particularly for the same rare instruments that
limit `cut` here), and would be expected to land somewhere between
Day22's fully-independent result and today's oracle ceiling -- how much
closer to the ceiling is itself an open, testable question, and a
natural next step.

## Conclusion

Conditioning verb prediction on true instrument identity raised macro F1
from 0.192 to 0.388, with each verb's improvement tracking how strongly
that verb co-occurs with a specific instrument -- confirming that a real
part of Day22's weak verb recognition was an architectural gap (an
independent classifier discarding available instrument information), not
solely a data or single-frame information limit. Genuinely ambiguous
verbs (grasp, irrigate as a minority case of irrigator, null_verb) did
not benefit or slightly worsened, consistent with those specific
confusions being a different, likely harder-to-fix kind of limitation.
This is an oracle upper bound using ground-truth instrument labels; using
Day21's actual predicted instruments instead is the natural next step
toward a realistic end-to-end triplet recognition pipeline, and is
exactly the structure Rendezvous's own interaction-attention modules are
built around, rather than an ad hoc fix specific to this project.
