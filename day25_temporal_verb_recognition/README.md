# Day25: Temporal Verb Recognition

## Objective

Day22 hypothesized that grasp/retract/null_verb -- all "grasper touching
tissue," visually near-identical in a single frame -- might be a genuine
single-frame information limit rather than a data problem. Day23's
oracle instrument-conditioning experiment supported this: even knowing
the true instrument barely moved grasp's F1 (0.434 -> 0.467). If the
missing information is "what happens over time" rather than "which
instrument," a model given several consecutive frames should do better
specifically on this ambiguous cluster. Today tests that directly: a
small GRU processes the past 8 frames' cached ResNet18 features (about 8
seconds of context, since CholecT50 frames are extracted at 1fps) and
predicts verb for the current frame from its final hidden state --
Day17's from-scratch RNN mechanism, now implemented in PyTorch and
applied to real visual features instead of symbolic triplet-states.

## Method

[`temporal_verb_recognition.py`](temporal_verb_recognition.py) extracts
and caches frozen ResNet18 features once for every frame of all 10
videos (~160s total), then builds (8-frame sequence -> verb label) pairs
for every frame with enough preceding history. A single-layer GRU
(hidden size 64) consumes each sequence; its final hidden state feeds a
linear head predicting all 10 verbs. Same video-level 8/2 split, same
verb label extraction, same evaluation as Day22, so any difference is
attributable to adding temporal context alone -- no instrument
conditioning here, deliberately isolated from Day23/24's question.

## Results

| Verb | Day22 F1 (single frame) | Day25 F1 (8-frame GRU) |
|---|---:|---:|
| grasp | 0.434 | 0.410 |
| retract | 0.692 | 0.774 |
| dissect | 0.652 | 0.700 |
| coagulate | 0.052 | 0.000 |
| clip | 0.000 | 0.352 |
| cut | 0.000 | 0.000 |
| aspirate | 0.045 | 0.022 |
| irrigate | 0.000 | 0.000 |
| pack | 0.000 | 0.000 |
| null_verb | 0.042 | 0.049 |
| **Macro F1** | **0.192** | **0.231** |
| **Macro accuracy** | 0.888 | 0.900 |

## Interpretation

**Temporal context produced a real, if modest, overall improvement**
(macro F1 0.192 -> 0.231), but not primarily by fixing the specific
ambiguity it was designed to test. The intended target, grasp, **did not
improve** (0.434 -> 0.410, essentially flat, arguably worse). Retract and
dissect both improved -- but these are the *majority* actions within
their respective instruments' repertoires (grasper -> retract 63.0%,
hook -> dissect 86.6%), so the model may simply be getting better at
confidently recognizing the common case, not at resolving the genuine
grasp-vs-retract confusion. That distinction matters: if the ambiguity
were actually being resolved, grasp's F1 should rise alongside retract's,
not stay flat while retract's rises.

**The most interesting result is one nobody was looking for: `clip` went
from undetectable (F1 0.000, Day22) to 0.352 -- entirely from temporal
context, with no instrument conditioning at all.** This is a genuinely
different mechanism than Day23/24's instrument-conditioning route to the
same verb. A plausible explanation: the clipping action itself has a
distinctive *motion signature* -- a snapping, closing motion applied to
a duct -- that a sequence of frames can capture even when a single frame
of the (small, easily confused) clipper instrument cannot reliably
identify what instrument is even present. Where Day23/24 tried to fix
`clip` indirectly (recognize the tool, infer the verb from that), Day25
may be catching a directly visible temporal cue for the action itself,
independent of nailing down instrument identity first.

**Coagulate and aspirate got worse** (0.052 -> 0.000, 0.045 -> 0.022).
Both are rare verbs tied to already-hard-to-detect instruments (bipolar,
irrigator). An 8-frame GRU has meaningfully more capacity and a harder
optimization problem than a single linear layer, and with very few
positive training sequences for these classes, that extra capacity looks
more likely to overfit or dilute a weak signal than to extract more
information from it -- a plausible, though unconfirmed, echo of Day21's
original rare-instrument, rare-verb data scarcity problem, now showing
up as a capacity/data mismatch rather than a detection failure per se.

## Reflection

The headline number (macro F1 up 20% relative) would be easy to report
as "temporal context helps, hypothesis confirmed." Breaking it down by
verb shows a more precise, more useful picture: temporal context helped,
but not the verb pair it was aimed at, and it helped a *different* verb
for a *different*, unanticipated reason. This is worth sitting with
rather than smoothing over -- the original hypothesis (grasp vs. retract
needs motion information) may still be correct in principle, but an
8-frame GRU over globally-pooled ResNet features, taking only its final
hidden state, is a fairly coarse way to extract "force and direction"
from a sequence. Global average pooling in particular discards most
spatial information (exactly where in the frame something is happening,
how a specific point is moving) that force/direction judgments plausibly
depend on -- so the negative result for grasp may say more about this
architecture's inability to represent fine motion than about whether
motion information could help in principle.

This also reinforces Day24's lesson about not assuming a fix's benefit
lands where intended: an intervention can produce a real net improvement
while missing its target and hitting something else entirely, and
checking the aggregate number without the per-class breakdown would have
missed both facts here.

## Conclusion

An 8-frame GRU over cached ResNet18 features improves verb recognition
overall (macro F1 0.192 -> 0.231) but does not resolve the grasp-vs-
retract ambiguity it was designed to test (grasp: 0.434 -> 0.410,
unchanged). Instead, most of the gain comes from a different, unexpected
source -- `clip` improving substantially (0.000 -> 0.352) through what is
plausibly a distinctive motion signature, independent of instrument
identity. Two already-weak, rare verbs (coagulate, aspirate) got worse,
consistent with added model capacity being a net cost rather than a
benefit when there is very little data to support it. The original
hypothesis about single-frame information limits for grasp/retract is
neither confirmed nor refuted by this result -- a coarser architecture
(final-hidden-state GRU over globally-pooled features) may simply be the
wrong tool to extract the specific force/direction information that
distinction plausibly requires, which is a separate, open question from
whether such information exists in the frame sequence at all.
