# Day35: Retrospective — the Self-Supervised Learning Arc (Day31-34)

## Objective

Day34 closed with an explicit flag: three different self-supervised
pretext tasks had now been tried against two downstream probes, with no
single method winning on both. Following the same discipline as Day30
(the retrospective that closed the symbolic and pixel-recognition arcs),
this is a deliberate pause before deciding what comes next -- not a
re-summary of each day, but a synthesis of what actually generalized
across the four days, and an honest accounting of what's still
unresolved.

## Four Days, Three Methods, Two Probes

**Day31** established the baseline result: SimCLR-style contrastive
learning (two augmented views of the same frame pulled together, other
frames pushed apart), fine-tuning only `layer4` on 8 unlabeled training
videos, closed 50% of the macro-F1 gap between frozen ImageNet features
(instrument F1 0.302) and full supervised fine-tuning (Day27, F1 0.512),
landing at F1 0.407. The recovery was concentrated in instruments Day26/27
had already diagnosed as feature-quality-limited (bipolar, clipper,
irrigator: ~60% of their individual gaps closed) and nearly absent where
frozen features were already strong (grasper: 4%).

**Day32** checked what that representation actually captured, reusing
Day16-19's linear-probe-and-PCA method on a real visual backbone for the
first time. Phase structure (never a training target, and the pretext
task had no temporal signal at all) improved modestly under contrastive
adaptation (0.511 -> 0.532) -- far less than instrument's gain -- and a
PCA plot showed a specific, interpretable change: a sharp cluster
appeared for `clipping-and-cutting`, the phase most tied to the clipper
instrument, exactly the instrument that improved most in Day31.

**Day33** tested Day31's own named limitation directly: doubling the
contrastive batch size (32 -> 64 images, 64 -> 128 views), with learning
rate scaled to match. Macro F1 was unchanged (0.407 -> 0.406) --
batch size, the most obvious hardware-constrained caveat, was not
actually the binding constraint within the range achievable on 8GB RAM.

**Day34** tried a structurally different pretext task: predict which of
two same-video frames comes later (a single "progress head" trained
with a pairwise ranking loss), using temporal position instead of
appearance similarity as the training signal. This produced the best
instrument result of the arc (F1 0.432, ~62% of the gap to supervised
fine-tuning closed) but a *worse* phase-probe result than even
unadapted ImageNet features (0.460 vs. 0.511) -- directly contradicting
the hypothesis that motivated the day. A follow-up diagnostic did not
cleanly explain why.

**Summary table, all four days, same 10 videos / video-level 8-2 split:**

| Backbone | Instrument macro F1 | Phase-probe accuracy |
|---|---:|---:|
| ImageNet frozen (Day21) | 0.302 | 0.511 |
| Contrastive, N=32/64 views (Day31) | 0.407 | 0.532 |
| Contrastive, N=64/128 views (Day33) | 0.406 | -- (not re-tested) |
| Temporal-order (Day34) | **0.432** | 0.460 |
| Supervised fine-tuning (Day27) | 0.512 | -- (not tested) |

## Cross-Cutting Lessons

**1. No pretext task is free of an inductive bias, and that bias doesn't
automatically align with every downstream task.** Contrastive learning
(appearance-only) helped both probes modestly; temporal-order (position-
only) helped instrument the most of any SSL method but actively hurt
phase. Each pretext task teaches the backbone to be sensitive to
whatever distinguishes its own training examples -- augmented-view
identity for contrastive learning, within-video chronology for
temporal-order -- and neither of those is the same thing as "everything
a human labeler would call meaningful." There is no free lunch: picking
a pretext task is already a bet on which structure in the data matters,
and Day34 shows that bet can pay off on one axis while losing on
another, simultaneously.

**2. A named limitation, tested directly, can turn out not to be the
real one (again).** Day33 continued the discipline Day26/27 established
for the supervised arc (test each proposed fix in isolation rather than
assume it matters): doubling batch size, the single most commonly-cited
SimCLR caveat, changed nothing. This is now the second time in this
project a plausible, literature-endorsed explanation was checked and
found wanting (the first being Day24-vs-25's disentangling of "needs
more data" from "needs temporal context" for verb recognition) -- a
reminder that even well-founded intuitions about what should matter are
worth verifying against this specific data and scale before acting on
them.

**3. Independent probes agreeing is stronger evidence than either alone
-- and independent probes disagreeing is just as informative.** Day31's
instrument-F1 breakdown and Day32's phase-PCA both pointed at the same
mechanism (clipper's feature separation improving), which was
reassuring precisely because the two analyses had no way to influence
each other. Day34's instrument-vs-phase split result is the mirror
image: two probes genuinely disagreeing about whether the same backbone
is "better," which is real information about the method (it has
uneven, task-specific effects) rather than a discrepancy to explain
away in favor of a single headline number.

**4. A 2D PCA plot's apparent structure and a linear probe's actual
accuracy can diverge, and only the probe answers the question that
matters.** This was Day34's most transferable finding: coloring a PCA
projection by phase showed a visually organized gradient even though
the quantitative phase probe (trained on all 512 dimensions) scored
below baseline comparisons. PCA shows the directions of largest
*variance*, not the directions most useful for a specific
classification -- the two can coincide (as they mostly seemed to in
Day17-19's symbolic-sequence probes) or diverge (as here), and this
project had not previously needed to distinguish the two explicitly.

## What's Still Open

- **No SSL method closed the full gap to supervised fine-tuning.** The
  best result (Day34, F1 0.432) still leaves roughly 38% of the Day21-
  Day27 gap unclosed, and none of the four days approached Day27's 0.512.
- **Day34's phase regression has no confirmed explanation.** The leading
  candidate (within-video overfitting) was not cleanly supported by the
  video-colored PCA; a more rigorous test (e.g., a probe trained on one
  set of videos and tested on a *third* held-out set, or explicitly
  checking which feature dimensions the progress head relies on) was not
  attempted.
- **Verb and target recognition were never evaluated under any SSL
  variant** -- everything in this arc was tested on instrument (all four
  days) and phase (Day32, Day34) only.
- **No method combined the two pretext signals** (appearance contrastive
  + temporal order) into a single objective, which the literature this
  project has been informally tracking (and the "no free lunch" lesson
  above) would suggest is a natural next thing to try, since the two
  showed complementary, non-overlapping strengths.
- **Batch size was only tested up to 128 views** -- Day33's negative
  result rules out that specific doubling mattering, not that batch size
  is irrelevant at the scale (thousands of views) published SimCLR
  results actually use, which remains outside this hardware's reach
  regardless.

## Reflection

Read together, these four days argue against expecting a single "winning"
self-supervised recipe and for treating each pretext task as answering a
narrower, more specific question than it first appears to. "Does
self-supervised learning help" turned out to be the wrong-shaped
question; the useful ones were narrower and each had a clean, if
sometimes surprising, answer: does *this* pretext task's specific signal
transfer to *this* specific downstream task, and does it degrade
gracefully or catastrophically for tasks it wasn't shaped around. That
reframing -- from "SSL: yes or no" to "which inductive bias, for which
downstream question" -- is the actual generalizable takeaway of this
arc, more than any single F1 number.

## Conclusion

Across three self-supervised pretext tasks (appearance contrastive at
two batch sizes, temporal order) and two downstream probes (instrument,
phase), the self-supervised arc found: real, task-specific value in
label-free adaptation (up to 62% of the supervised fine-tuning gap
closed on instrument recognition); a clean negative result for the most
commonly-cited limitation (batch size); and a genuine, unresolved
surprise (temporal-order pretraining actively hurting phase
recognition) that is being reported as open rather than forced into a
tidy explanation. This closes the arc's first phase. Extending to verb/
target, combining pretext signals, or moving to a different topic
entirely are all reasonable next directions, and the choice is deferred
to the next session rather than decided here.
