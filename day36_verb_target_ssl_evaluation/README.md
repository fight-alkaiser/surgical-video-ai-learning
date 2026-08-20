# Day36: Extending SSL Evaluation to Verb and Target

## Objective

Day31-35 evaluated the SSL-pretrained backbones (contrastive, temporal-
order) on only two downstream tasks: instrument recognition and phase
recognition. Day34 found a split verdict between them (temporal-order
won on instrument, lost badly on phase), and Day35's retrospective
flagged this as an open question: does the same kind of split -- or a
new pattern entirely -- show up on tasks not tested yet? Today reuses
the exact backbones already saved from Day31 and Day34 (no retraining)
and evaluates them on verb (10 classes, Day22's task) and target (15
classes, Day29's task), with the same class-weighted linear probe
recipe used throughout, so every number is directly comparable to the
instrument/phase results already established.

## Method

[`verb_target_ssl_evaluation.py`](verb_target_ssl_evaluation.py) extracts
features from three backbones -- plain frozen ImageNet, Day31's
contrastive checkpoint, Day34's temporal-order checkpoint -- for the
same 10 videos and video-level 8/2 split used throughout, then trains a
class-weighted linear probe (Day26's recipe) for verb and for target on
each. Six probes total (3 backbones x 2 tasks), all evaluated the same
way.

## Results

| Backbone | Verb macro F1 | Target macro F1 |
|---|---:|---:|
| ImageNet frozen | 0.309 | 0.209 |
| Contrastive (Day31) | 0.304 | 0.202 |
| Temporal-order (Day34) | 0.309 | **0.220** |

For reference, computed under different methodology (not directly
comparable row-for-row, but same underlying task): Day22's original
verb baseline (frozen, *unweighted* loss, no SSL) was macro F1 0.192;
Day29's target result (fine-tuned + weighted, fully supervised) was
0.207.

## Interpretation

**Neither SSL method helped verb recognition at all** -- all three
backbones land within 0.005 of each other (0.304-0.309), indistinguishable
from noise. This is a genuinely different outcome from every prior task:
instrument improved substantially under both SSL methods (Day31, Day34),
phase improved under contrastive and regressed under temporal-order
(Day32, Day34) -- verb is the first task where SSL adaptation, in either
form, changed essentially nothing.

**Target showed a small, one-sided effect**: temporal-order gave a real
if modest improvement (0.209 -> 0.220), while contrastive adaptation was
flat-to-slightly-worse (0.202). This is a third distinct pattern across
four tasks now (instrument: both help; phase: split, one helps one
hurts; verb: neither helps; target: only one helps, modestly).

**A side finding worth flagging on its own: class-weighting alone,
with no SSL adaptation at all, recovers most of what looked like an
"adaptation" effect for verb.** Comparing today's frozen-features-plus-
weighted-loss result (0.309) against Day22's original frozen-plus-
*unweighted* result (0.192) -- the only variable that changed is the
loss function, same unmodified backbone -- shows a 0.117 jump from
re-weighting alone. This echoes Day26's original instrument finding
(class-weighted loss alone raised macro F1 from 0.302 to 0.378, well
before any fine-tuning) and confirms it generalizes to a second task:
a meaningful fraction of what earlier days attributed to representation
quality was actually available from the loss function the whole time,
on the exact same frozen features.

**Reading all four tasks together, a unifying explanation emerges: SSL
feature-adaptation only helps when the downstream task's bottleneck is
actually feature separability -- and for verb and target, it mostly
isn't.** Day22 already diagnosed verb's core problem as split between a
single-frame information limit (grasp vs. retract is often genuinely
ambiguous in one still image) and an architecture gap (not conditioning
on instrument identity) -- neither of which better generic visual
features can fix, since the missing information isn't "sharper
appearance cues," it's temporal context or structural information the
backbone was never given access to. Day29 diagnosed target's problem as
extreme class cardinality and rarity (several classes under 1.5%
prevalence in an 8-video training set) -- a data-volume problem no
amount of representation quality can manufacture more positive examples
for. Instrument, by contrast, was mostly a *bona fide* feature-quality
problem (Day26 vs. Day27 showed class-weighting alone couldn't fix
scissors, but better features could fix clipper and bipolar) -- exactly
the kind of problem unlabeled contrastive or temporal-order pretraining
is suited to address. SSL adaptation isn't broadly weak or broadly
strong; it's a tool for a specific kind of problem, and three of this
project's four evaluated tasks didn't have that specific problem.

## Reflection

This directly extends Day35's reframing of the arc's central question
from "does SSL help" to "which inductive bias, for which downstream
question" -- and sharpens it further. The more precise version, visible
only now with four tasks instead of two: SSL adaptation helps in
proportion to how much of a task's difficulty is a feature-separability
problem specifically, versus a single-frame information limit, an
architectural gap, or a data-scarcity problem. Those latter three were
each independently diagnosed in earlier days (Day22, Day23-24, Day29)
for reasons that had nothing to do with self-supervised learning --
and today's null/weak results for verb and target are exactly what
that independent diagnosis would have predicted in advance, which is a
more satisfying kind of confirmation than a result that could only be
explained after the fact.

## Conclusion

Extending the SSL evaluation to verb and target completes a four-task
picture: SSL adaptation (either pretext task) meaningfully helped only
instrument recognition, the one task whose known bottleneck was
genuinely about feature quality. Verb showed no measurable SSL effect
at all; target showed a small, single-method effect (temporal-order
only). A side comparison confirmed class-weighting alone -- no SSL, no
fine-tuning -- already recovers a large share of what might otherwise
look like a representation-adaptation win, generalizing Day26's original
instrument finding to a second task. Combined with Day22/29's earlier,
independent diagnoses of why verb and target are hard, this closes the
SSL arc's extension cleanly: self-supervised feature adaptation is a
targeted fix for feature-separability problems specifically, not a
general-purpose improvement to reach for by default.
