# Day22: Verb Recognition from Raw Pixels

## Objective

Day21 recognized instruments (6 classes) from raw frames across 10
videos with a video-level split, and found the pipeline works well for
common instruments (grasper, hook) but struggles with rare ones
(bipolar, scissors, clipper, irrigator). Instrument is the easiest slice
of the instrument-verb-target triplet: instruments are large, visually
distinct metal shapes that look roughly the same regardless of what
they're doing. Today moves to the next slice -- verb (10 classes: grasp,
retract, dissect, coagulate, clip, cut, aspirate, irrigate, pack,
null_verb) -- the action being performed, using the exact same pipeline
(same 10 videos, same video-level 8/2 split, same frozen ResNet18 +
linear head, same evaluation) as Day21, so any difference in results is
attributable to the task itself, not the setup.

`null_verb` (an instrument on screen but not actively doing anything) is
included as one of the 10 classes, matching how CholecT50 defines it --
unlike Day12-19's state-segmentation code, which filtered `null_verb` out
because those days modeled active sub-tasks specifically. Here, the
model is asked to recognize whatever verb category is annotated,
including "idle."

## Method

[`verb_recognition_multi_video.py`](verb_recognition_multi_video.py) is
Day21's script with instrument (6-way) swapped for verb (10-way) --
identical model, training procedure, and video split (`VID05`, `VID12`
held out as test, matching Day21's split exactly since the same fixed
seed and video ID list are used).

## Results

| Verb | Accuracy | F1 | Baseline accuracy | Train prevalence | Test prevalence |
|---|---:|---:|---:|---:|---:|
| grasp | 0.758 | 0.434 | 0.785 | 0.251 | 0.215 |
| retract | 0.681 | 0.692 | 0.477 | 0.512 | 0.477 |
| dissect | 0.753 | 0.652 | 0.628 | 0.464 | 0.372 |
| coagulate | 0.947 | 0.052 | 0.951 | 0.057 | 0.049 |
| clip | 0.955 | 0.000 | 0.956 | 0.029 | 0.044 |
| cut | 0.961 | 0.000 | 0.981 | 0.028 | 0.019 |
| aspirate | 0.963 | 0.045 | 0.974 | 0.036 | 0.026 |
| irrigate | 0.998 | 0.000 | 0.998 | 0.005 | 0.002 |
| pack | 0.998 | 0.000 | 0.999 | 0.005 | 0.001 |
| null_verb | 0.867 | 0.042 | 0.883 | 0.123 | 0.117 |
| **Macro** | **0.888** | **0.192** | 0.863 | -- | -- |

Macro F1 (0.192) is markedly lower than Day21's instrument-recognition
macro F1 (0.302), on the same videos, same split, same model.

## Interpretation

**Only two verbs show a clean, real win: retract and dissect.** Both are
common (51.2% and 46.4% of training frames) and both clear their
baseline on accuracy *and* F1 (retract: 0.681/0.692 vs. baseline 0.477;
dissect: 0.753/0.652 vs. baseline 0.628) -- comparable to Day21's grasper
and hook.

**Grasp vs. retract is likely closer to a genuine ambiguity than a model
failure.** Its raw accuracy (0.758) is *below* the trivial baseline
(0.785, from always predicting "absent"), yet its F1 (0.434) shows the
model is doing real, if imperfect, work. Checking the instrument-verb
co-occurrence directly across all 10 videos: when a grasper is in use,
the annotated verb is `retract` 63.0% of the time, `grasp` 29.9%, and
`null_verb` 6.5%. Grasping and retracting tissue with the same
instrument can look nearly identical in a single still frame -- both are
"grasper touching/holding tissue" -- and likely differ mainly in force,
direction, or intent, which a static image may not capture and which
different human annotators might not even agree on consistently. A
model that leans toward predicting the majority action (`retract`) for
"grasper is doing something" frames, at the expense of the minority
action (`grasp`), is behaving reasonably given the data, not
malfunctioning -- this may be close to an irreducible ceiling for
single-frame verb recognition on this specific pair, not a limitation
more data alone would fix.

**In contrast, the tool-specific verbs (clip, cut, aspirate, irrigate)
are not inherently ambiguous, and their low F1 looks more like an
architecture gap than a data or information ceiling.** Checking the same
instrument-verb co-occurrence: clipper -> `clip` 94.9% of the time,
scissors -> `cut` 91.0%, bipolar -> `coagulate` 78.7%, hook -> `dissect`
86.6% -- all strongly concentrated on one verb. If a classifier already
knew which instrument was present (Day21's task), most of these verbs
would follow almost for free, with no need to separately learn subtle
action-specific visual cues from very few positive examples. But today's
verb classifier is a completely independent linear head trained directly
on the same generic image features, with no connection to instrument
identity at all -- it has to rediscover, from scratch and from a handful
of rare positive examples, a signal that is largely redundant with
"which instrument is this." That is a much harder learning problem than
it needs to be, and a natural, testable next step is to condition verb
prediction on detected (or true) instrument identity directly, and check
whether F1 for these classes jumps once that near-deterministic
structure is made available to the model rather than left for it to
re-derive unaided. (Irrigator is the one partial exception: aspirate is
its dominant verb at only 66.3%, with irrigate, retract, and null_verb
all making up meaningful shares -- so irrigator's own actions are
genuinely more varied, and conditioning on instrument alone would help
less there than for clipper or scissors.)

**A separate, genuinely temporal limitation still applies across the
board: verb is a fundamentally harder visual category than instrument,
because an action is inherently a temporal concept and this model only
ever sees a single, still frame.** This is exactly the reasoning that
motivated Day12's original move from raw per-frame triplets to
similarity-based state segments (S) in the symbolic pipeline -- frame-by-
frame triplet labels are noisy and context-dependent in a way that a
short, stable window of frames is not. An instrument's identity is
mostly a static property (its shape, size, metallic appearance don't
depend on what it's doing), so a frozen ImageNet backbone's generic
shape/texture features can plausibly capture it well; whether a grasper
is currently grasping, retracting, or idle depends on force, direction,
and intent none of which a single static frame contains -- so this
limitation specifically applies to the ambiguous, same-instrument verb
groups (grasp/retract/null_verb) discussed above, not to the
tool-specific verbs, which look more like an easier problem than this
pipeline is currently exploiting.

So the weak macro F1 (0.192) is really two different problems wearing
the same symptom. For grasp/retract/null_verb, more data likely helps
only a little -- the images themselves may not contain enough
information to separate them, and even the ground-truth labels may
embed some annotator-level judgment calls. For clip/cut/aspirate/
irrigate, the bottleneck looks architectural rather than fundamental:
these verbs are close to deterministic given instrument identity, so a
model that used Day21's instrument signal directly, instead of
re-deriving similar information from a few hundred raw pixels' worth of
rare examples, should do considerably better without needing more data
at all.

## Reflection

Comparing Day21 and Day22 side by side, with everything else held fixed,
isolates a real, task-level difference: recognizing *what tool this is*
generalizes better than recognizing *what it's doing* from the same
frozen visual features and the same amount of data. But splitting the
verb results further shows that difference isn't uniform across verbs --
some of it is a genuine, likely irreducible information limit (grasp vs.
retract vs. null_verb, indistinguishable in a still frame), and some of
it is simply that this pipeline doesn't yet use information it already
has (Day21's instrument signal) to make an easier problem easier. That
second part is a useful, concrete finding for thinking about how
Rendezvous-style triplet recognition is actually structured: instrument
recognition isn't just the visually easiest sub-problem to build first,
it's a signal the verb and target predictions are *meant* to be
conditioned on in the full triplet formulation -- treating verb
recognition as fully independent of instrument, as today's pipeline
does, throws away exactly the structure that would make tool-specific
verbs easy.

It also reinforces, for the third time now (Day20, Day21, Day22), that
raw accuracy under class imbalance is close to meaningless without a
baseline and an F1 (or similar) alongside it -- grasp's result would have
looked like a *regression* from Day21 by accuracy alone, when what
actually happened is the model found a real, moderately useful signal
that a coarser metric penalizes.

## Conclusion

Verb recognition (macro F1 0.192) is harder than instrument recognition
(Day21's macro F1 0.302) under an identical pipeline, video split, and
amount of data, but not for one single reason. Only the two common,
visually distinctive verbs (retract, dissect) show a clear real win.
Grasp vs. retract vs. null_verb looks close to a genuine, likely
irreducible information limit -- indistinguishable in a still frame, and
possibly not fully agreed-upon even by human annotators -- which more
data is unlikely to fix. Clip, cut, aspirate, and irrigate are a
different story: instrument-verb co-occurrence checked directly across
all 10 videos shows these are 79-95% determined by instrument identity
alone, so their weak F1 here looks like an architecture gap (an
independent classifier re-deriving redundant information from a handful
of rare examples) rather than a hard ceiling. The natural next step this
suggests is conditioning verb prediction on instrument identity directly
-- much closer to how Rendezvous's actual interaction-attention modules
are structured -- to see whether that alone recovers most of the gap for
tool-specific verbs, before reaching for more data or temporal context
across frames (which would remain the right next step for the
genuinely ambiguous verbs).
