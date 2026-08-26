# Day40: Retrospective — Closing the CholecT50 Series (Day01-39)

## Objective

Day30 closed the symbolic-modeling and pixel-recognition arcs (Day01-29).
Day35 closed the first phase of the self-supervised learning arc
(Day31-34). Day36-39 extended SSL evaluation to verb/target, then
returned to verb recognition's unresolved architecture questions three
more times (Day37-39). This is the project-level retrospective that
closes the whole thing: not a re-summary of Day30 or Day35 (both stand
as written), but the synthesis that only becomes visible by lining up
numbers *across* all four arcs, plus an honest account of what's still
unsolved as this series ends and the project moves to a new dataset
(JIGSAWS).

## Four Arcs, One Dataset

**Day01-19: symbolic sequence modeling.** Working from CholecT50's
triplet/phase annotations directly (no images), this arc built Markov
chains, embeddings, an RNN, attention, and a from-scratch Transformer
in numpy, on the philosophy that implementing a mechanism yourself is
how it stops being a black box. See Day30 for the arc's own
retrospective.

**Day20-29: pixel-based recognition, supervised.** Moving to the actual
CholecT50/Rendezvous task (raw endoscopic frames), this arc built up
instrument recognition (Day21: frozen ResNet18, F1 0.302), diagnosed
verb recognition's split cause -- information limit vs. architecture
gap (Day22) -- confirmed the architecture-gap half with an oracle test
(Day23: F1 0.192 -> 0.388), then closed the loop with realistic
instrument conditioning (Day24, Day28) and two orthogonal instrument
fixes (Day26: class-weighted loss, F1 0.378; Day27: fine-tuned
backbone, F1 0.512). Day29 applied the best recipe to target
recognition (F1 0.207) and closed the arc. See Day30 for the full
retrospective.

**Day31-36: self-supervised learning.** Three pretext tasks
(contrastive, contrastive at 2x batch size, temporal-order) evaluated
against instrument and phase probes (Day31-34, retrospective at Day35),
then extended to verb and target (Day36). The unifying finding: SSL
adaptation helped only instrument recognition, the one task whose
bottleneck was genuinely feature quality -- it did nothing for verb
(single-frame information limit) or target (data scarcity), diagnoses
Day22 and Day29 had already made independently before any SSL
evaluation existed.

**Day37-39: verb's architecture question, revisited three ways.** Day23/
28 (Day20-29 arc) had already shown instrument-conditioning helps verb
substantially under an *unweighted* loss (F1 0.192 -> 0.299, Day28).
Day37 re-tested the same idea against the *class-weighted* baseline
established later (Day36: class-weighting alone reaches verb F1 0.309,
already higher than Day28's fully-conditioned 0.299) and found the
oracle ceiling had grown too (F1 0.309 -> 0.484), but a realistic
predicted-instrument signal added nothing (0.305). Day38 tried a much
more accurate instrument predictor (F1 0.512 vs. 0.399) -- still
nothing (0.305 again). Day39 abandoned instrument-conditioning
entirely and tried temporal context instead (a 3-frame, ~2-second
window): verb F1 reached 0.332 -- the first realistic method in this
whole sub-thread to beat the class-weighting-alone baseline.

**The full numeric picture, four tasks, every method tried:**

| Task | Frozen baseline | Best supervised fix | Best SSL fix | Best realistic conditioning/context fix |
|---|---:|---:|---:|---:|
| Instrument | 0.302 (Day21) | **0.512** (Day27, fine-tuned) | 0.432 (Day34, temporal-order) | -- |
| Verb | 0.192 (Day22, unweighted) | 0.309 (Day26-style weighting, per Day36) | 0.304-0.309 (Day36, no effect) | **0.332** (Day39, temporal window) |
| Target | -- | 0.207 (Day29, fine-tuned+weighted) | 0.220 (Day36, temporal-order) | -- |
| Phase | 0.511 (Day21-era frozen, linear probe) | -- (never fine-tuned end-to-end) | **0.532** (Day32, contrastive) | -- |

## Cross-Cutting Lessons

**1. A loss-function fix and an architecture fix are not additive if
they target the same underlying failure mode.** This is the project's
least obvious finding, visible only by comparing Day28 against
Day36-39 side by side: Day28's fully-conditioned, fine-tuned-instrument
verb model (F1 0.299) scores *below* Day36's plain class-weighted
frozen-feature baseline with no conditioning at all (F1 0.309). Both
were trying to fix the same thing (verb's rare-class F1), from two
different angles -- and once class-weighting alone had already
captured most of the fixable "willingness to guess rare classes"
problem, adding instrument-conditioning on top (Day37, Day38) contributed
exactly nothing further, regardless of how accurate the instrument
signal was (F1 0.399 or 0.512, same result: 0.305). Two fixes for the
same bottleneck don't stack; a fix for a genuinely different bottleneck
(Day39's temporal context, addressing single-frame visual ambiguity
rather than class rarity) did add something on top.

**2. Diagnose before you fix, and check the diagnosis survives contact
with every later method.** Day22 split verb's difficulty into an
information limit and an architecture gap using nothing but co-
occurrence statistics, before any model beyond a frozen linear probe
existed. That diagnosis held up through Day23/24/28's conditioning
experiments, Day31-36's SSL evaluation (which affected neither cause,
as predicted), and Day37-39's more elaborate architecture tests (which
confirmed the architecture-gap half is real but practically unreachable,
and left the information-limit half as the one place a cheap fix
(temporal context) actually worked). A diagnosis made on Day22 was
still being tested, and still holding, seventeen days later.

**3. A named limitation, tested directly, is often not the real one.**
This recurred across every arc: Day14 ruled out a chronological-split
confound before it could contaminate later results; Day24-vs-25
disentangled "needs more data" from "needs temporal context" for verb;
Day33 tested SSL's most commonly-cited caveat (batch size) and found it
irrelevant; Day38 tested "the predictor is too noisy" and found it
false, even after a 30% relative improvement in predictor accuracy.
Each of these was a plausible, well-reasoned hypothesis that would have
been reasonable to just believe -- and each needed an actual experiment
to rule out or confirm.

**4. Fixes don't land where intended, and that's still useful
information.** Day25's temporal GRU was built to fix grasp-vs-retract
ambiguity; it left grasp flat and instead fixed clip, via a mechanism
(motion signature) nobody had targeted. Day39's temporal window was
motivated by an occlusion-tracking hypothesis for instrument
recognition; it barely moved instrument and instead fixed verb's
transient-action false positives. In both cases the *intervention*
was more broadly useful than the *hypothesis* that motivated it turned
out to be correct -- worth remembering when an experiment "succeeds"
for an unexpected reason.

**5. Explicit, cheap signals can beat expensive implicit ones.** Day39's
untrained 3-frame concatenation (verb F1 0.332) outperformed every one
of Day36's SSL-pretrained backbones (0.304-0.309), each the product of
a 60-100+ minute self-supervised pretraining run. Self-supervised
learning is trying to discover useful structure from unlabeled data
implicitly; here, structure (temporal adjacency) that was already
sitting in the dataset, free, and could be used directly without any
learning at all, did more for this specific task than the learned
representation did. This doesn't generalize to "SSL is a waste of
time" (it clearly helped instrument, per Day31/34), but it's a concrete
reminder to check whether a problem has a free, explicit solution
before reaching for a learned one.

## What's Still Open

- **Verb's realistic ceiling (F1 0.332) is still well below its oracle
  ceiling (0.484).** Temporal context and instrument-conditioning were
  never combined; it's unknown whether they'd stack (a genuine test of
  lesson #1 above, in the direction that might actually help).
- **Phase recognition was never fine-tuned end-to-end**, only ever
  evaluated via frozen or SSL-adapted linear probes (best: 0.532,
  Day32). Day27's instrument fine-tuning result (0.512) suggests phase
  might have similar unrealized headroom.
- **Day34's phase regression under temporal-order pretraining (0.511 ->
  0.460) was never explained**, and remains the project's one
  unresolved anomaly.
- **Scissors never got above F1 0.101** (Day27) under any intervention
  tried (class-weighting, fine-tuning) -- the clearest case in the
  project of a problem that looks like it needs more data, specifically,
  rather than a better technique on the same data.
- **Target recognition (F1 0.220 best) received the least attention of
  the four tasks** -- one diagnostic day (Day29) and one SSL evaluation
  (Day36), versus ten-plus days each for instrument and verb.
- **The occlusion hypothesis from Day38 was set aside, not confirmed
  false in general** -- Day39 refuted it as the explanation for the
  verb/instrument temporal-context asymmetry specifically, but CholecT50
  has no occlusion annotations to test it directly for any purpose.
- **No pretext task combined appearance and temporal signals**, despite
  Day35 flagging this as a natural next step -- superseded in practice
  by Day39's finding that explicit temporal context alone already beat
  every SSL variant on verb.

## Clinical Implications

Everything above evaluates this project on its own terms: macro F1
across instrument/verb/target/phase classes, treating every class and
every confusion as equally important. That metric is defensible for
what this project has actually been -- practice in building and
diagnosing representation-learning pipelines, in service of eventually
understanding surgical foundation models -- but it is not the same
question as "is this useful for real intraoperative decision support,"
and the two should not be conflated when reading the numbers above.

**Most of the errors this project spent the most effort on are
clinically close to irrelevant.** A single frame that can't be
confidently assigned to one instrument is not a real problem if the
frames around it can -- correct instrument identity is recoverable from
trajectory/continuity across a few seconds, which is a much easier
problem than single-frame classification and was never what any day in
this project actually modeled end-to-end. Similarly, confusing grasp
and retract is closer to an annotation/interpretation ambiguity than a
meaningful error -- nothing dangerous follows from a system reporting
one instead of the other. Target confusion (gallbladder vs. liver, for
instance) is much the same: for most practical purposes, which
anatomical label is attached matters far less than what is being done
to it.

**What would actually matter is asymmetric, and this project's metric
doesn't reflect that asymmetry at all.** Getting `clip` or `cut` wrong
at the moment it happens is a fundamentally different kind of error
than getting `grasp` vs. `retract` wrong, because those are the
moments where the action is close to irreversible and near
safety-critical structures -- yet macro F1 weighs a `clip`/`cut` error
exactly the same as a `grasp`/`retract` error. This project never built
or evaluated anything that reflects that difference in stakes.

**The more clinically useful shape of system is closer to a driver-
assistance system than a labeler.** Not "what instrument, what verb,
what target, frame by frame" but "is what's happening right now
dangerous," "what should happen next," "is a critical structure nearby
that needs caution" -- the surgical equivalent of forward-collision
warnings, lane-departure alerts, or a sign the driver might have missed.
That is a fundamentally different modeling target (anomaly/risk
detection and situational awareness) from anything this series built,
and none of the fixes explored here (class-weighting, fine-tuning, SSL,
instrument-conditioning, temporal context) were evaluated against it.

This doesn't invalidate the series -- fine-grained, per-frame accuracy
is a reasonable target if the goal is a general-purpose foundation-model
representation that some future safety-relevant task gets built on top
of, which has been this project's actual framing since the 2026-08-12
goal reset. But it means none of the F1 numbers in the table above
should be read as evidence of clinical readiness, or even progress
toward it, without that translation step -- and building that
translation (a risk/next-step/hazard-aware task, evaluated with a
stakes-weighted rather than class-uniform metric) is not something this
series attempted.

## Reflection

On 2026-08-12, discussing what should count as this project's actual
goal, the owner and Claude Code agreed to move away from "faithfully
reproduce a research history" (infeasible on 8GB RAM for most papers)
toward "each arc asks a clear question and gives an honest answer,
whether or not that answer is flattering." Day37-39 is the clearest
example of that principle actually operating: three consecutive days
where the starting hypothesis (verb needs instrument conditioning;
predictor accuracy is the bottleneck; occlusion explains the pattern)
was tested and found wanting each time, and each negative result
sharpened the next question rather than being treated as a failure to
paper over. The project's most useful finding (lesson #1 above) only
exists because Day28's result from three weeks earlier was kept around
and compared against, rather than each day being evaluated in isolation.

Closing a 40-day series on one dataset is also a good moment to name
what this project was never going to be: a leaderboard exercise, a
faithful reproduction of Rendezvous or any single paper, or a search
for one clean architecture that fixes everything at once. What it
consistently was: a sequence of small, falsifiable questions, most of
which came from something diagnosed several days earlier, answered
with the smallest experiment that could actually answer them. That
pattern is the thing to carry into JIGSAWS, more than any specific
number in the table above.

## Conclusion

Forty days on CholecT50 covered symbolic sequence modeling from
scratch, supervised pixel recognition across all three parts of the
instrument-verb-target triplet plus phase, three self-supervised
pretext tasks, and a focused three-day investigation into why verb
recognition resists every fix tried against it. The project's sharpest
finding emerged only at this closing distance: a loss-function fix
(class-weighting) and an architecture fix (instrument-conditioning)
turned out to target the same underlying failure mode and don't stack,
while a fix for a different failure mode (temporal context) did add
real value on top -- a distinction invisible from any single day's
result. Instrument recognition reached its best result (F1 0.512)
through supervised fine-tuning; verb's best realistic result (F1 0.332)
came from free temporal context, not from any labeled-data or
architecture investment; target and phase remain the least-explored
and least-resolved of the four tasks. None of these numbers should be
mistaken for clinical progress, though: the class-uniform metric this
whole series optimized treats a `grasp`/`retract` mix-up the same as a
`clip`/`cut` mix-up, when only the latter is close to the kind of error
that would matter in practice -- real intraoperative usefulness would
look like hazard and next-step awareness, not finer-grained labeling,
and this series never built or evaluated anything against that
different target. This closes the CholecT50 series. The project moves
next to JIGSAWS (robotic bench-top surgical gestures, paired with
synchronized kinematics) -- a deliberate shift toward a
task type (gesture recognition, sequence modeling over a genuinely
different data modality) this series never touched, rather than a new
dataset for the same kind of question.
