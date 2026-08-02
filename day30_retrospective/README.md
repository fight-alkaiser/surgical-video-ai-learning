# Day30: Retrospective — 29 Days on CholecT50

## Objective

Day29 closed the instrument-verb-target arc that began at Day20, and
with it, this project's two major campaigns on CholecT50 are both done:
a symbolic sequence-modeling arc (Day01-19) and a pixel-based
recognition arc (Day20-29). Before starting Day31's self-supervised
learning arc, this is a deliberate pause to look back across all 29
days -- not to re-summarize each day (every day's own README already
does that), but to pull out what actually generalized across them: the
methodological lessons that showed up more than once, in different
disguises, and the questions that stayed open when each arc ended.

## Two Arcs, One Dataset

**Arc 1 (Day01-19): CholecT50's annotations as given data.** Every day
in this arc started from human-provided triplet/phase labels, never
from pixels. Day01-11 was exploratory (phase timelines, triplet
frequencies, transition triggers, persistence, recurrence, frame
similarity). Day12-15 compressed frames into similarity-based "states"
(S) and modeled state-to-state transitions with a Markov chain, scaling
from one video (Day13) to all 50 (Day14), and comparing macro
(phase-level, 98.2% accuracy -- close to solved by clinical order alone)
against micro (state-level, 34.5% -- a much harder, closer-to-real
signal). Day16-19 asked whether more sophisticated sequence-modeling
machinery -- embedding, RNN, attention, a full Transformer block, all
implemented from scratch in numpy -- could push past that ~35% ceiling.
The RNN reached 40.5% and stayed the best of the four; attention alone
underperformed it, and a full Transformer block matched the RNN's
internal representation quality (phase-linear-probe accuracy) without
beating its raw accuracy.

**Arc 2 (Day20-29): the same dataset's actual task -- recognizing
triplets from raw endoscopic frames.** Day20-21 established (the hard
way) that evaluation protocol matters as much as model choice:
single-video, chronological evaluation looked broken until switched to
video-level splits across 10 videos. Day22-25 took on verb recognition,
found it split into two different failure modes (single-frame ambiguity
vs. an architecture gap from ignoring instrument identity), and tested
fixes for each (instrument conditioning: Day23-24; temporal context:
Day25). Day26-27 returned to instrument recognition's original
rare-class problem and found two complementary fixes (class-weighted
loss, backbone fine-tuning) that combined to more than double macro F1.
Day28 combined the two arcs' lessons (fine-tuned instrument classifier
conditioning verb prediction) and recovered most of the gap to an
earlier oracle ceiling. Day29 applied the same recipe to target
recognition (15 classes) and found it degrades further -- a clean
scale/cardinality effect, not a new failure mode.

**Headline numbers, both arcs:**

| Arc | Task | Best result | Ceiling / limiting factor |
|---|---|---|---|
| 1 | Next-state prediction (Markov, k=1) | 34.5% accuracy | Representation (triplet-state), not memory length |
| 1 | Next-state prediction (RNN, full history) | 40.5% accuracy | Best of 4 mechanisms; Transformer matched its representation quality, not its accuracy |
| 2 | Instrument recognition (10 videos) | Macro F1 0.512 (Day27) | Rare instruments (scissors) still capped by data scarcity |
| 2 | Verb recognition (10 videos) | Macro F1 0.299 (Day28) | Verbs tied to poorly-detected instruments; some verbs are single-frame-ambiguous by nature |
| 2 | Target recognition (10 videos) | Macro F1 0.207 (Day29) | Same fixes, weaker effect -- more classes, more of them extremely rare |

## Cross-Cutting Lessons

These are the patterns that reappeared across both arcs, in different
form each time -- the actual compounding return on 29 days, more than
any single day's number.

**1. Accuracy is close to meaningless under class imbalance, and this
project re-learned that lesson at every scale.** Day15 found phase-level
accuracy (98.2%) looked impressive but mostly reflected a low-entropy,
clinically-ordered task, not model insight. Day20-29 found it constantly:
a trivial train-majority baseline could match or beat a trained model's
raw accuracy while a per-class F1 breakdown told the real story (Day20's
hollow 1.000 scores for absent classes, Day22's grasp scoring *below*
baseline on accuracy while doing real work by F1, Day26's accuracy
*dropping* by design as a deliberate precision/recall trade-off). The
fix was always the same: never trust a macro number without its
per-class breakdown and a stated baseline.

**2. Evaluation protocol can matter as much as architecture.** Day14
split by video specifically to avoid patient-level leakage for the
symbolic pipeline; Day20 rediscovered why the hard way when a
single-video, chronological split produced a result (three of six
instrument classes with zero test-set positive examples) that couldn't
support any real claim. Day21's fix -- video-level splits, the same
discipline Day14 already had -- mattered more than any model change that
followed it.

**3. A falling loss curve is not proof of useful learning.** Day17's
RNN quietly mode-collapsed to always predicting the single most frequent
state (loss fell smoothly from 5.86 to 3.85; accuracy never moved off
the 12.1% baseline) because too-small weight initialization starved the
recurrent path of gradient. Day19's Transformer block later overfit in
the opposite direction -- falling training loss past epoch 40 while test
accuracy silently peaked and then declined. Both were caught only by
checking the metric that actually mattered, not the loss curve alone.

**4. Distinguish "the mechanism could represent this" from "the
mechanism actually learned this," and check, don't assume.** Day16's
embedding model didn't spontaneously encode phase structure and Day17's
RNN did -- verified not by eyeballing a PCA plot but with a linear probe
(RNN hidden state: 68.4% phase-decodable vs. embedding's much weaker
structure), a technique reused through Day19 and again in Day23's oracle
conditioning experiment (deliberately testing the *ceiling* a mechanism
could reach before testing what it reaches in practice).

**5. "Willingness to guess" and "ability to distinguish" are different
problems needing different fixes.** Day26's class-weighted loss fixed
rare-instrument recall by making the model more willing to guess "clipper
present" -- at a real precision cost, and it didn't help scissors at all,
because scissors' problem wasn't threshold calibration, it was that
frozen ImageNet features didn't separate it from other instruments in
the first place. Day27's backbone fine-tuning fixed *that* -- precision
and recall improved together, not traded off -- for every instrument
except scissors, whose ~500 total instances across the whole dataset
looks like a harder, more fundamental scarcity problem.

**6. Fixes compound, but errors propagate through pipelines too.**
Day23's oracle instrument-conditioning (true labels) more than doubled
verb macro F1; Day24's realistic version (Day21's actual, uneven
instrument predictions) recovered only about a quarter of that gain,
because the conditioning signal was only as reliable as its own weakest
instruments. Day28 showed the fix: improve the upstream signal (Day27's
fine-tuned classifier) and the downstream gain grows substantially (to
~55% of the oracle gap) -- without touching the verb classifier itself.

**7. An intervention's benefit doesn't always land where intended, and
checking only the aggregate number would miss that.** Day25's temporal
GRU was built to resolve grasp-vs-retract ambiguity; grasp barely moved,
but `clip` improved dramatically through what was plausibly a distinct
motion-signature cue, unrelated to the original hypothesis. Day28's
fine-tuned conditioning helped `coagulate` and `clip` sharply but
*regressed* `aspirate` even though the underlying instrument's own
detection had improved. Both are reminders that a positive net result
can hide a missed target and an unplanned side effect at the same time.

**8. Trade-offs should be made visible and chosen deliberately, not
absorbed silently into a "default."** Day26's loss re-weighting is the
clearest case: an unweighted loss isn't a neutral choice, it implicitly
accepts poor rare-class recall by default; naming the trade-off (recall
up, precision and overall accuracy down) turned an invisible default
into a decision that could be made on purpose, informed by what the
numbers would actually be used for.

**9. Honesty about what an experiment can and can't show is itself a
finding.** Day20's single-video result was reported and then explicitly
retracted as unable to support a claim, rather than spun. Day23 was
labeled an oracle test from the start, with its realistic version
deferred to Day24 rather than conflated with it. Day29 was deliberately
scoped as one lighter day rather than a new arc once the pattern from
Day26-27 made a full repeat unnecessary. None of these were failures --
they were the project consistently choosing not to overstate what a
result meant.

## What's Still Open

- **Scissors (instrument) and most target classes remain data-limited**,
  not technique-limited -- Day21's first-named fix, "more videos," was
  never directly tested at scale (only 10 of 50 videos were extracted
  locally).
- **No true end-to-end triplet model was built.** Day23-24/28
  conditioned verb on instrument one direction at a time; Rendezvous's
  actual interaction-attention modules (and target's own conditioning on
  instrument, never tried here) go further than this project reached.
- **Day25's negative result for grasp/retract was not followed up.** A
  GRU over globally-pooled features may simply be the wrong tool to
  extract fine force/direction information; whether richer temporal
  architectures (attention over a window, spatial feature maps instead
  of pooled vectors) would do better is untested.
- **Arc 1's Transformer (Day19) was a single block.** Multi-block
  stacking, the setting where Transformers are usually claimed to
  matter most, was never tried on the symbolic data.

## Reflection

Read end to end, the two arcs are more similar than they first appear.
Arc 1 spent four days (Day16-19) building increasingly sophisticated
machinery to ask "can a smarter mechanism unlock predictive power a
simple count table couldn't reach," and found the answer was mostly no
-- the ceiling was in what the data could express, not in the model.
Arc 2 spent ten days (Day20-29) building increasingly sophisticated
fixes to ask "can better technique unlock detection accuracy a simple
classifier couldn't reach," and found the answer was a qualified yes,
but bounded the same way -- by how much data existed per class, not by
which technique was applied to it. Neither arc's central lesson was
really about Markov chains or ResNets specifically; both were about the
same, more general fact: model sophistication is not a substitute for
what the data itself does or doesn't contain, and figuring out which of
those two is the actual bottleneck, before reaching for a bigger model,
was worth almost every day it took.

## Conclusion

29 days on one dataset produced two closed arcs and a small set of
lessons that outlasted both of them -- about evaluation design, about
what a falling loss curve does and doesn't prove, about the difference
between a model that's unwilling to guess and one that can't tell
classes apart, and about checking an intervention's actual effect
rather than assuming it landed where intended. Day31 moves to
self-supervised learning -- a genuinely new paradigm for this project,
not yet tested here -- but on the same underlying data and hardware
constraints, where these same habits (check the per-class breakdown,
verify the mechanism actually learned what it could represent, name the
trade-off before accepting it) should transfer directly, even though
the technique will be new.
