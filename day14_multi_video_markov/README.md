# Day14: Multi-Video State Vocabulary and Markov Prediction

## Objective

Day13 built a Markov transition model from a single video (VID01). That
model's transition "probabilities" were often just 1/1 or 2/3 — not real
statistics, just a coincidence of a single sample. Today we scale the same
idea up to all **50 videos** in CholecT50, build one shared vocabulary of
states, split the videos into train/test sets, and measure how well a
Markov model trained on some patients predicts the next state in patients
it has never seen.

This is the project's first proper **train/test evaluation**.

---

## Theory: train/test split and generalization

A model that is only ever checked against the data it was built from can't
tell you anything about how well it works on new cases — it might just be
memorizing quirks of that one sample. The standard fix is to **split the
data**: fit (train) the model on one subset, and measure its performance
on a separate, held-out subset (test) it never saw during fitting.

Here, the "model" is nothing more than: *for each state, what is the most
common next state seen during training?* Evaluating it the same way any ML
model is evaluated — accuracy on held-out data — is what turns Day13's
descriptive exercise into an actual (if very simple) predictive system.

This train/test idea does not go away later: it is exactly how RNNs and
Transformers are evaluated too, just with a far richer model in place of
"most common next state."

---

## Why this matters

* Day13 flagged that with only one video, transition probabilities can't
  really be trusted — most states were seen only a handful of times.
  Pooling 50 videos gives the transition counts real statistical weight.
* Building **one shared vocabulary** across all videos (instead of a
  separate vocabulary per video, as Day13 implicitly had) is the concrete
  "tokenizer" this project has been heading toward since Day12 — every
  video becomes a sequence of integers drawn from the same fixed alphabet
  of states, which is exactly the input format a sequence model expects.
* Comparing against a **baseline** (always predict the single most common
  next state, ignoring the current state entirely) is standard ML
  practice: a model is only useful if it beats doing something trivial.

---

## Completed image (what the output looks like)

1. Every video's frames are compressed into states exactly as in Day12/13.
2. All 50 videos' states are merged into one shared vocabulary
   (`state_vocabulary.json`), each with a stable integer ID.
3. Videos are split into training videos and test videos.
4. A transition model is built only from training videos.
5. The model's next-state predictions are checked against the test
   videos' real next states, and compared to a naive baseline.

---

## Method

* Reused the Day12/13 frame-triplet and Jaccard-based state segmentation
  logic, wrapped in functions (`build_frame_triplets`,
  `segment_into_states`, `load_state_sequence`) so it can run once per
  video instead of being copy-pasted 50 times.
* Built one global `signature -> id` vocabulary by scanning all 50 videos
  in sorted order. Note that similarity is only used within Day12's
  segmentation step (merging consecutive frames of one video into a
  state). Assigning IDs across different videos is an **exact match** on
  the compressed triplet-set signature, not a similarity threshold — two
  states only share an ID if their triplet sets are identical.
* Split the 50 videos 80/20 into 40 train / 10 test videos, using
  `random.seed(42)` so the split is reproducible.
* Counted state transitions **within each training video only**
  (transitions are never counted across a video boundary, since that
  would be a meaningless "jump" between two unrelated operations).
* For each state, the most frequent next state seen in training becomes
  its prediction (`most_likely_next`).
* Evaluated prediction accuracy on the test videos' real transitions, and
  compared against a baseline that always predicts the single most common
  next state overall.
* Exported the full vocabulary to `state_vocabulary.json` so later days
  can reuse it directly instead of recomputing it.

---

## Results

| Metric | Value |
|---|---:|
| Videos found | 50 |
| Train videos | 40 |
| Test videos | 10 |
| Vocabulary size | 358 states |
| Test predictions made | 1423 |
| Markov model accuracy | 0.345 |
| Baseline accuracy | 0.121 |
| Unseen current-state rate | 0.041 (59/1423) |

Test videos: VID06, VID111, VID13, VID23, VID27, VID35, VID40, VID42,
VID48, VID78.

---

## Interpretation

The Markov model is correct about **34.5%** of the time on patients it has
never seen, versus **12.1%** for a baseline that ignores the current state
entirely — nearly **3x** better than guessing the globally most common
next state. This confirms Day13's transition patterns are not just noise
from a single video: knowing the current state genuinely carries
predictive information about what a surgeon is likely to do next, and
that information generalizes across different patients and operations.

At the same time, **65% of predictions are still wrong**. A model that
only looks one step back cannot know, for example, whether the cystic duct
was already clipped earlier in this particular operation, or how this
surgeon tends to sequence steps — information a longer memory could use.
This gap is exactly the headroom that motivates moving beyond a 1-step
Markov model toward models with richer memory (RNN, Attention).

Only **4.1%** of test transitions started from a state that was never seen
during training. This is a reassuring finding in its own right: despite
different patients, anatomy, and surgeons, most of the 358 recurring
"states" in laparoscopic cholecystectomy are shared across operations
rather than being unique to one video — there is a real, common vocabulary
of surgical action to be learned.

---

## Reflection (surgeon's perspective)

Surgery does follow a coarse procedural order. For laparoscopic
cholecystectomy: port placement, dissection of Calot's triangle,
confirmation of the Critical View of Safety (CVS), clipping and cutting
the cystic duct and artery, dissection of the gallbladder off the liver
bed, and finally packaging and extraction. At that macro level — which
roughly corresponds to CholecT50's own phase labels — predicting "what
comes next" is largely reasonable. This macro level is really about
recognizing *events*: the moment one step of the procedure ends and the
next begins.

Recognizing those events reliably is known to be hard even at this coarse
level, which is exactly why prior work has gone beyond triplets and tried
to represent the scene as a graph over anatomy — spatial edges capturing
instrument-tissue relationships within a frame, temporal edges capturing
how those relationships change over time — instead of relying on
instrument-verb-target labels alone. If even macro-level event
recognition needed that much extra structure, it is not surprising that
predicting micro-level (per-state, S) transitions from triplets alone is
even harder.

At the micro level, what happens next during dissection often depends on
information a triplet label simply does not contain: whether tissue
tension at the current grasp point is adequate, whether unexpected
bleeding occurs, whether a structure has become clearly exposed. For
example, bleeding is often followed by irrigator use; a clearly exposed
cystic duct is often followed by clipper use. These are visual/anatomical
findings, not annotated categories in CholecT50.

This distinction matters for how the 34.5% accuracy number should be
read. The 65% of wrong predictions are a mix of at least two different
things:

1. **Genuine surgeon judgment calls** — decisions that would reasonably
   vary from case to case regardless of how much information were
   available.
2. **Transitions that are, in principle, predictable, but invisible to
   this model** — because CholecT50's triplet/phase vocabulary does not
   encode the visual finding (bleeding, exposure, tension) that actually
   drove the surgeon's decision.

CholecT50's annotations alone cannot separate (1) from (2). So 34.5%
should not be read as "surgical workflow is 34.5% predictable." It should
be read as "this is as predictable as it gets using only symbolic
triplet-and-phase labels and one step of memory." The ceiling measured
here is a property of this representation, not necessarily a property of
surgery itself.

---

## Conclusion

Scaling Day13's single-video Markov chain to 50 videos with a proper
train/test split turned a descriptive exercise into the project's first
genuinely evaluated predictive model, and produced a reusable artifact
(`state_vocabulary.json`) for future days. The Markov model clearly beats
a naive baseline, but its ~35% accuracy ceiling should be read carefully:
part of it comes from Markov's one-step memory, and part of it comes from
triplet/phase labels not encoding the visual and anatomical information
(bleeding, tissue tension, exposure) that surgeons actually use to decide
their next move.

## Next Step

Day15 will test the macro-vs-micro hypothesis directly: using the same
50-video train/test split, compare next-state prediction accuracy at the
**phase level** (CholecT50's own phase labels) against the **triplet-state
(S) level** used today. If macro-level transitions really are more
predictable than micro-level ones, phase-level accuracy should be
noticeably higher than the 34.5% measured here.
