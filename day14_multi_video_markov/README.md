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
  in sorted order.
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

## Conclusion

Scaling Day13's single-video Markov chain to 50 videos with a proper
train/test split turned a descriptive exercise into the project's first
genuinely evaluated predictive model, and produced a reusable artifact
(`state_vocabulary.json`) for future days. The Markov model clearly beats
a naive baseline, but its ~35% accuracy ceiling — set by looking only one
step into the past — is the concrete, measured motivation for the
sequence models coming next.

## Next Step

Day15: examine *where* the Markov model fails (which states are hardest
to predict, and why) before moving toward models with more memory than a
single previous state.
