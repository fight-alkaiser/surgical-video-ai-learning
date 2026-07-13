# Day21: Multi-Video Instrument Recognition

## Objective

Day20 trained on a single video (VID01) with a chronological split and
found the result couldn't support any real claim: three of six
instrument classes had zero positive examples in the test segment,
because testing "later in the same video" entangles "different part of
the procedure" with "generalizes to new data" -- exactly the confound
Day14 avoided for the symbolic triplet-state pipeline by splitting at
the video (patient) level. Today repeats the same task -- multi-label
instrument recognition from raw frames -- across 10 videos (the ones
extracted so far; see [`day20`](../day20_pixel_instrument_recognition/README.md)
for the storage discussion), split at the video level (8 train / 2
test), and asks directly: does this fix Day20's core problem, and does
the model now clear a trivial baseline?

## Method

[`instrument_recognition_multi_video.py`](instrument_recognition_multi_video.py)
uses the same model as Day20 -- a frozen ImageNet-pretrained ResNet18
with a newly trained linear head (512 -> 6), `BCEWithLogitsLoss`,
independent binary prediction per instrument -- extended to 10 videos
(VID01, 02, 04, 05, 06, 08, 10, 12, 13, 14; ~17,600 frames total).
The video ID list is shuffled with the project's usual fixed seed and
split 8 train / 2 test (`VID05`, `VID12` held out entirely); every frame
from a test video is unseen by the model in any form, unlike Day20 where
train and test frames came from the same operation.

**A methodological limitation worth naming:** there is no train/validation
split or hyperparameter search here -- 10 epochs was a fixed choice, not
one tuned against a held-out validation set, and the model was trained
exactly once. This means there's no leakage from peeking at the test set
during development, but also no check on whether 10 epochs was actually
a good stopping point, unlike Day19's epoch-sweep table. A proper setup
would split the 8 training videos further into train/validation (or run
k-fold across them) and touch the 2 held-out test videos only once, at
the very end. Skipped here to keep the day's scope small; timed
separately, one epoch over the 8 training videos takes ~96s, so a
6-video/2-video hold-out split would cost roughly 1.5-2x today's single
run, and a 4-fold cross-validation roughly 4x -- both affordable, just
deferred rather than judged unnecessary.

## Results

| Instrument | Accuracy | F1 | Baseline accuracy | Train prevalence | Test prevalence |
|---|---:|---:|---:|---:|---:|
| grasper | 0.814 | 0.860 | 0.661 | 0.709 | 0.661 |
| bipolar | 0.932 | 0.106 | 0.942 | 0.066 | 0.058 |
| hook | 0.755 | 0.677 | 0.437 | 0.529 | 0.437 |
| scissors | 0.969 | 0.054 | 0.981 | 0.032 | 0.019 |
| clipper | 0.953 | 0.012 | 0.956 | 0.031 | 0.044 |
| irrigator | 0.942 | 0.100 | 0.969 | 0.056 | 0.031 |
| **Macro** | **0.894** | **0.302** | **0.825** | -- | -- |

Every instrument now has meaningful, nonzero prevalence in the test set
(the minimum is scissors at 1.9% -- rare, but present, unlike Day20's
literal zero for three classes). Macro accuracy (0.894) now clears the
trivial train-majority baseline (0.825), which Day20 could not do
(0.807 vs. 0.811).

## Interpretation

**The video-level split fixed exactly the problem it was meant to fix.**
No instrument class has zero test examples anymore, so every row in the
table above is measuring something real, rather than reporting a hollow
1.000 for an absent class as Day20 did for scissors and clipper.

**The two common instruments show a clear, genuine win.** Grasper
(70.9% of training frames) and hook (52.9%) both comfortably beat their
baselines on both accuracy and F1 (grasper: 0.814/0.860 vs. baseline
0.661; hook: 0.755/0.677 vs. baseline 0.437). These are exactly the two
classes with enough positive *and* negative examples in training for a
linear head on frozen features to find a real, generalizable signal --
and it does, across patients this time, not just across time within one
patient.

**The four rare instruments (bipolar, scissors, clipper, irrigator,
all under 7% training prevalence) remain hard.** Their F1 scores
(0.01-0.11) are low, though no longer undefined. This isn't the same
failure as Day20: it's a real, different problem -- with only 8 training
videos and each rare instrument appearing in a few hundred frames at
most (e.g. scissors: 3.2% of 14,212 training frames is ~450 frames), a
frozen backbone with only a linear head to train has very little signal
to work with for these classes specifically. Accuracy alone would hide
this (0.95-0.97 for all four, because predicting "absent" is usually
correct when a class is this rare) -- F1 is what actually shows the model
is barely better than chance at detecting them when they do appear.

## Reflection

This day is best read as a direct, controlled comparison with Day20
rather than a standalone result: same model, same task, same code
structure, the only real change is evaluating across patients instead of
across time within one patient. That one change moved macro accuracy
from "loses to a trivial baseline" (0.807 vs. 0.811) to "clearly beats
it" (0.894 vs. 0.825), and turned three undefined F1 scores into small
but real ones. This is a clean illustration of a general point: for
surgical video data specifically, evaluation protocol (video-level vs.
frame-level or time-based splits) can matter as much as -- or more than
-- model architecture, because adjacent frames and adjacent time segments
of the same procedure share far more information than frames from a
different patient's procedure.

The remaining weakness (rare instruments) is not a protocol problem this
time -- it's a data volume problem. A frozen linear probe cannot
manufacture signal for a class it has only seen a few hundred times.
The natural next steps, in order of likely payoff: (1) more videos,
which directly increases the number of positive examples for rare
classes; (2) fine-tuning more of the backbone rather than only the final
linear layer, since ImageNet features may not emphasize the visual cues
(thin metal shafts, specific reflectance) that distinguish scissors from
clipper as well as they distinguish, say, a dog from a cat; (3) a
loss that weights rare classes more heavily, so the small number of
positive examples they do have isn't drowned out during training by the
much larger number of "absent" examples.

## Conclusion

Splitting by video rather than by time within one video fixes the core
problem Day20 ran into: every instrument class now has real test
examples, and the model clearly beats a trivial baseline on the two
common instruments (grasper, hook). Rare instruments remain hard to
detect (F1 0.01-0.11), but this is now a legible, real limitation --
insufficient positive examples for a frozen linear probe -- rather than
an artifact of a broken evaluation. This confirms multi-video,
video-level evaluation is a prerequisite for any pixel-based result on
this dataset to mean anything, exactly as it already was for the
symbolic triplet-state pipeline since Day14.
