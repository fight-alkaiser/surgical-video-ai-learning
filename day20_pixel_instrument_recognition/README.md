# Day20: Instrument Recognition from Raw Pixels

## Objective

Day01-19 always started from CholecT50's human-provided
instrument-verb-target triplet annotations -- a symbolic label handed to
the project, never something extracted from an image. The dataset's
actual task, and the one Nwoye et al.'s Rendezvous paper (2022, the paper
that introduced CholecT50) addresses, is the other half of the problem:
recognizing those triplets directly from raw endoscopic video frames.
Today starts that arc with its simplest slice -- multi-label instrument
recognition (6 classes: grasper, bipolar, hook, scissors, clipper,
irrigator) from a single frame -- deliberately leaving verb/target
recognition and the interaction-attention modules that connect them for
later days.

This is also the first day using PyTorch instead of a from-scratch numpy
implementation. That's a deliberate change, not a relaxation of standard:
convolution/pooling backprop is not a core mechanism this project has
been trying to internalize the way embedding lookups, BPTT, and attention
were (Day16-19) -- it's well-understood, separate engineering, and
training a CNN from scratch on real images without a GPU-accelerated
autograd engine isn't practical in a day's scope. Rendezvous itself uses
an ImageNet-pretrained CNN backbone and reserves its novelty for the
attention modules on top of it, so using a library for the backbone here
matches the paper's own design choice, not just convenience.

## Data scope

Only VID01 has raw frames downloaded locally (1734 frames, ~864MB) --
the other 49 CholecT50 videos are annotation-only in this project so far
(downloading all 50 videos' worth of frames would be a large download,
deferred rather than done without asking). Today is a **single-video
prototype**: everything here is trained and evaluated within one
patient's operation, which matters a great deal for how the results
below should be read (see Interpretation).

## Method

[`instrument_recognition.py`](instrument_recognition.py):

- **Split:** chronological, not random -- the first 80% of VID01's frames
  (in time) for training, the last 20% for testing. Adjacent frames in a
  surgical video are nearly identical, so a random frame-level split
  would put near-duplicate frames in both train and test, an easy way to
  get a falsely high accuracy. A chronological split avoids that specific
  leak, at the cost of a different problem discussed below.
- **Labels:** for each frame, a 6-dimensional multi-hot vector marking
  which instruments are active in any triplet annotated for that frame.
- **Model:** `torchvision`'s ResNet18, ImageNet-pretrained, with every
  layer frozen except a newly added final linear layer (512 -> 6),
  trained with `BCEWithLogitsLoss` (independent binary classification per
  instrument). This is the same "freeze a pretrained feature extractor,
  train a linear head on top" pattern as the linear probes in Day17-19 --
  applied here to a frozen visual backbone instead of a frozen sequence
  model's hidden state.

## Results

| Instrument | Accuracy | F1 | Baseline accuracy | Test prevalence |
|---|---:|---:|---:|---:|
| grasper | 0.689 | 0.799 | 0.827 | 0.827 |
| bipolar | 0.986 | 0.000 | 1.000 | 0.000 |
| hook | 0.271 | 0.271 | 0.135 | 0.135 |
| scissors | 1.000 | 0.000 | 1.000 | 0.000 |
| clipper | 1.000 | 0.000 | 1.000 | 0.000 |
| irrigator | 0.899 | 0.000 | 0.905 | 0.095 |
| **Macro** | **0.807** | **0.178** | **0.811** | -- |

"Baseline accuracy" is a trivial rule with no learning at all: for each
instrument, always predict whichever label (present/absent) was the
majority in the *training* frames. The trained model's macro accuracy
(0.807) does not beat this trivial baseline (0.811).

## Interpretation

This is a weak, honestly-reported result, and the reason is visible in
the per-class breakdown, not hidden by the macro average.

**Three instruments never appear at all in the test segment.** Checking
prevalence directly:

| Instrument | Train prevalence | Test prevalence |
|---|---:|---:|
| grasper | 0.616 | 0.827 |
| bipolar | 0.089 | 0.000 |
| hook | 0.534 | 0.135 |
| scissors | 0.012 | 0.000 |
| clipper | 0.074 | 0.000 |
| irrigator | 0.072 | 0.095 |

Bipolar, scissors, and clipper are used during specific mid-procedure
subtasks (dissection and clipping-and-cutting) that fall entirely within
the first 80% of this video -- so the test segment (the last 20%,
covering later phases like cleaning, extraction, and packaging) contains
*zero* positive examples of any of them. F1 is undefined/zero for these
classes not because the model failed at a hard problem, but because there
was nothing left to detect in the segment being tested. Their perfect
"accuracy" (1.000, 1.000, 1.000) is equally hollow -- predicting "absent"
always is correct when the class is always absent.

**Hook shows a real, large distribution shift**: used in 53.4% of
training frames but only 13.5% of test frames, because hook-based
dissection is concentrated earlier in the procedure than the test
segment covers. The training-majority baseline (predict "present",
matching train's >50% rate) becomes actively wrong most of the time on
the test segment (0.135 accuracy -- worse than a coin flip would give by
chance for a 13.5%-prevalence class), and the trained model, while better
(0.271), is still clearly struggling.

**Grasper is the only class with enough test-set signal to say anything
meaningful, and the model still doesn't clear the baseline** (0.689 vs.
0.827) on raw accuracy, though its F1 (0.799) suggests it is not simply
guessing at random -- accuracy and F1 disagree here because grasper is
present in 82.7% of test frames, so a model that's usually right about
the majority class but wrong on a meaningful chunk of the minority
"absent" frames can score worse on raw accuracy than "always predict
present" while still having reasonable precision/recall on the positive
class.

**The common thread is the evaluation protocol, not (necessarily) the
model.** A chronological split within a single video trades one problem
(near-duplicate frame leakage) for another: the training and test
segments cover different, non-overlapping stretches of the same
procedure, during which which instruments are even in use changes
substantially -- this is a real distribution shift, not measurement
noise, and it's largely a byproduct of testing generalization *within one
patient's video* rather than *across patients*. Every previous day in
this project (Day14 onward) was careful to split by video, specifically
to avoid exactly this kind of confound; today's single-video prototype
reintroduces it because only one video's frames are available locally.

## Reflection

The macro accuracy (0.807) looks like a solid number in isolation and
would be easy to report as "the model works." Breaking it down by class
shows that's not a fair reading: half the classes have no positive test
examples at all, and the one class with abundant, balanced test signal
(grasper) is the one where the model doesn't clearly beat a rule that
doesn't look at the image at all. This is the same lesson as Day16's
mode-collapse and Day19's overfitting-epoch table, in a new setting:
a single aggregate metric can look fine while hiding that nothing useful
is being measured underneath it, and the fix is the same each time --
break the average down by class/condition and check what a trivial
baseline gets before trusting a headline number.

This also clarifies what would actually need to change before this
result means something: not a better model or more training necessarily,
but a better-founded evaluation -- multiple videos with a proper
video-level train/test split, the same discipline Day14 already
established for the symbolic triplet-state work. A single video's
chronological split cannot separate "the model generalizes" from "the
test segment happens to contain different instruments than the training
segment," because in a single procedure, those two things are entangled
by construction.

## Conclusion

A frozen, ImageNet-pretrained ResNet18 with a linear head does not
clearly beat a trivial train-majority baseline at instrument recognition
within a single held-out segment of VID01, and three of six instrument
classes have no positive test examples at all under this split -- a
direct consequence of testing within one video rather than across
videos. The result itself is not the finding; the finding is that a
single-video, chronologically-split evaluation cannot support a real
claim about whether pixel-based instrument recognition works here.
Making that claim honestly requires what Day14 already did for the
symbolic pipeline: multiple videos, split by video, so a model has to
generalize across different patients and lighting conditions rather than
across two segments of the same operation.
