# Day31: Contrastive Self-Supervised Pretraining

## Objective

Day01-30 were all supervised: every day started from a human-provided
label. Foundation models are built the other way around -- learn a
useful representation from unlabeled data first, using a pretext task
invented from the data itself, then adapt cheaply to downstream labeled
tasks. Today tries the most influential modern version of that idea:
contrastive learning (SimCLR, Chen et al., 2020). Two randomly-augmented
views of the same frame are pulled together in embedding space; every
other frame in the batch is pushed apart. No instrument, verb, target,
or phase label is used anywhere in this pretraining.

This is deliberately a small, self-built reproduction of the
**mechanism**, not a real foundation model: full-scale SimCLR uses batch
sizes in the hundreds to thousands (more negatives per batch measurably
helps); this machine's 8GB RAM limits the batch size to a small fraction
of that (32 images -> 64 augmented views per batch here). The question
is narrower and still meaningful: starting from ImageNet-pretrained
weights, does adapting `layer4` to this surgical dataset with **no
labels at all** move Day21's frozen-feature baseline (instrument macro
F1 0.302) any closer to what instrument *labels* bought via supervised
fine-tuning (Day27, F1 0.512)?

## Method

Two scripts. [`contrastive_pretraining.py`](contrastive_pretraining.py):
`conv1`/`bn1`/`layer1`/`layer2`/`layer3` frozen (same split as Day27),
`layer4` trainable, plus a small projection head (512 -> 512 -> 128) used
only during pretraining and discarded afterward -- standard SimCLR
practice, since the representation used downstream is the layer *before*
the projection head, not the projection itself. Trained for 15 epochs
with the NT-Xent contrastive loss on the **8 training videos' frames
only** (14,212 frames) -- the same video-level split as every
instrument/verb/target day, so the 2 test videos are never touched, with
or without labels, keeping this comparable to every earlier evaluation.

[`linear_probe_evaluation.py`](linear_probe_evaluation.py) then freezes
the entire adapted backbone, caches its features (Day24's shortcut), and
trains a class-weighted linear probe (Day26's recipe) with instrument
labels -- used here for the first time in this backbone's existence, and
only at this evaluation stage, never during pretraining.

## Results

**Pretraining**: NT-Xent loss fell from 2.59 to 2.30 over 15 epochs
(~105 minutes total). For reference, random chance on a 64-view batch
(63 possible matches per row) would give a loss of ln(63) ≈ 4.14 --
pretrained ImageNet features already start well below that, and the
loss continued to decrease steadily, indicating real if gradual further
adaptation.

**Instrument recognition, linear probe on the frozen, contrastively-adapted backbone:**

| Instrument | Day21 F1 (ImageNet, no adaptation) | Day31 F1 (contrastive, no labels) | Day27 F1 (supervised fine-tune) | Gap closed by Day31 |
|---|---:|---:|---:|---:|
| grasper | 0.860 | 0.862 | 0.906 | 4.3% |
| hook | 0.677 | 0.719 | 0.907 | 18.3% |
| scissors | 0.054 | 0.069 | 0.101 | 31.9% |
| irrigator | 0.100 | 0.181 | 0.236 | 59.6% |
| clipper | 0.012 | 0.298 | 0.492 | 59.6% |
| bipolar | 0.106 | 0.310 | 0.431 | 62.8% |
| **Macro F1** | **0.302** | **0.407** | **0.512** | **50.0%** |

"Gap closed" = how far Day31 moved from Day21 toward Day27, as a
percentage of the total Day21-to-Day27 distance.

## Interpretation

**Label-free adaptation alone recovered half of what instrument labels
bought via fine-tuning** (macro F1 0.302 -> 0.407 -> 0.512), without the
contrastive objective ever seeing an instrument name. That's a real,
sizeable result for a deliberately scaled-down reproduction.

**The gap-closed percentage is not uniform across instruments, and the
pattern is informative rather than random.** `clipper` and `bipolar`
recovered ~60% of their respective gaps -- both were exactly the
instruments Day26/27 diagnosed as suffering from a genuine feature-
separability problem with frozen ImageNet features, not just a
detection-threshold problem. `grasper`, already near its ceiling under
plain frozen features (F1 0.860), moved almost nowhere (4.3%) -- there
was very little gap left to close. `hook`, despite being common and
well-detected under both frozen and fine-tuned features, closed only
18.3% of its gap -- suggesting whatever made hook detection jump so much
under supervised fine-tuning (0.677 -> 0.907) was a cue the contrastive
pretext task's "tell this frame apart from other frames" objective
wasn't particularly aligned with discovering on its own.

**Scissors remains the hardest case by a wide margin in absolute terms**
(F1 0.069), even though its *relative* gap-closed (31.9%) isn't the
lowest. This is consistent with Day27's conclusion that scissors'
extreme rarity (~500 total instances across the whole dataset) is a data
scarcity problem no technique tried so far -- supervised fine-tuning,
class weighting, or now label-free contrastive adaptation -- has been
able to fully overcome.

## Reflection

The headline framing here could easily overstate the result: "self-
supervised learning recovers half the value of labels" sounds like a
strong, general claim, but the per-instrument breakdown shows it's
concentrated in exactly the cases where Day26/27 already diagnosed a
feature-quality problem, and nearly absent where the frozen baseline was
already strong. That's a more precise and more interesting finding than
the macro number alone: contrastive pretraining isn't uniformly
"worth half a label" -- it's disproportionately valuable exactly where a
frozen, generic backbone's features were weakest to begin with, and close
to worthless where they were already good. This is also a reassuring
sanity check on the whole exercise: if the gap-closing had been uniform
or random across instruments, that would suggest something spurious (a
regularization side-effect, a lucky initialization) rather than the
contrastive objective genuinely learning a more useful visual
representation of this domain's images.

It's also worth being honest about what this result doesn't yet show.
Only 6 downstream classes (instruments) were tested, on a heavily
resource-constrained reproduction (batch size 32-64, vs. hundreds to
thousands in published SimCLR results), pretrained on only 8 videos'
worth of frames. Whether this pattern -- large relative gains
concentrated on feature-quality-limited classes -- would hold for verb
or target recognition, or with a larger batch size, is untested.

## Conclusion

A small, resource-constrained reproduction of SimCLR-style contrastive
learning -- adapting only `layer4` on 8 unlabeled training videos, no
instrument/verb/target/phase label ever used during pretraining --
recovers roughly half of the macro F1 gap between frozen ImageNet
features (Day21, F1 0.302) and full supervised fine-tuning (Day27, F1
0.512), landing at F1 0.407. The recovery is concentrated specifically
in the instruments Day26/27 identified as feature-quality-limited
(bipolar, clipper: ~60% of their gap closed) rather than the
already-strong ones (grasper: 4%), and scissors' extreme data scarcity
remains unresolved by this technique too. This is a genuinely
informative first result for the self-supervised arc opened today: not
that labels are unnecessary, but that a meaningful fraction of what
labels buy can be recovered without them, specifically where a generic
pretrained backbone's features were weakest.
