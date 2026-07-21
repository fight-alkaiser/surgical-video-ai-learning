# Day27: Backbone Fine-Tuning

## Objective

Day26 found that class-weighted loss fixes a model's *willingness* to
guess a rare instrument (clipper: F1 0.012 -> 0.291) but not scissors
specifically (0.054 -> 0.050, despite comparable recall gains) --
precision stayed near zero (0.026), suggesting the frozen ImageNet
backbone's features don't separate scissors from other instruments well
in the first place, regardless of loss threshold. Today tests Day21's
other named fix: unfreeze ResNet18's last residual block (`layer4`) and
fine-tune it, keeping Day26's class-weighted loss constant, so the
visual features themselves can adapt to this specific dataset instead of
staying fixed at whatever ImageNet happened to learn.

## Method

[`backbone_finetuning.py`](backbone_finetuning.py) keeps `conv1`, `bn1`,
`layer1`, `layer2`, `layer3` frozen (the earlier, more generic layers --
edges, textures, simple shapes) and unfreezes `layer4` plus a new final
linear layer, with a lower learning rate for the pretrained backbone
layer (1e-4) than the freshly-initialized head (1e-3) -- standard
practice to avoid destroying useful pretrained weights with large early
updates. Frozen layers' BatchNorm statistics are explicitly kept in eval
mode during training (a `set_training_mode()` helper), so they don't
silently drift even though their weights don't update. The class-weighted
`BCEWithLogitsLoss` from Day26 is unchanged.

This machine has 8GB RAM, which ruled out Day24-26's feature-caching
shortcut (caching `layer3`'s output for every frame would need several
GB by itself). Training here is a live loop like Day21's original
script -- forward through the whole network every batch, backward only
into `layer4` and the final layer -- at a smaller batch size (16) to
keep memory manageable. Same 10 videos, same video-level 8/2 split as
Day21/26.

## Results

| Instrument | Day21 F1 (frozen) | Day26 F1 (frozen, weighted) | Day27 F1 (fine-tuned, weighted) | Day27 precision | Day27 recall |
|---|---:|---:|---:|---:|---:|
| grasper | 0.860 | 0.859 | **0.906** | 0.876 | 0.939 |
| hook | 0.677 | 0.735 | **0.907** | 0.878 | 0.939 |
| bipolar | 0.106 | 0.182 | **0.431** | 0.484 | 0.389 |
| clipper | 0.012 | 0.291 | **0.492** | 0.481 | 0.503 |
| irrigator | 0.100 | 0.148 | **0.236** | 0.220 | 0.255 |
| scissors | 0.054 | 0.050 | 0.101 | 0.085 | 0.125 |
| **Macro F1** | **0.302** | **0.378** | **0.512** | -- | -- |
| **Macro accuracy** | 0.894 | 0.786 | **0.932** | -- | -- |

## Interpretation

**Fine-tuning improved every single instrument, including the two that
were already strong.** Grasper and hook -- already Day21's best
performers -- still gained (0.860 -> 0.906, 0.677 -> 0.907), showing this
wasn't only a rare-class fix; ImageNet features were leaving real
accuracy on the table even for the easy cases.

**Unlike Day26, this improvement did not come from trading precision for
recall -- both moved together.** Day26's clipper result (F1 0.291) came
from precision 0.190 and recall 0.629 -- a model made willing to guess,
often wrong. Day27's clipper result (F1 0.492, higher) comes from
precision 0.481 and recall 0.503 -- close to balanced, and both
substantially better than Day26's precision. This is the signature of a
genuinely better decision boundary in feature space, not just a
different threshold on the same one: the same `pos_weight` values are
in effect in both experiments, so the *only* variable that changed is
whether the features themselves could adapt. Macro accuracy improving
too (0.786 -> 0.932, now clearly above Day21's frozen unweighted result
as well) confirms this isn't the same accuracy-for-recall trade Day26
made -- fine-tuning bought a strictly better model, not a different
point on the same trade-off curve.

**Scissors improved the least, and differently from every other
instrument: precision rose (0.026 -> 0.085) while recall fell sharply
relative to Day26 (0.547 -> 0.125).** With fine-tuned features, the model
became far more conservative about predicting scissors -- guessing it
far less often, but somewhat more accurately when it does. Net F1 is
still barely above Day21's frozen baseline (0.101 vs. 0.054). Scissors
is the rarest instrument in this dataset by a wide margin (roughly 500
total co-occurrence instances across all 10 videos, concentrated in even
fewer of the 8 training videos) -- this looks like a case where even
better features run into a hard floor set by how few examples exist to
learn from at all, rather than a fixable feature-quality or
threshold-calibration problem.

## Reflection

Day26 and Day27 together tell a cleaner story than either would alone.
Day26 showed re-weighting the loss changes *where on the precision-
recall curve* a fixed set of features operates -- useful, but bounded by
how separable the classes already are in that fixed feature space.
Day27 shows that when the features themselves can move, the whole curve
shifts outward: precision and recall improve together rather than
trading off, for every instrument except the one (scissors) where the
underlying data is so scarce that no reasonable amount of fine-tuning
on 8 videos appears able to compensate. This is a fairly clean
confirmation of the distinction proposed after Day26 -- "willingness to
guess" vs. "ability to distinguish" -- with fine-tuning squarely
addressing the second, and doing so more effectively than expected, even
for instruments (grasper, hook) that weren't the ones motivating the
experiment.

It's also a reminder that "rare instrument" isn't one problem with one
severity: bipolar (F1 0.106 -> 0.431) and clipper (0.012 -> 0.492)
responded very differently from scissors (0.054 -> 0.101) to the exact
same intervention, despite all three being called "rare" in Day21's
original framing. Scissors' particular scarcity (fewer total instances
than any other instrument, concentrated further by the video-level
split) looks like a distinct, harder case that may need more videos
specifically, rather than a better technique applied to the same amount
of data.

## Conclusion

Fine-tuning ResNet18's last residual block, on top of Day26's
class-weighted loss, raises macro F1 from 0.378 to 0.512 and macro
accuracy from 0.786 to 0.932 -- both precision and recall improve
together for nearly every instrument, unlike Day26's pure re-weighting,
which traded one for the other within a fixed feature space. Scissors is
the exception: still the weakest performer (F1 0.101), improving only
modestly despite the same intervention that transformed bipolar and
clipper, consistent with its scarcity being severe enough that better
features alone cannot fully compensate. Between Day26 (fixes willingness)
and Day27 (fixes feature quality), most of Day21's original
rare-instrument problem now has a concrete, mostly-solved answer; what
remains -- scissors specifically -- points back to needing more data
rather than a different technique.
