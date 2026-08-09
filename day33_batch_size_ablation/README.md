# Day33: Does a Bigger Batch Size Actually Help?

## Objective

Day31 named its own biggest limitation explicitly: full-scale SimCLR
uses batch sizes in the hundreds to thousands, since more negatives per
contrastive batch is known to help the method, while this 8GB RAM
machine could only manage N=32 images (64 augmented views) per batch.
Today tests that specific, named caveat directly rather than leaving it
as an unverified disclaimer: does doubling the batch size to N=64 (128
views) -- as far as this machine's RAM comfortably allows -- move the
downstream result?

## Method

[`contrastive_pretraining_large_batch.py`](contrastive_pretraining_large_batch.py)
is Day31's script with one change: `BATCH_SIZE` doubled (32 -> 64
images, 64 -> 128 augmented views per batch), with both learning rates
scaled up 2x to match, following the standard "linear scaling rule" for
large-batch training (Goyal et al., 2017) -- without this, a bigger
batch just means fewer optimizer steps per epoch, which would confound
"does more negatives help" with "did we simply undertrain from fewer
updates." Same 8 training videos, same frozen/trainable layer split,
same NT-Xent loss, same 15 epochs.
[`linear_probe_evaluation.py`](linear_probe_evaluation.py) is Day31's
evaluation script unchanged, applied to this new backbone.

## Results

Training completed without memory issues, and was actually faster
wall-clock than Day31 (~68 minutes vs. ~105 minutes) despite processing
the same total number of images per epoch -- fewer, larger batches mean
less per-batch Python/loop overhead. NT-Xent loss values themselves
aren't directly comparable to Day31's (128 views means classifying among
127 candidates per row instead of 63, so the loss scale itself differs),
so the downstream linear probe is the fair comparison.

| Instrument | Day21 (frozen) | Day31 (N=32, 64 views) | Day33 (N=64, 128 views) | Day27 (supervised) |
|---|---:|---:|---:|---:|
| grasper | 0.860 | 0.862 | 0.877 | 0.906 |
| hook | 0.677 | 0.719 | 0.738 | 0.907 |
| bipolar | 0.106 | 0.310 | 0.265 | 0.431 |
| clipper | 0.012 | 0.298 | 0.318 | 0.492 |
| irrigator | 0.100 | 0.181 | 0.170 | 0.236 |
| scissors | 0.054 | 0.069 | 0.069 | 0.101 |
| **Macro F1** | **0.302** | **0.407** | **0.406** | **0.512** |

Doubling the batch size changed macro F1 by -0.001 -- indistinguishable
from no effect. Per-instrument, changes from Day31 to Day33 range from
-0.045 (bipolar) to +0.020 (clipper), much smaller than the original
Day21-to-Day31 gains they're being compared against (+0.002 to +0.286).

## Interpretation

**The named limitation does not explain the remaining gap to supervised
fine-tuning.** Going into this day, the natural hypothesis was "batch
size is probably part of why Day31 only closed half the gap to Day27";
doubling it and finding essentially no change rules that out, at least
across this doubling. This doesn't prove batch size would never matter
(published SimCLR results use 4-30x more views than even this larger
run), but it does mean the *first* doubling -- the one actually
achievable on this hardware -- bought nothing, so hardware-constrained
batch size specifically is not the place to look for the next
improvement.

**The small per-instrument fluctuations look like noise, not a
pattern.** Bipolar went down (0.310 -> 0.265) while clipper went up
(0.298 -> 0.318) and scissors didn't move at all (0.069 -> 0.069) --
there's no consistent direction, unlike the clear, uniform-looking gains
from Day21 to Day31 tied to which instruments had weak frozen features.
If a larger batch were genuinely adding useful signal, a more consistent
improvement pattern would be a more convincing sign of it; scattered,
small, bidirectional changes are the signature of run-to-run noise at
this data scale (8 training videos) more than of a real effect.

## Reflection

This is a valuable negative result precisely because it was a real,
falsifiable prediction stated in advance (Day31's own stated caveat),
not a hypothesis invented after seeing a disappointing number. Finding
that the correction didn't work is more informative than skipping the
test would have been: it rules out the most obvious, most frequently-
cited explanation (published SimCLR papers universally point to batch
size as important) and redirects attention to less obvious candidates --
total training epochs/compute, the augmentation strategy (color jitter
in particular may be poorly suited to surgical images, as flagged in
Day31), the temperature hyperparameter, or simply that 8 videos' worth
of visual diversity is the actual ceiling for what a frame-level
contrastive objective can learn here, independent of batch size. This
project's habit of testing a named caveat directly, rather than treating
it as a permanent excuse for a middling result, paid off with a cleaner
answer than assuming the caveat mattered would have given.

## Conclusion

Doubling the contrastive pretraining batch size (32 -> 64 images, 64 ->
128 views), with learning rate scaled to match, leaves the downstream
instrument-recognition macro F1 unchanged (0.407 -> 0.406). Day31's own
named limitation -- batch size, this machine's most obvious hardware
constraint relative to published SimCLR results -- is not, in fact, the
binding constraint on this result, at least within the range achievable
here. This closes the batch-size question cleanly and points the next
investigation (if this arc continues) toward other candidates: training
duration, augmentation choices, or a more fundamental ceiling set by
the amount and diversity of unlabeled data available, rather than
compute-bound batch size.
