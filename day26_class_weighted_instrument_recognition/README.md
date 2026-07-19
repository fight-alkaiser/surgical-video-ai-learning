# Day26: Class-Weighted Instrument Recognition

## Objective

Day21 named three candidate fixes for its rare-instrument problem
(bipolar F1 0.106, scissors 0.054, clipper 0.012, irrigator 0.100, vs.
grasper 0.860 and hook 0.677): more videos, backbone fine-tuning, and a
class-weighted loss. Today isolates the cheapest and fastest of the
three: everything stays identical to Day21 (same 10 videos, same frozen
ResNet18 + linear head, same split), except `BCEWithLogitsLoss` gets a
`pos_weight` per instrument -- the standard imbalanced-classification
technique of penalizing a missed positive example for a rare class more
heavily than one for a common class.

## Method

[`class_weighted_instrument_recognition.py`](class_weighted_instrument_recognition.py)
reuses Day24's feature-caching pattern (extract frozen ResNet18 features
once, train fast on cached vectors) with one change: `pos_weight[i] =
num_negative[i] / num_positive[i]`, computed from training prevalence,
passed to `BCEWithLogitsLoss`. In effect: bipolar's `pos_weight` is
14.27, scissors' is 30.72, clipper's is 31.15, irrigator's is 16.85 --
versus 0.41 for grasper and 0.89 for hook, which are already common
enough not to need boosting.

## Results

| Instrument | Day21 F1 (unweighted) | Day26 F1 (class-weighted) | Day26 precision | Day26 recall |
|---|---:|---:|---:|---:|
| grasper | 0.860 | 0.859 | 0.801 | 0.927 |
| hook | 0.677 | 0.735 | 0.811 | 0.673 |
| bipolar | 0.106 | 0.182 | 0.143 | 0.253 |
| clipper | 0.012 | **0.291** | 0.190 | 0.629 |
| irrigator | 0.100 | 0.148 | 0.084 | 0.613 |
| scissors | 0.054 | 0.050 | 0.026 | 0.547 |
| **Macro F1** | **0.302** | **0.378** | -- | -- |
| **Macro accuracy** | 0.894 | 0.786 | -- | -- |

## Interpretation

**Macro F1 improved meaningfully (0.302 -> 0.378) while macro accuracy
got clearly worse (0.894 -> 0.786, now below the trivial baseline's
0.825).** This is not a contradiction, it's the expected, deliberate
trade-off: `pos_weight` tells the loss "a missed rare-instrument
detection costs far more than a false alarm," so the model becomes much
more willing to predict "present" across the board. Recall jumps
dramatically for every rare instrument (clipper 0.629, irrigator 0.613,
scissors 0.547 -- versus near-zero recall implied by Day21's F1 of
0.012-0.106), at a real cost in precision (clipper 0.190, irrigator
0.084, scissors 0.026). Whether this trade is worth it depends entirely
on what the numbers are used for, exactly the point raised before
building this experiment: a lower accuracy number is not a regression
here, it's the visible cost of a deliberate choice to prioritize
catching rare instruments over not crying wolf on common frames.

**Clipper is the clear success story** (F1 0.012 -> 0.291, a 24x
improvement): its precision (0.190) is low but real, and its recall
(0.629) means the model now actually attempts detection instead of
almost never predicting "present" at all, which is what an F1 of 0.012
implies. **Bipolar and irrigator improved similarly**, roughly doubling
or better.

**Scissors is the outlier: F1 barely moved (0.054 -> 0.050) despite
recall rising to 0.547** -- comparable to clipper's recall gain, with an
almost identical `pos_weight` (30.72 vs. clipper's 31.15). But its
precision (0.026) is far worse than clipper's (0.190): roughly 38 false
positives for every true positive. Re-weighting fixed the model's
*willingness* to guess "scissors present," but couldn't fix a separate
problem -- the frozen ImageNet features apparently don't separate
scissors from other instruments well in the first place, so encouraging
more positive guesses mostly produces more wrong ones. This is a useful
negative result: `pos_weight` addresses a threshold/willingness problem,
not a feature-separability problem, and scissors looks like the latter.

**Grasper and hook, the two instruments that didn't need help, were not
harmed** (grasper F1 essentially unchanged at 0.859 vs. 0.860; hook
actually improved slightly to 0.735). Their `pos_weight` values (0.41,
0.89) are close to 1 by construction, since they're already common, so
this isn't surprising, but it's worth confirming directly: fixing the
rare classes didn't come at the strong classes' expense.

## Reflection

This is a concrete, small-scale instance of the trade-off question
raised before this experiment was designed: two goals (catch rare
instruments, keep overall accuracy high) pulling in different
directions, within a *single* model and a *single* choice (the loss
function's weighting). The resolution here wasn't to find a setting with
no cost -- there isn't one -- but to make the cost visible and legible
(precision fell in exchange for recall, tracked per class) so that
*which* point on that trade-off curve to choose is a decision that can
be made deliberately, informed by what the numbers will be used for,
rather than accepting whatever a default, unweighted loss happens to
produce. An unweighted loss is not a neutral, assumption-free choice --
it implicitly weights every class's errors equally in a way that, for
this imbalanced data, means rare-instrument recall is sacrificed by
default without anyone deciding that on purpose.

Scissors' non-result is the more scientifically interesting finding of
the day, precisely because it's a clean failure of the technique tried:
it shows `pos_weight` has a specific, limited mechanism (adjust how
readily the model guesses "present") that cannot substitute for feature
quality. That distinction -- willingness to guess vs. ability to
distinguish -- points directly at backbone fine-tuning (Day21's other
named fix, not yet tried) as the more likely next lever for scissors
specifically, rather than more aggressive re-weighting of the same
frozen features.

## Conclusion

Class-weighted loss alone -- no new data, no architecture change --
raises macro F1 from 0.302 to 0.378, mainly by making rare-instrument
recall usable at all (clipper: F1 0.012 -> 0.291). This comes at a real,
expected cost to precision and overall accuracy, which is a deliberate
trade-off rather than a flaw. Scissors' lack of improvement despite a
similar recall gain to clipper's shows this technique's limit: it fixes
a model's willingness to predict a rare class, not its ability to
visually distinguish that class in the first place, which is a separate
problem pointing toward backbone fine-tuning as the next fix to try.
