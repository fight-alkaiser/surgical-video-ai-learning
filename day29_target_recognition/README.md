# Day29: Target Recognition

## Objective

Instrument (Day20-21, 26-27) and verb (Day22-25, 28) each got a
multi-day arc working through the same progression: frozen baseline,
class-weighted loss, backbone fine-tuning, instrument-conditioning.
Target -- the anatomical structure a verb is applied to (15 classes:
`gallbladder`, `cystic_duct`, `cystic_artery`, `liver`, ... down to
`null_target`) -- is the third and last part of CholecT50's triplet, and
gets a deliberately lighter single day: rather than re-deriving the same
lessons step by step again, this applies the best known recipe directly
-- fine-tuned `layer4` (Day27) plus class-weighted loss (Day26) in one
run -- and asks what's different about target specifically, now that the
technique itself isn't the variable being tested.

## Method

[`target_recognition.py`](target_recognition.py) is Day27's script with
target's 15-way multi-hot label in place of instrument's 6-way one.
Same 10 videos, same video-level 8/2 split, same fine-tuning + `pos_weight`
recipe, no new technique introduced.

## Results

| Target | F1 | Precision | Recall | Test prevalence |
|---|---:|---:|---:|---:|
| gallbladder | 0.836 | 0.812 | 0.860 | 0.524 |
| liver | 0.594 | 0.454 | 0.859 | 0.084 |
| omentum | 0.466 | 0.483 | 0.450 | 0.073 |
| specimen_bag | 0.408 | 0.263 | 0.907 | 0.034 |
| cystic_artery | 0.293 | 0.317 | 0.272 | 0.057 |
| null_target | 0.235 | 0.193 | 0.300 | 0.117 |
| fluid | 0.188 | 0.233 | 0.157 | 0.026 |
| cystic_duct | 0.065 | 0.391 | 0.035 | 0.148 |
| abdominal_wall_cavity | 0.016 | 0.019 | 0.014 | 0.020 |
| cystic_plate | 0.000 | 0.000 | 0.000 | 0.010 |
| cystic_pedicle | 0.000 | 0.000 | 0.000 | 0.002 |
| blood_vessel | 0.000 | 0.000 | 0.000 | 0.005 |
| adhesion | 0.000 | 0.000 | 0.000 | 0.000 |
| peritoneum | 0.000 | 0.000 | 0.000 | 0.025 |
| gut | 0.000 | 0.000 | 0.000 | 0.001 |
| **Macro F1** | **0.207** | -- | -- | -- |
| **Macro accuracy** | 0.930 (baseline 0.928) | -- | -- | -- |

## Interpretation

**The same recipe that reached macro F1 0.512 for instrument (Day27) and
0.299 for verb-at-best (Day28) manages only 0.207 for target, and 6 of
15 classes (40%) are completely undetected.** This isn't a failure of
the technique -- it's what the technique's known limit looks like against
a harder version of the same problem. Target has more than double
instrument's class count (15 vs. 6) and several targets are rarer than
any instrument or verb seen so far: `cystic_pedicle` (0.2% test
prevalence), `gut` (0.1%), `adhesion` (0.0% in this particular test
split), `blood_vessel` (0.5%). With only 8 training videos, classes this
rare may have single-digit positive training examples, which no amount
of loss re-weighting or partial fine-tuning can manufacture more of --
exactly the ceiling Day27 already found for scissors (F1 0.101 even
after fine-tuning), now showing up for six targets at once instead of
one instrument.

**The targets that did work are the visually large, common anatomical
structures**: `gallbladder` (F1 0.836, the single most common target and
also large/visually distinctive), `liver` (0.594, also large), `omentum`
(0.466). `specimen_bag` (F1 0.408) is a notable case despite being rare
(3.4%): very high recall (0.907) with modest precision (0.263) -- a
plastic retrieval bag is visually unlike any tissue, so it's easy to
recognize whenever it's guessed, even though the aggressive `pos_weight`
(16.55, from its rarity) makes the model over-eager and produces some
false positives.

**`cystic_duct` is a specific, informative near-failure**: despite a
test prevalence (14.8%) higher than most other targets, its F1 (0.065)
is barely above the completely-undetected classes, driven by very low
recall (0.035) with moderate precision (0.391) -- when the model does
predict `cystic_duct`, it's usually right, but it almost never predicts
it. This is a plausible visual-similarity problem specific to the
"critical view of safety" anatomy (cystic duct, cystic artery, cystic
plate, cystic pedicle are all small, closely-clustered structures in the
same triangle of Calot, visually similar to each other and to
surrounding fat/connective tissue) rather than a pure rarity problem --
consistent with why this exact anatomy is clinically the hardest part of
the operation to identify correctly in the first place.

## Reflection

This day was deliberately scoped to be a single data point rather than
a new arc, and the result earns that choice: it doesn't reveal a new
technique-level lesson (fine-tuning and class weighting behave exactly
as Day26/27 would predict), it reveals a *scale* lesson -- the same
approach's effectiveness degrades as class count and worst-case rarity
both increase simultaneously. Instrument (6 classes, rarest ~2-3%
prevalence) responded very well; verb (10 classes, several under 3%)
responded partially; target (15 classes, several under 1%) responds
weakly for nearly half its classes. This is a fairly clean dose-response
relationship across the three label types this project touched, and it
reinforces -- for perhaps the clearest time yet -- that Day21's original
three fixes (more data, fine-tuning, class weighting) have real, but
bounded, power: they move the achievable ceiling, but the ceiling itself
still depends on how much data exists per class, which for the rarest
targets in an 8-video training set is very little.

## Conclusion

Target recognition reaches macro F1 0.207 with the best recipe developed
across Day26-27, well below instrument's 0.512 and verb's best combined
result of 0.299 (Day28) -- consistent with target simply being a harder
version of the same underlying problem (more classes, more of them
extremely rare) rather than requiring a different technique. Common,
visually distinctive anatomy (gallbladder, liver, specimen_bag) is
recognized reasonably well; the small, visually similar structures
around the triangle of Calot (cystic_duct, cystic_plate, cystic_artery,
cystic_pedicle) -- clinically the most important anatomy in this
operation to identify correctly -- are exactly the ones this pipeline
recognizes worst. This closes the instrument-verb-target arc (Day20-29):
the pattern found in Day21 (more data, better features, and
loss-weighting all help, but hit a floor set by data scarcity per class)
held for all three parts of the triplet, at three different scales of
difficulty.
