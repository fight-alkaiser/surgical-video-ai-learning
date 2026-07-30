# Day28: Fine-Tuned Instrument-Conditioned Verb Recognition

## Objective

Day24 conditioned verb prediction on Day21's *predicted* instrument
probabilities and reached only F1 0.241 -- far short of Day23's oracle
ceiling (0.388) -- because Day21's frozen-feature instrument classifier
was itself unreliable for exactly the rare instruments (clipper F1
0.012, bipolar 0.106) whose verbs stood to gain the most. Day27 then
showed fine-tuning fixes that unreliability directly (clipper F1 0.012
-> 0.492, bipolar 0.106 -> 0.431). Today combines both results to test
the obvious next question: if verb prediction is conditioned on Day27's
much-improved instrument classifier instead of Day21's, does more of
Day23's oracle ceiling become reachable in practice?

## Method

[`finetuned_instrument_conditioned_verb.py`](finetuned_instrument_conditioned_verb.py)
runs three stages on the same 10 videos and video-level 8/2 split as
every prior instrument/verb day: (1) fine-tune ResNet18's `layer4` for
instrument recognition with Day26's class-weighted loss, identical to
Day27; (2) freeze the now-improved backbone and cache its features for
every frame (Day24's shortcut, now applied to better features); (3)
train a verb classifier on [cached fine-tuned features + the fine-tuned
classifier's *predicted* instrument probabilities], with a plain,
unweighted verb loss matching Day22/24 exactly, so the comparison
isolates feature quality rather than mixing in a second change.

## Results

**Instrument classifier reproduced Day27 exactly** (0.932 mean per-class
test accuracy, vs. Day27's 0.932) -- confirms the fine-tuning setup is
stable and reproducible.

**Verb classifier, conditioned on the fine-tuned classifier's predictions:**

| Verb | Day22 F1 (frozen, no cond.) | Day24 F1 (frozen, predicted cond.) | Day28 F1 (fine-tuned, predicted cond.) | Day28 precision | Day28 recall |
|---|---:|---:|---:|---:|---:|
| grasp | 0.434 | 0.402 | **0.467** | 0.568 | 0.397 |
| retract | 0.692 | 0.744 | **0.772** | 0.709 | 0.846 |
| dissect | 0.652 | 0.663 | **0.723** | 0.846 | 0.631 |
| coagulate | 0.052 | 0.023 | **0.369** | 0.833 | 0.237 |
| clip | 0.000 | 0.119 | **0.377** | 0.656 | 0.265 |
| cut | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| aspirate | 0.045 | 0.169 | 0.000 | 0.000 | 0.000 |
| irrigate | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| pack | 0.000 | 0.250 | 0.250 | 0.333 | 0.200 |
| null_verb | 0.042 | 0.044 | 0.029 | 0.400 | 0.015 |
| **Macro F1** | **0.192** | **0.241** | **0.299** | -- | -- |

For reference, Day23's oracle (true-instrument) conditioning reached F1
0.388. Day24 closed about a quarter of the gap between Day22's floor and
that ceiling (0.049 of 0.196); Day28 closes about 55% of it (0.107 of
0.196) -- more than double Day24's share, using the exact same
conditioning architecture with only the instrument classifier's own
quality changed.

## Interpretation

**The central hypothesis holds clearly.** Verbs tied to instruments that
Day27's fine-tuning fixed well (bipolar, clipper) show the largest
gains: `coagulate` (needs bipolar) jumped from F1 0.023 to 0.369;
`clip` (needs clipper) from 0.119 to 0.377. Both now show high precision
(0.833, 0.656) with modest recall -- the model has become confident and
mostly correct when it does predict these verbs, a real, qualitatively
different result from Day24's noisier signal.

**Verbs tied to instruments fine-tuning could not fully fix stayed at
zero or regressed.** `cut` (needs scissors, Day27's F1 only 0.101 even
after fine-tuning) remained undetectable, consistent with Day27's own
conclusion that scissors' scarcity (~500 total instances) is a data
problem fine-tuning can't fully solve. `aspirate` (needs irrigator,
Day27's F1 a moderate 0.236) is the one puzzling regression -- it *fell*
from Day24's 0.169 to 0.000, despite irrigator detection itself
improving between Day21 and Day27. Irrigator's own verb distribution is
the least concentrated of any instrument (Day22 found aspirate is only
66.3% of irrigator's actions, with meaningful shares of irrigate,
retract, and null_verb too), so a modestly-improved but still imperfect
instrument signal may not be enough to reliably tip the balance for
irrigator's own internally ambiguous verb mix -- and with irrigator's
rare verbs, a single run's outcome for a class with this few positive
examples is also subject to real training-to-training variance that
this project has not attempted to average out.

## Reflection

This day is a satisfying, mostly-clean confirmation that the two
Track-2 fixes (Day26's re-weighting, Day27's fine-tuning) compound
usefully when carried into the Track-1/Track-3 question they were never
directly aimed at (verb conditioning). It also reinforces, one more
time, that "instrument detection got better" does not uniformly
translate to "every verb tied to that instrument got better" --
coagulate and clip benefited enormously, while aspirate got worse
despite irrigator's own detection improving, because verb-level success
depends on both the instrument signal's quality *and* how concentrated
that instrument's own verb distribution is. A rare instrument with one
dominant verb (clipper -> clip, 94.9%) is a much easier target for this
kind of conditioning to help than a rare instrument with a split verb
profile (irrigator's aspirate/irrigate/retract/null_verb mix).

## Conclusion

Conditioning verb prediction on a fine-tuned (rather than frozen)
instrument classifier's predictions raises macro F1 from 0.241 (Day24)
to 0.299 -- recovering roughly 55% of the theoretical gap to Day23's
oracle ceiling (0.388), more than double Day24's ~25%. The gains are
concentrated exactly where Day27's fine-tuning most improved instrument
detection (bipolar -> coagulate, clipper -> clip), while verbs tied to
instruments fine-tuning couldn't fully fix (scissors -> cut) or whose own
action repertoire is inherently split across several verbs (irrigator ->
aspirate) remain weak or noisy. This closes the loop opened in Day22:
most of the tool-specific verb recognition problem traces back to
instrument recognition quality, and improving that upstream signal
(Day26-27) is the more effective lever than anything tried directly on
the verb classifier itself (Day22-25).
