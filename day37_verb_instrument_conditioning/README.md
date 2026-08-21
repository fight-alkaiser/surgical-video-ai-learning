# Day37: Testing the Verb Recognition Architecture Gap

## Objective

Day22 diagnosed verb recognition's low ceiling as split between two
causes: a single-frame information limit (grasp vs. retract is often
genuinely ambiguous in one still image) and an architecture gap -- the
model was never given instrument identity, even though verb meaning is
instrument-dependent (e.g. "cut" only makes sense for scissors, "clip"
only for the clipper). Day31-36 showed neither contrastive nor
temporal-order SSL adaptation moves verb's macro F1 at all (0.304-0.309
across three backbones), which is consistent with the bottleneck not
being feature quality -- but doesn't test the architecture-gap half of
the diagnosis directly. Today does: does giving the verb probe access
to instrument identity close any of the gap?

## Method

[`verb_instrument_conditioning.py`](verb_instrument_conditioning.py) uses
the same 10 videos, video-level 8/2 split, frozen ImageNet ResNet18
features, and class-weighted linear probe recipe (Day26) used
throughout the project. SSL backbones are excluded here since Day36
already ruled them out for verb, keeping this test isolated to one
variable: instrument conditioning. Three conditions:

- **A (baseline)**: 512-dim frozen features only.
- **B (oracle)**: features concatenated with the **ground-truth**
  instrument one-hot label (6-dim) -- an upper bound on how much
  instrument identity could help in principle.
- **C (realistic)**: features concatenated with **predicted**
  instrument probabilities (6-dim) from a separately trained
  class-weighted instrument probe (same recipe as Day26) -- a
  realistic pipeline where instrument itself must be inferred, not
  read from the label file.

## Results

| Condition | Verb macro F1 |
|---|---:|
| A: baseline (features only) | 0.309 |
| B: oracle (+ ground-truth instrument) | **0.484** |
| C: realistic (+ predicted instrument) | 0.305 |

(Instrument probe itself, trained for condition C: macro F1 0.399,
consistent with Day26's 0.378 under a different split.)

Per-verb breakdown (A / B / C):

| Verb | A | B | C |
|---|---:|---:|---:|
| grasp | 0.454 | 0.526 | 0.432 |
| retract | 0.736 | 0.822 | 0.751 |
| dissect | 0.701 | **0.893** | 0.689 |
| coagulate | 0.287 | **0.616** | 0.270 |
| clip | 0.351 | **0.701** | 0.356 |
| cut | 0.061 | **0.444** | 0.051 |
| aspirate | 0.122 | **0.399** | 0.117 |
| irrigate | 0.029 | 0.028 | 0.026 |
| pack | 0.092 | 0.098 | 0.061 |
| null_verb | 0.257 | 0.311 | 0.296 |

## Interpretation

**The architecture-gap hypothesis is correct in principle, but not
exploitable with this project's instrument recognizer.** Condition B's
large jump (+0.175 macro F1) confirms that verb meaning is strongly
tied to instrument identity: the biggest gains land exactly on the
verbs that are near-deterministic given the instrument -- dissect
(0.701 to 0.893), clip (0.351 to 0.701), coagulate (0.287 to 0.616),
cut (0.061 to 0.444), aspirate (0.122 to 0.399). These are verbs each
performed by essentially one instrument in CholecT50's vocabulary
(scissors cut, clipper clips, hook/bipolar coagulate), so knowing the
instrument almost hands the model the verb. Verbs shared across
multiple instruments (grasp, retract) improved too, but far less
(+0.072, +0.086) -- consistent with instrument identity being only
partially informative there.

**Condition C shows this gain is entirely gated on instrument-recognition
accuracy, which this project doesn't have.** The realistic condition is
statistically indistinguishable from baseline (0.305 vs. 0.309), and
per-verb it's flat or slightly worse across the board -- including on
the exact verbs that gained the most under oracle conditioning
(dissect 0.701 to 0.689, clip 0.351 to 0.356, cut 0.061 to 0.051). The
instrument probe supplying this signal only reaches macro F1 0.399
(Day26's known ceiling for this feature/split combination) -- its
errors are correlated with the same visual ambiguity that made verb
hard in the first place (a probe unsure whether it's seeing bipolar or
hook is unsure for the same reason a verb model would be), so the
6-dim instrument-probability vector it hands to the verb probe carries
mostly noise rather than the clean signal the ground-truth label
provided.

**This refines rather than overturns Day22's diagnosis.** Verb's
difficulty is not simply "architecture gap" or "information limit" as
two independent, separately-fixable causes -- it's a dependency chain:
closing the architecture gap requires an instrument recognizer accurate
enough to be useful, and this project's best instrument recognizer
(Day26/27, macro F1 0.378-0.399) isn't there yet, particularly for the
rare instruments (bipolar, scissors, clipper) whose confusions are
exactly what would need to be resolved to unlock the corresponding
verbs (coagulate, cut, clip).

## Reflection

This is a cleaner result than most of the SSL-arc days: the oracle
condition validates the hypothesis unambiguously, and the realistic
condition pinpoints exactly why it doesn't yet pay off in practice,
rather than leaving an open question. It also connects two previously
separate threads in the project -- Day21/26's rare-instrument problem
and Day22's verb ceiling turn out to be the same underlying bottleneck
viewed from two tasks. Improving rare-instrument recognition (more
data, or a better feature extractor specifically for bipolar/scissors/
clipper -- not more generic visual features, which Day31-36 already
showed doesn't help) is now a concretely motivated next step, if the
project wants to pursue closing this gap rather than moving to new
material.

## Conclusion

Instrument identity is a real, substantial source of verb-recognition
difficulty (oracle conditioning: macro F1 0.309 to 0.484, concentrated
on instrument-specific verbs like clip, cut, coagulate, dissect), which
confirms the architecture-gap half of Day22's original diagnosis. But
this project's instrument recognizer isn't accurate enough to realize
that gain in practice -- feeding it predicted instrument probabilities
instead of ground truth recovers none of the improvement (macro F1
0.305, indistinguishable from baseline). The two problems are coupled:
fixing verb recognition via instrument conditioning is gated on first
fixing rare-instrument recognition, which SSL adaptation (Day31-36) has
already been shown not to solve.
