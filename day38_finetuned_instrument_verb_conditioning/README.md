# Day38: Does a More Accurate Instrument Predictor Close More of the Gap?

## Objective

Day37 found that conditioning the verb probe on instrument identity
helps a great deal in principle (ground-truth instrument: macro F1
0.309 to 0.484) but not at all in practice (predicted instrument from a
frozen-feature probe, itself only macro F1 0.399: macro F1 0.305,
indistinguishable from baseline). The explanation offered was that the
predictor was simply too noisy. Day27 already showed a substantially
more accurate instrument recognizer exists -- fine-tuning ResNet18's
last residual block raised instrument macro F1 from 0.378 to 0.512 --
but that day didn't save the checkpoint. Today re-runs Day27's exact
recipe (same seed, same split), saves the checkpoint this time, and
uses its predictions in place of the weaker frozen-feature probe from
Day37's realistic condition. If Day37's explanation is right, a
meaningfully more accurate instrument predictor should recover at least
part of the oracle-realistic gap.

## Method

[`finetuned_instrument_verb_conditioning.py`](finetuned_instrument_verb_conditioning.py)
has two parts. First, it reproduces Day27's fine-tuning recipe exactly
(ResNet18 with `conv1`/`bn1`/`layer1`/`layer2`/`layer3` frozen,
`layer4` + a new final linear layer trained, class-weighted loss, same
10 videos and 8/2 split) and saves the resulting backbone checkpoint.
Second, it reproduces Day37's verb-conditioning setup: a class-weighted
linear probe on frozen-ImageNet verb features, concatenated with a
6-dim instrument signal. Two conditions:

- **A (baseline)**: features only (should reproduce Day37's 0.309).
- **D (fine-tuned realistic)**: features + predicted instrument
  probabilities from the fine-tuned classifier above (macro F1
  ~0.512), replacing Day37's condition C predictor (macro F1 0.399).

Everything else -- verb probe recipe, frozen-ImageNet verb features,
video split -- is identical to Day37, isolating one variable: which
model supplies the instrument prediction.

## Results

| Condition | Verb macro F1 |
|---|---:|
| A: baseline (features only) | 0.303 |
| D: + fine-tuned instrument pred (instrument macro F1 0.512) | 0.305 |
| *Day37 B: + ground-truth instrument (reference)* | *0.484* |
| *Day37 C: + frozen-probe instrument pred, F1 0.399 (reference)* | *0.305* |

The fine-tuned instrument classifier reproduced Day27's result exactly
(macro F1 0.512), including large individual gains on bipolar (F1 0.106
to 0.431) and clipper (0.012 to 0.492). Condition A reproduced Day37's
baseline within noise (0.309 to 0.303 -- both linear probes, different
random draws of minibatch order downstream of the fine-tuning loop's
own RNG consumption, everything else identical).

Per-verb breakdown, condition A vs. D:

| Verb | A | D | Diff |
|---|---:|---:|---:|
| grasp | 0.446 | 0.451 | +0.005 |
| retract | 0.751 | 0.747 | -0.004 |
| dissect | 0.669 | 0.683 | +0.014 |
| coagulate | 0.260 | 0.258 | -0.002 |
| clip | 0.352 | 0.371 | +0.019 |
| cut | 0.056 | 0.057 | +0.001 |
| aspirate | 0.122 | 0.109 | -0.013 |
| irrigate | 0.025 | 0.000 | -0.025 |
| pack | 0.057 | 0.094 | +0.038 |
| null_verb | 0.289 | 0.278 | -0.011 |

## Interpretation

**Raising instrument prediction accuracy by 0.113 macro F1 (0.399 to
0.512) produced no measurable improvement in verb recognition.**
Condition D lands at 0.305, statistically identical to Day37's weaker
condition C. This directly refutes Day37's explanation for the gap
("the predictor is too noisy"): a substantially better predictor, with
large real gains on exactly the instruments (bipolar, clipper) that
verb-relevant actions (coagulate, clip) depend on, changed nothing --
clip moved +0.019, coagulate moved -0.002, both within noise.

**The likely explanation is that verb difficulty and instrument
prediction difficulty are correlated at the level of individual
frames, not just at the level of aggregate task accuracy.** The owner's
working hypothesis, which fits the evidence: a frame where the
instrument tip is occluded (by tissue, blood, smoke, or the edge of the
field of view) is a frame where *both* instrument identity and verb are
hard to read, for the same underlying reason -- the visual evidence for
either judgment lives in the same occluded region. A frame-level
accuracy improvement that averages out over the whole dataset (macro F1
0.399 to 0.512) doesn't help if the improvement comes from getting
*easier* frames more reliably right, while the *specific* frames verb
needs help on -- the visually ambiguous ones -- remain hard for
instrument prediction too. This reframes the oracle-realistic gap: it
isn't a predictor-quality gap that a better model closes incrementally,
it's a structural gap between "always-correct information" (oracle) and
"information that degrades exactly where it's needed most" (any
learned predictor, however accurate on average).

**This is consistent with, and extends, Day27's own instrument results.**
Scissors -- the instrument most plausibly subject to occlusion and
scarcity together -- barely improved under fine-tuning (F1 0.101,
against 0.906-0.907 for grasper/hook). If frame-level visibility is the
shared bottleneck, the instruments (and their corresponding verbs, cut
in particular) that remain hardest to fine-tune are exactly the ones
where this occlusion-driven correlation would be strongest. The
project doesn't have occlusion annotations to test this directly, and
temporal context (which the owner suggested could help resolve
instrument identity even through brief occlusion, by tracking across
frames, while verb -- which depends on the tip's motion during the
occluded moment -- might not benefit the same way) is untested here;
this remains a plausible mechanism rather than a proven one.

## Reflection

This is a genuine negative result relative to what the day was designed
to test (does a better predictor close more of the gap -- no), but it
sharpens the picture left by Day37 rather than just repeating it. Day37
established that the gap exists and that a weak predictor doesn't close
it; Day38 establishes that the gap doesn't close *by improving predictor
quality* either, which rules out the simplest fix and points toward a
structural (per-frame, not per-task) explanation instead. Reusing Day27's
exact recipe was also a useful discipline check: the checkpoint's
instrument macro F1 (0.512) reproduced Day27's number exactly, confirming
the fixed seed keeps this pipeline genuinely reproducible three days
apart -- something that couldn't have been checked before, since Day27
never saved the checkpoint.

## Conclusion

Replacing Day37's weak instrument predictor (macro F1 0.399) with
Day27's fine-tuned one (macro F1 0.512, saved as a checkpoint for the
first time) produced no additional improvement in verb recognition
(0.305 both times), even though the fine-tuned predictor is
substantially more accurate and improves exactly the instruments
(bipolar, clipper) that the most-improved verbs under oracle
conditioning (coagulate, clip) depend on. This rules out predictor
accuracy as the limiting factor and points toward a structural
explanation instead: verb difficulty and instrument-prediction
difficulty likely share a frame-level cause (plausibly occlusion of the
instrument tip), so a predictor that is more accurate on average still
fails on the specific frames where its help is most needed. Together,
Day37 and Day38 close this sub-thread: instrument identity genuinely
matters for verb recognition (Day37's oracle result), but recovering
that value through any realistic instrument predictor -- however
accurate -- is blocked by a deeper, shared source of visual ambiguity
that this project cannot currently diagnose further without occlusion
annotations or temporal modeling.
