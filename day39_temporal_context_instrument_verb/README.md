# Day39: Does Temporal Context Help Instrument or Verb More?

## Objective

Day38 found that a much more accurate instrument predictor (macro F1
0.512) recovered none of verb recognition's oracle-realistic gap,
ruling out predictor accuracy as the limiting factor and pointing to a
structural, frame-level explanation instead. The owner proposed a
specific, testable mechanism: a frame where the instrument tip is
occluded is hard for *both* instrument and verb recognition, but
temporal context should let a model recover instrument identity by
tracking across nearby frames, while verb -- which depends on the
tip's motion during the occluded instant itself -- might not benefit
the same way. Today tests this directly: does adding a small window of
neighboring frames help instrument recognition more than verb
recognition, as the occlusion-and-tracking hypothesis predicts?

## Method

[`temporal_context_instrument_verb.py`](temporal_context_instrument_verb.py)
uses the same 10 videos, video-level 8/2 split, frozen ImageNet
ResNet18 features, and class-weighted linear probe recipe used
throughout. CholecT50 frames are sampled at a fixed 1-second interval
(frame IDs increment by 1 within a video, confirmed directly from the
label files), so a 3-frame window (t-1, t, t+1) spans roughly 2 seconds
of real time. For each video's frames that have both neighbors present
(dropping each video's first and last frame), two probe inputs are
compared on the identical frame subset:

- **Single-frame**: 512-dim features from frame t only.
- **3-frame window**: 1536-dim features from t-1, t, t+1 concatenated.

Both instrument (6-way) and verb (10-way) are evaluated this way, so
the comparison isolates one question: which task benefits more from
the same added temporal context, on the exact same frames (excluding
edge frames without a full window, so any difference isn't from a
different data subset).

## Results

| Task | Single-frame | 3-frame window | Diff |
|---|---:|---:|---:|
| instrument | 0.388 | 0.403 | +0.015 |
| verb | 0.292 | 0.332 | **+0.040** |

Per-class breakdown (single -> windowed F1):

| Instrument | Single | Windowed | Diff |
|---|---:|---:|---:|
| grasper | 0.864 | 0.862 | -0.002 |
| bipolar | 0.193 | 0.192 | -0.001 |
| hook | 0.738 | 0.724 | -0.014 |
| scissors | 0.060 | 0.071 | +0.011 |
| clipper | 0.336 | **0.407** | +0.071 |
| irrigator | 0.136 | 0.161 | +0.025 |

| Verb | Single | Windowed | Diff |
|---|---:|---:|---:|
| grasp | 0.452 | 0.433 | -0.018 |
| retract | 0.742 | 0.749 | +0.007 |
| dissect | 0.647 | 0.645 | -0.002 |
| coagulate | 0.195 | **0.330** | +0.135 |
| clip | 0.317 | **0.420** | +0.103 |
| cut | 0.045 | 0.069 | +0.023 |
| aspirate | 0.116 | 0.113 | -0.002 |
| irrigate | 0.024 | 0.033 | +0.008 |
| pack | 0.094 | **0.222** | +0.128 |
| null_verb | 0.289 | 0.302 | +0.013 |

Precision/recall for the classes with the largest gains:

| Class | Single P / R | Windowed P / R |
|---|---|---|
| clipper (instr.) | 0.263 / 0.464 | 0.565 / 0.318 |
| coagulate (verb) | 0.284 / 0.148 | 0.418 / 0.272 |
| clip (verb) | 0.222 / 0.556 | 0.509 / 0.358 |
| pack (verb) | 0.051 / 0.600 | 0.250 / 0.200 |

## Interpretation

**The occlusion-and-tracking hypothesis is not supported: verb
benefited from temporal context roughly 2.5x more than instrument
(+0.040 vs. +0.015), the opposite of what that hypothesis predicts.**
Common, high-prevalence classes (grasper, hook, grasp, dissect) were
flat or slightly worse with the wider window; nearly all of the gain is
concentrated in a handful of rare, short-duration classes -- coagulate,
clip, pack on the verb side, clipper (and modestly scissors,
irrigator) on the instrument side.

**The owner's alternative mechanism -- low single-frame confidence on
rare classes causes the model to default to a safe, common guess, and
temporal context restores the confidence to commit to the rare class
correctly -- doesn't match the precision/recall pattern, though the
underlying intuition (rare, transient classes are where the gain
concentrates) is confirmed.** If that mechanism were right, recall
should rise with the wider window (the model becoming more willing to
predict the rare class). Instead, for every large-gain class, recall
*falls* while precision rises sharply: clip's precision more than
doubles (0.222 to 0.509) while recall drops (0.556 to 0.358); pack's
precision jumps nearly 5x (0.051 to 0.250) while recall falls by two
thirds. This is the opposite signature -- the single-frame model was
already eager to guess these classes (a known effect of the
class-weighted loss, Day26), often wrongly, on frames that briefly
*resemble* a coagulate/clip/pack moment without being one; the wider
window lets the probe correctly reject many of those false positives,
at some cost in recall on genuinely ambiguous ones. Coagulate is the
partial exception (both precision and recall rise), suggesting its
single-frame ambiguity runs in both directions rather than being purely
a false-positive problem.

**This is closer to Day27's "genuinely better features" signature than
Day26's "willingness" signature, applied to time instead of to the
loss function.** Day27 found that fine-tuning the backbone (letting
features adapt) fixed instrument classification's precision/recall
trade-off directly, rather than only shifting where the decision
threshold sits. Here, adding two seconds of temporal context does
something structurally similar for verb's rarest, most transient
classes: it doesn't make the model more willing to guess coagulate/
clip/pack, it makes it more able to tell a real instance apart from a
single ambiguous frame that merely looks similar.

**Why verb benefits more than instrument remains only partly
explained.** One plausible factor: coagulate, clip, and pack are all
*brief, punctual* actions (a clip closes in roughly a second; a
coagulation burst is short) whose single-frame appearance can closely
resemble an adjacent idle or transitional moment -- exactly the kind
of ambiguity that a couple of neighboring frames, showing the motion
building up or resolving, should help resolve. Instruments, by
contrast, tend to remain visible (and identifiable) across several
consecutive seconds once introduced, so a single frame already carries
most of the identifying information a 2-second window would add --
there is less ambiguity left for temporal context to remove. This is
consistent with the pattern but not confirmed by anything beyond it;
the original occlusion hypothesis is set aside as not supported by this
test.

## Reflection

This closes the Day37-39 sub-thread with a genuinely different
mechanism than any of the three days started with. Day37 showed
instrument identity matters for verb in principle; Day38 showed
predictor accuracy alone can't unlock that value; Day39 shows that a
much cheaper intervention than either -- two seconds of raw temporal
context, no retraining of any recognizer -- helps verb specifically,
for reasons that turn out to be about resolving single-frame visual
ambiguity on transient actions, not about instrument tracking through
occlusion. The owner's proposed mechanism, checked directly against
precision/recall rather than macro F1 alone, turned out to be inverted
from what the data shows -- a useful reminder that a plausible
narrative and the actual failure mode of a model are two different
things to verify separately.

One number is worth carrying into Day40's retrospective: this simple,
untrained 3-frame concatenation reaches verb macro F1 0.332 --
higher than any of Day36's three SSL-pretrained backbones on the same
task (0.304-0.309, all within noise of plain frozen ImageNet). An
explicit temporal signal, added with zero training, outperformed
several days of self-supervised pretraining aimed at learning useful
representations from unlabeled video. That contrast is a sharper
closing point for the SSL arc's implications than anything available
when Day35/36 were written.

## Conclusion

Temporal context helps verb recognition roughly 2.5x more than
instrument recognition (macro F1 +0.040 vs. +0.015 from a 3-frame,
~2-second window), refuting the hypothesis that instrument-tip
occlusion is the shared bottleneck between the two tasks. The actual
mechanism, visible in precision/recall, is that a handful of rare,
brief verbs (coagulate, clip, pack) generate single-frame false
positives that temporal context lets the model correctly reject --
closer to Day27's "better features" pattern than Day26's "willingness"
pattern. This gives verb recognition's single-frame information limit
(first diagnosed in Day22) a concrete, working fix that SSL adaptation
(Day31-36) never found, using nothing more than neighboring frames
already present in the dataset.
