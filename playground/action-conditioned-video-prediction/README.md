# Action-Conditioned Video Prediction (toy)

Day 61-62 and Day78-86 of the "surgeon learning surgical video AI" series. This is not
Cosmos-H-Surgical-Simulator, and it does not run it -- that model needs
about 65GB of GPU memory, far beyond what this Mac mini (Apple Silicon,
no CUDA) can do. This is a small model written from scratch, inspired by
reading Cosmos's source code (Day58-60 of the same series), that copies
one specific idea from it: how an action vector conditions video
generation. The data is real (Open-H); the model and weights are not
Cosmos's.

## What this is

Cosmos-H-Surgical-Simulator embeds a robot action vector with an MLP and
adds it into the diffusion transformer's timestep embedding, rather than
treating it like a text prompt (which gets cross-attended per-token).
This project reproduces that same idea -- inject the action as a
scale/shift on an image feature map -- in a much smaller, deterministic
next-frame predictor:

- up to 20 episodes, ~7700 frames total, 64x64 RGB (vs. Cosmos's full
  video diffusion pipeline trained on 32 datasets across 9 robot
  embodiments)
- deterministic pixel regression (vs. a diffusion/flow-matching
  objective)
- CPU/MPS on a Mac mini, not an A100/H100 cluster

This is a mechanism demo, not a benchmark or a claim of reproducing
Cosmos's actual results.

## Data

[Open-H Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment)
(nvidia, CC-BY-4.0), `Surgical/hamlyn/peg_transfer/episode_000000..000019`
(dVRK, peg transfer task). `prepare_data.py` extracts frames from each
episode video and pairs them with the 16-dim action vector (left/right
arm xyz + quaternion + gripper), matching the LeRobot-format schema
described in Cosmos-H-Surgical-Simulator's `README_ACTION_SPACE.md`.

## Model

`model.py`: a small conv encoder (64->32->16->8, channels 32->64->128)
feeds a bottleneck feature map. The action vector goes through an MLP
that outputs a per-channel scale and shift, applied as
`x = x * (1 + scale) + shift` -- the same shape as Cosmos's
`action_embedder_B_D` / `action_embedder_B_3D` pair, just applied to a
CNN feature map instead of a transformer's conditioning input. A conv
decoder then predicts a bounded delta, added back onto the input frame
(`pred = clamp(frame + delta, 0, 1)`) rather than reconstructing the
whole image from scratch -- see "what changed" below for why.

## Result (Day 61) -- six attempts, still short of a real win

All numbers are validation MSE against a trivial "copy the last frame
forward" baseline, expressed as a ratio (1.0 = tie).

| attempt | model / baseline |
|---|---|
| 1 episode, predict 1 frame ahead | 4.80x worse |
| 1 episode, predict 10 frames ahead | 1.19x worse |
| 1 episode, predict 15 frames ahead | 1.03x worse |
| 20 episodes (16 train / 4 held-out val), 10 frames ahead | 1.51x worse |
| same, + motion-saliency-weighted loss | 1.48x worse |
| same, + residual/delta prediction | 1.00x (tied) |

**Why the model kept losing to "do nothing":** the encoder-decoder
reconstructs the whole frame from scratch, so it pays a small blur cost
on the ~95% of the frame that never moves (static background), while the
copy-baseline is pixel-perfect there by construction. At 30fps the real
motion is small enough that this blur cost outweighs whatever the model
gains from correctly using the action. Longer horizons (more real motion
per step) narrowed the gap; a motion-saliency-weighted loss barely
helped; switching to residual/delta prediction (predict *change*, not
the whole frame) fixed the blur-cost problem outright -- an untrained
model with this design starts out exactly at the baseline.

**What the tie actually means:** qualitatively
(`outputs/qualitative_comparison_h10_residual.png`), the residual
model's predictions are visually almost identical to the input frame --
it converged to predicting a near-zero delta, i.e. it learned to copy
safely rather than to use the action signal in a way that generalizes to
held-out episodes. The conditioning mechanism runs end-to-end and trains
stably; getting it to actually pick up the action signal (rather than
collapsing to the safe no-op) is still unsolved.

## Result (Day 62) -- found and fixed a design flaw, then switched representations

**The design flaw:** the residual model's action modulation (`x = x *
(1+scale) + shift`) was applied to the bottleneck feature map, right
before `dec1`'s `GroupNorm`. A shuffle test (feed the same frame with its
real action vs. a random other episode's action) showed the difference
survived the modulation step (mean abs diff 0.28) but was exactly zero
after the next GroupNorm-containing block -- the normalization was
re-standardizing the activations and erasing the modulation, the mirror
image of how real AdaLN applies scale/shift *after* normalization, not
before. Moving the modulation to the very last feature map (after `dec3`,
right before `out_conv`, with no norm layer left downstream to undo it)
fixed this: the shuffle test now shows a real, nonzero sensitivity
(0.0030). Accuracy still didn't improve, though -- MSE with the real
action (0.000859), a shuffled action (0.000860), and no action at all
(0.000854) were statistically indistinguishable. The mechanism now
works; the model still hasn't learned anything useful to put through it.

**Switching to a JEPA-style latent comparison:** `jepa_model.py` /
`jepa_train.py` drop pixel reconstruction entirely -- an online encoder
maps frame_t to a latent vector, an EMA/no-gradient target encoder maps
frame_{t+H} to a target latent, and a small predictor tries to match the
two given the action. First attempt collapsed within 10 epochs (loss ->
~0, batch std of the latent -> ~0.002): the encoder learned to map every
frame to nearly the same point, which trivially "solves" a
normalized-MSE comparison without learning anything. Adding a
VICReg-style variance penalty (`variance_loss` in `jepa_model.py`,
penalizing any embedding dimension whose batch std drops below 1.0)
fixed the collapse (batch std stabilized around 1.7-2.2) and, for the
first time across both days, produced a real difference between
conditions:

| condition | latent-space error |
|---|---|
| real action | 0.0006 |
| action zeroed out | 0.0034 |
| shuffled (wrong) action | 0.0107 |
| copy z_t forward (do-nothing baseline) | 0.0006 |

Real action ties the do-nothing baseline, but a wrong action is ~18x
worse than the real one and zeroing it out is ~5.7x worse -- something
pixel-space comparison never showed, not even once, across six attempts
over two days.

**Caveat on where this goes next:** Cosmos-H-Dreams's actual purpose is
real-time, realistic video generation (for a person to watch, or drive
through VR) -- a JEPA-style latent is deliberately not meant to decode
back into a real image, so this may be the wrong representation for
*that* specific goal even if it's a better one for policy learning. See
the Day62 write-up for the fuller reasoning (and the I-JEPA note this
project's daily log links back to).

## Result (Day 78) -- replacing the JEPA predictor with Conditional Flow Matching

**Why revisit this.** Day62 ended on an unresolved observation: the JEPA
predictor's real-action loss tied the do-nothing baseline exactly. That is
the signature of a deterministic regressor collapsing to the conditional
*mean* of the future -- if the true next latent is not perfectly determined
by a 16-dim action vector (instrument jitter, camera noise), MSE's optimal
output is an average over possible futures, which looks like "barely change
anything." Day64-77 worked through DDPM, Score-based SDEs, Flow Matching,
Rectified Flow, DIAMOND (EDM), Self Forcing and OT-CFM -- all of them ways
to replace exactly this kind of point-estimate regression with a generative
objective. `cfm_model.py` / `cfm_train.py` apply that here: same
online/target-encoder pair and VICReg anti-collapse term as `jepa_model.py`,
but the predictor is replaced with a velocity field trained with the
standard Conditional Flow Matching loss (linear interpolant between a
source `x0` and the target latent `x1 = z_{t+H}`, regressing the constant
velocity `x1 - x0`).

**First attempt (`--source noise`, standard CFM, `x0 ~ N(0,I)`) --
evaluated the wrong way.** Comparing the *average* of several generated
samples to the one observed `z_{t+H}` showed real action still losing to
the do-nothing baseline (0.098 vs. 0.067) and even losing to a zeroed-out
action (0.083) -- CFM looked like it hadn't helped at all.

**Second attempt (`--source zt`, Rectified-Flow-style residual, `x0 = z_t`)
-- looked promising, then fell apart.** The first run (seed 0) showed real
action beating the do-nothing baseline for the first time in this project
(0.0012 vs. 0.0017). Repeating with different seeds killed that story:
seed 1 reversed the ordering, seed 2 collapsed completely (the velocity
network learned to output ~0 regardless of action -- the same
"safe no-op" failure as the Day61 pixel-space residual model, just in a
different parameterization), and a lower-LR run failed to converge to a
comparable scale at all. The seed-0 "win" was noise, not signal.

**The evaluation itself was the wrong tool.** Comparing a generated sample
(or worse, an average of several samples) to the one observed continuation
penalizes a generative model for doing its job: if the true future given
`(z_t, action)` is genuinely multimodal, a correctly-calibrated sampler
will land away from that one realization even when it's working, while a
plain deterministic regressor that always outputs the conditional mean
wins this comparison by construction -- exactly what CFM was supposed to
move away from. Two metrics avoid this: **paired_loss** (the training CFM
loss itself, evaluated against the real observed transition with each
candidate action -- a likelihood-style score that never converts the model
into a point estimate) and **best-of-N** (score the *closest* of several
samples to the target, rather than their average, asking whether the truth
is inside what the model considers plausible rather than penalizing
diversity).

**Re-run with the corrected evaluation (`--source noise`, 3 seeds) --
a stable, reproducible result at last.**

| seed | real | shuffled | zero |
|---|---|---|---|
| 0 | 0.4050 | 0.4077 | 0.3385 |
| 1 | 0.3708 | 0.4000 | 0.3174 |
| 2 | 0.3663 | 0.4105 | 0.3008 |

(`paired_loss`, lower is better.) `zero < real < shuffled` held across all
three seeds -- the first fully reproducible ordering in this project. The
`best_of_n_error` metric agreed in direction (mostly) but never approached
the do-nothing latent-copy baseline either.

**What this means.** `real < shuffled` shows the action pathway is reading
*something* real -- a wrong action explains the observed transition worse
than the right one. But `zero < real` shows that using the action at all,
even correctly, currently makes the prediction worse than ignoring it
entirely. The action pathway carries a directionally correct signal that
is still, net, harmful rather than helpful at this data scale (16 episodes,
~6200 training pairs) and this horizon (H=10 frames). Switching the
objective from deterministic regression to Conditional Flow Matching did
not fix the Day62 problem -- but it did finally produce an evaluation
methodology stable enough to say that with some confidence, instead of a
single-run result that looked like a win and evaporated under a seed
sweep.

## Result (Day 79) -- giving the action pathway an explicit opt-out gate

Day78 left two candidate explanations for "the action pathway reads real
signal but is net harmful": (1) not enough data for the network to learn
to weight the action correctly, or (2) an architecture problem -- the
plain `concat([z_s, t_emb, z_t, action]) -> MLP` design mixes action into
every hidden unit from the first layer, with no dedicated "how much should
I trust this" pathway, even though in principle the MLP could learn to
ignore it.

`GatedVelocityPredictor` (in `cfm_model.py`) tests (2) directly: the action
is embedded separately and injected additively through a learned sigmoid
gate, conditioned on the current hidden state and the action itself. The
gate's bias is initialized so training starts with the action mostly
gated off (close to the Day78 "zero action" regime, which was the
best-performing condition) -- the network has to actively learn to open
the gate where the action is worth using, rather than starting fully mixed
in and having to learn to suppress it.

| seed | real | shuffled | zero | mean_gate (real / shuffled / zero) |
|---|---|---|---|---|
| 0 | 0.2809 | 0.2780 | **0.2735** | 0.100 / 0.100 / 0.018 |
| 1 | 0.3161 | 0.3224 | **0.3132** | 0.105 / 0.103 / 0.013 |
| 2 | 0.2847 | 0.2913 | **0.2488** | 0.127 / 0.126 / 0.022 |

(`paired_loss`, lower is better, source=noise, 3 seeds, 100 epochs.)

**The gate does learn something real:** it opens roughly 5-10x wider for a
real or shuffled (nonzero-deviation) action than for a zeroed-out one,
consistently across all three seeds -- so the network can and does tell
"an action was given" from "no action was given."

**But it doesn't fix the problem.** `zero` remains the best-performing
condition in all three seeds, by a similar margin to the ungated Day78
run. Giving the model an explicit, cheap way to ignore the action when it
doesn't help did not make it stop being net-harmful when used. Seed 0 also
saw `real` and `shuffled` swap order (0.2809 vs 0.2780) -- something that
never happened in three ungated runs -- suggesting the extra gating
parameters added a bit of instability rather than removing it.

**What this means.** This weighs against the architecture-limitation
explanation and toward the data-scale explanation: the network isn't
being forced into using the action against its own judgment by the
`concat`+MLP design; giving it an easy way out doesn't change what it
concludes is optimal. At 16 training episodes (~6200 pairs), the action
signal available to learn from may simply not be reliable enough to be
worth using, independent of how it's architecturally fused in.

## Result (Day 80) -- testing the data-scale hypothesis directly

Day79 narrowed the "action pathway is net-harmful" problem down to two
explanations and weighed against the architecture one. The remaining
candidate: 16 training episodes (~6200 pairs) may simply not be enough
for the model to learn to weight the action correctly, independent of
how it's wired in.

Pulled 80 more episodes (episode_000020-000099) from the same Open-H
peg-transfer split used for the original 20 -- same task, same camera,
same format -- for 100 episodes total (~39,200 frames, ~36,900 training
pairs at H=10, up from ~6,200). The held-out val episodes were pinned to
the original four (episode_000002/4/6/19) so the comparison isn't
confounded by a different test set. Reran the Day78/79 experiment (3
seeds, both the plain and gated predictor) on this larger dataset.

| | 20 episodes (real / zero / shuffled) | 100 episodes (real / zero / shuffled) |
|---|---|---|
| ungated | 0.381 / 0.319 / 0.406 | 0.371 / 0.376 / 0.390 |
| gated | 0.294 / 0.279 / 0.297 | 0.269 / 0.282 / 0.278 |

(`paired_loss`, mean across 3 seeds, lower is better.)

**In all six runs (3 seeds x {ungated, gated}), `real` now beats `zero`.**
At 20 episodes `zero` was the best condition every time; at 100 episodes
`real` is. Nothing about the model or the training code changed --
only the amount of data. This is the first evidence in this project that
the action pathway can be net-beneficial rather than net-harmful, and it
lines up with the Day79 prediction: giving the model an opt-out gate
didn't fix the problem, but more data did.

One caveat: `best_of_n_error` (the sampling-based metric, ungated runs
only) didn't fully agree -- `zero` still edged out `real` there by a
small margin in all three seeds, even though `paired_loss` favored
`real`. `paired_loss` doesn't require generating a sample at all (see
below); `best_of_n_error` does, and is a noisier estimate over only 8
draws. Given `paired_loss` is the more direct measurement and agreed
across all 6 runs, it's the more trustworthy of the two here, but the
disagreement itself is worth flagging rather than picking whichever
number tells the cleaner story.

**A note on how `paired_loss` actually works**, since "compare a
generated frame to the real one" is an easy but inaccurate mental model
for it. The CFM predictor doesn't output a next-frame guess directly --
it outputs a *velocity*: a direction to move, step by step, from a
starting point toward the target latent, the way turn-by-turn GPS
directions guide a route rather than teleporting you to the
destination. Because the evaluation video is real footage, the true
next-frame latent (`x1`) is always known. `paired_loss` feeds the model
different claims about the action (real / shuffled / zero) against that
same known `x1`, and checks how well the resulting direction points
toward it -- reusing the training objective itself as the score, rather
than generating a sample and comparing it after the fact. That's also
why it doesn't have the "penalizes correct diversity" problem discussed
in Day78: it's not scoring a generated guess, just how well a given
piece of conditioning explains a real, known transition.

## Result (Day 81) -- is the paired_loss / best_of_n_error gap just integration error?

Day80's 100-episode result had a loose end: `real` beats `zero` on
`paired_loss`, but loses to it on `best_of_n_error`. One candidate
explanation was numerical, not semantic: `best_of_n_error` requires
integrating the ODE from noise with a 16-step Euler solver, and Euler is
first-order -- its error depends on how *curved* the velocity field's
path is, not just how accurate the field is pointwise. If `real`
action's paths are less straight than `zero`'s, a coarse solver would
penalize `real` specifically, independent of which condition the model
actually explains better.

Tested this directly against the Day80 checkpoints (`cfm_eval_steps.py`,
no retraining -- just re-running the sampling eval at more steps):

| steps | seed0 real / zero | seed1 real / zero | seed2 real / zero |
|---|---|---|---|
| 16  | 0.0859 / 0.0844 | 0.1388 / 0.1369 | 0.0880 / 0.0861 |
| 32  | 0.0886 / 0.0867 | 0.1426 / 0.1409 | 0.0892 / 0.0869 |
| 64  | 0.0898 / 0.0878 | 0.1440 / 0.1431 | 0.0900 / 0.0878 |
| 128 | 0.0898 / 0.0875 | 0.1440 / 0.1442 | 0.0904 / 0.0879 |

**The integration-error hypothesis is mostly not supported.** Going from
16 to 128 steps (8x finer) only closes the gap in seed 1 (0.1440 vs
0.1442, effectively tied); in seed 0 and seed 2 the gap holds steady or
widens slightly. A real discretization artifact should shrink
consistently across seeds as steps increase -- it didn't.

**An unexpected side finding:** `best_of_n_error` got *worse*, not
better, for every condition as steps increased (e.g. seed0 zero:
0.0844 -> 0.0875). More integration steps should only help if the
bottleneck is discretization; getting worse instead suggests the
velocity network's own pointwise error compounds over more integration
steps rather than washing out -- i.e. the field itself, not just the
solver, is the limiting factor.

**What this means.** The `paired_loss` / `best_of_n_error` disagreement
is probably not a cheap numerical artifact to wave away -- it likely
reflects a real difference in how the `real`- and `zero`-conditioned
sample distributions are shaped (e.g. `zero`'s samples may cluster more
tightly around a "safe" answer, giving best-of-8 more chances to land a
lucky hit, even though `real`'s distribution is a better fit to the true
transition on average per `paired_loss`). Understanding that shape
difference directly (rather than via a proxy like step count) is left
for a future day.

Framed in more ordinary ML terms: getting *worse* with more integration
steps is the signature of underfitting, not a bug -- the velocity
network is still only an approximate fit to the true dynamics at
~39,000 training pairs for a genuinely hard task, so each step's small
error compounds over repeated integration rather than washing out (the
same way dead-reckoning navigation drifts over a long walk without a
fixed landmark). Worth being precise about the fix, though: this
project already hit the opposite failure once (Day78's first `noise`
run overfit past ~epoch 30 on a small dataset), so "train longer on the
same data" is not the same lever as "train on more data" -- only the
latter is validated so far (Day80). Whether more data also closes this
specific `paired_loss`/`best_of_n_error` gap is a reasonable next
hypothesis, not yet tested.

## Result (Day 82) -- it's bias, not variance, and more samples don't fix it either

Day81 ruled out ODE step count as the explanation for the paired_loss /
best_of_n_error disagreement and left it as an open question: does the
gap come from `real`'s generated samples being *more spread out* than
`zero`'s (so best-of-8 luck favors the tighter cluster), or from
something else? Two things this can now rule in or out:

1. **Does more sampling close the gap?** (a sample-count analogue of
   Day81's step sweep) -- draw a pool of 256 samples per condition per
   pair and recompute best-of-N for N = 8, 16, 32, 64, 128, 256.
2. **Bias/variance decomposition** -- split each condition's expected
   squared error into `bias^2` (how far the *center* of its sample
   cluster is from the true target) and `variance` (how spread out the
   cluster is around its own center). These sum to (approximately) the
   single-sample expected error.

Both re-use the Day80 checkpoints, no retraining (`cfm_eval_distribution.py`).

| seed | zero bias² / var | real bias² / var | shuffled bias² / var |
|---|---|---|---|
| 0 | 0.0696 / 0.0247 | 0.0735 / 0.0240 | 0.0740 / 0.0242 |
| 1 | 0.1350 / 0.0392 | 0.1386 / 0.0364 | 0.1409 / 0.0368 |
| 2 | 0.0813 / 0.0172 | 0.0828 / 0.0166 | 0.0821 / 0.0162 |

**Variance is essentially the same across all three conditions in every
seed** (in 2 of 3 seeds `zero`'s variance is actually *higher* than
`real`'s, the opposite of the "tighter cluster" hypothesis). The entire
gap lives in `bias^2`: `zero`'s sample cluster is consistently centered
closer to the true target than `real`'s, by a small but reproducible
margin in all 3 seeds.

**More samples doesn't fix this either.** Going from N=8 to N=256 (32x
more draws) leaves `real` behind `zero` by roughly the same relative
margin throughout -- consistent with the gap being a bias, not a
coverage/variance problem. Drawing more samples only helps close a gap
caused by insufficient coverage of a genuinely wide distribution; it
cannot fix a systematic offset in where the distribution is centered.

**What this means.** This sharpens, rather than contradicts, Day81's
underfitting story. `paired_loss` measures whether the model's direction
is *locally* accurate when told the real answer already exists nearby
(a single check against a known point). Generating a full sample means
applying that same imperfect direction field open-loop, many times in a
row, with no such correction -- and apparently doing so introduces a
small but consistent drift specifically under the `real` condition, not
just added noise that more samples or finer steps would average out.
Whatever is different about `real`'s learned path, it's a bias baked
into the field itself, not a sampling artifact.

## Result (Day 83-84) -- more data (100 -> 200 episodes) doesn't extend Day80's win, and closes this arc

The natural next test after Day82's bias finding: does more data shrink
it, the same way 20 -> 100 episodes fixed the Day78/79 problem? Downloaded
80 more episodes (100 -> 200 total, `download_more_episodes.sh <start>
<end>`) and reran the Day80 experiment.

**First attempt was confounded by two stacked methodology bugs, not the
data itself.** With the epoch budget fixed at 100, val_loss now bottomed
out within ~10 epochs and rose for the rest of training -- more data at
the same epoch count meant more gradient updates per epoch, and the model
overfit faster, not slower. Fixed by saving the best-val_loss checkpoint
instead of the final one (`cfm_train.py` now tracks `best_epoch`/
`best_val_loss`). That surfaced a second problem: the CFM loss is
stochastic per batch, so a single epoch's val_loss is itself a noisy
read -- the "best" epoch it picked (0-2) turned out to be barely-trained,
not a genuine minimum. Fixed by averaging val_loss over 3 stochastic
draws and smoothing with a trailing moving average before comparing
epochs.

**With that fixed, extending training to 300 epochs confirmed a genuine
second convergence phase** (val_loss fell well below its epoch-0 value
in 2 of 3 seeds, after an overfitting hump in the middle) -- validating
that 100 epochs simply wasn't enough training at this data scale, not
that the model was stuck.

**Even so, `zero` (no action) won on `paired_loss` in all 3 seeds** once
training was genuinely converged (seed 2 came close: 0.1327 vs `real`'s
0.1336, but didn't flip). Day80's "more data -> real wins" trend did not
continue from 100 to 200 episodes -- if anything it partially reversed.
One seed (seed 1) also failed to reach the second convergence phase
within 300 epochs at all, best_epoch=2, underscoring how seed-sensitive
this training regime is at this scale.

**Closing this arc here.** Seven days (Day78-84) chasing whether the
action pathway helps produced an unresolved answer on that original
question, but a durable one on how to ask it: `paired_loss` (a
likelihood-style score against the real transition, no sampling needed),
`best_of_n_error` (best-of-N sampling, not average-of-N, to avoid
penalizing correct multimodality), bias/variance decomposition, and
smoothed best-checkpoint selection are all reusable pieces for any future
work on this predictor. Next up: Action Chunking or a return to reading
Cosmos-H-Surgical-Simulator's own source, not further iteration on this
specific question.

## Note (Day 85) -- rereading Cosmos-H-Surgical-Simulator's source after this arc

Day58-60 read Cosmos-H-Surgical-Simulator (CHSS)'s source at a surface
level (how it embeds actions, what data it trains on). Rereading it after
Day78-84, several of the open questions above turn out to already have
answers at production scale:

- CHSS's teacher model is trained with an actual Rectified-Flow
  objective -- the untried fix noted after Day82
- The distillation into the real-time student (Self Forcing) trains the
  student on its own generated rollouts instead of ground truth,
  addressing exposure bias directly -- a plausible cause for the drift
  found in Day81-82, though not one this project tested
- What this project already does -- flattening the action window into
  one vector -- turns out to be exactly what's called action chunking,
  and CHSS's own design (a fixed 12-step window, added uniformly) is
  structurally the same. What's genuinely untried on both sides is
  handling that window with a sequence model instead of flattening it
  (below) -- an open question, not catching up to a known-better design
- The real/shuffled/zero paired_loss comparison this project spent most
  of a week building doesn't appear to be the kind of ablation CHSS's
  own documentation reports

Reading the source alone wasn't enough to see any of this -- a week of
hands-on debugging is what made it visible. See the Day85 post for the
fuller writeup; next up is actually trying the sequence-model approach.

## Result (Day 86) -- encoding the action window with a GRU instead of flattening

Day85's genuinely-untried item: replace flattening the H-step action window
with a small sequence model. Added `ActionSequenceEncoder` (`cfm_model.py`)
-- a GRU that reads the window one step at a time and uses its final hidden
state as the action embedding -- selectable via `cfm_train.py --action-mode
{flatten,sequence}`. Same setup as Day83-84 otherwise (n=200 episodes, 300
epochs, 3 seeds), so results are directly comparable to that flatten
baseline.

| seed | mode | real best_of_n | zero best_of_n |
|---|---|---|---|
| 0 | flatten | 0.00359 | 0.00347 |
| 0 | sequence | 0.0061 | 0.0060 |
| 1 | flatten | 0.00547 | 0.00539 |
| 1 | sequence | 0.0040 | 0.0039 |
| 2 | flatten | 0.00306 | 0.00290 |
| 2 | sequence | 0.0037 | 0.0037 |

Zero action still ties or beats real action in all three seeds with the
GRU encoder -- the sequence model didn't flip the Day83-84 result. Absolute
loss values were also noisier than the flatten version (e.g. seed0's
paired_loss jumped from ~0.15 to ~0.19), with a wider train/val gap
suggesting mild overfitting from the added GRU parameters at this data
scale (200 episodes).

Rereading earlier notes on the pi0 paper (arXiv:2410.24164) afterward: pi0
also chunks actions, but on the *output* side (its Action Expert generates a
chunk as a policy) rather than the *input/conditioning* side (this
project's world-model-style setup), and it encodes the chunk with
bidirectional attention over the whole window, not step-by-step
recurrence -- a more expressive architecture than the GRU tried here. ACT's
chunking also leans on temporal ensembling across overlapping chunk
predictions to suppress compounding error, a different lever than the
encoder architecture change made today. So this result should be read
narrowly: "this GRU didn't help," not "sequence modeling can't help."
Attention-based encoding, or the Day81-82 velocity-field accuracy itself,
remain more likely places to look next.

## Next steps (not yet done)

- Try masked/cropped instrument-region evaluation with an actual
  detector instead of a precomputed motion-saliency heuristic
- Try an attention-based (Transformer) action-window encoder instead of
  the GRU tried in Day86, closer to pi0's Action Expert design
- Read the ACT (Action Chunking with Transformers) paper properly and
  check whether its temporal-ensembling idea applies to the Day81-82 drift
- Test whether Self Forcing-style training (condition on the model's own
  generated rollouts, not ground truth) reduces the Day81-82 drift,
  since CHSS uses this to address what looks like the same failure mode

## Files

- `prepare_data.py` -- extract frames + actions from raw episodes
- `download_more_episodes.sh <start> <end>` -- fetch additional Open-H
  peg-transfer episodes into `data/raw/`; Day80 used `20 99`, Day83 used
  `100 199`
- `model.py` -- the pixel-space action-conditioned predictor (FiLM-style
  action injection applied post-normalization + residual/delta output)
- `train.py` -- pixel-space training loop; supports `--horizon`,
  `--weighted-loss`; episode-level train/val split
- `jepa_model.py`, `jepa_train.py` -- the latent-space (JEPA-style)
  variant: online/target encoders, predictor, variance-based
  anti-collapse regularization
- `cfm_model.py`, `cfm_train.py` -- Day78: replaces the JEPA predictor with
  a Conditional Flow Matching velocity field on the same latent space;
  supports `--source {noise,zt}`, `--seed`, `--lr`; evaluation reports
  `paired_loss` (likelihood-style, both source modes) and `best_of_n_error`
  / `sample_spread` (source=noise only). Day79: `--gated` swaps in
  `GatedVelocityPredictor`, which injects the action through a learned
  sigmoid gate instead of a plain concat; adds `mean_gate` to the eval
  output. Train/val split is pinned to a permutation of the original 20
  episodes, so the held-out set stays fixed regardless of how many
  episodes are on disk. Day83: tracks `best_epoch`/`best_val_loss` and
  restores that checkpoint instead of the final epoch, with val_loss
  averaged over 3 stochastic draws and smoothed (trailing moving average)
  before comparing epochs -- fixes both overfitting-past-the-optimum and
  noisy single-epoch checkpoint selection. Day86: `--action-mode
  {flatten,sequence}` picks how the `(H, action_dim)` window becomes one
  embedding -- `flatten` (default, unchanged) concatenates all H steps;
  `sequence` runs it through `ActionSequenceEncoder`, a small GRU, instead
- `cfm_eval_steps.py` -- Day81: reloads a saved checkpoint and re-runs the
  sampling-based eval at several `--steps` values, without retraining;
  used to test whether the paired_loss/best_of_n_error gap is an ODE
  discretization artifact
- `cfm_eval_distribution.py` -- Day82: reloads a saved checkpoint and draws
  a large sample pool per condition to (1) recompute best-of-N at several
  `N` values and (2) decompose expected error into bias² vs. variance;
  also dumps a 2D PCA scatter of one example pair's samples
- `make_day78_summary.py` .. `make_day82_summary.py`,
  `make_day86_summary.py` -- regenerate the cross-seed comparison figures
  from `outputs/history_cfm_*.json`
- `outputs/` -- loss curves, qualitative comparisons, training history;
  `day61_experiment_summary_v2.png` (six pixel-space attempts compared),
  `day62_jepa_action_sensitivity.png` (latent-space action sensitivity),
  `day78_cfm_paired_loss_summary.png` (CFM real/shuffled/zero across
  seeds), `day79_gating_summary.png` (gated vs ungated + gate values),
  `day80_data_scale_summary.png` (20 vs 100 episodes, both architectures),
  `day81_step_sweep_summary.png` (best_of_n_error vs. ODE step count),
  `day82_distribution_summary.png` (sample-count sweep + bias/variance),
  `day82_sample_distribution_pca_*.png` (2D PCA of one example pair)
- `data/raw/`, `data/episodes/` -- source parquet + mp4 + extracted
  frames/actions per episode (200 episodes as of Day83)
