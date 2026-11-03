# I-JEPA Representation Learning (toy)

Day 93-95, Day99, Day101 and Day153 of the "surgeon learning surgical video AI" series. A deliberate
pivot away from ../action-conditioned-video-prediction/, which spent 15
days (Day78-92) on whether conditioning on the robot action helps a small
predictor and concluded that the negative result was most likely a
data/compute scale limit, not something a cleverer trick would fix (see
that project's README, Day92 section). This project asks a different
question that shouldn't have the same scale-sensitivity: can a small
Transformer learn useful single-image representations with I-JEPA's
masked-patch-prediction objective (Assran et al., 2023), no actions, no
frame pairs involved at all.

Same data source as the other project (Open-H peg_transfer episodes),
reused directly from `../action-conditioned-video-prediction/data/` --
but every frame is used independently here, since there's no horizon or
action window to build pairs around.

## Architecture

- `patchify`: (3, 64, 64) frame -> 64 patches of 8x8 pixels
- `context_encoder`: small Transformer (3 layers), sees only the visible
  (non-masked) patches -- gets gradients
- `target_encoder`: same architecture, EMA copy of the context encoder,
  sees all patches, no gradients, output detached
- `predictor`: context tokens + learnable mask tokens (at target
  positions) -> predicted target-encoder-space embeddings at those
  positions
- masking (`masking.py`): one shared mask per batch (not per-image, a
  deliberate simplification -- see that file's docstring), following
  I-JEPA's multi-block scheme: ~4 target blocks (scale 0.15-0.2 of the
  image, aspect 0.75-1.5) removed from one large context block (scale
  0.85-1.0)

## Result (Day 93) -- two collapse blind spots found in tooling used since Day61

First run's val_loss was ~1000x lower than train_loss -- checked why, and
found the target encoder's output had collapsed completely: cosine
similarity of 1.0 between totally different images' patch embeddings.
The `variance_loss` anti-collapse term this whole series has used since
Day61-62 (`../action-conditioned-video-prediction/jepa_model.py` and
elsewhere) checks *raw-magnitude* variance across a flattened batch, but
the actual training loss (`normalized_mse_loss`) only ever compares
L2-normalized *directions*. A vector can vary substantially in magnitude
while always pointing the same way -- satisfying the old check while the
representation is still fully collapsed. This blind spot has likely been
present in every use of `variance_loss` in this series, not just here;
it just happened not to matter enough to notice in a whole-image
CNN-encoder setting the way it does when patch-level direction is the
entire point.

Fixed by applying the variance term to L2-normalized vectors instead,
with `gamma` rescaled for unit-norm vectors (`1/sqrt(embed_dim)`).
Retrained -- and found a second, more specific collapse the first fix
couldn't see: different *images* were now distinguishable, but every
patch *within* one image still mapped to the identical vector regardless
of position, exactly the axis I-JEPA's task depends on. The first fix
pools batch and patch dimensions together before computing variance, so
this axis was invisible to it. Added `within_image_variance_loss`
(`ijepa_model.py`), which computes variance across patches *within each
image* separately instead of pooling everything -- this collapse
resolved too (cosine similarity across patches within an image dropped
from 1.0 to ~0, sometimes slightly negative, over a few epochs).

Retraining after both fixes (100 epochs, 2 seeds) surfaced a third,
still-unresolved issue: in both seeds, `val_loss` oscillates wildly
between near-zero and near-maximum (the metric's ceiling, ~4.0, meaning
predicted and target directions are nearly opposite) throughout the
entire run, never settling. The smoothed-best-checkpoint logic
(inherited from `../action-conditioned-video-prediction/cfm_train.py`)
picks up a lucky low point, but neither seed shows genuine convergence.
Cause not yet identified -- candidates include the learning rate being
too high for this small Transformer, or the EMA decay (0.996) not being
well matched to how fast the context encoder's representation is
changing.

No working representation model to show for today, but two genuine,
previously-invisible blind spots found in anti-collapse tooling this
project has relied on since Day61.

**Revised by Day99**: the within-image fix described above only held
with the encoder in `train()` mode (dropout active during the
collapse check). In `eval()` mode -- what every downstream use of this
checkpoint actually calls -- the same "fixed" checkpoint is fully
collapsed (cosine similarity 1.0000 across all patches). See the
Day99 section below.

## Result (Day 94) -- oscillation is a real tradeoff, and the trained encoder loses to a random one

Added `--clip-grad` to `ijepa_train.py` and swept `--lr` (0.001, 0.0003,
0.0001) plus clip thresholds (1.0, 0.5) at 30 epochs each, tracking
`ctx_cos_sim` (across-image) alongside `val_loss`:

| config | frac(val_loss>3.0) | val_loss std | cos_sim(across-img) |
|---|---|---|---|
| lr=0.001 | 0.23 | 1.591 | ~0 by epoch 3 |
| lr=0.0003 | 0.00 | 0.237 | stuck 0.6-0.9 |
| lr=0.0001 | 0.00 | 0.167 | stuck 0.82-0.92 |
| lr=0.001, clip=1.0 | 0.13 | 1.296 | ~0 |
| lr=0.001, clip=0.5 | 0.17 | 1.412 | ~0 |

Not a simple fix: high LR develops healthy, diverse representations fast
but oscillates badly; low LR removes the oscillation but representation
diversity never develops in 30 epochs -- a genuine tradeoff, not a bug.
Gradient clipping (clip=1.0) partially helps (23%->13%) but
non-monotonically (clip=0.5 was worse than clip=1.0).

Rather than keep tuning around that noise, added
`probe_representation_quality.py`: mean-pools the context encoder's
patch embeddings (full, unmasked image, per I-JEPA's own evaluation
convention) and probes them against the real per-frame action, comparing
the trained checkpoint (lr=0.001, clip=1.0, best_epoch=29/30) to a
randomly initialized encoder of identical architecture.

| | val_mse | R² vs. mean-action baseline |
|---|---|---|
| mean-action baseline | 0.8099 | -- |
| random (untrained) encoder | 0.6285 | 0.224 |
| trained encoder | 0.8008 | 0.011 |

The trained encoder is worse than a random one at this downstream
signal. No collapse (both across- and within-image diversity checks
pass) does not mean the representation is useful -- necessary, not
sufficient. This lands the project in the same place as
`../action-conditioned-video-prediction/`'s Day78-92 work, reached by a
completely different route: training from scratch on 200 episodes on a
CUDA-less Mac mini doesn't beat a naive baseline here either.

## Result (Day 95) -- a frozen, never-trained-on-this-data backbone beats everything built for this project

Added `probe_pretrained_resnet18.py`: loads torchvision's ImageNet-
pretrained ResNet18 (`ResNet18_Weights.IMAGENET1K_V1`, ~44MB, downloaded
once and cached), drops the classifier head, uses the 512-dim pooled
feature frozen (no fine-tuning -- inference only, lighter on this Mac
mini than training anything from scratch), and runs the same probe
methodology as Day91/94 against it.

| encoder | val_mse | R² vs. mean-action baseline |
|---|---|---|
| mean-action baseline | 0.8099 | -- |
| Day94 random I-JEPA encoder | 0.6285 | 0.224 |
| Day94 trained I-JEPA encoder | 0.8008 | 0.011 |
| **pretrained ResNet18 (frozen)** | **0.2500** | **0.691** |

By far the clearest positive result either investigation in this series
has produced. Per-dimension breakdown (same method as
`../action-conditioned-video-prediction/probe_action_per_dimension.py`,
Day92):

| group | R² |
|---|---|
| left_xyz | 0.905 |
| right_xyz | 0.892 |
| left_quat | 0.552 |
| right_quat | 0.527 |
| left_gripper | 0.483 |
| right_gripper | 0.384 |

Even gripper open/close -- the one dimension every from-scratch encoder
in this project (Day92's CNN, Day93-94's I-JEPA Transformer) failed to
pick up at all -- is now clearly recoverable. A backbone that has never
seen a surgical video, this task, or this data at all outperforms
everything built specifically for it here. Both investigations in this
series (action-conditioning, Day78-92; from-scratch representation
learning, Day93-94) ran into the same wall by different routes: not
enough data/compute on this Mac mini to learn useful vision from
nothing. The fix was never "train harder" -- it was "don't train the
vision part at all; borrow it."

## Result (Day 99) -- the Day93 "fix" only held in train mode; eval mode was never checked

The day before Day100's retrospective was spent verifying earlier claims
concretely instead of trusting summary numbers -- and that turned up
something significant that revises the Day93/94 story.

Day93 reported the within-image collapse fixed: `ctx_within_cos_sim`
dropped from 1.0 to ~0 during training. To compare concretely,
`repro_day93_collapse_for_review.py` reproduces the original bug (a
short run with `variance_loss` applied to raw, non-normalized
`ctx_tokens`, exactly as the first Day93 attempt did -- the original
collapsed checkpoint no longer exists, each re-run overwrote the same
filename). Loading the actual "fixed" checkpoint
(`model_ijepa_seed0_lr0.001_ema0.996_clip1.0.pt`) and checking a real
image's 64 patches pairwise: **cosine similarity 1.0000 across every
pair, std 0.0000** -- fully collapsed, not fixed at all.

The cause: `model.train()` vs `model.eval()`. Every training-time
collapse check (`ctx_cos_sim`, `ctx_within_cos_sim` in `ijepa_train.py`)
ran during the training loop, with `PatchEncoder`'s Transformer in train
mode -- dropout actively injecting noise into every forward pass. Every
actual downstream use of this checkpoint (Day94's
`probe_representation_quality.py`, and this check) calls `model.eval()`
first, as literally any real use of a trained model does. Switching the
same loaded checkpoint from eval to train mode on the same image:

| mode | mean cosine sim | std |
|---|---|---|
| eval (dropout off -- what every downstream check actually uses) | 1.0000 | 0.0000 |
| train (dropout on -- what the training-time monitor measured) | -0.0082 | 0.8216 |

The model never learned to genuinely distinguish patches. It leaned on
dropout's randomness to satisfy the anti-collapse loss during training,
and reverts to a constant output -- true collapse -- the moment that
noise is turned off. This fully explains Day94's strangest finding (the
trained encoder scoring worse than a random one, R²≈0.01 vs ≈0.22): at
eval time it wasn't a badly-learned representation, it was no
representation at all, the same output regardless of input -- consistent
with the concrete example checked today, where the trained encoder's
probe predicted the same gripper value (0.005) for six real examples
whose true values ranged from -1.234 to 1.150, while a random-init
encoder's predictions at least varied with the input.

The anti-collapse check itself had a blind spot as real as the one it
was built to catch: verified only in the one mode nothing downstream
actually uses.

## Result (Day 99, continued) -- checking the pretrained backbone against Day93's own collapse test

**Done**: swapping the frozen pretrained ResNet18 into
`../action-conditioned-video-prediction/`'s CFM predictor was completed
in Day96-98 -- real action now beats zero action reproducibly across 3
seeds, and the Day82 bias/variance story flips in real's favor too. See
that project's README (Day96/97/98 sections) for the full result.

Separately, closed a loose end specific to this project: does the
pretrained backbone exhibit the exact collapse Day93 found and fixed in
the from-scratch encoder -- every patch within one image mapping to
(nearly) the same direction regardless of position? Added
`probe_pretrained_patch_collapse.py`, which applies Day93's own cosine-
similarity check to ResNet18's pre-pool feature map (`layer4` output,
7x7x512 for a 224x224 input) instead of the from-scratch `PatchEncoder`.

| check | mean cosine sim | range |
|---|---|---|
| within-image (different patches, same image) | 0.587 | 0.309 - 0.954 |
| across-image (different images, same patch position) | 0.834 | 0.758 - 0.978 |

No exact collapse (nothing pinned at 1.0, unlike Day93's original bug),
but real, substantial correlation is present -- higher than the
from-scratch encoder's post-fix state (~0, sometimes slightly negative).
This isn't a defect to fix: ResNet18 was never trained to decorrelate
patch-level features the way a JEPA-style anti-collapse objective
demands, and neighboring 7x7-grid positions have overlapping receptive
fields, so some shared structure is expected. It does mean the pretrained
backbone's spatial features are "structured but correlated" rather than
"maximally spread," a different character than either failure mode
found earlier in this project.

## Result (Day 101) -- removing the dropout crutch doesn't buy a non-collapsed encoder; it just removes the disguise

Day99 found that the Day93 "fix" only avoided collapse because
`TinyTransformer` never set `dropout` explicitly, so `nn.TransformerEncoderLayer`'s
default (0.1) was silently active in train mode and off in eval mode -- the
anti-collapse loss was satisfied by dropout noise, not by genuinely
distinguishable patches. The natural next question: with that disguise
removed (`dropout=0.0`, so train mode and eval mode see the same network),
can this architecture actually learn a non-collapsed representation?

Three runs, seed 0, otherwise unchanged from Day93's setup:

| run | var_weight | what happened |
|---|---|---|
| baseline | 5.0 (unchanged) | fully collapsed by epoch 10 (`ctx_std`, `train_loss` -> 0.0000) and stayed there for all 100 epochs |
| stronger penalty | 15.0 | same -- fully collapsed by epoch 10, 60 epochs, no recovery |
| stronger still | 30.0, 150 epochs | collapses by epoch 10, later escapes into a loud, unstable oscillation in train-mode loss (epoch ~65-95, val_loss swinging between 0.05 and 1.6), then collapses again and stays collapsed through epoch 150 |

The var_weight=30 run's oscillation looked promising in train-mode metrics
alone (`ctx_std` reached 0.80, train-mode `cos_sim(across-img)` dropped to
0.47) -- but `outputs/day101_collapse_check_seed0_lr0.001_ema0.996_vw30.0.png`
(new per-epoch eval-mode logging added this session, see below) shows the
eval-mode across-image cosine similarity never leaves 1.0000 for the entire
150-epoch run, including during the loud oscillation. Only within-image
similarity dips briefly (1.0 -> ~0.93, epoch 88-95) before returning to 1.0.
Whatever instability var_weight=30 introduced in training dynamics, it never
translated into a genuinely non-collapsed representation under the
conditions that matter (`model.eval()`).

**Methodological side-finding**: the training loop's automatic checkpoint
selection (lowest smoothed `val_loss`) is fooled by collapse the same way
the old anti-collapse check was -- a fully collapsed model trivially scores
`val_loss` at or near 0.0, so on all three runs the auto-selected
"best_epoch" was a collapsed checkpoint, even in the var_weight=30 run where
a genuinely different (if unstable) regime existed for ~30 epochs. Fixed by
adding periodic checkpoint saving (`outputs/checkpoints/`, every 10 epochs)
and a second plot (`day101_collapse_check_*.png`) showing `val_loss` next to
the eval-mode collapse curve, so a checkpoint can be picked by looking at
both instead of trusting `argmin(val_loss)` alone.

**Reading**: removing the dropout disguise didn't turn up a fixable bug to
patch -- at this architecture and data scale (196 training episodes'
frames, 3-layer/64-dim Transformer, Mac mini), the anti-collapse loss did
not reliably out-compete the collapse-to-constant-output solution, even at
6x its original weight. This is consistent with Day95's lesson but adds a
separate point: it's not just that a from-scratch encoder underperforms a
pretrained one here -- at this scale, a from-scratch encoder trained with
this objective did not reliably converge to any non-degenerate solution at
all in these runs, independent of the dropout-monitoring bug Day99 found.

## Result (Day 153) -- LeJEPA's SIGReg replaces EMA, but label-free model selection picks the collapsed checkpoint when val batches are consecutive frames

Paper: Balestriero & LeCun, *LeJEPA: Provable and Scalable Self-Supervised
Learning Without the Heuristics*, arXiv 2511.08544 (2025, not peer reviewed;
read on Day151-152). Its two claims tested here on this project's toy:

1. SIGReg alone (no EMA teacher, no stop-gradient, no extra variance terms)
   prevents collapse.
2. The training objective itself, `(1-λ)·pred + λ·SIGReg`, tracks
   representation quality, so a checkpoint can be picked without labels.

**What changed** (`--anti-collapse sigreg`; the default `variance` mode is
untouched):

- `ijepa_model.py`: `sigreg_loss()` -- the paper's Algorithm 1 (random 1-D
  projections, Epps-Pulley characteristic-function test against N(0,1), 17
  knots on [-5, 5], directions resampled every step), single-device, with
  cos/sin instead of complex tensors because MPS has none.
  `IJEPAModel.forward_online_target()` -- the context encoder produces both
  the prediction target and the full-image tokens, with gradients; no EMA.
- `ijepa_train.py`: plain MSE as in the paper, λ=0.05, 256 slices.
  `--sigreg-on image` applies SIGReg to one mean-pooled vector per image (as
  in the paper's code); `--sigreg-on patch` applies it to every patch token.
  Checkpoint selection uses smoothed val `(1-λ)·pred + λ·SIGReg`.

Sanity check of `sigreg_loss` on known 64-dim inputs: standard
Gaussian 1.1, all-zeros (full collapse) 26, 3x-scaled Gaussian 38,
anisotropic 14 -- it does flag collapse.

**Runs**: seed 0, 30 epochs each, otherwise the Day101 setup (dropout 0.0).

| run | across-image cos (train) | within-image cos (train) | auto-selected epoch |
|---|---|---|---|
| `sigreg-image` | 0.97 at epoch 0 -> 0.005 by epoch 29 | **1.000 throughout** | **0 (collapsed)** |
| `sigreg-patch` | 0.85 -> 0.30 | -0.015 | 29 |

`sigreg-image` collapsed in the first epoch, then escaped by epoch 10 (so a
pre-run guess -- that SIGReg's gradient nearly vanishes at full collapse,
~0.004, and so can't pull the model out -- did not hold here). But every
patch inside one image stayed the same vector: image-level SIGReg separates
images and says nothing about patches, the gap noted on Day151.

**Main finding -- the selection criterion was fooled by correlated frames.**
`sigreg-image`'s val SIGReg *rose* over training (25 -> 46-50) while train
SIGReg fell (36 -> 3.4), so the label-free criterion picked the collapsed
epoch 0. The cause is how val batches are drawn: `np.arange(i, i+64)` over 4
concatenated val episodes, i.e. 64 consecutive, near-identical frames of one
episode. SIGReg tests whether a batch looks like i.i.d. samples from an
isotropic Gaussian; a batch of near-duplicates fails that test however good
the encoder is. Re-scoring the saved checkpoints with shuffled val batches
(`day153_val_batch_check.py`, `outputs/day153_val_batch_check.txt`, eval
mode, fixed projection seed; SIGReg, lower is better):

| checkpoint | consecutive batches | shuffled batches |
|---|---|---|
| image, epoch 0 | 26.3 | 26.3 |
| image, epoch 10 | 38.5 | 23.5 |
| image, epoch 20 | 51.6 | 19.4 |
| image, epoch 29 | 48.4 | **13.1** |
| patch, epoch 0 | 48.9 | 40.9 |
| patch, epoch 29 | 33.3 | **18.8** |

(Re-running the script moves these by up to ~1 on MPS despite the fixed
projection seed; the ordering does not change.)

`outputs/day153_consecutive_vs_shuffled.png` (`day153_figure.py`) shows
what the two kinds of batch actually look like on these frames, next to the
image-run numbers above.

With shuffled batches the criterion improves monotonically with training and
would no longer pick the collapsed checkpoint. The same artifact explains why
Day101's eval-mode collapse probe reports across-image cos ~1.0 for every run
here: its 8 probe images are consecutive val frames.

**Reading**: LeJEPA's label-free model selection assumes the evaluation
batch is a sample of independent images. Surgical video (and these robot
episodes) breaks that by default -- frames are consecutive, and even
shuffled val here has only 4 episodes (val SIGReg ~13 vs train ~3.4 over 196
episodes, plausibly the effective-sample-size gap, not yet tested). So in
this setting the criterion is only meaningful if the evaluation batches are
sampled across cases, not along time. This is the Day152 concern ("300,000
frames from 100 cases are closer to 100 cases of variation") showing up
directly in the loss value.

**Not measured**: representation quality (no probe yet -- only collapse);
one seed only; λ, slice count and lr not varied; no projector/BatchNorm
before SIGReg (LeVLJEPA reported SIGReg working poorly right after a final
LayerNorm, which is what `TinyTransformer`'s post-norm output is).

## Next steps (not yet done)

- Day154: select checkpoints with shuffled val batches, check whether that
  choice holds across seeds, and run the Day94 action probe on the selected
  checkpoint to see whether "lower shuffled SIGReg" means a better
  representation, not just a less collapsed one.

Earlier thread (Day101) closed with nothing outstanding: This project and
`../action-conditioned-video-prediction/` independently hit the same wall
(training from scratch on ~200 episodes on a CUDA-less Mac mini doesn't
beat simple baselines) and independently resolved it the same way (borrow
a pretrained backbone instead of training one) -- Day101 confirmed that
wall goes deeper than the Day99 dropout-monitoring bug: strengthening the
anti-collapse loss up to 6x doesn't produce a stable non-collapsed
from-scratch encoder either.

## Files

- `ijepa_model.py` -- `PatchEncoder`, `Predictor`, `IJEPAModel`,
  `normalized_mse_loss`, `variance_loss`, `within_image_variance_loss`.
  Day101: `TinyTransformer`/`PatchEncoder`/`IJEPAModel` take an explicit
  `dropout` argument (default 0.0) instead of relying on
  `nn.TransformerEncoderLayer`'s unset 0.1 default -- the root cause behind
  Day99's train/eval mode discrepancy
- `masking.py` -- I-JEPA-style multi-block context/target mask sampling
  on the 8x8 patch grid (one shared mask per batch)
- `ijepa_train.py` -- training loop; pools every frame from the training
  episodes (no pairing/horizon needed); tracks `ctx_cos_sim`
  (across-image) and `ctx_within_cos_sim` (within-image) as direct
  directional-collapse monitors, not just raw std. Day94: `--clip-grad`
  (max grad norm; 0 disables). Day101: `--dropout` (default 0.0, see
  `ijepa_model.py` above); `eval_mode_collapse_check()` runs the same
  cosine-similarity check with `model.eval()` on a fixed probe batch every
  epoch (`ctx_cos_sim_eval`/`ctx_within_cos_sim_eval` in the saved
  history), since Day99 showed the train-mode-only monitor can't be
  trusted; saves a checkpoint every 10 epochs to `outputs/checkpoints/`
  and a second plot (`day101_collapse_check_*.png`, val_loss next to the
  eval-mode collapse curve) because the automatic best-epoch selection
  (lowest `val_loss`) is itself fooled by collapse
- `probe_representation_quality.py` -- Day94: mean-pools the context
  encoder's patch embeddings and probes them against the real per-frame
  action, comparing the trained checkpoint to a randomly initialized
  encoder of identical architecture -- the real test of whether training
  helped, independent of the training loss curve's own noise
- `probe_pretrained_resnet18.py` -- Day95: frozen, ImageNet-pretrained
  ResNet18 (torchvision) as the encoder instead of anything trained on
  this project's data, probed the same way -- the comparison point that
  finally beat the mean-action baseline by a wide margin
- `probe_pretrained_patch_collapse.py` -- Day99: applies Day93's own
  within-image / across-image cosine-similarity collapse check to
  ResNet18's pre-pool `layer4` feature map (7x7 spatial grid) instead of
  the from-scratch `PatchEncoder`
- `repro_day93_collapse_for_review.py` -- Day99: reproduces the original
  Day93 bug (unnormalized `variance_loss`) for a short run, purely so
  the collapsed and "fixed" checkpoints can be compared side by side on
  real images -- the original collapsed checkpoint no longer exists
  (each Day93 re-run overwrote the same filename)
- `day99_concrete_patch_comparison.py` -- Day99: picks one real frame, a
  background patch and an instrument-region patch, and reports/plots the
  encoder's actual cosine similarity between them (collapsed-repro vs.
  the "fixed" checkpoint) -- the check that surfaced the train/eval mode
  discrepancy
- `day153_figure.py` -- Day153: real val frames from one consecutive vs one
  shuffled batch, next to the SIGReg each gives the same checkpoints
  (`outputs/day153_consecutive_vs_shuffled.png`)
- `day153_val_batch_check.py` -- Day153: re-scores the saved SIGReg
  checkpoints' val SIGReg and across-image cosine similarity with
  consecutive vs shuffled val batches (output in
  `outputs/day153_val_batch_check.txt`)
- `outputs/` -- loss curves, training history (`history_ijepa_seed*.json`
  includes both collapse-fix runs' full curves and the Day94 LR/clip
  sweep), `day94_probe_results.json`, `day95_probe_results.json`,
  `day99_patch_collapse_check.json`, `day99_full_pairwise_heatmap.png`,
  `day99_concrete_patches.png`
