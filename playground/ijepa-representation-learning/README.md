# I-JEPA Representation Learning (toy)

Day 93-95 and Day99 of the "surgeon learning surgical video AI" series. A deliberate
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

## Next steps (not yet done)

None outstanding for this specific thread. Both this project and
`../action-conditioned-video-prediction/` independently hit the same
wall (training from scratch on ~200 episodes on a CUDA-less Mac mini
doesn't beat simple baselines) and independently resolved it the same
way (borrow a pretrained backbone instead of training one).

## Files

- `ijepa_model.py` -- `PatchEncoder`, `Predictor`, `IJEPAModel`,
  `normalized_mse_loss`, `variance_loss`, `within_image_variance_loss`
- `masking.py` -- I-JEPA-style multi-block context/target mask sampling
  on the 8x8 patch grid (one shared mask per batch)
- `ijepa_train.py` -- training loop; pools every frame from the training
  episodes (no pairing/horizon needed); tracks `ctx_cos_sim`
  (across-image) and `ctx_within_cos_sim` (within-image) as direct
  directional-collapse monitors, not just raw std. Day94: `--clip-grad`
  (max grad norm; 0 disables)
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
- `outputs/` -- loss curves, training history (`history_ijepa_seed*.json`
  includes both collapse-fix runs' full curves and the Day94 LR/clip
  sweep), `day94_probe_results.json`, `day95_probe_results.json`,
  `day99_patch_collapse_check.json`, `day99_full_pairwise_heatmap.png`,
  `day99_concrete_patches.png`
