# I-JEPA Representation Learning (toy)

Day 93+ of the "surgeon learning surgical video AI" series. A deliberate
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

## Next steps (not yet done)

- Diagnose the val_loss oscillation (try a lower learning rate, a
  different EMA decay, or logging what fraction of batches land near the
  0 vs. ~4 ends of the range)
- Once training is stable, evaluate representation quality -- e.g. a
  linear probe from pooled patch embeddings to the actual per-frame
  action (reusing this series' own action labels purely as a downstream
  evaluation signal, not as a training input), analogous to the probes
  built in `../action-conditioned-video-prediction/probe_action_from_latents.py`

## Files

- `ijepa_model.py` -- `PatchEncoder`, `Predictor`, `IJEPAModel`,
  `normalized_mse_loss`, `variance_loss`, `within_image_variance_loss`
- `masking.py` -- I-JEPA-style multi-block context/target mask sampling
  on the 8x8 patch grid (one shared mask per batch)
- `ijepa_train.py` -- training loop; pools every frame from the training
  episodes (no pairing/horizon needed); tracks `ctx_cos_sim`
  (across-image) and `ctx_within_cos_sim` (within-image) as direct
  directional-collapse monitors, not just raw std
- `outputs/` -- loss curves, training history (`history_ijepa_seed*.json`
  includes both collapse-fix runs' full curves)
