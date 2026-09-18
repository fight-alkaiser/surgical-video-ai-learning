# Stomach Phantom Navigation (toy)

Day 106+ of the "surgeon learning surgical video AI" series. A new dataset,
not a continuation of `../action-conditioned-video-prediction/`'s dVRK
peg-transfer data -- Day105 found that Open-H-Embodiment (the source
dataset for this whole `playground/` series) has grown substantially since
Day78, including real clinical surgical data and several GI-relevant
phantom/ex-vivo datasets (see Obsidian: `Ideas/Open-H-Embodiment v2
データセット調査メモ.md`). This project picks up the simplest of those new
options first.

**Task**: Endoscopy/cuhk/openh_dataset_full/find_greater_curvature (Open-H
Dataset, `nvidia/PhysicalAI-Robotics-Open-H-Embodiment`, HuggingFace,
CC-BY-4.0). A custom 2-motor cable-driven soft robotic endoscope navigating
a silicone stomach phantom, task: "Search the greater curvature for a white
oval suspicious region." 462 episodes total, 640x480 @ 20fps,
teleoperated by an endoscopic surgeon expert. Action space is just 2
continuous dims (antagonistic motor speed commands) -- much simpler than
the dVRK project's 16-dim action or the CMR Surgical clinical data's
100-dim one (see the Open-H survey note above).

**Question**: same one `../action-conditioned-video-prediction/` asked from
Day78-98 -- does conditioning on the real motor-speed action improve
prediction of the future latent frame representation, compared to a
zeroed-out action? Unlike that project, this one starts directly from
Day95's conclusion (frozen, ImageNet-pretrained ResNet18 beats any
from-scratch encoder at this data scale) instead of re-discovering it.

## Day106 -- pipeline built, first quick run (not yet decisive)

- `download_episodes.sh`, `prepare_data.py`: same pattern as
  `../action-conditioned-video-prediction/`'s, adapted for this dataset's
  camera key (`observation.images.endo三`, non-ASCII, needs percent-encoding)
  and 2-dim action.
- `cfm_model.py`: a fresh, deliberately simplified rewrite of that
  project's `cfm_model.py` -- frozen `PretrainedResNet18Encoder` +
  Conditional Flow Matching `VelocityPredictor` only. None of that
  project's from-scratch-encoder, gated-action, or GRU/Transformer
  action-encoder variants are included here; those were explorations of
  problems (real action actively hurting predictions, mainly) that haven't
  been established on this dataset yet. Start simple.
- Hit an environment problem before any of this would even import:
  `torchvision` failed because this Mac's pyenv-built Python 3.11.5 has no
  `_lzma` C extension (`xz` was installed via Homebrew after Python was
  built, not before). `pyenv install --force 3.11.5` to rebuild hit an
  unrelated wall -- the build recipe references the now-deprecated
  `openssl@1.1`, and the rebuild's `ensurepip` step segfaulted. Pursuing
  that further risked breaking the `.venv` shared with
  `../action-conditioned-video-prediction/` and
  `../ijepa-representation-learning/`, to fix a torchvision code path
  (`torchvision.datasets`' optical-flow dataset loaders) this project
  never uses. Worked around it locally instead: `cfm_model.py` stubs out
  `sys.modules["lzma"]` before importing `torchvision.models`, so the
  system Python is untouched.
- First run: 20 episodes, seed 0, 40 epochs (~62s/epoch on MPS -- the
  frozen ResNet18's forward pass, run on both context and target frames
  every batch, dominates). Real action did not clearly beat zero:

  ```
       real -- paired_loss: 0.9774   best_of_n_error: 0.7089
   shuffled -- paired_loss: 0.9772   best_of_n_error: 0.7133
       zero -- paired_loss: 0.9762   best_of_n_error: 0.7092
  ```

  Expected at this scale: even with the pretrained encoder,
  `../action-conditioned-video-prediction/`'s Day96-98 turnaround only
  showed up once that project scaled past 20 episodes to 200. No collapse,
  nothing broke -- just not enough data yet to see an effect.

## Day107 -- 10x more data didn't move the needle either

Downloaded episodes 20-199 (180 more, 200 total) and reran training twice:
25 epochs, then 50 epochs, both at `--batch-size 128` (larger batches cut
wall-clock time a lot with a frozen encoder that has to run every batch --
same fix `../action-conditioned-video-prediction/` made on Day97). Neither
run showed real action clearly beating zero:

```
25 epochs:      real -- paired_loss: 0.8848   best_of_n_error: 0.6135
             shuffled -- paired_loss: 0.8849   best_of_n_error: 0.6139
                 zero -- paired_loss: 0.8843   best_of_n_error: 0.6133

50 epochs:      real -- paired_loss: 0.8667   best_of_n_error: 0.6069
             shuffled -- paired_loss: 0.8680   best_of_n_error: 0.6071
                 zero -- paired_loss: 0.8668   best_of_n_error: 0.6062
```

real and zero stayed within ~0.0001 of each other on paired_loss both
times -- essentially tied, with shuffled a hair worse (so the model isn't
completely indifferent to the action's content, just not helped by it
specifically). This is not what happened in
`../action-conditioned-video-prediction/` when that project scaled past 20
episodes with a pretrained encoder (Day96-98's turnaround); here, 10x more
data didn't change the picture.

Not a clean negative result yet, though: at 50 epochs, val_loss was still
slowly decreasing between epoch 40 (0.8755) and epoch 49 (0.8659), not
clearly converged. Two explanations remain open -- undertrained, or
something about this task's action representation (2D motor-speed
commands, vs. the dVRK project's absolute position/orientation) that a
single before/after frame pair can't use regardless of training budget.
Timing note for reproducing: 25 epochs + evaluation took ~2h8m wall-clock
on this Mac mini (`--batch-size 128`, MPS); 50 epochs + evaluation took
correspondingly longer.

## Day108 -- probe: the encoder isn't the bottleneck

Before spending more compute on longer training, ruled out the more
discouraging explanation for Day107's tied result first: that the frozen
ResNet18 encoder simply doesn't preserve this task's action-relevant
information at all, in which case no amount of additional training could
fix it. Same method as
`../action-conditioned-video-prediction/probe_action_from_latents.py`
(Day91 there): freeze the encoder, train a small separate probe to regress
the real 2-dim action window from `(z_t, z_t+H)` alone, no predictor
involved (`probe_action_from_latents.py`).

```
mean-action baseline (ignores z entirely): 0.9665
probe on shuffled actions (z unrelated to target): 0.9631-0.9682
probe on real actions:                             0.7269

R^2 vs. mean-action baseline: 0.2480
```

The shuffled-action control stayed near the mean-action baseline, as
expected (no relationship to recover). The real-action probe scored
clearly better. R^2~0.25 is lower than the ~0.69 this same frozen encoder
achieved on `../action-conditioned-video-prediction/`'s dVRK task (Day95
there), but well above the shuffled/mean-baseline noise floor -- the
encoder is not blind to this task's action.

**Reading**: Day107's tied real/zero result is more likely explained by
the predictor not yet exploiting a signal that does exist in the
representation, or by undertraining, rather than a structural dead end.
Basis for running a longer (100-epoch) training run next rather than
abandoning this task.

## Day109 -- 100 epochs still tied on paired_loss, but bias/variance shows a small, consistent signal

The overnight 100-epoch run (200 episodes, seed 0) finished. val_loss kept
decreasing through the very last epoch (0.8515 at epoch 90 to 0.8503 at
epoch 99), so it wasn't obviously undertrained, but the headline metrics
were still essentially tied:

```
     real -- paired_loss: 0.8519   best_of_n_error: 0.6124
 shuffled -- paired_loss: 0.8525   best_of_n_error: 0.6126
     zero -- paired_loss: 0.8502   best_of_n_error: 0.6127
```

Reran Day82's bias/variance decomposition on this checkpoint
(`cfm_eval_distribution.py`) rather than accepting the tied paired_loss as
the final word -- that probe question (Day108) was post-hoc (given both
`z_t` and `z_t+H`, can the action be explained?), a different and easier
question than what the predictor actually faces (predicting `z_t+H` from
`z_t` and the action alone, without ever seeing the answer). This
decomposition asks the forward-facing question directly:

```
      real  bias^2=0.1879  variance=0.5377  sum=0.7256
  shuffled  bias^2=0.1911  variance=0.5413  sum=0.7324
      zero  bias^2=0.1895  variance=0.5383  sum=0.7278
```

The ordering **real < zero < shuffled** holds on bias^2, variance, and
best_of_N at nearly every N from 8 to 256 (`outputs/day109_distribution_h20_n200_seed0.json`)
-- small differences, but consistent across every angle this script
checks, not noise in one metric. Shuffled action (a real but mismatched
action) scoring worst throughout is the more informative part: the model
distinguishes a correct action from an incorrect one, it isn't simply
indifferent to action content the way the raw paired_loss numbers alone
would suggest. A PCA plot of one example pair's 256 samples per condition
(`outputs/day109_sample_distribution_pca_h20_n200_seed0.png`) shows the
same story as Day82's: all three conditions' clusters overlap heavily
around the true target at the single-pair level -- this effect only shows
up once bias is averaged across many pairs.

**Reading**: real action does carry real, usable predictive signal on this
task, not just post-hoc explanatory signal (Day108) -- but the effect size
is small enough that it doesn't show up in the coarser paired_loss/best_of_n
comparison used through Day106-107. Given the current 2D motor-velocity
action representation, 64x64 downsampled frames, and this encoder, that
may be close to the ceiling for this exact setup; whether a different
action representation, resolution, or horizon would widen the gap is not
yet tested.

**Also fixed this session**: overnight, `caffeinate -disu` was used to keep
the display and system fully awake, based on an initial (mistaken) theory
that display sleep was stalling MPS computation -- the training run
actually completed normally over the full ~7h53m span, most of which had
no caffeinate assertion active at all, so that theory wasn't supported by
the outcome. Switched to `caffeinate -i` (prevents idle *system* sleep
only, lets the display sleep normally) for future long runs, since keeping
the display on for many hours serves no purpose here and needlessly wears
the hardware. This Mac mini's `pmset` was already configured with system
sleep disabled (`sleep 0`), so the safety margin `caffeinate -i` adds is
mostly a low-cost backstop, not a hard requirement.

## Next steps (not yet done)

- Test whether the small real-vs-zero gap widens with a different action
  representation (e.g. an encoded/sequence action window instead of flat
  concatenation), a shorter horizon, or higher-resolution input frames --
  the current setup may simply be near its ceiling for this task

## Files

- `download_episodes.sh <start_ep> <end_ep_inclusive>` -- fetch
  `find_greater_curvature` episodes into `data/raw/`
- `prepare_data.py` -- extract 64x64 frames + 2-dim actions into
  `data/episodes/`, matching `../action-conditioned-video-prediction/`'s
  output format
- `cfm_model.py` -- `PretrainedResNet18Encoder`, `VelocityPredictor`,
  `CFMActionModel` (Conditional Flow Matching on the frozen ResNet18
  latent space); stubs `lzma` before importing `torchvision.models` (see
  Day106 above)
- `train.py` -- training loop + real/shuffled/zero action evaluation
  (`paired_loss`, `best_of_n_error`, same metrics as
  `../action-conditioned-video-prediction/`'s Day78/96-98)
- `probe_action_from_latents.py` -- Day108: freezes the encoder and trains
  a small MLP probe to regress the actual action window from `(z_t,
  z_t+H)` alone (no predictor involved) against a mean-action baseline and
  a shuffled control -- tests whether the encoder itself discards
  action-relevant signal
- `cfm_eval_distribution.py` -- Day109: reloads a trained checkpoint and
  draws a large sample pool per condition to (1) recompute best-of-N at
  several `N` values and (2) decompose expected error into bias^2 vs.
  variance; also dumps a 2D PCA scatter of one example pair's samples
  (same method as
  `../action-conditioned-video-prediction/cfm_eval_distribution.py`,
  Day82 there)
- `outputs/` -- loss curves, training history, logs
- `data/raw/`, `data/episodes/` -- source parquet + mp4 + extracted
  frames/actions per episode (not committed to git, see `.gitignore`)
