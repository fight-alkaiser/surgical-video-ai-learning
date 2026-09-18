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

## Next steps (not yet done)

- 100-epoch run at 200 episodes, seed 0 (running overnight as of Day108) --
  see if training converges to a clear real-vs-zero gap given more budget

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
- `outputs/` -- loss curves, training history, logs
- `data/raw/`, `data/episodes/` -- source parquet + mp4 + extracted
  frames/actions per episode (not committed to git, see `.gitignore`)
