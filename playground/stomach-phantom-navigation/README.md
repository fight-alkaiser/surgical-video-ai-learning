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

## Next steps (not yet done)

- Download more episodes (this dataset has 462 available, vs. the 20 used
  so far) and rerun training -- Day107

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
- `outputs/` -- loss curves, training history, logs
- `data/raw/`, `data/episodes/` -- source parquet + mp4 + extracted
  frames/actions per episode (not committed to git, see `.gitignore`)
