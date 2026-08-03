# Action-Conditioned Video Prediction (toy)

Day 61 of the "surgeon learning surgical video AI" series. This is not
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

## Open question raised by today's results

Comparing predicted vs. real video frame-by-frame in raw pixel space may
itself be a bad fit for this task -- the same argument JEPA (read on Day
27 of this series) makes for representation learning generally: predict
and compare in a learned latent space, not pixels. Today's results are
consistent with that argument (pixel MSE structurally rewards copying
over using the action) but do not prove it either way.

## Next steps (not yet done)

- Understand why the residual model's delta collapses to ~0 on held-out
  episodes instead of learning a generalizable action-dependent signal
- Consider evaluating (or training) in a latent/embedding space instead
  of raw pixels
- Try masked/cropped instrument-region evaluation with an actual
  detector instead of a precomputed motion-saliency heuristic

## Files

- `prepare_data.py` -- extract frames + actions from raw episodes
- `model.py` -- the action-conditioned predictor (FiLM-style action
  injection + residual/delta output)
- `train.py` -- training loop; supports `--horizon`, `--weighted-loss`;
  episode-level train/val split
- `outputs/` -- loss curves, qualitative comparisons, training history,
  and `day61_experiment_summary_v2.png` (all six attempts compared)
- `data/raw/`, `data/episodes/` -- source parquet + mp4 + extracted
  frames/actions per episode
