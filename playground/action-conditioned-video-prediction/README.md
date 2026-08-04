# Action-Conditioned Video Prediction (toy)

Day 61-62 of the "surgeon learning surgical video AI" series. This is not
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

## Next steps (not yet done)

- The JEPA-style model still hasn't beaten the do-nothing baseline --
  only shown that it's sensitive to getting the action right
- Try masked/cropped instrument-region evaluation with an actual
  detector instead of a precomputed motion-saliency heuristic
- Per the plan agreed with the project owner, this line of exploration
  pauses here; upcoming days return to reading
  Cosmos-H-Surgical-Simulator's own source code

## Files

- `prepare_data.py` -- extract frames + actions from raw episodes
- `model.py` -- the pixel-space action-conditioned predictor (FiLM-style
  action injection applied post-normalization + residual/delta output)
- `train.py` -- pixel-space training loop; supports `--horizon`,
  `--weighted-loss`; episode-level train/val split
- `jepa_model.py`, `jepa_train.py` -- the latent-space (JEPA-style)
  variant: online/target encoders, predictor, variance-based
  anti-collapse regularization
- `outputs/` -- loss curves, qualitative comparisons, training history;
  `day61_experiment_summary_v2.png` (six pixel-space attempts compared),
  `day62_jepa_action_sensitivity.png` (latent-space action sensitivity)
- `data/raw/`, `data/episodes/` -- source parquet + mp4 + extracted
  frames/actions per episode
