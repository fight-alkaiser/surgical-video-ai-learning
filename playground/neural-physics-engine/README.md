# Neural Physics Engine — Playground

Small architecture-level playground, separate from the day-numbered
CholecT50 challenge (`day01_...` onward), in the same spirit as
`playground/object-centric-world-models`. The paper discussion itself
lives in the Day34 LinkedIn post; this folder is a small piece of code
to check that understanding by building something concrete on real
CholecT50 data.

**Paper:** Chang, Ullman, Torralba, Tenenbaum. *A Compositional
Object-Based Approach to Learning Physical Dynamics (Neural Physics
Engine).* ICLR 2017 / arXiv:1612.00341.

## Core idea being tested

NPE predicts a scene's Dynamics by applying **one shared pairwise
Interaction function** to every Object pair, then combining the
effects. Because the same function is reused for every pair, it
generalizes to scenes with a different number of Objects than it was
trained on — this is the paper's central compositionality claim.

CholecT50 has no bounding boxes in this release (position/velocity are
not available), so this is not a reproduction of the NPE architecture.
Instead, `interaction_persistence_demo.py` builds the simplest possible
stand-in for that shared function using data this project already has:

- **Object** = an instrument or an anatomical target
- **Edge** = an (instrument, target) pair currently interacting (the
  verb is dropped — we only ask "are these two Objects interacting
  right now?")
- **Interaction function** `f(instrument, target)` = empirical
  probability that an edge present in the current state is still
  present one state later, estimated by counting on 40 training videos
  and applied identically to every edge in every test state, regardless
  of how many other edges are present
- **Dynamics** = the predicted next state's edge set, obtained by
  applying `f` to each current edge independently

It reuses the Day12/13/14 frame-triplet and state segmentation logic,
and the Day14 train/test video split.

Run it with:

```
python3 interaction_persistence_demo.py
```

## What it found

- The shared per-edge function predicts persistence at 62.7% accuracy
  on held-out videos, vs. 56.3% for a baseline that only knows the
  overall persistence rate (no per-pair structure). A modest but real
  improvement from treating instrument-target pairs as distinct units.
- Compositional generalization check: on test videos, 99.9% of
  individual edges had been seen during training, vs. 98.1% of whole
  state signatures (Day14-style exact match). The gap is smaller than
  expected — CholecT50's instrument/target vocabulary is small enough
  that most whole-state combinations already recur across 50 videos.
  The direction (edge-level ≥ whole-state coverage) is consistent with
  the paper's claim that compositional, pairwise representations
  generalize better across scene configurations, but this dataset does
  not show it as dramatically as the paper's variable-object-count
  experiments do.
- The demo print at the bottom applies the exact same `f` to a 1-edge
  state and a 3-edge state with no retraining in between — the toy
  version of "the same interaction function works whichever number of
  Objects are in the scene."

## Not in scope here

No neural network, no true position/velocity (not available in this
CholecT50 release), no multi-step rollout. This only checks the
single-hop, per-pair persistence idea — not a reproduction of the
paper's architecture or its rigid-body experiments.

## Conclusion: a stretch of a fit

Redefining the state `S` as (instrument, target) edges instead of full
triplets, and re-running the same "predict the next `S` from the current
`S`" question, lands back on the same wall as Day13-15: next-state
prediction in surgical video is hard, no matter how the state is sliced.

The honest reason is not that this implementation was too simple — it's
that NPE's compositionality trick works because its domain gives it three
things surgical triplets do not: a (near) **complete** physical state
(position, velocity, mass fully determine what happens next), a
**deterministic** law governing it (Newtonian mechanics), and a
**continuous** space where similar states lead to similar futures. A
CholecT50 triplet is a symbolic label a surgeon retrofits onto a much
richer, only partially observed process (tissue stiffness, inflammation,
force, intent) — closer to balls that occasionally ignore the laws of
physics for reasons the video doesn't show.

So: applying an object-relation physics engine to surgical video was a bit
of a reach in hindsight (🙂) — but a useful one, since it pinned down
*why* it's a reach rather than just assuming it would work.
