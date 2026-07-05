# Day13: State Transition Matrix

## Objective

Until Day12, a surgical video was represented as a sequence of **states**
(consecutive frames with similar active triplet sets). Today we ask a new
question: given the current state, what state tends to come next?

This is the first step away from *describing* the video and toward
*predicting* it.

---

## Theory: Markov chains

A **Markov chain** models a sequence of states where the probability of the
next state depends only on the **current** state, not on the full history
before it. This assumption ("the past only matters through the present") is
called the **Markov property**.

For a pair of states `i` and `j`, the transition probability is:

```
P(next = j | current = i) = count(i -> j) / count(i -> anything)
```

Collecting these probabilities for every pair of states forms a
**transition matrix**.

---

## Why this matters

Every previous analysis (Day01-12) was static or descriptive: frequencies,
durations, similarities. A transition matrix is the first model that makes a
**prediction** — it commits to a belief about the future.

It also exposes a limitation that motivates everything coming next in this
project:

* A Markov chain only looks **one step back**. It cannot represent "this is
  the second time the gallbladder has been grasped" or "clipping already
  happened earlier".
* Real surgical workflow often depends on more than just the immediately
  preceding state (e.g. whether the cystic duct was already clipped changes
  what a later grasping action means).
* This exact limitation is why sequence models (RNNs) and eventually
  **self-attention / Transformers** exist: instead of only looking at the
  previous step, a Transformer lets every position in a sequence look back
  at *any* earlier position, with a learned weight for how much attention
  to pay to each one. A transition matrix is the "1-step-memory" baseline
  that Attention will later be compared against.

---

## Completed image (what the output looks like)

1. Every distinct triplet-set signature seen in Day12's state segmentation
   is assigned a stable integer ID (`S0`, `S1`, ...) the first time it
   appears — a small **vocabulary** of 32 states.
2. The video becomes a sequence of these IDs, for example:
   `S0 -> S1 -> S25 -> S1 -> S0 -> S13 -> ...`
3. For every state, we count how often each *next* state followed it, and
   turn those counts into probabilities.

---

## Method

* Reuse Day12's frame-level triplet sets and Jaccard-based state
  segmentation (threshold 0.6) — 1734 frames compress into 148 state
  occurrences.
* Assign a unique ID to each distinct triplet-set signature the first time
  it is seen. Because some signatures reappear later in the video (e.g.
  "irrigator aspirates fluid" happens several separate times), the 148
  occurrences map onto only **32 unique states**.
* Count transitions between consecutive state IDs in the 148-long sequence.
* Normalize counts into probabilities for each state.

---

## Results

* Sequence length: 148 state occurrences
* Vocabulary size: 32 unique states
* 15 states always transition to exactly one next state (fully
  predictable given this single video).
* 17 states branch into two or more possible next states.

Two states act as clear **hubs**:

| State | Meaning | Outgoing behavior |
|---|---|---|
| S0 | grasper, grasp, gallbladder | seen 24 times; goes idle 50% of the time, but branches into 8 different possible next states |
| S1 | idle (no active triplet) | seen 36 times; most often returns to grasping the gallbladder (39%) or moves on to the specimen bag (19%), but branches into 9 different next states |

In contrast, task-specific states are much more deterministic, e.g.:

* `S28 (grasper, pack, gallbladder) -> S27 (grasp specimen_bag + pack)` with
  probability 1.00
* `S16 (clipper, clip, cystic_duct) -> S1 (idle)` with probability 1.00

---

## Interpretation

The most frequent, generic actions (grasping the gallbladder, being idle)
are exactly the states with the **least predictable** future — they can be
followed by almost anything, because they occur in many different surgical
contexts. Rare, task-specific actions (clipping, packaging) are highly
predictable, because they only ever occur in one narrow context.

This mirrors Day06's finding about triplet-phase predictability: frequency
and predictability are not the same thing, and generic actions carry less
information about "what happens next" than specific ones.

It also confirms the Markov chain's core weakness in a concrete way: from
state S0 alone, we cannot tell *why* it sometimes goes idle and sometimes
goes straight to clipping — that depends on context further back in the
video that a 1-step model cannot see.

---

## Conclusion

Representing the video as an integer state sequence and counting
transitions gives the first genuine **predictive** model in this project.
Its blind spot — no memory beyond one step — is exactly what motivates the
next stages: sequence modeling with more memory (RNNs), and ultimately
Attention, which lets a model weigh *all* previous states instead of only
the last one.

## Next Step

Day14: turn the state sequence into a clean, reusable token sequence
(a proper vocabulary + integer sequence export) and start evaluating how
well a simple Markov predictor actually performs.
