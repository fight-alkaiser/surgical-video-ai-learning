# Day12 - State Segmentation

## Goal

Until Day11, we compared consecutive frames using Jaccard similarity and detected abrupt changes in surgical activity.

Today, we moved one step further by representing a surgical video as a sequence of **states** rather than individual frames.

---

## Method

For every frame:

- Extract the set of Instrument-Verb-Target (IVT) triplets.
- Ignore missing labels and `null_verb`.

A state is defined as a consecutive sequence of frames having sufficiently similar triplet sets.

Similarity between consecutive frames is calculated using **Jaccard similarity**:

$begin:math:display$
J\(A\,B\)\=\\frac\{\|A\\cap B\|\}\{\|A\\cup B\|\}
$end:math:display$

Frames whose similarity exceeds a threshold are merged into the same state.

---

## Output

The program reports

- Total number of frames
- Number of detected states
- Compression ratio
- Duration of every state
- IVT triplets contained in each state

---

## Observation

Using a similarity threshold reduced the number of states slightly by absorbing small annotation fluctuations.

However, many state transitions remained because changes of only one IVT triplet often lowered the Jaccard similarity below the threshold.

This suggests that simple set similarity alone is insufficient for robust workflow representation.

---

## What I learned

This exercise clarified the difference between:

- Frame-level annotations
- State representations
- State sequences

Rather than focusing on perfectly defining each state, modern Transformer-based surgical workflow models primarily learn **state transitions over time**.

Today's work serves as a bridge from frame-level analysis toward sequence modeling.

---

## Next Step

Generate a sequence of state IDs that can later be used as input tokens for Transformer-based workflow understanding.
