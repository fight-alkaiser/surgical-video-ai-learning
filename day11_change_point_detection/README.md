# Day11 Change Point Detection

## Goal

Detect important transition points in a surgical video by comparing
triplet sets between consecutive frames.

## Method

- Build triplet set for every frame
- Compute Jaccard similarity
- Detect frames where similarity < 0.5
- Report added and removed triplets
- Export results to CSV

## Result

Detected 15 change points in VID01.

Many change points corresponded to:

- instrument replacement
- action change
- transition between surgical subtasks

## Insight

Instead of treating every frame independently,
it may be more meaningful to represent a surgical video
as a sequence of semantic events.

This idea naturally connects to Transformer-based sequence modeling.
