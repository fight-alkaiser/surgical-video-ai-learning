"""I-JEPA's multi-block masking, simplified for a tiny 8x8 patch grid.

Real I-JEPA samples a fresh random mask per image. This toy version samples
one mask per *batch* (shared across all examples in that step) -- a
deliberate simplification, not an oversight: it keeps context/target patch
counts fixed-shape within a batch (no padding/ragged-batch logic needed) and
is cheap to justify at this scale, since a fresh mask is drawn every batch
anyway (thousands of masks seen over a training run either way).
"""

import random

GRID = 8
NUM_PATCHES = GRID * GRID


def _sample_block(scale_range, aspect_range):
    area = random.uniform(*scale_range) * NUM_PATCHES
    aspect = random.uniform(*aspect_range)
    h = max(1, min(GRID, round((area / aspect) ** 0.5)))
    w = max(1, min(GRID, round((area * aspect) ** 0.5)))
    top = random.randint(0, GRID - h)
    left = random.randint(0, GRID - w)
    return {(top + i) * GRID + (left + j) for i in range(h) for j in range(w)}


def sample_mask(num_target_blocks: int = 4, min_context: int = 8):
    """Returns (context_indices, target_indices), both sorted lists of patch
    indices in [0, 64). Target blocks: I-JEPA's default scale (0.15, 0.2),
    aspect (0.75, 1.5). Context block: scale (0.85, 1.0), aspect ~1, with
    target-block patches removed (so the model can't peek at what it's
    predicting). Retries if that removal leaves too little context."""
    for _ in range(10):
        target = set()
        for _ in range(num_target_blocks):
            target |= _sample_block((0.15, 0.2), (0.75, 1.5))
        context = _sample_block((0.85, 1.0), (1.0, 1.0)) - target
        if len(context) >= min_context:
            return sorted(context), sorted(target)
    # fallback: extremely unlikely at these scales, but don't hang if it happens
    context = _sample_block((0.85, 1.0), (1.0, 1.0))
    return sorted(context), sorted(target)
