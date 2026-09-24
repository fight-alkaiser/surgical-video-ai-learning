# ORQA inference exploration (toy)

Day112-113+. Not a training project like the other `playground/` entries --
ORQA ("Specialized Foundation Models for Intelligent Operating Rooms",
[egeozsoy/ORQA](https://github.com/egeozsoy/ORQA), arXiv:2505.12890) trains
a Qwen2-VL-based multimodal foundation model for operating-room
understanding. Training it needs a single NVIDIA A40 for about a week and a
CUDA-locked pipeline (bitsandbytes 4-bit, flash-attn, spconv for point
clouds) -- not reproducible on this Mac mini. See Obsidian:
`Papers/ORQA - Specialized Foundation Models for Intelligent Operating
Rooms.md` for the full paper notes (Background/Method/Novelty/Limitations/
Clinical implications).

## Day113 -- trying inference (not training) on Google Colab

Goal: run the smallest released distilled checkpoint (Dist-S,
`..._pkd_050_depthreduce4.zip`) purely for inference, entirely inside a
Colab VM so nothing about the base model or checkpoint touches local
storage. `day113_orqa_colab_inference.ipynb` is the notebook -- upload it to
colab.research.google.com, set the runtime to a T4 GPU, run top to bottom.

Uses LLaMA-Factory's own `ChatModel` Python API (the code path ORQA's
inference is actually built on) rather than hand-rolled model loading, so
it reuses tested logic for combining the quantized base model with the LoRA
adapter.

**Status: not yet working end to end.** Every attempt so far got stuck in
dependency setup, in this order, each fix moving the failure to a new,
shallower spot rather than resolving it outright:

1. `torch==2.4.1` no longer resolvable on Colab's pip index (Colab's index
   has moved past it) -- fixed by not pinning torch at all
2. `wandb`'s interactive login prompt during LLaMA-Factory's install left a
   blank Colab input box with no visible question (pip's `-q` flag
   swallowed the prompt text) -- fixed by setting `WANDB_DISABLED=true`
   before install
3. `flash-attn` failed to build from source on Colab (no prebuilt wheel
   matches Colab's current torch/CUDA/Python combo) -- skipped entirely,
   falls back to SDPA attention at inference time (the training config's
   own comment warns SDPA "does not work" for *training*; untested whether
   that also holds for inference only)
4. Re-running cells without a full runtime restart nested the git clone
   inside itself (`.../LLaMA-Factory/ORQA/Qwen2-VL/LLaMA-Factory/...`) --
   fixed by pinning every cell to an absolute `/content/...` path and
   making the clone step idempotent
5. `numpy` binary-incompatibility errors, in both directions, across
   several attempts -- some dependency in this pinned 2024-era stack
   expects numpy's pre-2.0 ABI, another expects a numpy>=2.0-only feature
   (`numpy.dtypes.StringDType`). Installing everything in one combined pip
   call (instead of two separate invocations fighting each other) plus
   explicitly pinning `numpy==2.0.2` got past this specific error
6. Most recent error: `ModuleNotFoundError: No module named 'trl'` -- a
   missing plain dependency, not a version conflict; the easiest kind of
   failure in this whole sequence, and where Day113 stopped

**Reading**: this is less about ORQA specifically and more a case study in
research-code bit-rot. ORQA's `requirements.txt` pins versions from
roughly two years before this attempt -- reasonable at the time (exact
pins protect a paper's reported numbers from silently drifting if a
library's internals change), but nobody has strong incentive to keep a
research repo's dependencies current after publication, while the
ecosystem underneath keeps moving. numpy 2.0 (2024) was an unusually
disruptive breaking release across the ML ecosystem, and this is a
textbook case of the gap it left behind.

## Next steps (not yet done)

- Install `trl` and continue past wherever the next dependency gap turns
  up, or decide the dependency chase itself has run its course

## Files

- `day113_orqa_colab_inference.ipynb` -- Colab notebook, image+text
  inference against ORQA's Dist-S checkpoint via LLaMA-Factory's
  `ChatModel` API; see Day113 notes above for its fix history
- `ORQA/` -- upstream clone (not committed, see `.gitignore`) used locally
  only to inspect config files, checkpoint paths, and `requirements.txt`
  while writing the notebook
