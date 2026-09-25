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
6. `ModuleNotFoundError: No module named 'trl'` even after adding it to the
   combined pip install (Day113) -- installing it in its own separate cell,
   right before it's needed, worked
7. `llamafactory.data.collator` also unconditionally imports `open3d`
   (point-cloud library) even for a pure image+text run -- added it too
   (plain prebuilt wheels, unlike spconv/torch-scatter)
8. `ImportError: cannot import name 'AutoModelForVision2Seq' from
   'transformers'` -- installing `trl` pulled in `trl==1.14.0`, which
   requires `transformers>=4.56.2`, silently upgrading transformers to
   5.16.1 and removing the older `AutoModelForVision2Seq` API
   LLaMA-Factory's 2024-era code depends on. ORQA's own requirements.txt
   pins `trl>=0.8.6,<=0.9.6` for exactly this reason
9. (Day114) `pip install --force-reinstall "transformers==4.46.1"
   "trl==0.9.6"` to fix #8 cascaded into a much bigger mess: it also bumped
   torch to 2.14.0 (breaking torchvision, which wants 2.11.0) and dropped
   numpy back to 1.26.4 (breaking a dozen of Colab's own preinstalled
   tools -- jax, cudf, opencv, shap, etc., all via "pip's dependency
   resolver does not currently take into account all the packages already
   installed" warnings) -- and the *original* AutoModelForVision2Seq error
   was still there afterward. Stopped here; this runtime's package state is
   corrupted enough that further patching in place isn't worth pursuing.

**Reading**: this is less about ORQA specifically and more a case study in
research-code bit-rot, and in a mistake of our own. ORQA's `requirements.txt`
pins versions from roughly two years before this attempt -- reasonable at
the time (exact pins protect a paper's reported numbers from silently
drifting if a library's internals change), but nobody has strong incentive
to keep a research repo's dependencies current after publication, while the
ecosystem underneath keeps moving. numpy 2.0 (2024) was an unusually
disruptive breaking release across the ML ecosystem, and this is a textbook
case of the gap it left behind.

The mistake on our side: every install in this notebook went straight into
Colab's shared system Python (`/usr/local/lib/python3.13/dist-packages`),
which is *also* where Colab's own large preinstalled stack lives (jax,
cudf, google-colab's own tooling, a modern numpy/pandas/transformers).
Every attempt to pin this 2024-era stack's versions fought that shared
environment in both directions -- our old pins broke Colab's tools, and
Colab's newer already-installed packages (like the `trl` that quietly
pulled transformers forward) broke our old code. None of this would happen
inside an isolated virtual environment.

## Next steps (not yet done)

- Start over on a *freshly deleted* Colab runtime (Runtime -> Disconnect
  and delete runtime, not just Restart -- this one's package state is too
  tangled to keep patching), and this time create an isolated virtual
  environment inside it (`python -m venv` or similar) before installing
  any of ORQA's pinned dependencies, so this old stack never touches
  Colab's own preinstalled packages in either direction

## Files

- `day113_orqa_colab_inference.ipynb` -- Colab notebook, image+text
  inference against ORQA's Dist-S checkpoint via LLaMA-Factory's
  `ChatModel` API; see Day113 notes above for its fix history
- `ORQA/` -- upstream clone (not committed, see `.gitignore`) used locally
  only to inspect config files, checkpoint paths, and `requirements.txt`
  while writing the notebook
