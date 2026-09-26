# ORQA inference exploration (toy)

Day112-115. Not a training project like the other `playground/` entries --
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

**Status (Day113-114): not yet working end to end** -- resolved on Day115, see below. Every attempt so far got stuck in
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

## Day115 -- isolated venv, authors' own loading path, and it runs

`day113_orqa_colab_inference.ipynb` was rewritten (same filename, so the
links above still work). Every step now raises and stops on failure
instead of `!cmd`, which never raises on a non-zero exit code.

**Environment.** A plain `python -m venv` would not have been enough:
Colab's Python is 3.13, and `torch==2.4.1` (like `open3d==0.18.0` and the
prebuilt `torch-scatter` wheels) only ships up to Python 3.12. The
authors used Python 3.10 (their README mentions
`ENV/lib/python3.10/site-packages`), so the venv is built on a
standalone CPython 3.10 via `uv` (`uv venv --python 3.10`). Nothing
ORQA-related is installed into Colab's own Python; the model runs as a
script (`orqa_infer.py`) executed by the venv's interpreter.
Re-reading the earlier errors with this in mind:
- #1 (`torch==2.4.1` "not resolvable") was most likely the missing
  Python 3.13 wheel, not Colab's pip index moving on
- #6 (`No module named 'trl'`) was most likely an unsatisfiable install:
  the combined call pinned `numpy==2.0.2`, while LLaMA-Factory's own
  `requirements.txt` says `numpy<2.0.0`, so pip installed nothing, and
  `-q` hid the resolver error
- The dependency set was checked offline first (`uv pip compile
  --python-version 3.10 --python-platform x86_64-manylinux_2_28`):
  transformers 4.46.1, trl 0.9.6, numpy 1.26.4, spconv-cu121,
  torch-scatter `+pt24cu121` all resolve with prebuilt wheels

**What following the authors' code (instead of `ChatModel`) turned up.**
Loading now mirrors `qwen2_vl_helpers.load_pretrained_model` (`_pkd`
branch), `ORQAWrapperQA`, and `web_demo_orqa.py`:
1. **Dist-S is not a LoRA adapter.** It is a complete, shrunken
   Qwen2-VL (LLM hidden 768 vs 1536, 8 of 28 LLM layers kept by
   `depthreduce4`) plus ORQA's custom visual pooler. Day113's
   `ChatModel(adapter_name_or_path=...)` could not have loaded it, and
   LLaMA-Factory's inference loader never reads `visual_block.pt`
   (`model/loader.py:173` only does so when a training argument is set)
2. **The authors' loader needs an unreleased `_pkd_teacher` checkpoint**
   as an architecture template. `Qwen/Qwen2-VL-2B-Instruct` stands in;
   the script asserts the rebuilt architecture matches the checkpoint's
   own `config.json` and counts parameters the checkpoint did not
   overwrite (result: 0)
3. **ORQA replaces transformers' `Qwen2VLImageProcessor`** via a
   `sys.modules` patch at the top of `scene_graph_prediction/main.py`;
   without it preprocessing fails (`batch_images_to_idx` /
   `pixel_values_to_batch_idx`). Copied verbatim, then asserted
4. **`PointTransformerV3` asserts `flash_attn` is importable in its
   constructor**, even for image-only use; a placeholder gets past
   construction, and the point-cloud branch is deleted immediately
   afterwards (asserted)
5. **The released Dist-S `visual_block.pt` is truncated** at exactly
   734,003,200 bytes (700 MiB); `torch.load` fails with "failed finding
   central directory". PKD trains with `only_llm=True`, which freezes
   every `visual.*` parameter, so the file should duplicate the visual
   weights in `model.safetensors`. The script walks the truncated zip's
   local headers and compares every readable tensor: 222 compared,
   0 different
6. T4 constraints: `eager` attention (FlashAttention-2 needs sm80+; eager
   is the authors' own non-CUDA fallback) and fp32 (no native bf16)

Measured size: the Dist-S checkpoint is 938.1M parameters including the
vision tower. The paper's "Dist-S 278M" most likely counts only the
language model.

**Result (2 images x 3 questions, greedy).**

| image | question | answer |
|---|---|---|
| OR photo (thoracoscopy) | List all entities in the OR. | head surgeon, drill, instrument table, assistant surgeon, drape, operating table, patient, anaesthetist, mps station, nurse, head surgeon, saw, mako robot |
| OR photo | What action is being performed at this time? | They are currently incision. |
| OR photo | Describe what you see in this image. | The position of hammer is 303, 443, 418, 724 in the image. |
| endoscope frame | List all entities in the OR. | The OR has mouth gag, other hands left. |
| endoscope frame | What action is being performed at this time? | They are currently incision. |
| endoscope frame | Describe what you see in this image. | The position of mouth gag is 622, 998, 428, 998 in the image. |

OR photo: "Operating Room", National Cancer Institute (NIH),
photographer John Crawford, public domain, via
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Operating_room.jpg).
Endoscope frame: frame 50 of the Day106 stomach-phantom episode.

**Reading (preliminary -- two out-of-domain images, not an evaluation).**
Answers come back in the label vocabulary of ORQA's training datasets.
Some fit the photo (surgeons, nurse, instrument table, drape), but it
also lists a drill, saw, MAKO robot and hammer -- typical of MM-OR's knee
arthroplasty -- for a thoracoscopy, and answers "incision" for both
images. The endoscope frame switches to EgoSurgery vocabulary (mouth
gag, other hands), so the image does influence the output; the model
seems to map an unfamiliar scene onto the closest training dataset's
frame of reference. This echoes object hallucination / language-prior
effects described for VLMs (CHAIR, Rohrbach et al. 2018; POPE, Li et
al. 2023) and, loosely, the action-conditioned toy model's
real-vs-zero-action finding -- but unlike that toy model, nothing here
was tested in-distribution, so it cannot yet separate "ORQA ignores the
image" from "ORQA is out of its domain".

## Next steps (not yet done; implementation paused from Day116)

Same three-condition design as the action-conditioned toy model
(real / shuffled / zero):
- **real**: in-distribution frames cropped from ORQA's own
  `figures/teaser.jpg` (4D-OR / MM-OR), no data request needed
- **zero**: a uniform gray image -- if it still gets "incision" and a
  similar entity list, those answers come from the prior alone
- **shuffled**: a frame from a different scene

## Files

- `day113_orqa_colab_inference.ipynb` -- Colab notebook (rewritten on
  Day115): isolated Python 3.10 venv, ORQA's own loading path, Dist-S
  image+text inference; Day113-114 history above
- `orqa_outputs.json` -- Day115 raw outputs (answers, token counts,
  timings, and the load-time verification results in `meta`)
- `ORQA/` -- upstream clone (not committed, see `.gitignore`) used locally
  only to inspect config files, checkpoint paths, and `requirements.txt`
  while writing the notebook
