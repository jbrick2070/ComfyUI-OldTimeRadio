# LTXAVTextEncoderLoader Access Violation Diagnosis

## Question

What is the most likely root cause of a `Windows fatal exception: access
violation` thrown by `LTXAVTextEncoderLoader.execute()` while loading
Gemma 3 12B + LTX 2.3 22B via `comfy.sd.load_clip`, and what is the cheapest
diagnostic to disambiguate between the two top candidates?

## Hard facts

* Platform: Windows 10, RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120.
* Software: ComfyUI Desktop 0.21.1, Python 3.12.11 (uv-managed), torch
  2.10.0+cu130, safetensors 0.7.0.
* The crashing node is `LTXAVTextEncoderLoader` (lives in
  `comfy_extras/nodes_lt_audio.py` despite the filename -- "lt" is "LTX-V",
  not "load_torch"). Source:

  ```python
  @classmethod
  def execute(cls, text_encoder, ckpt_name, device="default"):
      clip_type = comfy.sd.CLIPType.LTXV
      clip_path1 = folder_paths.get_full_path_or_raise(
          "text_encoders", text_encoder)
      clip_path2 = folder_paths.get_full_path_or_raise(
          "checkpoints", ckpt_name)
      model_options = {}
      if device == "cpu":
          model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
      clip = comfy.sd.load_clip(
          ckpt_paths=[clip_path1, clip_path2],
          embedding_directory=folder_paths.get_folder_paths("embeddings"),
          clip_type=clip_type,
          model_options=model_options)
      return io.NodeOutput(clip)
  ```

  Line 203 of `nodes_lt_audio.py` (the line in the crash stack frame) is
  the `clip = comfy.sd.load_clip(...)` call.

* Widgets passed (workflow node id 57):
  * `text_encoder`: `gemma_3_12B_it_fp4_mixed.safetensors`  (file size:
    9,447,702,218 bytes = 9.45 GB on disk)
  * `ckpt_name`: `ltx-2.3-22b-dev.safetensors`              (file size:
    46,149,344,974 bytes = 46.15 GB on disk)
  * `device`: `"default"`

* Crash stack (most-recent-first):

  ```
  Windows fatal exception: access violation
  Stack (most recent call first):
    File "...\torch\storage.py", line 468 in __getitem__
    File "...\ComfyUI\comfy\utils.py", line 136 in load_torch_file
    File "...\ComfyUI\comfy\sd.py", line 1241 in load_clip
    File "...\ComfyUI\comfy_extras\nodes_lt_audio.py", line 203 in execute
    File "...\comfy_api\latest\_io.py", line 1833 in EXECUTE_NORMALIZED
    File "...\comfy_api\internal\__init__.py", line 149 in wrapped_func
    File "...\ComfyUI\execution.py", line 297 in process_inputs
    ... [normal execute() dispatch]
  ```

* Pre-load GPU state (from the comfy log just before crash):
  * FLUX loaded "full load: True", 22,700.13 MB resident (on a 16 GB
    physical card -- dynamic offloader in effect).
  * AudioEncoderLoader had already loaded `whisper_large_v3_fp16.safetensors`
    earlier in the run (the file is canonical Comfy-Org/HuMo_ComfyUI byte-
    identical, 3 GB, encoder + decoder both present).
  * `Found quantization metadata version 1` + `Using MixedPrecisionOps for
    text encoder` lines appear immediately before the access violation,
    consistent with fp4_mixed quant being applied on the Gemma 3 12B
    weights.

* Same workflow JSON ran end-to-end successfully at tag
  `v2.0-alpha-cleanbreak` (commit `1aed66d`, 2026-05-12). The
  `nodes_lt_audio.py` / `LTXAVTextEncoderLoader` class is bundled with
  ComfyUI Desktop and may have been added/changed in a version bump
  between 2026-05-12 and 2026-05-17.

* `comfy.sd.load_clip(ckpt_paths=[gemma_3_12B_fp4_mixed, ltx_2.3_22b_dev],
  clip_type=CLIPType.LTXV, ...)` is being asked to walk a 9.45 GB fp4-
  mixed encoder file AND a 46.15 GB diffusion checkpoint and reconstruct
  a CLIP-shape text encoder out of their combined weights.

## Top two candidate root causes

**(P1) Quantization-aware load path defect.** safetensors 0.7.0 + torch
2.10.0+cu130 + comfy.sd.load_clip on a fp4_mixed quantized text encoder
where `Found quantization metadata version 1` triggers a code path that
mis-allocates tensor storage. The Blackwell sm_120 architecture is new
enough that path may not be widely exercised. Storage `__getitem__`
indexing into wrong-shape pages reads unmapped memory -> access
violation.

**(P2) 46 GB checkpoint mmap / sharded-load issue.** `comfy.sd.load_clip`
called with two ckpt_paths walks both files. The 46.15 GB
`ltx-2.3-22b-dev.safetensors` is far above what CLIP loaders typically
handle. Either an `int32` offset overflow inside safetensors header /
load_torch_file or an mmap window that exceeds the 16 GB VRAM ceiling
causes storage indexing into invalid memory.

## Cheapest disambiguating test

A cold-launch isolation run with ONLY the `LTXAVTextEncoderLoader` node +
a trivial `CLIPTextEncode` consumer (force the CLIP to actually resolve)
will tell us:

* Crash with same signature -> bug is intrinsic to the loader / file /
  version triple. Co-residence with FLUX is not the cause. Repro for
  upstream issue.
* Clean load -> bug is co-residence + dynamic offloader fragmentation.
  Fix lives in HuMo's pre-LTX unload sequence or workflow execution
  order, NOT in the loader. Workflow JSON correctly wires LTX but the
  graph schedules LTX after FLUX without unloading.

## What I want from the round-robin

1. Among (P1) and (P2), which is more likely given the four hard
   constraints (ComfyUI 0.21.1, torch 2.10.0+cu130, safetensors 0.7.0,
   Blackwell sm_120) and the fact the same workflow + files worked five
   days ago at `v2.0-alpha-cleanbreak` on the same machine?
2. Are there other root cause candidates I haven't considered (CUDA
   driver regression, ComfyUI Desktop version bump between 2026-05-12
   and 2026-05-17, etc.)?
3. If the isolation test in Track 1 produces a clean load, what's the
   most targeted second test to isolate the co-residence interaction
   without spinning a full episode? (Some incremental adds of FLUX,
   Whisper, etc. before LTX, in a small graph, to find the exact
   trigger.)
4. If the isolation test crashes with the same signature, what's the
   minimal upstream-issue repro shape that ComfyUI maintainers would
   accept?

## Hard stops (already locked)

* No model swap.
* No workflow data edit.
* No version bump until root cause is locked.
* No `wrong_model` failure class in the harness.
* No code change to `scripts/otr_api.py` or harness beyond commit
  `0facea7` (worker process-death detection).
