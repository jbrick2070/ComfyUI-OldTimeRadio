# Qwen-Image removal review state

## Operator decision

Remove the optional Qwen-Image still engine completely. It is not a dependency
of OTR, Z-Image-Turbo, Qwen3/Qwen2.5 coding or writer models, WAN, HuMo, LTX,
Flux, Lumina, or the canonical workflow.

## Scope

Remove the Qwen-Image adapter and its registration/capability surface,
engine-specific smoke and CPU tests, and stale engine-only prose. Preserve the
literal `CLIPLoader(type="qwen_image")` contract in Z-Image-Turbo: that is the
ComfyUI loader type for Z-Image's Qwen3-4B text encoder, not the removed
Qwen-Image engine. Preserve all Qwen3 LLM references.

## Current implementation edits in the working tree

- Deleted `nodes/_otr_image_engines/qwen_image.py`.
- Deleted `scripts/_otr_qwen_image_smoke.py`.
- Deleted `tests/test_image_engine_c3.py`.
- Removed the Qwen-Image import from
  `nodes/_otr_image_engines/__init__.py`.
- Removed the `qwen_image` capability row from
  `nodes/_otr_image_engines/registry.py`.
- Removed Qwen-Image assumptions from capability, dependency-pilot,
  model-slot, dropdown, and workflow-apply tests.
- Cleaned one HiDream descriptive comparison that named Qwen-Image.
- Updated the model-slot audit, go-forward plan, and failure inventory to mark
  the engine removed while retaining historical receipts.

The active overnight campaign was started before this removal and its loaded
temporary matrix still contains historical Qwen pairings. Do not edit that
running file mid-campaign. Once it releases, remove or replace the stale
`still_flat`, `word_razzle`, and `humo_1.7B` Qwen image coverage rows; those are
test selections only, not approved production pairs. The separate video and
image dropdowns remain independently user-controlled.

## Required review

Read this state file, the real repository, and `git diff`. Verify that the
removed engine cannot still be selected or imported, that capability rows and
registry names remain one-to-one, and that no test still requires the removed
engine. Distinguish engine-owned `qwen_image` references from Z-Image's
ComfyUI CLIP loader type. Identify any stale engine-only docs or environment
variables. Do not edit code, workflow JSON, or docs. Do not remove Qwen3 LLM
support or Z-Image's loader type.
