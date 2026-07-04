# Manual review packet for Antigravity (independent second take)

Paste this whole file to agy (or run: agy -p "<this file's contents>") from the repo root:
C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio

---

You are an independent reviewer. Read the REAL repository files yourself -- do not trust the document's claims. REVIEW ONLY: do not edit any file except writing your review to kibitz-runs\2026-07-03-widget-audit\antigravity_manual.md.

DOCUMENT UNDER REVIEW: kibitz-runs\2026-07-03-widget-audit\r1\final.md
(a widget-surface audit + cleanup plan for workflows\otr_scifi_16gb_full.json)

Context you must verify, not assume:
- ComfyUI litegraph JSON: widgets_values is POSITIONAL; removing a mid-list widget from INPUT_TYPES shifts every later saved value.
- Key files: nodes\otr_caption_burn.py, nodes\otr_post_upscale_procgen_blend.py, nodes\_otr_voice_node_common.py, nodes\cast_lock.py, nodes\_otr_delivery_profiles.py, nodes\otr_video_director.py, nodes\otr_video_render_batch.py, nodes\OTR_LedgerScriptWriter.py, nodes\OTR_LedgerFreezeCascade.py, nodes\_otr_workflow_apply.py, workflows\otr_scifi_16gb_full.json.

Your review format:
VERDICT: build-ready / yes-with-fixes / no
MUST-FIX BEFORE BUILD: numbered, each with file:line evidence you read yourself
SHOULD-FIX: numbered, with evidence
MISREADS IN THE DOC: anything the document claims that the code contradicts
CUT THESE: scope/over-engineering

Write the review to: kibitz-runs\2026-07-03-widget-audit\antigravity_manual.md
