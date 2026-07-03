# Antigravity manual review — credits-enrichment plan

Paste the block below into `agy` (or `agy -p`) from the repo root
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`.
Agy reads the real repo itself — it does not need the plan pasted, just the path.
When it returns, hand me its output and I will ground every claim against the code
and fold only the survivors in (I am the judge; agy is one panelist).

---
BEGIN PROMPT

You are an independent, code-grounded reviewer. Open and crawl this repo yourself:
C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio

Review this plan for correctness and implementability:
  docs/2026-07-03-credits-enrichment/GO_FORWARD_CREDITS.md

It enriches the end-roll credits (the Telemetry HUD in nodes/video_engine.py, node 12
OTR_SignalLostVideo, green-blended by node 93 OTR_PostUpscaleProcgenBlend). Operator
rules you MUST hold the plan to:
  - CLEANBREAK: rip the old surface + its fallbacks out, paste the new one in, fix the
    seams up- and down-stream in the SAME change. No dual paths.
  - NO FALLBACKS during the rip: a silent placeholder like "(not recorded)" or a blank
    voice is a BUG to remove, not a safety net. Temporary breakage between rip and paste
    is accepted; the branch is green only at each committed chunk.
  - Any node/widget/wiring change must land IN workflows/otr_scifi_16gb_full.json in the
    same change (litegraph; widgets_values is POSITIONAL, append-only).

Focus your review on:
  1. Is the render-ORDER diagnosis right? Node 12 renders the HUD before nodes 91
     (OTR_ImageGenDispatcher) and 92 (OTR_VideoRenderBatch) run. Confirm from the JSON
     link/order and the node code. Does that truly force the engine credits to a LATE
     render (S3) rather than a node-12 rewire?
  2. The S3 seam: is extending node 93 / adding a terminal OTR_CreditsRoll fed by
     clip_manifest_json (node 92 slot 1) + patched_ledger_json (node 91 slot 0) sound?
     Any hazard with the node-12 episode finalize/rename (~video_engine.py 2286-2315,
     2430+) if order changes?
  3. Data availability: is meta.voice_cast_decision + cast_voice_slots actually in node
     12's frozen ledger (stamped pre-freeze)? Is meta.render_engines saved to the
     production-ledger singleton but meta.image_engines only on a wire? Is CastLock's
     voice_engine/voice_ref_id wire-only (no singleton save)?
  4. The green-only channel constraint (node 93 colorchannelmixer zeroes R+B) — does the
     proposed multi-color roll survive as green luminance?
  5. The 45s mux tail budget (OTR_MAX_CREDITS_TAIL_S) vs the +50% font / longer roll (S0).
  6. Any hidden test that pins a HUD/credits/dossier widget or line COUNT that the rip
     would break (grep with ignore rules OFF).
  7. Best rip/paste ORCHESTRATION: which slices (S0 font, S1 Cast&Voices, S2 durability
     stamps, S3 late seam, S4 polish) can go truly parallel vs must serialize, and the
     safest order.
  8. AUDIO/MUX SEAM (highest concern): the mux is mux-LAST (node 85 OTR_MasterAudioMux
     takes node 93 silent video + master audio, fails loud if the silent video exceeds
     the master audio by >OTR_MAX_CREDITS_TAIL_S=45s, otr_master_audio_mux.py:149-153).
     Moving the credit roll to a late node changes final-video length and where the
     credit-roll audio comes from. Does the plan's audio contract (extend master audio to
     cover the roll, OR fit inside 45s; credit audio in the MASTER mix not a discarded
     clip; preserve what the mux consumes; keep test_audio_byte_identical green) actually
     hold against the code? Is there a way the late roll silently truncates or desyncs
     audio, or trips the 45s guard? This must not break the deliverable.

Output format:
  VERDICT: yes / yes-with-fixes / no
  MUST-FIX BEFORE BUILD: numbered, each with file:line and a concrete fix.
  SHOULD-FIX: numbered, file:line.
  CUT / over-engineering: anything not worth building.
  Ground every claim in files you actually read. Mark anything you could not verify.

END PROMPT
---

After it runs, give me agy's text. I will ground it against the real files (Desktop
Commander, never the Linux mount), discard misreads, and merge the survivors into the
hardened plan alongside codex's review and my own anchor.
