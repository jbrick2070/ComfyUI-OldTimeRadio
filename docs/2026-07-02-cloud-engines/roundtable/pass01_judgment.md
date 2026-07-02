# R1 judgment (Claude, sole judge)

Panel: gemini-3.1-pro-preview (TRUNCATED at 2k tokens -- verdict + top
point only; rerun at 12k from R2), gpt-5.5, deepseek-v4-pro (both full at
12k after the known reasoning-token retry). Anchor: claude (grounded).
R1 spend: $0.0694 + $0.1124 = ~$0.1818.

ACCEPTED (grounded):
- Split provider surfaces A/B; quarantine B; named fallbacks (GPT#1, DS#3,
  Gemini#1, anchor#2). CONFIRMED: _otr_comfy_backend.py is chat-only.
- Per-role reactivity matrix replaces blanket AUDIO-REACTIVE (GPT#2, DS#2,
  anchor#1).
- Import-gated registration replaces requires_flag trust (GPT#3).
  CONFIRMED registry.py ~line 151: NO GATED_BY_FLAG case; class docstring
  above is STALE (contradicts itself) -- flagged for a future docs fix.
- Adapters invoke bundled partner-node classes in-process; S0 schema
  pinning via /object_info (GPT#5, DS#1, anchor#3).
- Media canonicalization contract (GPT#6). Auth-broker + hidden-input
  wiring design; "no node surgery" claim retracted -- additive hidden
  inputs are node surgery (GPT#7).
- Billing cache / idempotent re-runs (anchor#4). Budget/fallback policy
  matrix decided (GPT#8). CHEAP labels demoted to candidate until priced +
  smoked (GPT#4). ToS audit as S0 deliverable (DS#5). Reactive smoke moved
  into S0 (GPT SHOULD-5). Structural (not byte) acceptance for cloud audio
  (anchor SHOULD-2). Profile default-override mechanism specified (DS#4).
  Audio registry described as parallel frozen impl, not shared base
  (GPT SHOULD-6). Voice cloning deferred post-S2 (unanimous).
  ElevenLabsTextToDialogue demoted to experiment flag (anchor CUT-1:
  breaks per-line captions/ledger granularity). SoniloVideoToMusic cut
  (anchor + GPT CUT-3). Meshy-rig aspiration sentence cut (anchor CUT-2).

REJECTED (with reason):
- GPT CUT-1 "cut 3D entirely": operator brief says "maybe 3d". Kept as
  docs-only appendix -- zero build cost, no registry tokens, no sprint.
- DS "no cuts identified": superseded by accepted cuts above.

VERIFY-AT-BUILD: consolidated as pass01_plan.md section 9 (9 items).

CONVERGENCE: R1 surfaced material arc changes (surfaces split, gating
redesign, canonicalization layer). NOT converged; proceed to R2 (coding).
