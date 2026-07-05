# 2026-07-04 JSON Prompt Transplant -- kibitz hardening arc

**Status:** Kibitz-hardening pass in progress on the REAL upstream plan.

## Anchor docs (source of truth)

The real py-to-JSON transplant plan lives in the SIBLING repo, not here:

- `ComfyUI-OTR-UpstreamStoryLab` on branch `main` at commit `7df7c80`
- Absolute path on this machine:
  `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OTR-UpstreamStoryLab`
- Anchor files:
  - `docs\R1_ARCHITECTURE_AND_CODING_PLAN_V2.md`
  - `docs\JSON_CONTENT_PYTHON_BEHAVIOR_R1_R4_REWRITE.md`
- Supporting:
  - `docs\todays-plan-handoff.md`
  - `docs\story-engine-map-brief.md`
  - `docs\FABLE_FINAL_REVIEW_2026-07-02.md`
  - `docs\FABLE_FINAL_REVIEW_PROMPT_BEFORE_TRANSPLANT.md`
  - `docs\GO_FORWARD_PLAN.md`
  - `docs\PACK_AUTHOR_CHECKLIST.md`

## SUPERSEDED

The v0 anchor docs previously committed here under
`anchors/R1_PHASE_A_EXTRACTION.md` and `anchors/PHASE_A_INVARIANT_CONTRACT.md`
(and their earlier names before the Phase A rename) were **invented drafts
written before the real anchors were located**. They are **SUPERSEDED** by
the real anchors listed above.

They remain in place per operator directive (do NOT delete via history
rewrite) but do NOT rely on them for content. The kibitz arc is anchored on
the REAL docs in the sibling repo.

## Phase A vs Phase B carve-out

Operator directive: this arc is hardening PHASE A only.

- **Phase A:** py-to-JSON extraction of the 12 named seams + sci-fi lifted
  as a JSON profile alongside news / cinematic / radio. Byte-identical
  audio. No production code touched until Phase A ships green.
- **Phase B:** the full architectural transplant described by the real R1
  doc (4 orthogonal axes, bank + pipeline + style registries, bridge
  artifact, C1-C5 chunks). Gated on Phase A shipping soak-green.

## Final deliverable

`PHASE_A_JSON_EXTRACTION_PLAN_FINAL.md` in this folder, chunk-by-chunk
executable, committed + pushed to `v2.0-alpha` at the end of the arc.

`PHASE_B_STUB.md` is a placeholder for the follow-on architectural sprint.
