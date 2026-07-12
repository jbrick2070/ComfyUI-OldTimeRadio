# r1 judgment -- dynamic-story-visual (Claude, sole judge)

Panel: codex `gpt-5.6-sol` @ reasoning `ultra` (operator directive: sol ultra),
antigravity `gemini-3.5-pro`. Driver anchor written first (driver_anchor.md).
Scope: r1 only, per operator directive ("R1 /kibitz codex 5.6 sol"); r2-r4
belong to the implementation phase (Codex owns all code).

## Grounding verdicts

### Codex (sol ultra)
- M1 story-binding unimplementable at the meta-only seam + raw arrays mutate
  post-freeze: **CONFIRMED** (get_visual_style meta-only,
  _otr_visual_styles.py:378-390; overlay_audio_timing mutates lines in place,
  otr_shot_lock.py:169-221; CastLock assigns voice post-freeze, acknowledged in
  _otr_ledger_freeze.py:493-502). FOLDED: immutable story PROJECTION hash +
  ledger-aware resolve seam at consumer entries.
- M2 ShotLock swallows the style failure 6.1 must abort: **CONFIRMED**
  (bare `except Exception: pass` around finish_visual_prompt,
  otr_shot_lock.py:626-636). FOLDED as must-fix.
- M3 two look authorities (brief beats pack): **CONFIRMED** (brief-first
  precedence, pack era_tail is only the fallback,
  _otr_story_brief_helpers.py:356-370, 401, 414, 428). FOLDED: dynamic lane
  makes the artifact pack the sole final-look authority; brief = evidence.
- M4 LLM must not author engine-safety pack fields: **CONFIRMED**
  (v2 safety-adjacent fields _otr_visual_styles.py:74-98, 117-123; talking-lane
  law otr_meta_brief_image_prompt.py:160-168). FOLDED: vetted safety base +
  look-only whitelist; Python-owned geometry lint.
- M5 evidence map incomplete / existence!=support: CONFIRMED as written.
  FOLDED (complete per-field evidence map; factual vs rationale split), kept
  lean.
- M6 replay/reroll contradictions (created_utc inside the hash; re-runs
  re-freeze): **CONFIRMED**. FOLDED: semantic_sha256 vs envelope split; reroll
  CUT from v1.
- M7 model receipt false premises (temp=0 rejected; no seed; virtual vs
  provider ids): **CONFIRMED** (otr_shot_lock.py:687-692 comment). FOLDED:
  effective-sampling receipt, "attempts", creative slot recommended.
- M8 VRAM teardown: **CONFIRMED** (FreezeCascade unload in finally +
  freeze_unload_ok stamp, OTR_LedgerFreezeCascade.py:453-478). FOLDED.
- M9 registration + three links: **CONFIRMED** (insertion needs
  LFC->Direction NEW link + two rewires; __init__.py registration). FOLDED.
- CUT scenes[]/shots[]: **PARTIALLY REJECTED** -- the operator product intent
  explicitly requires per-scene/per-shot decisions. Judge ruling: scenes[] cut
  (non-canonical key); shots[] KEPT but only with a MANDATORY consumer on the
  dynamic lane (MetaBrief beat stills), beat_id-keyed. Stored-but-unconsumed
  would indeed be dead scope (feedback_rip_legacy_dead_code).
- CUT continuity/wardrobe/motifs/clue_visual/composition_rules: **PARTIALLY
  FOLDED**. continuity CUT (meta.continuity already owned,
  OTR_LedgerScriptWriter.py:4721-4746; character look = cast rows). wardrobe
  stays orthogonal (agy agrees). motifs/clue_visual KEPT as non-executable,
  evidence-bound rationale (operator intent), clearly non-authoritative.
  composition_rules demoted to rationale (geometry law).
- CUT reroll, placeholder-pack D1(b), gate_in/done/credits extras: FOLDED
  (done gate kept ONLY as the standard opaque ordering idiom -- zero cost,
  matches CastLock/ShotLock; credits integration cut).

### Antigravity (gemini-3.5-pro)
- M1 stale-hash deadlock from post-freeze timing/voice mutations: **CONFIRMED**
  (same as codex M1). FOLDED via projection hash (field whitelist per table).
- M2 meta-only seam cannot hash arrays: **CONFIRMED**. Its peek_ledger() fix is
  **REJECTED** -- consumers parse the WIRE ledger (e.g. MetaBrief json.loads,
  otr_meta_brief_image_prompt.py:2134-2144); the singleton can lag or be absent
  in that seam. Codex's ledger-aware entry resolve adopted instead.
- M3 sentinel gate deadlock at the writer: **CONFIRMED**
  (resolve_visual_style raises pre-story, OTR_LedgerScriptWriter.py:3334-3339).
  PROMOTED from D1-consequence to must-fix; code-side sentinel exemption.
- M4 ComfyUI re-run caching deadlock for rerolls: CONFIRMED as a real ComfyUI
  property; moot for v1 (reroll cut) -- FOLDED as a note on the future widget.
- S1 VRAM: duplicate of codex M8. FOLDED.
- S2 import isolation: **CONFIRMED** law (repo posture). FOLDED one line.
- S3 _STYLES cache poisoning: **CONFIRMED** design constraint
  (_otr_visual_styles.py:170, 355-359). FOLDED: dynamic pack never enters the
  module cache.
- OPT per-shot notes stay artifact-side: agrees with judge ruling on D3.

## Deliverable
Survivors folded into docs/2026-07-12-dynamic-story-visual-scope.md (rev 2).
final.md in this folder is the rev-2 copy.
