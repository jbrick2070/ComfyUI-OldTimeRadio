# Driver anchor review -- r1 (high-level arc) -- dynamic_story visual scope

Reviewer: Claude (Cowork driver; anchor panelist + sole judge).
Target: docs/2026-07-12-dynamic-story-visual-scope.md (docs-only design).
Grounding: all claims checked against the real Windows tree this session.

VERDICT: SOUND ARC, with three must-fix design gaps I already see from the
code, listed below. The core moves are correct: (1) enter through the existing
`VisualStyle` resolution seam so zero consumers fork; (2) store at namespaced
`meta.visual_direction` (the ONLY location that survives `_merge_with_disk` --
CONFIRMED at nodes/production_ledger.py:1387-1393 vs :1403-1413); (3) bind the
artifact to the freeze stamps + content hashes and fail closed on skew
(CONFIRMED freeze stamps at nodes/_otr_ledger_freeze.py:806-811).

## MUST-FIX (anchor)

- A1. **The doc's insertion point creates a THIRD consumer of FreezeCascade
  out[1] it does not rewire.** CONFIRMED in workflows/otr_canonical.json: LFC
  out[1] also feeds nothing else visual, but ShotLock in[0] (link 252) and
  MetaBrief in[0] (link 255) are BOTH listed for rewire in section 7.2 --
  correct -- yet OTR_ImageGenDispatcher receives its ledger from ShotLock
  out[0] (link 256), so the dispatcher inherits the artifact only if ShotLock
  passes meta through unchanged. CONFIRMED it does (`led` is parsed and
  re-serialized whole, nodes/otr_shot_lock.py:1040, 1142). No gap in behavior,
  but the doc should STATE the dispatcher's artifact copy arrives via
  ShotLock's re-serialization, or a reader will hunt for a missing wire.
- A2. **Pack `style_id` uniqueness vs the registry.** The artifact embeds
  `style_id: "dynamic_story"`; `_validate_row` checks id format, and the
  registry loader asserts filename==style_id for DISK packs
  (nodes/_otr_visual_styles.py:323-349). A synthesized in-memory pack with the
  same id as the dropdown sentinel risks colliding with option D1(b) (a
  placeholder pack file named dynamic_story.json). The doc's D1 must resolve
  this: if (b) is chosen, the synthesized pack needs a distinct id (e.g.
  `dynamic_story__<episode_id>`). Flag as a decision consequence, not a free
  choice.
- A3. **still_word typography/backdrop dict key sets are FIXED
  (noir/sci-fi/western/pulp/default, nodes/_otr_visual_styles.py:94-97), and
  the genre SELECTOR is Python-locked (otr_meta_brief_image_prompt.py:997-1002).**
  The LLM authoring a full v2 pack must fill all five keys of BOTH dicts even
  though only one genre is ever selected per episode. The doc has D5 but
  should make the cost explicit: 10 dict values authored, 1 used. Cheap
  mitigation: instruct the LLM to author the selected genre key richly and
  fill the others with the episode's default row.

## SHOULD-FIX (anchor)

- B1. Credits/dossier: the doc says credits may quote the artifact; verify the
  credits reader only consumes meta keys it knows -- additive key is safe
  (credits reads named keys, e.g. meta["credits_source_line"],
  OTR_LedgerScriptWriter.py:3646-3649). Confirmed safe; keep as a note.
- B2. The wire-size concern (D8) is real but understated: `script_json` is
  re-serialized at every hop (ShotLock, dispatcher); a multi-KB artifact rides
  each time. Fine at 30-420 words; cap notes anyway.
- B3. Section 6.1's "no fallback to sci_fi_radio" should also cover the
  DEFAULT lane explicitly: `get_visual_style` treats absent/empty as default
  (nodes/_otr_visual_styles.py:386-389) -- the dynamic branch must key on the
  SENTINEL VALUE, never on artifact presence, or a stray stale artifact could
  activate dynamic styling under a named pack. The doc implies this
  (precedence list) but should state artifact-presence-is-never-a-trigger.

## Anchor claims audit

- meta per-key merge survival: CONFIRMED (production_ledger.py:1403-1413).
- Top-level key drop risk: CONFIRMED (TOP_PRESERVE :1387-1393).
- Freeze stamps + verdicts: CONFIRMED (_otr_ledger_freeze.py:806-811).
- Resolve-once threading: CONFIRMED (otr_meta_brief_image_prompt.py:1601-1609).
- Dispatcher cache key contains prompt_hash: CONFIRMED
  (otr_image_gen_dispatcher.py:117-129).
- Writer stamp point + gate: CONFIRMED (OTR_LedgerScriptWriter.py:3334-3339,
  3651-3655).
