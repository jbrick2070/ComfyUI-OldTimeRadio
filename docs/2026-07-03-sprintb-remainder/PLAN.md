# Sprint B remainder -- NEXT-CODE decision + build plan (kibitz target 2026-07-03)

Harden two things: (1) the SEQUENCING between `ideo_word` and B6, and (2) the
build contracts of whichever goes first. Anchor+judge = Cowork Claude; panel =
codex + antigravity (no claude CLI). Hard rules: NO fallbacks / dropdown-only
defaults; workflow-JSON same-change rule (BUG-LOCAL-097 positional widgets);
UTF-8 no BOM; SFW; suite + Bug Bible + push per green chunk; prod/main GATED.

## Already shipped this session (context, do not rebuild)

- Sprint A no-fallback rip (8de5862d); Sprint B S1 stills core B1-B5 (b5ef58bc);
  S1+1 `ideo` plain Ideogram engine (1bf2a2d2). HEAD 4ef78318, all pushed.
- Cloud image adapters: `nodes/_otr_image_engines/eng_cloud_image.py`
  (_CloudImageBase + Recraft/FluxPro/NanoBanana2/Seedream2/Ideo; render_image ->
  invoke_partner_node -> canonicalize_image -> str(png_path)).
- `nodes/_otr_shared/cloud_media_canonical.py::canonicalize_image` (cover+crop to
  exact canvas, sRGB PNG, sha256).
- `nodes/_otr_shared/cloud_model_ids.py` (V3 model-id resolver, never forwards
  the placeholder).
- `tests/test_cloud_partner_conformance.py` (billed-row coverage + emitted-kwargs
  declared; `_engine_by_node_key()` maps node_key -> ONE engine, last wins).
- Ideogram speed->price map + `_ideogram_speed()`/`_ideogram_est_usd()` already
  in eng_cloud_image.py (reusable by ideo_word).

## DECISION (proposed): build `ideo_word` BEFORE B6

Rationale: (a) `ideo_word` is the operator-priority "main build" of the S1+1 doc
and is cohesive with the just-shipped stills lane (same node_key cloud_ideogram_v4
as `ideo`, reuses the speed->price map, cache module from S0 c2). (b) B6's
portrait-mint gate exists to gate the 3D flag, and the 3D lanes are PARKED
(GO_FORWARD section 8: 3D GPU lanes held until S-3D-0 + operator green light) --
so B6 protects a feature that cannot yet run; lower urgency. (c) B6 touches the
beat loop + ShotLock ordering (higher blast radius) whereas ideo_word is
additive in the image namespace. Open question for the panel: is there any
ordering hazard that makes B6 a prerequisite for ideo_word? (Anchor says no.)

## `ideo_word` build contracts (the chosen-first build)

Source of truth: `docs/GO_FORWARD_NEXT/2026-07-02-ideogram-lyric-stills.md`.

1. **Two adapter classes, one node_key.** `ideo_word` is the words specialist for
   the video roles. Prefer TWO small classes (one per mode) over a dual-mode
   class (doc CUT-2). Both use node_key `cloud_ideogram_v4`. Emit the pinned
   {prompt, rendering_speed, resolution, seed}. CAPABILITIES rows + guarded
   __init__ import in the SAME change. Registered EMPTY default_roles.

2. **Prompt modes keyed by role, built in the composer (not the adapter).** A new
   `kind=lyric_card` path in the meta-brief image-prompt composer
   (compose_still_prompt today builds from meta/role/beat):
   - `lyric_text` (character/announcer): prompt text = the beat line excerpt via a
     PURE deterministic helper (first clause boundary, capped at 8 words) with
     tests for quotes/ellipses/em-dashes/stage-directions/empty. BYPASSES the
     scene composer's NO_TEXT_CLAUSE (text is the point); quoted-text shape per
     the existing ideogram profile.
   - `title_mood` (music_visual): wordless old-radio card from the procgen
     episode title + Meta brief + era tail (finish_visual_prompt chain), KEEPS
     the no-text directive. Verify-at-build the exact ledger/meta key for the
     episode title (new plumbing, small).

3. **Cost.** estimated_usd via the SHIPPED speed->price map (TURBO default). No
   new pricing code -- reuse `_ideogram_est_usd()`.

4. **Cache.** Wire `cache_lookup`/`cache_store` (S0 c2 cloud_media_cache.py)
   around the Ideogram call -- verify the RequestCacheKey pre-submit contract fits
   an image request. If it doesn't fit cleanly, drop the cache claim (doc codex
   r2 #8) rather than force it.

5. **Family invariant.** Worded cards (lyric_text) are per-beat unique, NEVER
   pooled. Wordless (title_mood) is the only pool-safe mode.

6. **Conformance test debt (MUST-FIX, found by anchor).** When `ideo_word` joins
   `ideo` on cloud_ideogram_v4, `_engine_by_node_key()` (last-wins) will only
   check ONE of them. Change it to iterate ALL engines per node_key so every
   adapter's emitted kwargs are conformance-checked.

7. **Workflow JSON.** v1 expects NO new widgets (V-11); the ImageDirector combo
   is dynamic (all_engine_names()), so registering ideo_word makes it selectable
   with no JSON change. VERIFY the selector exposes it from the saved
   otr_scifi_16gb_full.json; if not, JSON change in the SAME commit.

## Verify-at-build checklist

- Excerpt helper is PURE + deterministic + covered (5 edge cases).
- lyric_text BYPASSES, title_mood KEEPS, NO_TEXT_CLAUSE -- proven by tests.
- No fallbacks anywhere; a missing episode title fails LOUD (title_mood) rather
  than silently dropping to a generic card.
- Suite + Bug Bible + B7 green; push per green chunk; HEAD==origin.

## Open questions for the panel

- Sequencing: any real dependency making B6 a prerequisite? (anchor: no)
- Does splitting lyric_text/title_mood into two engine NAMES (both node_key
  cloud_ideogram_v4) collide with anything (registry keys are the engine name,
  not node_key -- anchor says safe)?
- Cache: does cloud_media_cache's RequestCacheKey fit an image request pre-submit,
  or is it video/duration-shaped?

---

## KIBITZ R1 OUTCOME (2026-07-03; panel = codex + antigravity, Claude judge)

Both agents CONVERGED (all claims grounded by me against the real code):

- DECISION CONFIRMED: build `ideo_word` before B6 (B6 gates PARKED 3D; lower urgency).
- ONE public engine `ideo_word` (name "ideo_word", node_key cloud_ideogram_v4),
  branch internally -- NOT two engine names (registry keys by engine name ->
  two names = two dropdown entries in every role).
- title_mood title source = `meta["episode_title"]` (writer-stamped; read first by
  video title resolution). Empty -> fail LOUD.
- Conformance `_engine_by_node_key()` last-wins -> change to node_key -> list,
  assert ALL engines (MUST-FIX same commit as ideo_word).
- Doc correction: image-model dropdowns live on `OTR_VideoDirector` (node 87),
  combo dynamic from `all_engine_names()`; `OTR_ImageDirector` has NO image
  widgets. Registering ideo_word auto-appears; defaults stay flux_gen1 (no JSON
  change) + ADD a test that "ideo_word" is in the combo.
- CUT cloud_media_cache from v1 (lyric cards per-beat unique -> near-zero
  cross-episode hit rate; the dispatcher's in-run content hash already stops
  same-run regen). Accepted both agents.

### LOAD-BEARING FORK (panel-surfaced, grounded; needs a call)

The image dispatcher resolves engine PURELY BY ROLE -> slot
(`otr_image_gen_dispatcher.py:151-173`); the prompt composer builds objects with
`compose_still_prompt(meta, kind=tgt["kind"], role=tgt["role"])`
(`otr_meta_brief_image_prompt.py:1286-1308`) and never sees image_models. A role's
object set includes PORTRAITS + scene stills + plates, not just beat cards. So
selecting `ideo_word` for a slot under today's role-only routing turns EVERY still
in that role typographic -- it cannot be "just the beat cards" without new wiring.

Two ways forward (operator/next-round call):
- **(A) Whole-role typographic treatment (simplest):** `ideo_word` selected for a
  role = every still in that role becomes a period typographic card. Composer still
  must learn image_models to emit `kind=lyric_card` (lyric_text for character/
  announcer, title_mood for music) so the PROMPT matches. No dispatcher change.
- **(B) Per-object routing (bigger):** lyric cards route to ideo_word while
  portraits/plates in the same role stay on the role's default engine -- requires
  extending the role-only dispatcher to a per-kind override. More blast radius.

Anchor lean: (A) for v1 (matches "offered for the video roles" framing, no
dispatcher surgery), revisit (B) if eyeball wants portraits preserved. Composer
plumbing (pass image_models into generate() -> derive objects, emit lyric_card
when the role's engine == ideo_word) is required either way.
