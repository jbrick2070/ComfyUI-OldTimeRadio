# No-Fallbacks Rip — stack-wide "if it fails, it fails hard"

**Operator directive (2026-07-03):** no fallbacks for ANY model — all LLMs, all
video, all image, all audio, all TTS. A model failure is a hard stop (fail loud),
never a silent swap to another model or a degraded output.

Grounded inventory: three read-only fan-out audits (audio/tts, video/image,
llm/cloud), 2026-07-03. Cloud voice (S1 @ 925438e2) + cloud music (S5 @ c7da53b1)
were BUILT no-fallback from the start — this rip is about the LOCAL model lanes.

---

## A. RIP — true model fallbacks (silent swap / soft-fail that hides a failure)

### Audio / TTS
1. **Bark missing-ref net** — `nodes/_otr_voice_node_common.py:27-44, 462-552`.
   A cloning engine (indextts2/chatterbox/dia) with no usable voice_ref renders
   that line on **bark** instead. RIP → raise `EngineUnusable(MISSING_MODEL/REF)`.
   Drop `missing_ref_fallback` from adapter metadata + the whole `_bark_fb` branch.
2. **`_resolve_clone_ref_path` gender/any-ref best-effort** — `_otr_voice_node_common.py:76-135`.
   Never raises; picks ANY ref as last resort. RIP → raise when no matching ref.
3. **`_resolve_character_voices_fail_soft`** — `nodes/cast_lock.py:387-513`.
   Never raises; orphan lines fall to "node-81 engine fallback" (:511). RIP →
   fail loud on an unvoiceable character line.
4. **`_fallback_voice_identity`** — `cast_lock.py:351-369`. Deterministic
   `v2/en_speaker_N` synthesis for a missing preset. RIP → raise.
5. **Kokoro voice-id swap** — `nodes/_otr_audio_engines/eng_kokoro.py:158-174`.
   Missing `.pt` → swap to the seeded episode voice. RIP → raise `EngineUnusable`.
6. **Stage-direction silence** — `_otr_voice_node_common.py:430-442`. Empty
   prepared text → 0.3s silence, skip engine. JUDGMENT (see C): a beat with no
   dialogue is arguably legit, but it IS a silent substitution.

### Image
7. **Per-role slot → other_beats fallback** — `nodes/otr_image_gen_dispatcher.py:158-159`.
   Empty named slot silently uses the global other_beats image model. RIP → raise
   on an unresolved explicitly-named slot.
8. **Scene-still-missing soft-degrade** — `nodes/_otr_video_engines/render_driver.py:1025-1029`.
   image_to_video/static_motion with no scene still → "pre-spine init" fallback.
   RIP → hard raise (the rest of render_driver is already no-fallback).

### LLM / Writer
9. **Voice-preset healthcheck swap + pool exhaustion** — `OTR_LedgerScriptWriter.py:682-759`.
   Disabled preset → same-gender sibling; exhaustion logs, no raise. RIP → raise.
10. **body-score-never-fails** — `OTR_LedgerScriptWriter.py:1603-1659`. Every
    feature error → score 0, biasing the reroll decision silently. RIP → raise
    (or at minimum log ERROR + surface) so a scoring break can't ship unnoticed.
11. **Contract / pitch / grammar soft-fails** — `OTR_LedgerScriptWriter.py:3169,
    3209, 3409, 3513, 3666, 3715, 3900, 3932, 4210, 4264, 4308`. Bare
    `except: continue  # never break the writer`. RIP → raise ValidationError.
12. **News degrade** — `story_orchestrator.py:2856-2915`. Retry budget exhausted
    → `meta["news"]=None`. RIP → raise when news is required (toggle-gated).
13. **Title / announcer-outro / news-coda template fallbacks** —
    `story_orchestrator.py:4100-4182, 4765, 4924-4928`. LLM fail → deterministic
    template. RIP → raise (these are model-output fallbacks).
14. **Character portrait 3-tier fallback** — `story_orchestrator.py:5393`. RIP →
    raise when all tiers exhausted.
15. **OpenRouter model-gone remote fallback** — `_otr_openrouter_backend.py:1045-1060`.
    404 → one remote fallback slug. JUDGMENT (see C): already one-shot + loud, but
    it IS a model swap.

---

## B. KEEP — NOT model fallbacks (ripping these breaks correctness, not a swap)

- **INPUT_TYPES / `build_engine_combo` / `load_resolver` C-5 safety** —
  `_otr_voice_node_common.py:162-180`, `_otr_engine_profiles.py:342-354`. A widget
  list must NEVER crash or ComfyUI can't load the node pack. Dispatch path already
  fail-loud via `require_resolver()`. KEEP (add a debug log only).
- **Transient network retry ladders** — `_otr_openrouter_backend.py:1037-1143`,
  `_otr_ollama_backend.py:239-266`. Retrying the SAME model on 429/503 is not a
  fallback. KEEP (already fail-closed after budget).
- **Teardown `except: pass`** — `_otr_voice_node_common.py:587-603` etc. Cleanup
  must not mask the render result. KEEP.
- **`empty_audio_batch` for a zero-dialogue scene** — `_otr_resolved_request.py:158`.
  No model involved; silence is the correct output for a scene with no lines. KEEP
  (make the log LOUD).
- **Engine import-time `except: pass`** — `_otr_video_engines/__init__.py`,
  `_otr_audio_engines/__init__.py`. A missing optional dep must not break the pack
  import. KEEP (add a warning log so the drop is visible).
- **Observability best-effort** — heartbeat/provenance/settlement swallow
  (`cloud_media_invoke.py:238, 530, 577`). KEEP (does not hide a model failure).

---

## C. JUDGMENT CALLS — RESOLVED by operator 2026-07-03

1. **Stage-direction-only silence (#6): RIP the silence.** Operator was surprised
   the ledger even carries stage-direction-only beats. Decision: it must NOT emit
   silence. Fail LOUD if an empty-after-clean (stage-direction-only) line reaches
   the voice gate, so the writer never silently ships silence and such lines can't
   creep into dialogue. Future idea PARKED in `docs/ROADMAP_IDEAS.md`: route a
   stage-direction beat to a NEW media engine (overlay video / procgen / 3D /
   still) instead of a voice — re-add to the ledger then.
2. **OpenRouter model-gone (#15): keep a CONSTRAINED backstop tied to "latest".**
   The dropdown should offer dynamic **"latest"** aliases as the DEFAULT plus a
   few standard version pins (~last 3) expected to stay available; the model-gone
   path may fall back only to those REAL valid pins (never an invented slug). This
   is why the operator wanted "latest": it resolves dynamically so a dead pin is
   rare. Folds into the dropdown-validity workstream below.
3. **rank-chain / local auto-select: KEEP.** It only fires when no explicit engine
   is chosen and already picks only valid registered engines (cloud excluded). Not
   a mid-render model swap. The operator's "latest / only-valid-models" directive
   applies to the CLOUD model dropdowns, not the local engine rank-chain.

## C2. NEW workstream (operator 2026-07-03) — valid-models-only dropdowns (R5)
The OpenRouter model list AND the Comfy cloud model dropdowns must expose ONLY
real, currently-valid models. Dead / stale model ids must not appear. Tracked
separately from the fallback rip (R5).

**Default = PARK a "latest" alias (operator 2026-07-03).** The four slot widgets
(`openrouter_slot_a_model`, `openrouter_slot_b_model`, `comfy_slot_a_model`,
`comfy_slot_b_model`) currently default to a `(enable ...)` placeholder that would
hard-fail if left unset. Decision: PARK a **"latest"** alias entry as the default —
a selectable dropdown value that resolves to the newest valid model at run time —
instead of the fail-on-unset placeholder. So the box-fresh default is "latest",
plus the ~last-3 real version pins as explicit picks (the model-gone backstop, C-2).

**Operator note / open tension:** ideally there'd be a CHEAP dynamic call to fetch
the live valid-model list so "latest" always resolves to a real current id. Since a
live per-run model-list fetch costs an API call + latency, "latest" is a PARKED
alias resolved from a cached/pinned map (refreshed occasionally), not a live call on
every episode. A cheap dynamic refresh (periodic/cached) is the nice-to-have.

---

## D. Sprint sequencing (each chunk green + committed+pushed to v2.0-alpha)

- **R1 — audio voice rip:** #1-#5 (+#6 per C1). One coherent change to
  `_otr_voice_node_common.py` + `cast_lock.py` + `eng_kokoro.py` + the adapter
  `missing_ref_fallback` metadata. Retire the bark-fallback tests, add fail-loud
  tests. Full suite + Bug Bible.
- **R2 — image rip:** #7-#8.
- **R3 — LLM/writer rip:** #9-#14 (largest; many tests pin the soft-fail).
- **R4 — convergence:** kibitz r2/r4 + Fable final grounded gate (CLAUDE.md §9,
  high-stakes structural), workflow-JSON audit, no new must-fix.

Every RIP replaces a silent swap with a NAMED loud raise (EngineUnusable /
ValueError / RenderError) — never a bare `raise`. UTF-8, no BOM, SFW.

---

## E. r2 kibitz hardening (Codex panel + Claude anchor, grounded 2026-07-03)

Codex ran read-only against the real files; Claude wrote the anchor and grounded
every claim. ACCEPTED (CONFIRMED against code):

- **E1 [enum taxonomy].** There is NO `MISSING_REF` in `EngineUsabilityReason`
  (values: GATED_BY_FLAG, MISSING_MODEL, MISSING_HF_TOKEN, INCOMPATIBLE_PROFILE,
  NONCOMMERCIAL_BLOCKED, MALFORMED_CONFIG). The missing-ref fail-loud raise uses
  `EngineUnusable(engine, role, EngineUsabilityReason.MISSING_MODEL, detail="no
  usable voice reference for cloning engine <engine>: <char/line>")` — NOT an
  invented enum value.
- **E2 [split, do not blanket-rip `_resolve_character_voices_fail_soft`].** It
  bundles THREE behaviors (`cast_lock.py:387-513`): (3a) missing-preset synthesis
  [FALLBACK → RIP], (3b) mis-stamped-announcer REROUTING [CORRECTNESS routing, NOT
  a fallback → KEEP, move out], (3c) true-orphan reassignment [FALLBACK → RIP].
  Raise loud ONLY for a true character row still unvoiceable after CastLock; keep
  3b. Rename the function after (the `_fail_soft` name becomes false).
- **E3 [stale R3 citations — RE-GROUND before coding].** `story_orchestrator.py`
  is 2711 lines; the plan's `:4100-5393` news/title/outro/portrait citations are
  WRONG (inventory subagent hallucinated them). Real homes: writer soft-fails +
  outro in `OTR_LedgerScriptWriter.py`; line/body scoring in `_otr_line_composer.py`;
  character portrait fallback in `otr_meta_brief_image_prompt.py:derive_image_prompts`
  (contract literally says "never raises; never emits an empty prompt" — update the
  contract + callers/tests). R3 MUST re-grep every LLM-lane target before touching it.
- **E4 [`_otr_body_score` raise is inert unless the caller catch is narrowed].**
  The reroll/score block is wrapped in `except Exception` that keeps the original
  line. Ripping body_score to raise does nothing until that catch is narrowed to
  the LLM reroll call only. Same "swallowed by an outer catch" risk applies to the
  contract/pitch/grammar soft-fails — check each call site's surrounding catch.
- **E5 [no `ValidationError`].** The writer imports no project-local
  `ValidationError`; pydantic's is not a generic message exception. Use `ValueError`
  or a new named `WriterValidationFailure`, raised `... from exc`.
- **E6 [collapse R1a+R1b into ONE audio-voice commit].** cast_lock repair FEEDS the
  voice nodes; ripping the bark net (R1a) alone makes a repaired-orphan path raise
  before R1b lands, breaking the intermediate suite. Ship the bark-net + cast_lock
  + kokoro + stage-direction rips as one green commit.
- **E7 [stage-direction: add a WRITER-side assert too].** Failing only at the voice
  gate is late — the writer scrub keeps original text when stripping would empty the
  line, so parenthetical-only text can still flow. Add a pre-freeze assert for
  voiced lines emptied by the scrub; keep the TTS-side raise as defense-in-depth.
- **E8 [image slot raise — define "explicit"].** `resolve_engine_for_role` treats
  absent-key / empty-dict / empty-string alike. Define explicit-but-unresolved as
  `slot in image_policy["image_models"]` AND `_eid(models[slot]) == ""` → raise
  there; define absent-key behavior separately.
- **E9 [verify `_bark_health_check_for_cast` has a caller].** It is defined in
  `story_orchestrator.py` but grep finds no writer call — verify it is live before
  spending R3 time ripping it.

REJECTED / SCOPED OUT:
- **Workflow-JSON audit in R4:** CUT for R1-R3 (runtime-behavior rips change no
  node input/widget). It applies ONLY to the R5 dropdown work (which does change
  widget values). Moved to R5.
- **C2 valid-model dropdowns not implementable "live":** confirmed — there is no
  live catalog source and V3 combos are excluded from the pin. Resolved by the
  operator's "park a latest" alias (cached, not a per-run live fetch); R5 defines
  the refresh source. Keep C2 OUT of R1-R4.

Next gate: the LIMITED internal Fable pass runs as the FINAL grounded gate on the
EXECUTED R1 audio-voice diff (CLAUDE.md §9 — after codex, right before merge), not
on this doc. Antigravity (agy) manual review is queued (AGY_MANUAL_PROMPT.md).

## F. R1 EXECUTED + Fable gate (2026-07-03)
R1 shipped (bark missing-ref net + _resolve_clone_ref_path-None + cast_lock
preset-synthesis + orphan-reassign + kokoro voice-id swap + stage-direction
silence all -> named loud raises; announcer reroute KEPT). Full suite 6140 + Bug
Bible green; 6 pinning tests inverted to assert the raises. Fable final gate =
**SHIP** (no build-breakers; no dangling refs; happy path byte-safe; announcer
reroute still reached; kokoro imports in scope).

**R1c — NEXT rip target (Fable caught, inventory missed):**
`scene_sequencer.py:846-866` has a SURVIVING silent **"Inline-Bark fallback"** —
when pre-rendered clip counts run short the sequencer inline-generates bark. Happy
path is unaffected (counts match), but it contradicts the global no-fallback
directive. Pinned by `test_sequencer_ledger.py:293`. Rip it next (fail loud on a
clip-count shortfall). Also a legacy-graph note: ledgers with no
`meta.cast_contract.cast_seed` now raise VoiceCastingError unconditionally
(intended — not a regression).

## G. r2 agy round (Antigravity manual panelist, grounded by Claude 2026-07-03)
Operator ran the agy manual review (AGY_MANUAL_PROMPT.md); Claude grounded it.
VERDICT: SOUND. It CONFIRMS R1 was safe + refines the R2/R3 map. Grounded survivors:

- **[CONFIRMED, 2 panels] E4 body_score swallow — exact caller.** Ripping
  `_otr_body_score` (OTR_LedgerScriptWriter.py:1603-1659) to raise is INERT: the
  caller at `OTR_LedgerScriptWriter.py:4689-4700` wraps the reroll block in
  `except Exception as _bg_exc:  # never break audio` and keeps the original line.
  R3 MUST narrow that catch (or let a WriterValidationFailure propagate).
- **[CONFIRMED] Dead bark health checks — DELETE, don't refactor.**
  `_bark_health_check` (story_orchestrator.py:649) + `_bark_health_check_for_cast`
  (:691) have NO live callers (only self-referential docstrings). Delete both in
  R3 rather than spend rip/test effort (supersedes plan item #9's "rip").
- **[CONFIRMED] R1 was safe because the writer produces valid casts.**
  `lock_cast` (OTR_LedgerScriptWriter.py:3005) guarantees every cast row a valid
  preset and maps lines only to existing cast IDs by construction — so the R1
  cast_lock fail-loud raises stay dormant on real episodes. NOTE: `repair_orphans`
  (_otr_cast_repair.py:194) is UNWIRED (no caller), so post-R1 any LLM dialogue-tag
  DRIFT that invents an orphan now fails loud as VoiceCastingError (was: silent
  reassign to another character). This is the intended no-fallback outcome.

**R2 test-inversion checklist (invert in the SAME commit, never delete):**
- `test_still_spine_helpers.py::test_slot_fallback_is_flagged` (:569-574) — empty
  `music_visual` slot must RAISE, not return `fell=True`.
- `test_still_spine_helpers.py::test_st4_still_index_and_family_init` (:724-733) —
  missing scene still must RAISE, not fall back to the `"portrait"` init source.

## H0. R3 DESIGN DECIDED (operator 2026-07-03) — explicit writer_fallback, no silent template
Fable split the writer soft-fails into (Class 1) model->template SILENT swaps and
(Class 2) defensive polish catches. Operator's resolution (better than pure
hard-fail): the objection is to HIDDEN swaps, not to a chosen fallback. So:

- NEW visible widget `writer_fallback` on the writer node: `[fail_hard | <a
  specific backup LLM>]`, DEFAULT = `fail_hard`. Add it to INPUT_TYPES + the
  workflow JSON (source of truth, CLAUDE.md §0) in the same change.
- The 5 CLASS-1 sites (news degrade, title, announcer-outro, news-coda, character
  portrait): when the PRIMARY writer LLM fails/returns junk -> if a backup LLM is
  chosen, RETRY on it -> if the backup ALSO fails (or writer_fallback=fail_hard),
  FAIL LOUD (named). The canned templates are REMOVED -- no silent filler EVER.
- CLASS 2 defensive polish catches: KEPT, but each must be LOUD (ledger + log
  stamp) so a skipped enhancement is never silent (Fable).
- DELETE dead `_bark_health_check` + `_bark_health_check_for_cast`.

Build order: (R3a) find the central writer LLM-invoke seam + add the
fallback-then-hard-fail wrapper gated on writer_fallback; (R3b) remove the 5
template fallbacks -> route through the wrapper; (R3c) Class-2 loud-stamp audit +
dead-code delete; workflow JSON widget in the same change; invert the mapped
tests; Fable gate before merge (§9).

## H. R3 grounded map (agy 2026-07-03, Claude-grounded) + scope split
Two CLASSES of R3 site, which matters for scope:

MODEL->TEMPLATE fallbacks (a MODEL failed -> silent template/degrade; the faithful
"no model fallbacks" targets):
- news degrade: OTR_LedgerScriptWriter.py:2939 (required->halt) + :2956 (optional
  ->meta['news']=None, degrade to raw seed). Test: test_news_briefs_required.py.
- title template fallback: LedgerScriptWriter.py:5251-5252 (regen empty->outline.title)
  + the swallowing catch :978-984 (_generate_title_from_script returns ""). Test:
  test_writer_title_scratchpad.py.
- announcer-outro template: _otr_line_composer.py:3519-3524/3600-3602/3657-3662
  (catch :3652). Tests: test_announcer_passes.py::test_compose_announcer_outro_
  llm_raises_falls_back + ::_multiline_output_falls_back.
- news-coda template floor: _otr_line_composer.py:3446-3457. Test:
  test_announcer_kill2_c3.py (retries_then_floor / floor_deterministic / varies).
- character portrait 3-tier: otr_meta_brief_image_prompt.py derive_image_prompts
  :1113-1156 (contract says "Never raises; never emits an empty prompt", :1045).
  Tests: test_image_platform_c1.py::test_meta_brief_prompt_temp0_hash_reseed_
  fallback + ::test_meta_brief_consistency_gate_fallback;
  test_brief_prompt_finishing.py::test_image_person_guard_then_finish_no_retrigger.

DEFENSIVE-COMPUTATION soft-fails (a helper computation errored -> skip an
ENHANCEMENT; NOT a model swap -- ripping these makes the writer abort on any
transient hiccup): body_score-never-fails :1603-1659 + its swallowing caller
:4689-4700 ("# never break audio"); contract :3169; pitch :3209; grammar :3409;
crisis-telemetry :3494; L1/L2 :3513; arc-shape :3666; dramatic-state :3715; slot-
drama :3900/:3932; tension-ramp :4210; next-turn :4264; a5-fields :4308.

Dead code to DELETE (both scopes): _bark_health_check + _bark_health_check_for_cast
(story_orchestrator.py:649/:691, no callers).

**R3 test-inversion checklist (invert in the SAME commit):**
- `test_announcer_passes.py::test_compose_announcer_outro_llm_raises_falls_back` +
  `::test_compose_announcer_outro_multiline_output_falls_back` (:355-384) — assert
  the raise, not `announcer_outro_fallback`.
- `test_image_platform_c1.py::test_meta_brief_prompt_temp0_hash_reseed_fallback`
  (:427) + `::test_meta_brief_consistency_gate_fallback` (:451) — empty/inconsistent
  LLM visual prompt must RAISE, not remap to the template.
- `test_brief_prompt_finishing.py::test_image_person_guard_then_finish_no_retrigger`
  (:90-97) — non-person prompt must RAISE, not return `template_person_guard`.
