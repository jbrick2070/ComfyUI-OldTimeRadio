# ANNOUNCER REDESIGN + NEWS CODA + KILL-2 -- HARDENED (pass01, post-R1)

R1 (creative arc) converged hard: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro +
the Claude anchor ALL independently landed the same three roots. This plan folds
the verified survivors. Section ids are stable for traceability.

OPERATOR THESIS (unchanged): the show TEACHES. Drama delivers; the NEWS is the
payload, explicit at the very end -- framed deliberately, never stealing the
characters' climax. All behind the `story_scaffold` flag; byte-identical off.

---

## 0. GROUNDED SEAM INDEX (verified this session)
- Open beat intent hardcoded `_otr_outline.py:1591` -- but the open's CONTENT is
  composed by `compose_announcer_intro` (`_otr_line_composer.py:2709`), which
  reads ONLY `script_brief`. Editing the intent does NOT change the open.
- `fallback_announcer_intro` (`:2614`) echoes `script_brief` verbatim.
- `compose_announcer_outro` (`:2778`) carries `news_close_brief` (the REAL news)
  + `ending_change` (the FICTIONAL outcome) + `final_character_line`. The F3
  "State this outcome plainly" branch (`:2854`) is about the FICTIONAL ending.
- `_ANNOUNCER_OUTRO_SYSTEM` (`:2536-2555`) FORBIDS news-summary/lesson framing +
  demands a concrete final image -- hostile to an explicit news coda as written.
- `render_style_grammar` (`:678`) ZERO callers; catalog docstring (`:5-7`):
  `sound_world` = "concrete audio palette (feeds dialogue mood + visualizer/LTX
  render prompt)"; `story_engine` = the conflict shape; `ending_mode` = the climax
  landing. The fields ARE concrete -- but `sound_world` is AUDIO/scene vocabulary,
  not dialogue content.
- The writer already computes `_climax_beat_id` (`OTR_LedgerScriptWriter.py
  :3266-3273`) and injects the ending template ONLY on that beat (`:4166-4171`).
  Today climax == last char beat (forced), so a climax-keyed outro is
  byte-identical NOW and future-proofs KILL 3.
- KILL-4 enrich gate `if beat_role in (PERSONAL_STAKE, IRREVERSIBLE_CHOICE)`
  (`_otr_story_quality_l12.py:795`); `[:_INTENT_MAX]` truncation AFTER enrichment
  (`:800`).

---

## 1. KILL 2 -- StoryContract: shape the body where it can be shaped (REWRITTEN per R1)

The R1 verdict (unanimous): injecting the full grammar into every prompt is the
same single-prior trap KILL 1 disproved AND -- grounded against the repo's
stage-direction-leak history (L3/L4 ACTION-strip, docs/2026-06-22-stage-direction-
leak) -- injecting `sound_world` AUDIO vocabulary into character-dialogue line
prompts will make weak models LEAK sound cues ("[ticking clock]") into dialogue.

So inject the style WHERE IT BELONGS, by layer:
- **OUTLINE (macro / phase / beat-intent)**: render `story_engine` (conflict
  shape) + `ending_mode`. This is the structural lever; it shapes WHAT happens.
- **MOOD / RENDER**: route `sound_world` to beat mood + the visualizer/LTX prompt
  (its documented home), NOT to spoken dialogue.
- **LINE (character beats)**: pass ONLY a COMPACT register/tone tag + the
  per-beat `conflict_object` obligation. Do NOT inject `sound_world`/`story_engine`
  prose into LineRequest (R1 cut: Gemini, confirmed; GPT compact-obligations).

DETERMINISTIC TEETH (the answer to ask #3): the only per-line lever that is
honestly gate-able is the `conflict_object` -- already deterministic via
`assign_conflict_slot` + grounded by KILL 1's body gate. Do NOT add a per-line
"style-marker present" gate: the markers are audio vocabulary and gating for them
would PUSH the leak. KILL 2 is therefore scoped honestly as a STRUCTURAL STEER
(outline + mood + render) plus the existing deterministic conflict-object teeth;
its success is measured at re-soak, not asserted.

Build: one frozen `StoryContract(slug,label,sound_world,story_engine,ending_tag,
ending_template,grammar)` + `build_story_contract` in `_otr_style_catalog`, built
ONCE after cast-lock (seed=`cast_seed`) + news interpretation, BEFORE
`OutlineRequest`, from `script_brief or news_seed`. Reuse in F2 (delete the late
`select_style(outline.premise,...)` -- FIRST verify no other caller; if other
callers exist, stop calling it from the outline path only). Add style fields to
`OutlineRequest` (rendered in `_build_macro/phase/beat_user_prompt`). ADD
`meta.story_contract`; do NOT overwrite `resolved["style"]`/`meta.style`/
`visual_plan.style` (defer the collapse).

ACCEPTANCE (objective, not "read N episodes"): (1) `build_story_contract` is
CALLED + `meta.story_contract` records slug; (2) outline prompts contain
story_engine/ending_mode under flag; (3) every body beat carries its
`conflict_object` and the KILL-1 gate passes; (4) delete-it test reverts; (5)
re-soak: two different styles on the same news produce measurably different
conflict objects + structure (NOT "vibe").

DEFERRED out of first build: "premise-specific conflict objects beyond the
domain pool" (R1 cut, GPT/DeepSeek) -- the existing seeded domain slot + KILL-1
premise grounding is enough to prove the contract reaches the body; a
premise->object map is its own pass.

---

## 2. ANNOUNCER REDESIGN -- three jobs (HARDENED)

### JOB 1 -- THE OPEN: deterministic no-spoiler by INPUT STARVATION (primary)
R1 unanimous: a post-gate alone cannot reliably detect a spoiler (false
positives). PRIMARY mechanism = input starvation -- the open prompt literally
cannot contain the ending.
- Build a `SafeOpenBrief` from outline + contract: era/`time_of_day`, `setting`,
  cast names+roles, `opening_status_quo` (a NEW outline field = the situation at
  the START), contract tone. NOTHING outcome-bearing.
- SEVER `script_brief` from `compose_announcer_intro` under `story_scaffold` (new
  structured params). The deterministic fallback is built ONLY from `SafeOpenBrief`
  -- it must NOT read `script_brief`.
- BELT (post-gate): reject the generated open if it token-overlaps the
  `ending_change` / `news_close_brief` outcome vocabulary; reroll once; else the
  structured fallback. (Belt, not the primary guarantee.)
- Rewrite `_ANNOUNCER_INTRO_SYSTEM` for the cold-open structure (sentence 1
  orients: era/time/place/cast/status-quo; sentence 2 = intrigue, no outcome
  terms).

ACCEPTANCE: the open names setting+era+characters + states the opening situation,
contains NO outcome/twist/climax token, and is produced WITHOUT `script_brief` in
its inputs (assert the call site).

### JOB 2 -- THE CHARACTER CLOSE + outro DECOUPLING (do now; byte-identical)
The dramatic climax lands in the character's voice. The outro must bridge off the
CLIMAX beat, not the last beat. R1 unanimous coupling find: `compose_announcer_outro`
uses `final_character_line` (`:2786/2852`) which == last line. Since
`_climax_beat_id` already exists (`:3266`), pass the CLIMAX beat's line as
`climax_character_line`. Today climax==last => byte-identical; future-proofs KILL 3.
Stop treating "last line == resolution".

### JOB 3 -- THE NEWS CODA: reconcile the hostile outro voice (MUST, was missed in pass00)
The news coda is NOT merely "framing an already-wired close": the current outro
PROMPT actively forbids it. Under `story_scaffold`:
- Rewrite `_ANNOUNCER_OUTRO_SYSTEM` to a two-part close: (a) the character-close
  reflection (keep the concrete-image discipline), then (b) a deliberate pivot to
  the REAL news as a plain fact (the teaching payload), sourced from
  `news_close_brief`.
- GATE OFF the resolved-FICTION branch (`:2854-2867` "State this outcome plainly")
  under the flag -- the announcer must NOT restate the fictional `ending_change`
  as the coda; pass `ending_change` only as "do NOT restate this". Keep the branch
  intact when the flag is OFF (byte-identical).
- DETERMINISTIC fixed lead-in (ask #1, R1 unanimous: small models blend fiction +
  news unless the pivot is fixed): inject the lead-in as a PREFIX, not LLM
  discretion. Operator leans "The real story:"; RECOMMEND a period-appropriate,
  in-voice lead-in to protect the OTR fiction (e.g. "From tonight's headlines:",
  "The true account:", "What the record shows:"), optionally a small CLOSED
  seed-keyed set of 3-5 recognizable variants to avoid mechanical repetition.
  The label is teachability; the news fact is the payload. (Final wording is the
  operator's creative call.)
- Coda-specific length budget + validator (label required + `news_close_brief`
  fact required + no fictional `ending_change` restatement). The existing
  14-34-word band may be too tight; set a coda band (e.g. 18-45 words).

GROUNDING WIN (corrected): the news PULL (`news_close_brief -> compose_announcer_
outro`) is wired; the WORK is (a) rewriting the hostile outro voice, (b) the
deterministic lead-in, (c) gating off the fictional-outcome branch, (d) the
climax decoupling. KILL 5's old "suppress" is replaced by "frame + protect".

ACCEPTANCE: the coda delivers the real fact as a recognizable teaching beat AFTER
the character climax; the announcer never restates the fictional outcome under
the flag; lead-in present deterministically; byte-identical when off.

---

## 3. KILL 4 -- un-starve the body (HARDENED)
Role-keyed enrichment map for setup / pressure / personal_stake + every
CLIMAX_CLASS_ROLES member (class-specific text). Fix truncation ORDER: truncate
the ORIGINAL intent to `_INTENT_MAX - len(enrichment)` FIRST, then append the
enrichment (reserve the tail); if reserve is negative, truncate the enrichment
slot. DO NOT DELETE `consequence` enrichment (R1 catch, GPT): pass04 cut it as
"unreachable under climax-last", but KILL 3 will make it reachable -- mark it
DEFERRED, do not delete.

---

## 4. KILL 3 -- climax POSITION = spine-driven (DEFERRED; principle settled)
Remove the FORCE (`assign_beat_roles` `i==n-1->climax` `~511`; `validate_beat_
roles` `~558`), let the spine/ending class decide; keep last-beat valid + common.
DEFERRED to its own build AFTER KILL 2/announcer. The outro decoupling (Job 2) is
the only KILL-3 prerequisite pulled forward (and it is byte-identical now).

---

## 5. BYTE-IDENTICAL DISCIPLINE (explicit flag boundary -- R1 GPT#8)
When `story_scaffold` is off: NO `StoryContract` construction, NO new prompt text,
NO `meta.story_contract`, NO changed fallback/outro text, NO request-shape change
visible to old paths, NO outro-prompt rewrite, NO open input change. Add OFF-flag
golden-output tests (open line, outro line, ledger meta) alongside the existing
`test_audio_byte_identical`.

## 6. TELEMETRY (feature-specific; provable via the 3-test "baked in" check)
`meta.story_quality.{open_spoiler_rerolls, open_gate_failed, open_safe_fallback,
news_coda_emitted, news_coda_fallback, story_contract_slug}`. Distinct flags, not
generic `announcer_intro_fallback`.

## 7. BUILD ORDER (R1-adjusted)
KILL 2 (StoryContract: outline + mood/render injection) -> announcer redesign
(open input-starvation + news-coda outro rewrite + outro climax-decoupling) +
KILL 4 (together; both touch the body/outro pipeline) -> LIVE re-soak (gemma +
mistral) -> KILL 3 (its own later build). Hand to a coder window after R4.

## 8. PIPELINE / DATA HANDOFF (R1 GPT#4 -- specify, verify at build)
1) cast-lock (cast_seed) -> 2) news interpretation -> 3) `build_story_contract`
(pre-outline) -> 4) `OutlineRequest` (carries contract style fields) -> 5) outline
emits SAFE open fields incl. `opening_status_quo` -> 6) announcer OPEN composed
post-outline from SafeOpenBrief + contract (NO script_brief) -> body lines carry
compact register + conflict_object -> 7) outro composed from news_close_brief +
climax_character_line + lead-in. VERIFY-AT-BUILD: cast-lock truly precedes
`OutlineRequest`; if not, move contract selection to the earliest stable seed.

## INVARIANTS (a fix that breaks one is rejected)
Behind `story_scaffold` -> byte-identical off; audio spine FROZEN
(`test_audio_byte_identical` green, mux-LAST); suite + Bug Bible per chunk (5
pre-existing 267a53e fails not ours); commit+push per green chunk to v2.0-alpha;
100% local; determinism (seed-keyed); LOUD fallbacks; UTF-8 no BOM; SFW;
prod/main + tags GATED.
