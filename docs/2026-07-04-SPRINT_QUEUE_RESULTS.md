# July-4 code-ready sprint queue -- RESULTS (autonomous run)

Branch `v2.0-alpha`. Rebased on `bb5d84b2`. All work below is green (full suite +
Bug Bible) and pushed; HEAD == origin at each step.

## Sprint 1 -- radio-face-logic  ✅ DONE (commit `f8b6ebd2`)
`docs/2026-07-04-radio-face-logic/PLAN.md`, implemented in full:
- New FACELESS `radio_object` style in `build_radio_host_prompt` (object/material
  language only, anatomy-free object anchor, no overtness clause, no person
  framing anchor) + full still era/grade finish.
- CONDITIONAL synthetic-announcer style pick keyed on the RESOLVED announcer
  engine family AFTER `OTR_FORCE_ENGINE_MAP` (`_effective_announcer_family`):
  HuMo + audio_driven_face -> `console_face` dial-face; static / forced /
  default / unresolved -> faceless `radio_object`.
- Cross-node provenance: `radio_host_style` stamped on the announcer object,
  propagated through the dispatcher generated + cache-hit rows, and a fail-CLOSED
  render guard (`RenderError` for announcer_visual + audio_driven_face +
  char_id=="announcer" whenever the style is not `console_face`, INCLUDING a
  missing field on a frozen ledger).
- Retired `radio_head_person` (style + `_RADIO_HEAD_PERSON` + `RADIO_HEAD_PERSON_NEG`
  + `ANNOUNCER_PORTRAIT_ANCHOR`); both prompt + negative helpers now fail LOUD on
  it. Pruned the dead MUSIC radio-face mint (`_LTX_RADIO_FACE_ROLES` narrowed to
  announcer-only). Doc comments scrubbed. Watcher script repointed (gitignored).
- Full named test surface updated + added. Suite 6165 green + Bug Bible 16.
- Invariants honored: token `character`; SFX defenses untouched; positive
  closed-set style dispatch; NO workflow-JSON node/widget change was required (the
  work is prompt-selection + one ledger provenance field + a render guard).

## Sprint 2 -- still_word prompts  ✅ DONE (commits `e821d6fd`, `41f64185`)
`docs/2026-07-04-still-word-prompts/roundtable/pass01_plan.md`, core + follow-on:
- Per-episode LOCKED lettering (`_STILL_WORD_TYPOGRAPHY`, capitals-locked) +
  COOL backdrop (`_STILL_WORD_BACKDROP`, colour/light/density, no objects), keyed
  by one genre classifier over the radio_form haystack (absent/failed brief ->
  neutral default).
- Beat-mood ALLOWLIST (`_still_word_mood_from_line`): only the enumerated lemmas
  emit {urgent|tense|somber|hopeful|calm}; "neutral"/None/voice-only -> NO token;
  word-boundary, danger-forward priority; character cards only.
- `compose_still_word_prompt` now takes a DICT or STR (param name `beat_line`
  kept); call site passes the line dict. `_STILL_WORD_CARD_STYLE` split into a
  genre-neutral legibility guard + a positive text guard (appended LAST).
- MANDATORY word-mode era-tail text-summoner scrub; inner-double-quote FOLD in
  `_still_word_clean_line` + the music title (closes the nested-quote ambiguity +
  the prompt-injection); DETERMINISTIC line-length reduction (never the hard
  abort that would kill every episode); blank line still fails LOUD. Music path
  UNTOUCHED (wordless abstract title).
- Follow-on: `_still_word_roles_from_policy` honors `OTR_FORCE_ENGINE_MAP` (force
  into/away-from still_word; env-unset byte-identical); `lettering_style` +
  `backdrop_family` provenance propagated through the dispatcher rows.
- Suite 6180 green + Bug Bible 16.
- SCOPED CODE-ONLY (per plan §5 r4 permission): the composer is engine-agnostic
  and works whenever a role's video engine is `still_word`; NO node-87 JSON wiring
  was made, so no workflow-JSON edit was required this run.
- **DEFERRED (separable follow-on, NOT done):** the Ideogram 16:9-resolution fix
  (`eng_cloud_image.py`). It needs the EXACT allowed `resolution` COMBO value read
  from the live `cloud_ideogram_v4` node + a LIVE 16:9 Ideogram QA card to confirm
  no words are cropped -- neither is possible headless/offline in this run. When a
  role is wired to `still_word` + `ideo` in node 87, that JSON edit + the Ideogram
  resolution fix land together in one commit (invariant 6).

## Sprint 3 -- UpstreamStoryLab handoff, items 1 & 2

### Item 2 -- SFX ledger rip  ✅ DONE (commit `0ceaf1c2`)
Traced EVERY `sfx` reference in `nodes/OTR_LedgerScriptWriter.py` +
`nodes/production_ledger.py` (and the socket names `sfx_plan_json` /
`sfx_audio_clips` / `sfx_offset_ms`, `set_sfx`, `apply_sfx`, `sfx_cue`,
`sfx_timings`). Finding: the LIVE sfx ledger plumbing was ALREADY ripped (S27
cleanbreak-tail + rip-sfx-broll 2026-07-01) -- the BUG-108 merge `ROW_KEYED` map
already excludes `sfx`, and every remaining reference is a comment/tombstone or
docstring, NOT a live gate. The only out-of-sync residual was one stale docstring
that still listed `sfx[i]` as a copied row shape; corrected it to match the live
`lines/clips/music` code (a clean seam, no behavior change, no silent hole).
Grep-confirmed: generator gone (no `[SFX:]` in `_otr_period_prompts.py`), all four
`speaker_role=="sfx"` rejection sites LIVE, 3 forbidden-socket tombstones intact
(invariant 5 preserved). Suite 6180 green + Bug Bible 16.

### Item 1 -- "big LLM prompt update across the whole workflow"  ⛔ BLOCKED: no code-ready spec
This is the ONLY queue item I did not code, and deliberately so. Unlike Sprints 1
& 2 -- each of which had a fully hardened, converged PLAN with exact prompt
strings and named MUST-FIXes -- item 1 has NO concrete, converged, code-ready
diff. Tracing it:
- `docs/todays-plan-handoff.md` describes item 1 only as "Sweeping prompt changes
  touching many nodes" -- a CATEGORY of work, not a specification.
- The underlying design is the UpstreamStoryLab JSON-owns-content story-engine
  transplant (`R1_ARCHITECTURE_AND_CODING_PLAN_V2.md`,
  `JSON_CONTENT_PYTHON_BEHAVIOR_R1_R4_REWRITE.md`,
  `UPSTREAM_STORY_LAB_CODE_READY_BRIEF.md`). That is a large architectural rewrite
  (move story/prompt content into JSON packs), NOT a bounded prompt edit.
- Those same docs EXPLICITLY gate production: "Production is not edited until the
  explicit transplant chunk" / "The lab ... must NOT edit the production workflow
  until a later explicit transplant chunk." No such explicit transplant chunk was
  authorized for tonight, and no exact-string prompt diff exists to execute.

Sweeping, unspecified changes to the story LLM prompts are the highest-risk,
most-quality-sensitive, hardest-to-revert kind of change (the operator's core
deliverable; Fable §9 territory). Fabricating one blind -- with no converged spec,
no acceptance criteria, and the source docs themselves gating production -- would
violate the converged-plan / fix-at-root discipline and almost certainly force a
debugging-and-revert loop. That is a genuine specification blocker, so per the
autonomy banner ("halt only for a true hard blocker") I stopped item 1 rather than
invent it, while completing everything else in the queue.

**To unblock item 1, one of:** (a) point me at the concrete converged prompt-diff
spec if one exists, (b) authorize the "explicit transplant chunk" and I will run
the roundtable/kibitz arc to converge a hardened PLAN first (as Sprints 1 & 2
had), then code it, or (c) scope item 1 down to a specific, enumerated set of
production prompt sites + target text.

## Deferred (per the queue's explicit scope -- NOT started, correctly)
- Handoff item 3 (story-engine map + assertion inventory): read-only analysis for
  a FRESH window AFTER the coding closes (`story-engine-map-brief.md`).
- The Python->JSON assertion move: future work; first needs a declarative-rule
  enforcer node built (today's `_otr_workflow_validator.py` only audits litegraph
  structure).
