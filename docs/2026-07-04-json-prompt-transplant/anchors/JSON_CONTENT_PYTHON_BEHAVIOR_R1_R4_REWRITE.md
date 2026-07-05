# JSON CONTENT / PYTHON BEHAVIOR -- R1..R4 REWRITE

**Companion doc to** `R1_ARCHITECTURE_AND_CODING_PLAN_V2.md`.
Same anchor date, branch, HEAD. Same non-goals.

This doc is the principle-level rewrite. It fixes the shape of the JSON
profiles, the Python behavior surface, and the specific policy calls that
kibitz r1..r4 will pressure-test.

---

## 1. Principle statement

> **JSON owns content and configuration. Python owns validation, routing,
> and execution.**

Content = the strings that a language model or a downstream renderer will
see verbatim: system prompts, user templates, example blocks, label
vocabularies. Configuration = data-shaped choices that a non-Python editor
should be able to change without touching code: profile lists, site
overrides, per-profile knobs.

Python still owns everything about HOW those strings get used: what LLM they
go to, how they're batched, how caching / IS_CHANGED / VRAM tenancy behaves,
how outputs are validated, how errors surface. None of that moves.

The failure mode this is preventing: a genre / period / tone change (e.g.
"give me a noir 1970s version") requires a Python edit today. Post-transplant,
it's a new `noir_1970s.json` under `config/prompt_profiles/` and a catalog
entry. Zero Python edits, zero regressions on other profiles.

---

## 2. Content vs. behavior boundary (concrete)

**Moves to JSON (content):**

- Every `_SYSTEM_PROMPT`, `_MACRO_SYSTEM_PROMPT`, `_PHASE_SYSTEM_PROMPT`,
  `_BEAT_SYSTEM_PROMPT`, `_INVENTOR_SYSTEM`, `_CHOOSER_SYSTEM`,
  `_SLOT_JOB_SYSTEM_PROMPT`, `_NEWS_CODA_SYSTEM`, `_NEWS_CODA_SYSTEM_V2_EXAMPLES`.
- The inline `sys_msg` at `OTR_LedgerScriptWriter.py:937`.
- The inline system prompt at `_otr_pitch_room.py:183`.
- The inline system prompt at `_otr_story_select.py:165`.
- The inline system prompt built in `news_interpreter.py:704`.
- The user-template segments that are pure content (no interpolation of
  runtime state -- see section 4 for policy on this).
- The genre keyword tuples in `otr_meta_brief_image_prompt.py:614` and the
  typography / backdrop maps at L631 / L642.

**Stays in Python (behavior):**

- `resolve_creative_system_prompt(repo_id, phase)` -- becomes a lookup, not a
  literal branch, but the FUNCTION stays.
- `build_pitch_prompt`, `_build_user_prompt` (all of them), `generate_title`
  -- these are prompt ASSEMBLERS; they stitch profile content + runtime data.
- `compose_news_coda` including the concat pattern
  (`_NEWS_CODA_SYSTEM + _NEWS_CODA_SYSTEM_V2_EXAMPLES` at
  `_otr_line_composer.py:3407`) -- concat stays in Python; the two strings
  live independently in the JSON profile.
- All grounding-lever gating (`req.conflict_object` branch at
  `_otr_line_composer.py:1621` and its fallback). The two rider TEXTS move
  to JSON; the IF stays in Python.
- All env-flag reads (`OTR_ENABLE_PITCH_ROOM`, `OTR_GROUNDING_LEVER`, etc.).
- Schema validation of loaded JSON (Python-owned).
- Test assertion logic in `tests/test_period_prompts.py` -- the STRINGS
  under test come from the loaded profile now; the assertions
  ("must contain '1940s'", "must not contain '[SFX:'") are Python.

---

## 3. Profile inventory

| Profile ID | Role | Content sourcing | Default? |
|------------|------|------------------|----------|
| `sci_fi` | current baked-in genre; carries lifted payload verbatim | direct extract from HEAD `a7bdc42d` Python constants | YES (default fallback) |
| `news` | overrides `interpret` + `news_coda_system` only; inherits sci_fi elsewhere | direct extract from current news_interpreter.py + `_NEWS_CODA_SYSTEM` | no |
| `cinematic` | reserved scaffold; empty; inherits sci_fi everywhere | none this sprint | no |
| `radio` | period-locked 1940s OTR; unifies with existing `otr_1940s_v1` catalog value | extract from `OTR_PERIOD_SYSTEM_PROMPT` (referenced in `_otr_creative_prompt_router.py:11`) | no |

**Sci-fi is not deprecated, migrated, or ripped.** It becomes the reference
profile. Any profile that omits a site inherits sci_fi for that site, with a
loader warning. This inheritance rule is the single biggest audio-invariant
guarantee: as long as sci_fi carries a byte-identical extract of the current
Python payload, the default code path is unchanged.

---

## 4. Templates vs. static strings

Some sites are pure static system prompts (e.g. `_SLOT_JOB_SYSTEM_PROMPT`).
Others weave runtime state into a template (e.g. `casting_brief_seam` uses
`{name}`, `{style_str}`, `{story_text}`).

Rule:

- **Static system content** -> `sites.<key>.system: str`. Straight copy.
- **Templated user content** -> `sites.<key>.user_template: str` with
  `str.format`-compatible `{slot}` placeholders. Loader validates that every
  `{slot}` the template names is present in a Python-side slot registry
  for that site (fail-fast on schema load).
- **Example / few-shot blocks** -> `sites.<key>.examples: List[str]` OR
  `sites.<key>.v2_examples: str` for a single-blob variant matching current
  `_NEWS_CODA_SYSTEM_V2_EXAMPLES` shape.

This keeps assembly logic in Python (which still validates slot presence
and formats the final string) while moving the content to JSON. It also means
the loader can reject a profile that references a nonexistent slot at
load time, not at prompt time.

---

## 5. Failure modes and where they surface

| Failure | Where it manifests | Guardrail |
|---------|---------------------|-----------|
| JSON syntax error in a profile | `PROFILE_REGISTRY` load at import | `PromptProfileValidationError`, fails ComfyUI node registration |
| Missing site in a non-default profile | first call to `get_prompt(profile_id, site_key)` | Loader falls back to `sci_fi` for that site, logs WARNING with `profile_id` + `site_key`; NEVER silent |
| Missing slot in a `user_template` | schema load | `PromptProfileValidationError` names the missing slot |
| Content drift (post-transplant string != pre-transplant string) | Chunk I golden-fixture test | Test fails; commit rejected pre-push |
| Two profiles both claim to be default | schema load | `PromptProfileValidationError`; exactly one profile has `is_default: true` |
| Sci-fi profile itself is missing | schema load | Hard error; sci-fi is required |

---

## 6. Migration rules (byte-identical requirement)

For each of the 15 sites in the R1 doc, the migration is:

1. Read the current Python string literal exactly as it appears in HEAD `a7bdc42d`.
2. JSON-escape (`"` -> `\"`, real newlines -> `\n`, tabs -> `\t`). No other
   normalization. No trimming. No case folding. No unicode form conversion.
3. Insert as the value of `sites.<key>.system` (or `.user_template`, per
   section 4) in the appropriate profile JSON.
4. Delete the Python constant OR remove the inline literal, replacing the
   original reference with `get_prompt(profile_id, "<site_key>").system`.
5. Run the chunk's regression test (audio-invariant regression episode +
   golden-fixture string equality).
6. Commit; push to `v2.0-alpha`; verify HEAD == origin, no 0-byte files,
   no BOM, AST parse clean.

The audio-invariant regression is the tightest test. If it fails on any
chunk, revert and re-check the JSON escaping. String equality is necessary
but not sufficient -- audio identity is the ground truth.

---

## 7. Sci-fi payload catalog (what the sci_fi.json profile carries)

Enumerated so kibitz r2..r4 can produce line-diff-level extractions.
Line refs are in HEAD `a7bdc42d`.

- **outline_system:** `_otr_outline.py:532-...` -- opens with "You are a
  story editor for short science-fiction audio dramas grounded in real
  science."
- **outline_macro_system:** `_otr_outline.py:1102-...` -- opens with "You
  plan short science-fiction audio dramas."
- **outline_phase_system:** `_otr_outline.py:1115-...` -- "You plan one phase
  of a science-fiction audio drama."
- **outline_beat_system:** `_otr_outline.py:1130-...` -- "You flesh out one
  beat of a science-fiction audio drama."
- **pitch_room_system:** `_otr_pitch_room.py:183-...` -- "pitches for a
  short science-fiction audio drama".
- **story_select_system:** `_otr_story_select.py:165-...` -- "You are a
  tough story editor grading a short science-fiction audio ...".
- **title_system:** `OTR_LedgerScriptWriter.py:936-...` -- "You are titling
  a single episode of a sci-fi radio drama."
- **style_pick_inventor:** `_otr_style_picker.py:296-...` -- "You are a
  sci-fi radio drama showrunner."
- **style_pick_chooser:** `_otr_style_picker.py:329-...` -- "You are a
  strict radio drama editor." (no sci-fi baked in the CHOOSER system, but
  the CHOOSER user block at L335 opens "Choose the single best descriptor
  for adapting the article into a sci-fi radio drama.")
- **labels:** `otr_meta_brief_image_prompt.py:614-...` -- the four-key genre
  tuple `("noir", ...), ("sci-fi", ...), ("western", ...), ("pulp", ...)`
  plus typography L631 and backdrop L642 maps that reference `"sci-fi"` as
  a top-level key.

---

## 8. News payload (what the news.json profile carries)

- **interpret:** `nodes/news_interpreter.py:704-723` (system prompt is built
  inline in `_build_user_prompt`; extraction needs isolating the system
  portion from the templated user portion).
- **news_coda_system:** `_otr_line_composer.py:3275-3292` (base) plus
  `_otr_line_composer.py:3297-...` (v2 examples block).

Every other site inherits `sci_fi` -> byte-identical to current for news
content that doesn't touch these two sites.

---

## 9. Radio payload (what the radio.json profile carries)

The `radio` profile carries the OTR 1940s content that currently lives
behind `prompt_profile="otr_1940s_v1"` in `_otr_model_catalog.py:85`.
`nodes/_otr_creative_prompt_router.py:98` reads this and returns
`OTR_PERIOD_SYSTEM_PROMPT`. Extraction target: locate `OTR_PERIOD_SYSTEM_PROMPT`
definition (not confirmed at this anchor; kibitz r2 must ground the file:line
and inclusion path).

Radio profile also carries the assertions guarded by
`tests/test_period_prompts.py` -- the string it loads must contain
`"1940s"`, `"Suspense"`, `"NARRATOR"`, `"CHARACTER:dialogue"`,
`"Family-broadcast safe"`; must NOT contain `"[SFX:"` or `"[SFX"`; must NOT
contain modern-slang tokens. Radio's content passes those assertions by
construction (extracted from the current Python constant that already
passes them).

---

## 10. Cinematic payload

Empty this sprint. `sites: {}`. Inherits sci_fi for every site.
Sprint 3 Item N (future) fills in `outline_system`, `pitch_room_system`,
`title_system`, `style_pick_inventor` with cinematic language.
Documented here so the profile enum stops changing shape after this sprint.

---

## 11. What kibitz r1..r4 is verifying

- **r1 (arc):** does the content-vs-behavior split hold up? Are the four
  profiles the right axis? Should the extension point be `content_profile`
  or a two-axis (`content_profile` x `prompt_profile`) matrix?
- **r2 (coding):** are the 9 chunks the right decomposition? Are the exact
  string diffs mechanically extractable for each site? Any site that resists
  extraction (e.g. requires runtime state to reconstruct)?
- **r3 (wiring):** does the loader contract survive Python import-order,
  ComfyUI IS_CHANGED, and test-time mocking? Is `resolve_creative_system_prompt`'s
  generalization backward-compatible with existing catalog rows?
- **r4 (convergence):** residual defects only. Is any invariant from R1
  section 7 at risk? Is the golden-fixture test structured correctly?
