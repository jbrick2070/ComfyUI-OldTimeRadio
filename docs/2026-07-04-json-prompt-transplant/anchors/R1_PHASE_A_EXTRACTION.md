# R1 PHASE A -- MECHANICAL PROMPT EXTRACTION

**Sprint:** 2026-07-04 JSON Prompt Transplant, **Phase A only**.
**Branch:** `v2.0-alpha`. **Anchor HEAD:** `a7bdc42d`.

## 0. Phase gate

This document scopes **Phase A** only. Phase A is the **mechanical relocation**
of hardcoded prompt strings from Python source files into JSON config files.
There is no content change, no architecture change, no genre semantics
change, no story-model refactor, no variable viz style.

Phase B (the big transplant / architectural rewrite) is out of scope for this
sprint arc and is gated on Phase A shipping green (audio byte-identical soak
clean). See `PHASE_B_STUB.md` in this folder -- do not populate it here.

The success criterion for Phase A is **behavior-preserving**:

- Every LLM call receives the exact same rendered prompt string it received
  pre-Phase-A, byte-for-byte, at HEAD `a7bdc42d`.
- Audio output remains byte-identical against the fixed-seed regression episode.
- No changes to ledger schema, no changes to `ROW_KEYED` merge, no changes to
  critic / reroll seams.

If any of those slip, Phase A is not done.

---

## 1. What Phase A does

For each of the 15 prompt sites enumerated in section 3, Phase A:

1. **Extracts** the current Python string literal verbatim (post JSON-escape).
2. **Places** it into a versioned JSON profile file.
3. **Replaces** the Python literal with a loader call
   (`get_prompt(profile_id, site_key)`) that resolves the same string at
   runtime.
4. **Verifies** byte-identical prompt output against a captured snapshot.
5. **Verifies** byte-identical audio output against the regression episode.

There is exactly one profile per genre lane, and each profile carries the
CURRENT content of that lane. The profile files exist to make future edits
(Phase B) config-side rather than code-side, but Phase A does not exercise
that capability. Phase A only proves the wiring.

---

## 2. What Phase A does NOT do

- Does NOT change any prompt string. Escaping-only transforms are allowed
  (`"` -> `\"`, real newlines -> `\n`, tabs -> `\t`); no other edits.
- Does NOT add new prompt sites, new profiles beyond the four
  (news / sci-fi / cinematic / radio + shared), or new dispatch axes.
- Does NOT generalize `_otr_creative_prompt_router.py:67` beyond
  what the mechanical move requires. If a site currently branches on
  `prompt_profile == "otr_1940s_v1"`, the post-Phase-A code still branches
  on the same condition (the branch reads its strings from JSON instead of
  from Python constants).
- Does NOT deprecate the existing `prompt_profile` catalog field. Coexistence
  is fine for Phase A; unification is a Phase B concern.
- Does NOT introduce content variation between profiles beyond the exact
  content variation that already exists in Python today.
- Does NOT modify any test assertion logic, nor the `l3-2026-05-14` ledger
  schema, nor the critic / reroll seams.

---

## 3. Prompt sites in scope (15 total)

Verified against HEAD `a7bdc42d` by `git grep -n` on 2026-07-04.
12 sites from the operator's original enumeration + 3 outline-hierarchy
siblings that must move atomically with `outline_system`.

| # | Site key | Current file:line | Current holder | Currently profile-branched? |
|---|----------|-------------------|----------------|-----------------------------|
| 1 | `interpret` | `nodes/news_interpreter.py:704` | inline in `_build_user_prompt` (no constant) | no |
| 2 | `outline_system` | `nodes/_otr_outline.py:532` | `_SYSTEM_PROMPT` | via `resolve_creative_system_prompt` (modern vs otr_1940s_v1) |
| 3 | `outline_macro_system` | `nodes/_otr_outline.py:1102` | `_MACRO_SYSTEM_PROMPT` | no |
| 4 | `outline_phase_system` | `nodes/_otr_outline.py:1115` | `_PHASE_SYSTEM_PROMPT` | no |
| 5 | `outline_beat_system` | `nodes/_otr_outline.py:1130` | `_BEAT_SYSTEM_PROMPT` | no |
| 6 | `pitch_room_system` | `nodes/_otr_pitch_room.py:183` | inline in `build_pitch_prompt` | no |
| 7 | `story_select_system` | `nodes/_otr_story_select.py:165` | inline (grader) | no |
| 8 | `dramatic_state_system` | `nodes/_otr_slot_drama_contract.py:304` | `_SLOT_JOB_SYSTEM_PROMPT` | no |
| 9 | `line_grounding_rider` | `nodes/_otr_line_composer.py:1621` | conditional block in `_build_user_prompt` (with a fallback rider at ~1631) | branches on `req.conflict_object` (Python) |
| 10 | `casting_brief_seam` | `nodes/_otr_casting.py:288` | `_build_user_prompt` assembles `casting_brief` | branches on `casting_brief` presence (Python) |
| 11 | `news_coda_system` | `nodes/_otr_line_composer.py:3275` (+ `_V2_EXAMPLES` at L3297; concat at L3407) | `_NEWS_CODA_SYSTEM` + `_NEWS_CODA_SYSTEM_V2_EXAMPLES` | no |
| 12 | `title_system` | `nodes/OTR_LedgerScriptWriter.py:937` | inline `sys_msg` in `generate_title` | no |
| 13 | `style_pick_inventor` | `nodes/_otr_style_picker.py:296` | `_INVENTOR_SYSTEM` | no |
| 14 | `style_pick_chooser` | `nodes/_otr_style_picker.py:329` | `_CHOOSER_SYSTEM` | no |
| 15 | `labels` | `nodes/otr_meta_brief_image_prompt.py:614` | `_STILL_WORD_GENRE_KEYWORDS` + typography L631, backdrop L642 | keys include `noir`, `sci-fi`, `western`, `pulp`, `default` |

Sites 3, 4, 5 (outline macro / phase / beat) are not in the operator's original
12 but MUST move together with site 2 (`outline_system`) or the outline
pipeline splits mid-lifecycle. Included as one atomic chunk.

Site 15 (`labels`) is image-side only (typography and backdrop; consumed at
`otr_meta_brief_image_prompt.py:715`). It does not feed the dialogue prompts.
Included in Phase A because it also matches the "hardcoded string map that
should be config" shape.

---

## 4. Target JSON schema

One file per profile under `config/prompt_profiles/`:

```
config/prompt_profiles/sci_fi.json
config/prompt_profiles/news.json
config/prompt_profiles/cinematic.json
config/prompt_profiles/radio.json
config/prompt_profiles/_shared.json     # optional shared block, see below
```

Per-profile file shape:

```json
{
  "profile_id": "sci_fi",
  "profile_version": 1,
  "inherits_from": null,
  "sites": {
    "interpret":              { "system": "...", "user_template": "..." },
    "outline_system":         { "system": "..." },
    "outline_macro_system":   { "system": "..." },
    "outline_phase_system":   { "system": "..." },
    "outline_beat_system":    { "system": "..." },
    "pitch_room_system":      { "system": "..." },
    "story_select_system":    { "system": "..." },
    "dramatic_state_system":  { "system": "..." },
    "line_grounding_rider":   { "system": "..." },
    "news_grounding_rider":   { "system": "..." },
    "casting_brief_seam":     { "user_template": "..." },
    "news_coda_system":       { "system": "...", "v2_examples": "..." },
    "title_system":           { "system": "..." },
    "style_pick_inventor":    { "system": "..." },
    "style_pick_chooser":     { "system": "..." },
    "labels":                 { "genre_keywords": {...}, "typography": {...}, "backdrop": {...} }
  }
}
```

- `profile_id` and `profile_version` are required. Loader rejects unknown
  versions. Version 1 is the Phase A schema.
- `inherits_from` is `null` for `sci_fi` (the canonical baseline) and can
  point to another `profile_id` for the other three. Missing sites fall back
  to the inherited profile's site. If both are missing, loader raises.
- `sites` is an object; every key that appears MUST match a known site key.
  Unknown keys are a load-time error.

For Phase A, each of the four profiles carries only the exact content that
already exists in Python today, split across profiles the same way genre
selection currently splits it in code:

- `sci_fi.json`: carries all 15 sites, populated from the current Python
  literals (this preserves the current default behavior verbatim).
- `news.json`: overrides `interpret` and `news_coda_system` with the current
  news-side Python content; `inherits_from: "sci_fi"` for everything else.
- `cinematic.json`: empty `sites: {}`; `inherits_from: "sci_fi"`. Scaffolding
  only for Phase A; do not populate until Phase B.
- `radio.json`: carries `outline_system` (extracted from
  `OTR_PERIOD_SYSTEM_PROMPT` referenced at
  `_otr_creative_prompt_router.py:11`); `inherits_from: "sci_fi"` for
  everything else.
- `_shared.json` (optional): if any two of the above profiles have literally
  identical content at a site, the site can be lifted into `_shared.json`
  and referenced by both. Not required for Phase A; use only if it removes
  duplication. Default: do not create.

---

## 5. Python loader / validator / router

New module: `otr/config/prompt_profile_loader.py`.

Responsibilities:

1. **Discovery.** Scan `config/prompt_profiles/*.json` at import time.
2. **Parse.** Load each file as JSON. Any JSON syntax error is fatal.
3. **Validate against schema.** Every file must satisfy the shape in
   section 4. Version must equal 1. Failures raise
   `PromptProfileValidationError` naming the file path and the offending key.
4. **Register.** Populate a frozen `PROFILE_REGISTRY: Dict[str, ProfileBundle]`.
5. **Public API:**

   ```python
   def get_prompt(profile_id: str, site_key: str) -> PromptBundle
   def list_profiles() -> List[str]
   def list_sites(profile_id: str) -> List[str]
   ```

   `PromptBundle` is a small typed struct with `system: Optional[str]`,
   `user_template: Optional[str]`, `v2_examples: Optional[str]`, and site-
   specific extension fields (e.g. `genre_keywords` / `typography` /
   `backdrop` for `labels`).

6. **Inheritance resolution.** `get_prompt` walks `inherits_from` chain until
   the site key is found. Missing site in the terminal profile (usually
   `sci_fi`) is a hard error.
7. **No content mutation.** The loader never reformats, trims, or reencodes
   the loaded string. It returns the exact bytes stored in the JSON file
   (decoded as UTF-8 text).
8. **Import-time only.** No hot reload. `IS_CHANGED`-safe by construction
   (loader is a module-level immutable singleton after import).

Router changes at `nodes/_otr_creative_prompt_router.py:67`
(`resolve_creative_system_prompt`): the function keeps its signature and
its dispatch condition (`row.prompt_profile == "otr_1940s_v1"`). It changes
only its internals: instead of returning a Python constant, it maps the
current `prompt_profile` value to a `profile_id` (`modern` -> `sci_fi`,
`otr_1940s_v1` -> `radio`) and returns
`get_prompt(profile_id, "outline_system").system`. No new fields on the
model catalog. No new dispatch axis. Phase B may generalize this; Phase A
does not.

At sites without existing profile routing (2, 3, 4, 5, 6, 7, 8, 12, 13, 14
in section 3), the call is a fixed `get_prompt("sci_fi", "<site_key>")` --
matching current Python behavior which unconditionally uses the baked-in
sci-fi string. Only site 1 (`interpret`) and site 11 (`news_coda_system`)
resolve against `get_prompt("news", ...)` (matching current news-specific
behavior). Site 15 (`labels`) is a data-map lookup, not a system prompt --
handled by an equivalent `get_labels(profile_id) -> LabelsBundle`.

---

## 6. Chunk plan (target for R2 coding elaboration)

R2 will produce per-file exact-string diffs. Chunks:

- **A. Scaffold:** create the four profile files with empty sites +
  `inherits_from` wiring. Add loader module with schema validation. Add
  unit tests that assert schema-load succeeds. Chunk ships even though
  it changes no LLM behavior yet -- pure infrastructure.
- **B. sci_fi baseline (11 sites):** extract sites 2, 3, 4, 5, 6, 7, 8,
  12, 13, 14, and 15 into `sci_fi.json`. These are the sites currently
  hardcoded as sci-fi in Python. Replace each Python literal with a
  `get_prompt("sci_fi", "<key>")` call. Verify byte-identical output +
  audio.
- **C. News overrides (2 sites):** extract sites 1 and 11 into `news.json`.
  Wire the two call sites to `get_prompt("news", ...)`. Verify byte-
  identical output + audio.
- **D. Radio wiring (1 site):** move `OTR_PERIOD_SYSTEM_PROMPT` (referenced
  at `_otr_creative_prompt_router.py:11`) into `radio.json` as
  `outline_system`. Update the router to look it up. This is the only site
  that has existing profile-branching in Python; verify branch behavior
  unchanged (both `modern` and `otr_1940s_v1` catalog values still route to
  their equivalent JSON profiles).
- **E. Dialogue-tail sites (2 sites):** extract sites 9 (`line_grounding_rider`
  including the news-grounding fallback rider at L1631-1636) and 10
  (`casting_brief_seam`) into `sci_fi.json`. Python retains all conditional
  gating; only the rider TEXTS move.
- **F. Golden-fixture regression + Bug Bible:** add a test module that
  captures a snapshot of every resolved prompt string on pre-Phase-A HEAD
  and asserts equality on post-Phase-A HEAD, per profile. Add the Bug Bible
  commands for the audio-invariant regression episode. Freeze.

Chunks A, B, C, D, E, F are each one commit each with a full local test run
before push, per the standing rules. Push per chunk to `v2.0-alpha`.

---

## 7. Verification harnesses (concrete)

- **Byte-identical prompt harness.** New test:
  `tests/test_prompt_profile_extraction.py`. For each site key, resolves
  the current prompt via `get_prompt` and asserts equality against a
  snapshot captured from pre-Phase-A HEAD `a7bdc42d`. Snapshots live under
  `tests/snapshots/prompt_profiles/<profile>/<site>.txt`.
- **Audio byte-identical harness.** Existing suite:
  `tests/test_audio_byte_identical` (or the closest existing name -- kibitz
  r3 will confirm). Runs the fixed-seed regression episode end-to-end and
  compares audio checksums. MUST stay green after each chunk.
- **ROW_KEYED merge invariants.** Existing tests under `tests/` that
  exercise ledger merge behavior. MUST stay green.
- **Test asserts in `tests/test_period_prompts.py`.** These currently assert
  on the Python constant. Post-Chunk D they assert on the resolved JSON
  string via `get_prompt("radio", "outline_system").system`. Assertion set
  is unchanged; only the source of the string changes.
- **Critic / reroll seam smoke:** run the existing critic + reroll tests.
  MUST stay green.

Any chunk that fails any harness is reverted before push.

---

## 8. Sci-fi payload handling in Phase A

The current Python code bakes "science-fiction" / "sci-fi" into 12 of the
15 sites. In Phase A this content moves to `sci_fi.json` VERBATIM. The
strings still say "science-fiction audio drama"; nothing is neutralized.

This is a mechanical relocation, not a rewrite. Phase B may edit the sci-fi
content or introduce alternate profile content -- Phase A does not.

Under Phase A, `cinematic.json` and (mostly) `radio.json` inherit sci-fi
content by default. This is a wiring artifact, not a design statement. It
mirrors the current Python behavior where any code path that reaches the
sci-fi constant sees the sci-fi string.

---

## 9. Open questions for kibitz r1..r4

- Is the four-profile split the right shape when only two profiles (sci_fi,
  news) actually carry content in Phase A? Could we ship Phase A with only
  those two + defer cinematic / radio scaffolding until they carry payload?
  (Trade-off: schema stability vs. minimalism.)
- Should `casting_brief_seam` live in the profile file at all, given it is
  a Python-formatted template with `{name}` / `{style_str}` / `{story_text}`
  slot substitution? Or should Phase A leave it in Python as pure
  formatter and only migrate it in Phase B?
- Does `_shared.json` earn its keep in Phase A, or should we drop it and
  revisit in Phase B?
- Does the golden-fixture harness (section 7) need per-chunk incremental
  snapshots, or is one snapshot at Chunk F sufficient with per-chunk audio
  as the load-bearing test?
- Is `PromptProfileValidationError` the right failure surface for schema
  errors given ComfyUI's node-registration lifecycle, or should the loader
  degrade to a WARN + skip so the node pack still loads for other work?
  (Phase A default: fail fast. Kibitz r3 wiring detail.)

These questions ARE the r1..r4 hardening target.
