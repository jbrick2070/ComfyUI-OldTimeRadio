# R1 ARCHITECTURE AND CODING PLAN V2

**Sprint context:** Sprint 3 Item 1 -- "JSON owns content, Python owns behavior"
prompt transplant. Move hardcoded prompt strings out of Python into
JSON profile blocks. No behavior change; audio byte-identical.

**Scope:** Docs-only in this sprint arc. The kibitz r1-r4 arc hardens THIS plan
before any Python is touched. Production code stays frozen at HEAD `a7bdc42d`
on `v2.0-alpha` until TRANSPLANT_PLAN_FINAL.md is committed.

**Anchor date:** 2026-07-04
**Branch:** `v2.0-alpha`
**HEAD at drafting:** `a7bdc42d`

---

## 1. Target state (why this move)

Current: every LLM-facing system/user prompt lives as a Python string literal
(either a module-level `_SYSTEM_PROMPT`-style constant or inline in a
prompt-builder function). Sci-fi genre is baked into most of them as an
uncontestable assumption ("science-fiction audio drama", "sci-fi radio drama
showrunner", etc.).

Target: every prompt string moves into a versioned JSON profile file. Python
retains ONLY:

1. Validation (schema-check the loaded JSON against a Pydantic/dataclass model).
2. Routing (dispatch on `content_profile` -- news / sci-fi / cinematic / radio).
3. Execution (call the LLM with the resolved prompt).

Sci-fi is NOT ripped. It becomes the first-class default profile
(`content_profile="sci_fi"`). The current hardcoded sci-fi payload lifts into
a `sci_fi.json` block so the migration is content-preserving.

---

## 2. Profile schema (target)

Each profile is a JSON file under `config/prompt_profiles/<profile>.json`.
Profile names planned:

- `sci_fi` (default -- carries the current hardcoded payload verbatim)
- `news` (news-interpreter, news-coda)
- `cinematic` (future -- reserved key; empty until a content sprint fills it)
- `radio` (period-locked OTR 1940s content -- overlaps with existing
  `prompt_profile="otr_1940s_v1"` in `_otr_model_catalog.py:85`; unify carefully)

Top-level schema (per file):

```json
{
  "profile_id": "sci_fi",
  "profile_version": 1,
  "sites": {
    "interpret":            { "system": "...", "user_template": "..." },
    "outline_system":       { "system": "..." },
    "outline_macro_system": { "system": "..." },
    "outline_phase_system": { "system": "..." },
    "outline_beat_system":  { "system": "..." },
    "pitch_room_system":    { "system": "..." },
    "story_select_system":  { "system": "..." },
    "dramatic_state_system":{ "system": "..." },
    "line_grounding_rider": { "system": "..." },
    "news_grounding_rider": { "system": "..." },
    "casting_brief_seam":   { "user_template": "..." },
    "news_coda_system":     { "system": "...", "v2_examples": "..." },
    "title_system":         { "system": "..." },
    "style_pick_inventor":  { "system": "..." },
    "style_pick_chooser":   { "system": "..." },
    "labels":               { "genre_keywords": [...], "typography": {...}, "backdrop": {...} }
  }
}
```

`profile_version` is an integer; loader rejects unknown versions. Every site
key is optional -- if a profile omits a site, loader falls back to
`sci_fi` (the reference profile) for that site with a diagnostic warning
(NEVER silently -- log at INFO+ with the missing site name).

---

## 3. Ground truth: 15 prompt sites at HEAD `a7bdc42d`

Verified against HEAD by `git grep -n` on 2026-07-04. Each site row lists the
current file:line and the target profile key.

| # | Site key | Current file:line | Current holder | Sci-fi baked? |
|---|----------|-------------------|----------------|---------------|
| 1 | `interpret` | `nodes/news_interpreter.py:704` | inline in `_build_user_prompt` (no constant) | no |
| 2 | `outline_system` | `nodes/_otr_outline.py:532` | `_SYSTEM_PROMPT` | yes (L533) |
| 3 | `outline_macro_system` | `nodes/_otr_outline.py:1102` | `_MACRO_SYSTEM_PROMPT` | yes (L1103) |
| 4 | `outline_phase_system` | `nodes/_otr_outline.py:1115` | `_PHASE_SYSTEM_PROMPT` | yes (L1116) |
| 5 | `outline_beat_system` | `nodes/_otr_outline.py:1130` | `_BEAT_SYSTEM_PROMPT` | yes (L1131) |
| 6 | `pitch_room_system` | `nodes/_otr_pitch_room.py:183` | inline in `build_pitch_prompt` | yes (L185) |
| 7 | `story_select_system` | `nodes/_otr_story_select.py:165` | inline (grader system) | yes (L165) |
| 8 | `dramatic_state_system` | `nodes/_otr_slot_drama_contract.py:304` | `_SLOT_JOB_SYSTEM_PROMPT` | no |
| 9 | `line_grounding_rider` | `nodes/_otr_line_composer.py:1621` | conditional block in `_build_user_prompt` | no |
| 10 | `casting_brief_seam` | `nodes/_otr_casting.py:288` | `_build_user_prompt` (assembles `casting_brief`) | no |
| 11 | `news_coda_system` | `nodes/_otr_line_composer.py:3275` | `_NEWS_CODA_SYSTEM` (+ `_V2_EXAMPLES` at L3297; concat at L3407) | no (news-locked) |
| 12 | `title_system` | `nodes/OTR_LedgerScriptWriter.py:937` | inline `sys_msg` in `generate_title` | yes (L937) |
| 13 | `style_pick_inventor` | `nodes/_otr_style_picker.py:296` | `_INVENTOR_SYSTEM` | yes (L297) |
| 14 | `style_pick_chooser` | `nodes/_otr_style_picker.py:329` | `_CHOOSER_SYSTEM` | yes (L335 in user block) |
| 15 | `labels` | `nodes/otr_meta_brief_image_prompt.py:614` | `_STILL_WORD_GENRE_KEYWORDS` (+ typography L631, backdrop L642) | keys include `"sci-fi"` |

Sites 3, 4, 5 were not in the operator's original 12-site enumeration. They
belong to the outline hierarchy (macro -> phase -> beat) and MUST move
together with `outline_system` or the outline pipeline splits between the two
paradigms. Included in the R2 coding plan as one atomic chunk.

Site 15 (`labels`) is used in `otr_meta_brief_image_prompt.py:715` inside
`for genre, keys in _STILL_WORD_GENRE_KEYWORDS:` -- consumers are image-side,
not dialogue-side. Migration is separable from dialogue sites; treat as its
own chunk.

---

## 4. Extension point already present

`nodes/_otr_creative_prompt_router.py:67`:

```python
def resolve_creative_system_prompt(repo_id: str, phase: Phase) -> str:
    # ...
    if row is not None and row.prompt_profile == "otr_1940s_v1":
        return OTR_PERIOD_SYSTEM_PROMPT
    # ...
```

Dispatches on `row.prompt_profile` from `_otr_model_catalog.py:85` where
`prompt_profile: Literal["modern", "otr_1940s_v1"] = "modern"`.

The transplant EXTENDS this router. Add a `content_profile` field to
`_otr_model_catalog.py` (Literal keys) and generalize the resolver to look up
`(profile_id, site_key)` in a loaded ProfileRegistry rather than branching on
literal profile IDs. Existing `otr_1940s_v1` behavior maps to a new
`radio` content profile with the OTR period prompt as its `outline_system`.

**Backward-compat rule:** the existing `prompt_profile` field stays through
this sprint; a follow-up sprint deprecates it after all catalog rows migrate.

---

## 5. Loader contract

`otr/config/prompt_profile_loader.py` (new module):

- Loads all JSON files under `config/prompt_profiles/` at import time.
- Validates each against the schema (Pydantic v2 or dataclass + jsonschema).
- Registers a singleton `PROFILE_REGISTRY: Dict[str, ProfileBundle]`.
- Public API:
  - `get_prompt(profile_id: str, site_key: str) -> PromptBundle` -- returns
    the fully-resolved prompt for a site, with `sci_fi` fallback for missing
    keys and a WARNING log.
  - `list_profiles() -> List[str]`
  - `list_sites(profile_id: str) -> List[str]`
- On validation failure at import: raise `PromptProfileValidationError` with
  file path and JSON pointer to the offending node. Fail-fast; never fall back
  to Python literals silently.
- `IS_CHANGED`-safe: reads once at module import, then is frozen.

---

## 6. Non-goals (this sprint arc)

- No prompt-content edits. The transplant is byte-for-byte string moves
  (post-normalization for JSON string escaping only: `"` -> `\"`, real
  newlines -> `\n`, tabs -> `\t`, unicode preserved as-is).
- No new profiles beyond the four listed. `cinematic` ships as empty
  scaffolding (`sites: {}`), inheriting sci-fi for every site.
- No changes to `IS_CHANGED`, VRAM, or model-management contracts.
- No changes to `OTR_ENABLE_PITCH_ROOM`, `OTR_GROUNDING_LEVER`, or any other
  env-flag gating. Those remain Python-owned routing decisions.
- No test rewrites beyond adding the golden-fixture regression that asserts
  post-transplant strings equal pre-transplant strings verbatim.

---

## 7. Invariants the plan MUST preserve

The kibitz panel and the QA fanout will verify each of these against the
final plan. Any proposed diff that risks one is rejected.

1. **Audio byte-identical.** Post-transplant audio checksums must match
   pre-transplant on a fixed-seed regression episode. This is the load-bearing
   contract for the entire sprint.
2. **`ROW_KEYED` merge semantics.** Any ledger-row merge behavior in
   `OTR_LedgerScriptWriter.py` is untouched; prompt content moves, merge logic
   does not.
3. **Test asserts in `tests/test_period_prompts.py`.** The three anchor tokens
   (`"1940s"`, `"Suspense"`, `"NARRATOR"`), the `[SFX:` prohibition, and the
   modern-slang blacklist all continue to pass after the migration. These
   tests get parameterized over the loaded `radio` profile (post-transplant)
   rather than the current Python constant, but their assertion set is
   preserved verbatim.
4. **Critic / reroll seam.** `_otr_story_critic.py` and `_otr_ledger_reviewer.py`
   are not in the 15-site set; they must remain unmodified by any diff this
   plan produces.
5. **News-agnostic interpreter.** `news_interpreter.py:704` (`interpret`)
   currently has no profile branching. Post-transplant it still resolves via
   the same code path (loader returns the `news` profile's `interpret` block);
   no new branch is added at the call site.

---

## 8. Sprint chunks (target for R2 coding plan)

R2 will elaborate each chunk into per-file exact-string diffs. R1 lists them
in order:

- **Chunk A:** Create `config/prompt_profiles/{sci_fi,news,cinematic,radio}.json`
  scaffolds. Sci-fi carries lifted content; the other three are minimal
  (news for `interpret` + `news_coda_system`; radio inherits the existing
  OTR period prompt; cinematic empty).
- **Chunk B:** Add `otr/config/prompt_profile_loader.py` with schema,
  validation, and `get_prompt(profile_id, site_key)` API. Unit-tested against
  scaffolds from Chunk A.
- **Chunk C:** Migrate 5 outline sites atomically
  (`outline_system`, `_MACRO`, `_PHASE`, `_BEAT`, plus the outline user-prompt
  builder at L579).
- **Chunk D:** Migrate 4 style / pitch / title sites
  (`_INVENTOR_SYSTEM`, `_CHOOSER_SYSTEM`, `pitch_room_system`, `title_system`).
- **Chunk E:** Migrate 2 news sites (`interpret`, `news_coda_system` including
  the V2 examples concat). This chunk verifies the `sci_fi`-fallback path for
  a profile that DOES override.
- **Chunk F:** Migrate 3 remaining dialogue sites (`dramatic_state_system`,
  `line_grounding_rider`, `casting_brief_seam`).
- **Chunk G:** Migrate `story_select_system` (the grader). Isolated because
  it's a scoring-adjacent site and needs its own audio-invariant verification
  pass.
- **Chunk H:** Migrate `labels` (image-side only). Separable; ships last.
- **Chunk I:** Add golden-fixture regression test that loads each site from
  the profile registry and byte-compares against a snapshot captured on
  pre-transplant HEAD.

Each chunk is one commit. Each commit runs the full test suite locally
before push. Push per chunk to `v2.0-alpha`. No `--no-verify`, no `main`.

---

## 9. Open questions (for kibitz r1-r4 to resolve)

- Should `outline_macro_system`, `outline_phase_system`, `outline_beat_system`
  be nested under a single `outline` sub-object (`outline.system`,
  `outline.macro`, `outline.phase`, `outline.beat`) rather than four flat
  site keys? Impacts schema simplicity vs. loader lookup complexity.
- Does `news_coda_system` need `v2_examples` as a separate string, or is
  concat done in Python and the profile only carries the base + example
  strings independently? (Kibitz r3 wiring detail.)
- Do we unify `prompt_profile` (existing model-catalog field) with
  `content_profile` (new field) in this sprint, or defer to a follow-up?
  Unifying is cleaner but expands blast radius.
- Does `casting_brief_seam` belong in the profile at all, given it's a user
  template with `{name}` / `{style_str}` / `{story_text}` substitution
  slots rather than a system prompt? Options: (a) leave in Python as a
  formatter, (b) move to profile as a template string with a schema-declared
  slot list, (c) split -- carrier stays Python, tokens move to profile.

These questions ARE the r1 arc-hardening target.
