# Multi-Modal Story Schema -- STAGE 1 HARDENED SUB-PLAN (v2, post-kibitz r1)

Date: 2026-07-04. Branch: `v2.0-alpha`. Parent: `docs/multimodal-story-schema/BUILD_PLAN.md`.
Status: DRAFT, hardened through kibitz r1 (codex + antigravity, Claude judge);
advancing r2..r4 then the Fable structural gate, THEN code.

## 0. Operator intent (governs every decision)

> "the current prompts need to more or less survive so i can run the new
> workflow when all is done and it works" -- operator, 2026-07-04.

The science/sci-fi lane must render **the same episode** through the new schema.
So Stage 1 is a *provably-safe, dormant* foundation: it lands the loader + the
first pack + machine-proven byte-identity, and changes NOTHING about how the
sci-fi path runs. The risky part (wiring a live consumer) is quarantined into
**Stage 1b** behind its own gate, because the current code has object-identity
coupling that a naive JSON swap would silently break (kibitz r1, confirmed).

Core law: `JSON owns content+config. Python owns validation/routing/execution.
No fallbacks. No hidden models/engines. Unknown id = hard error.`

## 1. Two safety guarantees (why the sci-fi run cannot move)

1. **Dormant in Stage 1.** No production call site changes. The loader is built,
   validated, and byte-pinned, but not yet consumed. Sci-fi output is byte-for-byte
   identical because no code path changed.
2. **Byte-identity pin.** For every seam authored into the pack, a test
   AST-extracts the live Python constant(s) and asserts the pack value equals the
   **assembled runtime string** (char-exact `str == str`; never compared to
   `bytes`). Drift becomes a RED test, not a silent episode change. (Lab-proven:
   `_sibling-archive/tests/test_phase_a_byte_identity.py` + `extractor.py`.)

When Stage 1b wires the first consumer, the no-fallback rule applies: a MIGRATED
seam that is missing/empty RAISES; the Python constant is the byte-identity ORACLE
(test-time), never a runtime fallback.

## 2. In-repo shape (NO new package)

- **Behavior (one module):** `nodes/_otr_story_pack.py` -- strict **stdlib-only**
  loader + validator (NO pydantic; `news_interpreter.py:66-70` only imports it
  behind a v1 fallback and nothing pins it -- a hand-rolled validator is the
  quiet, dependency-free, v1/v2-proof choice). Zero content literals.
- **Content (data):** `nodes/story_packs/<bank>/<model>.json`. Stage 1 ships one:
  `nodes/story_packs/science_news/science_news_default.json`.
- **No** banks/pipelines/visual_styles in Stage 1 (routing = Stage 2, visual =
  Stage 3). Module file + data dir under `nodes/`, not a new top-level package.

## 3. Contract + loader (`nodes/_otr_story_pack.py`)

`StoryPack` = a hand-rolled dataclass with an explicit strict validator:

- Known field set (rejects unknown top-level keys -> typos fail loud):
  `source_bank_id, story_model_id, story_pipeline_id, label, status,
  prompt_stages, examples, tone_guardrails, forbidden_plot_patterns,
  forbidden_leakage_terms, source_requirements, ledger_validation_notes,
  schema_version`.
- Stage 1 VALIDATES + USES only: `source_bank_id/story_model_id/story_pipeline_id`
  (opaque validated strings -- NO id resolution against banks/pipelines; that
  resolver is Stage 2), `schema_version` (must be in a hardcoded known set, e.g.
  `{"v2.0"}` -> unknown version raises), `prompt_stages` (see section 4). The
  remaining fields are tolerated **inert** (kept for forward-compat, not validated
  or consumed in Stage 1). No leakage scanner in Stage 1.
- `prompt_stages: dict[str,str]` -- every KEY must be in `PRODUCTION_SEAM_ALLOWLIST`
  (section 4); an unknown seam key raises `UnknownSeamError`. Python polices
  container/keys, stays blind to the VALUES.
- Loader:
  - `load_pack(path) -> StoryPack`: read text, `json.loads` with an
    `object_pairs_hook` that **rejects duplicate keys** (json.load silently keeps
    the last -- codex r1), validate, return. Missing file / malformed JSON /
    unknown key / unknown seam / unknown schema_version each raise a typed
    `StoryPackError` naming the path.
  - `get_pack_prompt_or_none(pack, seam) -> str | None`: value if present+non-empty
    else None (for a NOT-YET-migrated seam; caller keeps its Python constant).
  - `get_pack_prompt(pack, seam) -> str`: RAISES if absent/empty (for a MIGRATED
    seam -- no hidden fallback). Seam must be in the allowlist (else raise).
  - Errors: `StoryPackError` (base), `UnknownSeamError`.
- No code from JSON (no eval/import-by-string), no user paths, read-only, no RNG/
  network/VRAM. Determinism + security preserved.

## 4. Seams -- runtime-message granularity (byte-identity target)

kibitz r1 confirmed several runtime prompts are COMPOSITE. Seam keys are defined
at the granularity of the constant that composes the runtime message; the
byte-identity test asserts the ASSEMBLED runtime string equals the pack-assembled
value. `PRODUCTION_SEAM_ALLOWLIST` (exact literal to ship in code):

```
outline_system
line_composer_system
coda_system              coda_examples
announcer_intro_system   announcer_intro_safe_system
announcer_outro_system   announcer_outro_resolved_tail
style_pick_inventor_system   style_pick_inventor_user
style_pick_chooser_system    style_pick_chooser_user
```

Live byte sources (extraction truth -- pull from HERE, not the design docs /
`_sibling-archive` copies / the schema-example placeholders):

| seam | live constant(s) | runtime assembly |
|------|------------------|------------------|
| outline_system | `_otr_outline.py:532` `_SYSTEM_PROMPT` | as-is |
| line_composer_system | `_otr_line_composer.py:1174` `_SYSTEM_PROMPT` | as-is |
| coda_system + coda_examples | `_otr_line_composer.py:3275` `_NEWS_CODA_SYSTEM` + `:3297` `_NEWS_CODA_SYSTEM_V2_EXAMPLES` | joined at `:3407` |
| announcer_intro_system / _safe | `_otr_line_composer.py:2905` `_ANNOUNCER_INTRO_SYSTEM` / `:2926` `_..._SAFE` | routed at `:3195/:3227` |
| announcer_outro_system + _resolved_tail | `_otr_line_composer.py:~2945` `_ANNOUNCER_OUTRO_SYSTEM` + inline tail `:3517` | concatenated at `:3517` |
| style_pick_inventor_system / _user | `_otr_style_picker.py:296/301` `_INVENTOR_SYSTEM` / `_INVENTOR_USER_TEMPLATE` | separate roles |
| style_pick_chooser_system / _user | `_otr_style_picker.py:329/334` `_CHOOSER_SYSTEM` / `_CHOOSER_USER_TEMPLATE` | separate roles |

`PRODUCTION_SEAM_ALLOWLIST` as SHIPPED = the above keys PLUS the broader R1 sec-4
canonical names reserved for later stages (`interpret, pitch_room_system,
dramatic_state_system, story_select_system, line_grounding, casting_brief_seam,
title_system, style_pick_chooser_user_template, labels`) so future stages add
seams without touching allowlist logic. Only the granular keys above are AUTHORED
in the Stage 1 science pack.

STAYS PYTHON in Stage 1 (do NOT extract): `line_grounding` conditional f-strings
(`_otr_line_composer._build_user_prompt` ~1258-1345); the inline source-label/
develop-verb branch (`_otr_outline._build_user_prompt` ~570-577); `NewsBriefs`
interpreter prompt body (Stage 2); visual tails + `_LTX_MOTION_PROMPT_BY_ROLE`
(Stage 3, `visual_style` axis). Validator/critic prompts (`_CONTINUITY`, `_QA`,
`EDITOR_CONSTRAINTS`, `_AUDITOR`, `_CRITIC`) = acknowledged deferred debt, out of
the creative-seam scope.

## 5. Chunking

**Stage 1 = Chunk 1 ONLY (dormant foundation, zero behavior change).**
- `nodes/_otr_story_pack.py`: contract + loader + `PRODUCTION_SEAM_ALLOWLIST` +
  typed errors (section 3).
- `nodes/story_packs/science_news/science_news_default.json`: the granular seam
  keys (section 4) authored byte-for-byte from the live constants (assembled where
  composite).
- Tests (move WITH the schema, same commit): (a) AST byte-identity -- extract each
  live constant, assemble where composite, assert `== pack[seam]`; (b) loader
  fail-loud -- unknown seam key, unknown schema_version, unknown top-level key,
  duplicate key, malformed JSON, missing file each raise; (c) extractor semantics
  -- `get_pack_prompt_or_none` None for absent seam + exact str for present;
  `get_pack_prompt` raises on absent; (d) **workflow no-diff GATE** -- sha256 of
  `workflows/otr_scifi_16gb_full.json` unchanged this chunk.
- NO production call-site change; NO workflow-JSON change.

**Stage 1b (SEPARATE, own kibitz + Fable gate) -- first live consumer.**
Precondition (must land in the same chunk): retire the object-identity coupling --
convert `_otr_outline.py:1847 if resolved is _SYSTEM_PROMPT` and
`test_creative_prompt_router.py:62 out is expected` from `is` to `==`, with a repo_id
-> pack-coordinates map in `_otr_creative_prompt_router.py` (not a scattered
hardcode). Then wire ONE seam (`outline_system` or `line_composer_system`) via
`get_pack_prompt(...)`, proving byte-identity through the existing story +
`test_audio_byte_identical` regressions plus a new equivalence test. This is where
the risk lives, so it is NOT part of Stage 1 acceptance.

## 6. Security posture ("quietest + most secure")

Fail-loud + strict known-field validation; stdlib-only (no dep surface); no code/
paths from JSON; duplicate-key rejection; dormant (a broken/missing pack cannot
alter the sci-fi run in Stage 1 -- there is no consumer). No-fallback in Stage 1b
(migrated seam missing -> raise). Deterministic, read-only.

## 7. Gates per chunk (CLAUDE.md)

Full suite (`.venv` python, `PYTHONUTF8=1`, `pytest -q -p no:cacheprovider`) +
Bug Bible (survival-guide repo, relative path) + B7 -- GREEN. UTF-8 no BOM; AST-parse
touched `.py`; commit AND push per green chunk to `v2.0-alpha`; verify HEAD==origin,
no 0-byte. Workflow-JSON edits (none in Stage 1) go same-commit + re-validate.
prod/main + tags operator-GATED.

## 8. Acceptance for Stage 1 "done"

- `nodes/_otr_story_pack.py` loads+validates the science pack fail-loud (unknown
  seam/version/key/dup/JSON/missing all raise; suite proves it).
- The science pack holds the granular seams byte-identical to the assembled live
  runtime strings (AST pin green).
- Workflow no-diff GATE green; `otr_scifi_16gb_full.json` untouched.
- Foundation ready for Stage 1b (first consumer) + Stage 2 (routing) without
  reopening Stage 1. NOTE for operator expectation: the NEW workflow surface
  (source_bank/visual_style dropdowns) is Stage 2+; Stage 1 is invisible in the JSON.

## 9. Open questions carried into r2..r4

1. Stage 1b first seam: `outline_system` (behind the router + the `is`->`==` +
   overlay-detection fix) vs a consumer with no identity coupling -- which is the
   true lowest-blast-radius pilot?
2. Composite seams (coda/outro): author split keys + assert the joined runtime
   string, or author one pre-joined seam? (Draft: split keys, assert joined.)
3. Confirm no OTHER `is <constant>` identity checks exist on the seams we author
   (grep before Stage 1b).
