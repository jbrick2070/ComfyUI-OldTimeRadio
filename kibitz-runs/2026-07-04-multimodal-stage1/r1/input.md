# Multi-Modal Story Schema -- STAGE 1 HARDENED SUB-PLAN

Date: 2026-07-04. Branch: `v2.0-alpha`. Plan of record parent:
`docs/multimodal-story-schema/BUILD_PLAN.md` (Stage 1 = Content -> JSON foundation).
Status: DRAFT for kibitz (codex + antigravity) + Fable structural gate, THEN code.

## 0. Operator intent (governs every decision here)

> "the current prompts need to more or less survive so i can run the new
> workflow when all is done and it works" -- operator, 2026-07-04.

Stage 1 is the *quietest, most secure* foundation: the science/sci-fi lane must
produce **the same episode** through the new schema. So the mechanism is
**passthrough-by-default + machine-proven byte-identity**, NOT a rewrite of the
prompt text. Nothing about how the sci-fi path renders changes in Stage 1.

Core law (adopted verbatim from `design-reference/`):
```
JSON owns content + configuration. Python owns validation/routing/execution.
No fallbacks. No hidden models/engines. Unknown id = hard error.
```

## 1. The safety mechanism (why the sci-fi run cannot break)

Two guarantees, both proven by tests in the same commit:

1. **Passthrough default.** The loader exposes
   `get_pack_prompt_or_none(bank, model, pipeline, seam) -> str | None`.
   A seam absent/empty in the pack returns `None`, and the production caller
   keeps its **existing Python literal** (`pack_value or <PY_CONST>`). A missing
   or empty override can never silently change output -- worst case it is `None`
   and the current constant wins.
2. **Byte-identity pin.** For every seam we DO move into the pack, a test
   AST-extracts the live Python constant and asserts
   `pack.prompt_stages[seam] == <live constant bytes>`. This is the lab's proven
   `test_phase_a_byte_identity` pattern (archived at
   `_sibling-archive/tests/test_phase_a_byte_identity.py` + `extractor.py`). Any
   future drift between the pack and the code is a RED test, not a silent bug.

Net: extracted seams are the identical bytes; un-extracted seams fall through to
the identical Python literal. Either way the LLM sees the same prompt.

## 2. In-repo shape (NO new package -- operator hard rule)

The lab prototyped this as a standalone `src/upstream_story_lab/` package. We do
NOT bring that package. We build into existing `nodes/`:

- **Loader module (behavior):** `nodes/_otr_story_pack.py` -- one file. Holds the
  strict contract + fail-loud JSON loader + `get_pack_prompt_or_none`. Zero
  content literals (pure behavior). Reuses **pydantic v2** (already a dependency;
  `nodes/news_interpreter.py` uses `BaseModel`) with `extra="forbid"`.
- **Content (data):** `nodes/story_packs/<bank>/<model>.json`. Stage 1 ships
  exactly one: `nodes/story_packs/science_news/science_news_default.json`.
- **No** `banks.json` / `pipelines.json` / `visual_styles/` in Stage 1 -- those are
  the routing (Stage 2) and visual (Stage 3) axes. Stage 1 is story-prompt
  content only. (Keeps the four axes cleanly separated per R1.)

This is a module file + a data dir under `nodes/`, not a new top-level package.

## 3. The contract (`StoryPack`, trimmed to Stage 1)

Pydantic v2 `BaseModel`, `model_config = ConfigDict(extra="forbid")`. Fields
(adapted from `schema-examples/story_packs/science_news/science_news_default.json`):

- `source_bank_id: str`, `story_model_id: str`, `story_pipeline_id: str`
- `label: str`, `status: str` (must be a known value; e.g. `ready_fixture`)
- `prompt_stages: dict[str, str]`
- `examples: list[str]`, `tone_guardrails: list[str]`
- `forbidden_plot_patterns: list[str]`, `forbidden_leakage_terms: list[str]`
- `source_requirements: list[str]`, `ledger_validation_notes: list[str]`
- `schema_version: str` (registry refuses unknown versions -- no silent migration)

Validation (all fail-loud, naming the offending file):
- Unknown top-level key -> pydantic `extra=forbid` raises.
- Every `prompt_stages` KEY must be in `PRODUCTION_SEAM_ALLOWLIST` (section 4);
  an unknown seam name is a hard error. (Python is blind to the VALUES, only
  polices that the container/keys are well-formed -- the "structural parser, not
  a content reader" directive from the audio review.)
- Unknown `schema_version` -> hard error.
- Malformed JSON / missing file -> hard error naming the path.

Deliberately DEFERRED to later stages (do not build in Stage 1): cross-id
validation against banks/pipelines, template-variable allowlist, source packets,
PD-manifest path-safety, visual policy. Stage 1 validates ONE pack's structure.

## 4. Canonical seam allowlist -- design name -> REAL live constant

The R1 seam NAMES are aspirational; the grounding pass mapped them to the actual
live constants (this is the extraction source of truth -- pull bytes from HERE,
never from the design docs or the `_sibling-archive` copies):

| seam (pack key)        | live constant (byte source)                                             |
|------------------------|--------------------------------------------------------------------------|
| `outline_system`       | `nodes/_otr_outline.py:532` `_SYSTEM_PROMPT` (primary; 3 re-defs @1102/1115/1130 are secondary passes) |
| `line_composer_system` | `nodes/_otr_line_composer.py:1174` `_SYSTEM_PROMPT` (static half) |
| `coda_system`          | `nodes/_otr_line_composer.py:3274` `_NEWS_CODA_SYSTEM` |
| `announcer_intro_system`| `nodes/_otr_line_composer.py:~2905` `_ANNOUNCER_INTRO_SYSTEM` (+ `_SAFE` @~2926) |
| `announcer_outro_system`| `nodes/_otr_line_composer.py:~2945` `_ANNOUNCER_OUTRO_SYSTEM` |
| `style_pick_inventor`  | `nodes/_otr_style_picker.py:~296/300` `_INVENTOR_SYSTEM` + `_INVENTOR_USER_TEMPLATE` |
| `style_pick_chooser`   | `nodes/_otr_style_picker.py:~329/334` `_CHOOSER_SYSTEM` + `_CHOOSER_USER_TEMPLATE` |

STAYS PYTHON in Stage 1 (do NOT extract -- conditional/generated, not static prose):
- `line_grounding` per-beat user f-strings (`_otr_line_composer._build_user_prompt`
  ~1258-1345) -- conditional on `conflict_object`. BUILD_PLAN defers this explicitly.
- The inline source-label / develop-verb branch (`_otr_outline._build_user_prompt`
  ~570-577) -- the "labels" set is scattered inline literals today, not constants.
- `NewsBriefs` interpreter prompt body (`news_interpreter.py`) -- interpreter/binding
  lane, Stage 2.
- Visual tails (`_otr_story_brief_helpers.py:331-353`) + `_LTX_MOTION_PROMPT_BY_ROLE`
  (`render_driver.py:529-543`) -- these are `visual_style`, Stage 3.

`PRODUCTION_SEAM_ALLOWLIST` in code = the full canonical seam list from R1 sec 4
(superset), so future stages can add seams without touching the allowlist logic.

## 5. Chunking (each chunk: suite + Bug Bible + B7 green, commit+push)

**Chunk 1 -- dormant foundation (zero behavior change).**
- `nodes/_otr_story_pack.py`: contract + `load_pack(path)` + `get_pack_prompt_or_none`
  + `PRODUCTION_SEAM_ALLOWLIST` + typed errors (`StoryPackError`, `UnknownSeamError`).
- `nodes/story_packs/science_news/science_news_default.json`: real extracted bytes
  for the seams in section 4 that are single clean module constants
  (`outline_system`, `line_composer_system`, `coda_system`,
  `announcer_intro_system`, `announcer_outro_system`, `style_pick_inventor`,
  `style_pick_chooser`). Values are copied byte-for-byte from the live constants.
- Tests (move WITH the schema): (a) AST byte-identity -- extract each live constant,
  assert `== pack[seam]`; (b) loader fail-loud -- unknown seam key, unknown
  schema_version, extra top-level key, malformed JSON, missing file each raise;
  (c) `get_pack_prompt_or_none` returns None for an absent seam and the exact bytes
  for a present one.
- NO production call-site change. NO workflow-JSON change. Sci-fi run is byte-for-byte
  untouched (the loader is validated but not yet consumed). This is the safest
  possible "byte-identical start".

**Chunk 2 -- wire the FIRST consumer (still byte-identical).**
- Single integration seam: `nodes/_otr_creative_prompt_router.py` already centralizes
  the outline + line-composer system prompts (`_MODERN_BY_PHASE`). Change ONLY that
  lookup to `get_pack_prompt_or_none(...) or <existing _MODERN_* constant>`.
  PRESERVE the `otr_1940s_v1 -> OTR_PERIOD_SYSTEM_PROMPT` swap branch exactly (period
  path is untouched; only the science/modern default consults the pack).
- Prove byte-identity: existing story + `test_audio_byte_identical` regressions stay
  green, PLUS a new equivalence test asserting the router returns the identical
  string with vs without the pack present.
- Still NO workflow-JSON change (no node/widget added -- the loader is called inside
  an existing node's code path). Re-confirm the canonical JSON is untouched (R4).

**Chunk 3+ -- extend consumers seam by seam** (coda, announcer, style-picker), each
its own byte-identical passthrough wire + equivalence test. Order chosen for lowest
blast radius; coda/announcer touch the no-fallback raise sites (2026-07-03 rip) so
they get extra care + a Fable spot-check.

## 6. Security posture ("quietest + most secure")

- **Fail-loud, `extra=forbid`:** malformed/unknown/duplicate content raises naming
  the file; no fallback, no invented prose.
- **No code from JSON:** Stage 1 packs are pure data -- no binding names executed,
  no `eval`, no import-by-string (interpreters/bindings are Stage 2, still an
  explicit allowlist then).
- **No user file paths:** Stage 1 loads only the in-repo shipped pack; PD-manifest
  path-safety (absolute/`..`/symlink-escape) lands with the PD lane in Stage 2.
- **Blast-radius floor:** passthrough default means a broken/missing pack cannot
  quietly alter the sci-fi output -- it either raises loudly or yields the current
  Python literal.
- **Determinism preserved:** loader is pure/read-only; no RNG, no network, no VRAM.

## 7. Gates per chunk (CLAUDE.md)

- Full suite (`.venv` python, `PYTHONUTF8=1`, `pytest -q -p no:cacheprovider`) +
  Bug Bible (survival-guide repo, relative path) + B7 forbidden-sweep -- GREEN.
- UTF-8 no BOM; AST-parse touched `.py`; commit AND push per green chunk to
  `v2.0-alpha`; verify HEAD==origin, no 0-byte files. Do NOT push unprompted beyond
  the standing per-green-chunk rule; prod/main + tags stay operator-GATED.
- Any workflow-JSON edit (none expected in Stage 1) goes in the SAME commit +
  re-validate (`OTR_WorkflowValidator` + round-trip + link/widget audit).

## 8. Acceptance for Stage 1 "done"

- `nodes/_otr_story_pack.py` loads + validates the science pack fail-loud; unknown
  seam/version/key/JSON all raise; suite proves it.
- The science pack holds the extracted seams **byte-identical** to the live
  constants (AST pin test green).
- At least one production consumer (Chunk 2 router seam) reads prompts through the
  loader with a proven-identical result; sci-fi episode output unchanged
  (`test_audio_byte_identical` + story regressions green).
- Canonical `workflows/otr_scifi_16gb_full.json` untouched.
- Foundation is ready for Stage 2 (banks/pipelines routing + new lanes) to build on
  without reopening Stage 1.

## 9. Open questions for the panel (kibitz)

1. Wire in Chunk 1 or keep Chunk 1 fully dormant (this draft: dormant -> safest)?
2. First consumer = `_otr_creative_prompt_router` outline/line seam -- is that the
   lowest-blast-radius integration point, or is a single self-contained constant
   (e.g. `coda_system`) a cleaner pilot despite touching the no-fallback raise site?
3. pydantic `extra=forbid` StoryPack vs a hand-rolled strict dataclass validator --
   pydantic is already a dep (news_interpreter) and gives us `extra=forbid` free;
   confirm no import-time cost concern in the node package.
4. Should `get_pack_prompt_or_none` raise on an unknown BANK/MODEL/PIPELINE triple
   (lab behavior) or also passthrough in Stage 1? (This draft: raise on unknown
   triple = loud; None only for an absent SEAM within a known pack.)
